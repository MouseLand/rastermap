import os, sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore, mode
import torch
from torch import nn
from rastermap import Rastermap , utils

from rl_zoo3.utils import ALGOS
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


# setup model with learning_rate OFF
custom_objects = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0,
    "optimize_memory_usage": False
}

vis_cnn = {}
def hook_fn_cnn(m, i, o):
    vis_cnn[m] = o.detach().clone().cpu()

vis_mlp = {}
def hook_fn_mlp(m, i, o):
    vis_mlp[m] = o.detach().clone().cpu()

def get_all_layers(model, hook_fn):
    for name, layer in model._modules.items():
        if isinstance(layer, nn.Sequential):
            get_all_layers(layer, hook_fn)
        else:
            layer.register_forward_hook(hook_fn)

def get_all_activations_qrdqn(cnn_net, mlp_net, obs_torch):
    ilayer = np.zeros((0,), "int")
    cnn = np.zeros((0,), np.float32)
    mlp = np.zeros((0,), np.float32)
    out = cnn_net(obs_torch)
    out2 = mlp_net(out)
    i = 0
    for layer_name in vis_cnn.keys():
        if "ReLU()" in str(layer_name):
            act = vis_cnn[layer_name].flatten().detach().numpy()
            cnn = np.concatenate((cnn, act), axis=0)
            ilayer = np.concatenate((ilayer, i*np.ones(len(act), "int")), axis=0)
            i += 1
    mlp = out2.cpu().flatten().detach().numpy()
    ilayer = np.concatenate((ilayer, i*np.ones(len(mlp), "int")), axis=0)
    return cnn, mlp, ilayer

def run_qrdqn(model_folder, root, env_id, n_seeds=10, device=torch.device("cuda")):
    algo = "qrdqn"

    log_path = os.path.join(model_folder, algo, env_id+"_1")
    model_path = os.path.join(log_path, f"{env_id}.zip")

    print(f"using Atari env {env_id}")
    env = make_atari_env(env_id, n_envs=1, seed=0)
    # Frame-stacking with 4 frames
    env = VecFrameStack(env, n_stack=4)
    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects)
    cnn_net = model.policy.quantile_net.features_extractor
    print(cnn_net)
    mlp_net = model.policy.quantile_net.quantile_net
    print(mlp_net)
    get_all_layers(cnn_net, hook_fn_cnn)
        
    spks, eps_len, actions = [], [], []
    for seed in range(n_seeds):
        print(f"seed {seed}")
        env = make_atari_env(env_id, n_envs=1, seed=seed)
        # Frame-stacking with 4 frames
        env = VecFrameStack(env, n_stack=4)
        obs = env.reset()

        deterministic = False
        state = None
        episode_reward = 0.0
        ep_len = 0
        n_iterations = 4000  

        # don't show game
        render = False

        obs_all, actions_all, states_all = [], [], []
        iteration = 0
        for _ in range(n_iterations):
            
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs_torch = torch.from_numpy(obs.transpose(0,3,1,2).astype(np.float32).copy()).to(device)
            preprocessed_obs = preprocess_obs(obs_torch, 
                                                model.policy.observation_space, 
                                                normalize_images=model.policy.normalize_images)
            obs_all.append(env.get_images()[0])
            #actions = torch.from_numpy(action[np.newaxis,:]).to(device)
            #out = cnn_net(preprocessed_obs)
            cnn, mlp, ilayer = get_all_activations_qrdqn(cnn_net, mlp_net, preprocessed_obs)
                
            if iteration==0:
                activations = np.zeros((len(cnn)+len(mlp), n_iterations), "float32")
                print(activations.shape, ilayer.max()+1)
            activations[:len(cnn), iteration] = cnn
            activations[len(cnn):, iteration] = mlp
                
            obs, reward, done, infos = env.step(action)
            
            if render:
                env.render("human")

            episode_reward += reward[0]
            ep_len += 1
            iteration += 1
            actions_all.append(action)
            states_all.append(state)

            if infos is not None:
                episode_infos = infos[0].get("episode")
                if episode_infos is not None:
                    print(f"Atari Episode Score: {episode_infos['r']:.2f}")
                    print("Atari Episode Length", episode_infos["l"])
                    break;
        activations = activations[:, 4:iteration]
        env.close()
        spks.append(activations)
        actions.append(np.array(actions_all))
        eps_len.append(activations.shape[1])

    obs = np.stack(tuple(obs_all), axis=0).squeeze()[4:]
    spks = np.concatenate(tuple(spks), axis=1)
    actions = np.concatenate(tuple(actions), axis=0)
    eps_len = np.array(eps_len)

    np.savez(os.path.join(root, "simulations/", f"qrdqn_{env_id}.npz"),
            spks=spks, ilayer=ilayer, obs=obs, actions=actions,
            eps_len=eps_len)

def sort_spks(root, env_id):
    dat = np.load(os.path.join(root, "simulations/", f"qrdqn_{env_id}.npz"))
    spks = dat["spks"]
    obs = dat["obs"]
    ilayer = dat["ilayer"]
    eps_len = dat["eps_len"]

    x_std = spks.std(axis=1)
    igood = x_std > 1e-3
    print(igood.mean())

    S = zscore(spks[igood], axis=1)
    rm_model = Rastermap(time_lag_window=10, locality=0.75).fit(S[:,:])
    isort = rm_model.isort

    # show last episode
    bin_size = 50
    X_embedding = zscore(utils.bin1d(S[:,-eps_len[-1]:][isort], bin_size, axis=0), axis=1)
    #if env_id=="EnduroNoFrameskip-v4":
    #    X_embedding = X_embedding[:,780:]
    #    obs = obs[780:]
    nn, nt = X_embedding.shape
    if env_id=="EnduroNoFrameskip-v4":
        nt = nt-900
        iframes = np.linspace(780 + nt*0.1, 780 + nt*0.9, 4).astype("int")
    else:
        iframes = np.linspace(nt*0.1, nt*0.9, 4).astype("int")
    print(iframes)
    emb_layer = mode(ilayer[igood][isort][:nn*bin_size].reshape(-1, bin_size), axis=1, keepdims=False).mode
    ex_frames = obs[iframes]
    print(ex_frames.shape)
    np.savez(os.path.join(root, "simulations/", f"qrdqn_{env_id}_results.npz"),
             X_embedding=X_embedding, emb_layer=emb_layer, 
             ex_frames=ex_frames, iframes=iframes)