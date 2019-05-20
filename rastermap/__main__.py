from rastermap import Rastermap
from rastermap import gui
from scipy.stats import zscore
import numpy as np
import argparse
import os

def main():
    S = np.load('spks.npy')
    model = RMAP()
    return model.fit_transform(S)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='spikes')
    parser.add_argument('--S', default=[], type=str, help='spiking matrix')
    parser.add_argument('--ops', default=[], type=str, help='options file')
    parser.add_argument('--iscell', default=[], type=str, help='which cells to use')
    args = parser.parse_args()

    if len(args.S)>0:
        S = np.load(args.S)
        ops = np.load(args.ops, allow_pickle=True).item()
        if len(args.iscell) > 0:
            iscell = np.load(args.iscell)
            if iscell.ndim > 1:
                iscell = iscell[:, 0].astype(np.bool)
            else:
                iscell = iscell.astype(np.bool)
            if iscell.size == S.shape[0]:
                S = S[iscell, :]
                print('iscell found and used to select neurons')
        print('size of rastermap matrix')
        print(S.shape)
        if len(S.shape) > 2:
            S = S.mean(axis=-1)
        S = zscore(S,axis=1)
        ops['mode'] = 'basic'
        model = Rastermap(n_components=ops['n_components'], n_X=ops['n_X'], nPC=ops['nPC'],
                          init=ops['init'], alpha=ops['alpha'], K=ops['K'], constraints=ops['constraints'],
                          annealing=ops['annealing'])
        if ops['end_time']==-1:
            ops['end_time'] = S.shape[1]
            ops['start_time'] = 0
        train_time = np.zeros((S.shape[1],)).astype(bool)
        print(train_time.shape)
        train_time[np.arange(ops['start_time'], ops['end_time']).astype(int)] = 1
        model.fit(S[:, train_time])
        proc  = {'embedding': model.embedding, 'uv': [model.u, model.v],
                 'ops': ops, 'filename': args.S, 'train_time': train_time}
        basename, fname = os.path.split(args.S)
        np.save(os.path.join(basename, 'embedding.npy'), proc)
        #os.path.dirname(args.ops)
        np.save('embedding.npy', proc)
    else:
        gui.run()
