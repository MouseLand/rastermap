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
        ops = np.load(args.ops)
        ops = ops.item()
        if len(args.iscell) > 0:
            iscell = np.load(args.iscell)
            if iscell.ndim > 1:
                iscell = iscell[:, 0].astype(np.bool)
            else:
                iscell = iscell.astype(np.bool)
            if iscell.size == S.shape[0]:
                S = S[iscell, :]
                print('iscell found and used to select neurons')
        print(S.shape)
        S = zscore(S,axis=1)
        model = Rastermap(ops['n_components'], ops['n_X'], ops['n_Y'], ops['nPC'],
                          ops['sig_Y'], ops['init'], ops['alpha'], ops['K'])
        model.fit(S)
        proc  = {'embedding': model.embedding, 'usv': [model.u, model.sv, model.v],
                 'cmap': model.cmap, 'A': model.A, 'ops': ops, 'filename': args.S}
        basename, fname = os.path.split(args.S)
        np.save(os.path.join(basename, 'embedding.npy'), proc)
    else:
        gui.run()
