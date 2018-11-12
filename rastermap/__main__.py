from rastermap import Rastermap
from rastermap import gui
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
    args = parser.parse_args()

    if len(args.S)>0:
        S = np.load(args.S)
        print(S.shape)
        ops = np.load(args.ops)
        ops = ops.item()
        model = Rastermap(ops['n_components'], ops['n_X'], ops['n_Y'], ops['nPC'],
                          ops['sig_Y'], ops['init'], ops['alpha'], ops['K'])
        embedding = model.fit_transform(S)
        proc  = {'embedding': embedding, 'ops': ops, 'filename': args.S}
        basename, fname = os.path.split(args.S)
        np.save(os.path.join(basename, 'embedding.npy'), proc)
    else:
        gui.run()
