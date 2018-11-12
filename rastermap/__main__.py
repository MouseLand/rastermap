from rastermap import RMAP
from rastermap import gui
import numpy as np
import argparse

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
        model = RMAP()
        embedding = model.fit_transform(S)
        np.save('embedding.npy', embedding)
    else:
        gui.run()
