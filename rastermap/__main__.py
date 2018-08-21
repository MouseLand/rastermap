from rastermap import mapping
import numpy as np

def main():
    S = np.load('spks.npy')
    mapping.main(S)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='spikes')
    parser.add_argument('--S', default=[], type=str, help='spiking matrix')
    args = parser.parse_args()

    if len(args.S)>0:
        S = np.load(args.S)
        mapping.main(S)
