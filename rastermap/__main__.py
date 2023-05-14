from scipy.stats import zscore
import numpy as np
import argparse
import os

try:
    from rastermap.gui import gui
    GUI_ENABLED = True
except ImportError as err:
    GUI_ERROR = err
    GUI_ENABLED = False
    GUI_IMPORT = True
except Exception as err:
    GUI_ENABLED = False
    GUI_ERROR = err
    GUI_IMPORT = False
    raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="spikes")
    parser.add_argument("--S", default=[], type=str, help="spiking matrix")
    parser.add_argument("--proc", default=[], type=str,
                        help="processed data file 'embedding.npy'")
    parser.add_argument("--ops", default=[], type=str, help="options file 'ops.npy'")
    parser.add_argument("--iscell", default=[], type=str,
                        help="which cells to select for processing")
    args = parser.parse_args()

    if len(args.ops) > 0 and len(args.S) > 0:
        from rastermap.mapping import Rastermap
        S = np.load(args.S)
        ops = np.load(args.ops, allow_pickle=True).item()
        if len(args.iscell) > 0:
            iscell = np.load(args.iscell)
            if iscell.ndim > 1:
                iscell = iscell[:, 0].astype("bool")
            else:
                iscell = iscell.astype("bool")
            if iscell.size == S.shape[0]:
                S = S[iscell, :]
                print("iscell found and used to select neurons")
        print("size of rastermap matrix")
        print(S.shape)
        if len(S.shape) > 2:
            S = S.mean(axis=-1)
        S = zscore(S, axis=1)
        model = Rastermap(**ops)

        if ("end_time" in ops and ops["end_time"] == -1) or "end_time" not in ops:
            ops["end_time"] = S.shape[1]
            ops["start_time"] = 0
        train_time = np.zeros((S.shape[1],)).astype(bool)
        print(train_time.shape)
        train_time[np.arange(ops["start_time"], ops["end_time"]).astype(int)] = 1
        model.fit(S[:, train_time])

        proc = {
            "embedding": model.embedding,
            "isort": model.isort,
            "U": model.U,
            "V": model.V,
            "ops": ops,
            "filename": args.S,
            "train_time": train_time
        }
        basename, fname = os.path.split(args.S)
        fname = os.path.splitext(fname)[0]
        try:
            np.save(os.path.join(basename, f"{fname}_embedding.npy"), proc)
        except Exception as e:
            print("ERROR: no permission to write to data folder")
            #os.path.dirname(args.ops)
            np.save("embedding.npy", proc)
    else:
        if not GUI_ENABLED:
            print("GUI ERROR: %s" % GUI_ERROR)
            if GUI_IMPORT:
                print(
                    "GUI FAILED: GUI dependencies may not be installed, to install, run"
                )
                print("     pip install rastermap[gui]")
        else:
            # use proc path if it exists, else use S path
            filename = args.proc if len(args.proc) > 0 else None
            filename = args.S if len(args.S) > 0 and filename is None else filename
            gui.run(filename=filename, proc=len(args.proc) > 0)
