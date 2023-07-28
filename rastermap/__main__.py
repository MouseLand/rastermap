"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
from scipy.stats import zscore
import numpy as np
import argparse
import os
from rastermap import Rastermap
from rastermap.io import load_activity

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
        X, Usv, Vsv, xy = load_activity(args.S)
        ops = np.load(args.ops, allow_pickle=True).item()
        if len(args.iscell) > 0:
            iscell = np.load(args.iscell)
            if iscell.ndim > 1:
                iscell = iscell[:, 0].astype("bool")
            else:
                iscell = iscell.astype("bool")
            if iscell.size == X.shape[0]:
                X = X[iscell, :]
                print("iscell found and used to select neurons")
        
        if Usv is not None and Usv.ndim==3:
            Usv = Usv.reshape(-1, Usv.shape[-1])
        
        model = Rastermap(**ops)
        train_time = np.ones(X.shape[1] if X is not None 
                             else Vsv.shape[0], "bool")
        if X is not None:
            if ("end_time" in ops and ops["end_time"] == -1) or "end_time" not in ops:
                ops["end_time"] = X.shape[1]
                ops["start_time"] = 0
            else:
                train_time = np.zeros(X.shape[1], "bool")
                train_time[np.arange(ops["start_time"], ops["end_time"]).astype(int)] = 1
                X = X[:, train_time]

        model.fit(data=X, Usv=Usv, Vsv=Vsv)

        proc = {
            "filename": args.S,
            "save_path": os.path.split(args.S)[0],
            "isort": model.isort,
            "embedding": model.embedding,
            "user_clusters": None,
            "ops": ops,
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
