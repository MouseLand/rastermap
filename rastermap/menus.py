from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
from scipy.stats import zscore

# ------ MENU BAR -----------------
def mainmenu(parent): 
    loadMat =  QtGui.QAction("&Load data matrix", parent)
    loadMat.setShortcut("Ctrl+L")
    loadMat.triggered.connect(lambda: load_mat(parent, name=None))
    parent.addAction(loadMat)
    # run rastermap from scratch
    parent.runRMAP = QtGui.QAction("&Run embedding algorithm", parent)
    parent.runRMAP.setShortcut("Ctrl+R")
    parent.runRMAP.triggered.connect(parent.run_RMAP)
    parent.addAction(parent.runRMAP)
    parent.runRMAP.setEnabled(False)
    # load processed data
    loadProc = QtGui.QAction("&Load processed data", parent)
    loadProc.setShortcut("Ctrl+P")
    loadProc.triggered.connect(lambda: parent.load_proc(name=None))
    parent.addAction(loadProc)
    # load a behavioral trace
    parent.loadBeh = QtGui.QAction(
        "Load behavior or stim trace (1D only)", parent
    )
    parent.loadBeh.triggered.connect(parent.load_behavior)
    parent.loadBeh.setEnabled(False)
    parent.addAction(parent.loadBeh)
    # export figure
    exportFig = QtGui.QAction("Export as image (svg)", parent)
    exportFig.triggered.connect(export_fig)
    exportFig.setEnabled(True)
    parent.addAction(exportFig)

    # make mainmenu!
    main_menu = parent.menuBar()
    file_menu = main_menu.addMenu("&File")
    file_menu.addAction(loadMat)
    #file_menu.addAction(loadProc)
    file_menu.addAction(parent.runRMAP)
    file_menu.addAction(parent.loadBeh)
    file_menu.addAction(exportFig)

def export_fig(parent):
    parent.win.scene().contextMenuItem = parent.p0
    parent.win.scene().showExportDialog()

def load_mat(parent, name=None):
    if name is None:
        name = QtGui.QFileDialog.getOpenFileName(
            parent, "Open *.npy", filter="*.npy"
        )
        parent.fname = name[0]
        parent.filebase = name[0]
    else:
        parent.fname = name
        parent.filebase = name
    try:
        print("Loading", parent.fname)
        X = np.load(parent.fname)
        print(X.shape)
    except (ValueError, KeyError, OSError,
            RuntimeError, TypeError, NameError):
        print('ERROR: this is not a *.npy array :( ')
        X = None
    if X is not None and X.ndim > 1:
        iscell, file_iscell = parent.load_iscell()
        parent.file_iscell = None
        if iscell is not None:
            if iscell.size == X.shape[0]:
                X = X[iscell, :]
                parent.file_iscell = file_iscell
                print('using iscell.npy in folder')
        if len(X.shape) > 2:
            X = X.mean(axis=-1)
        parent.p0.clear()
        parent.sp = zscore(X, axis=1)
        del X
        parent.sp = np.maximum(-4, np.minimum(8, parent.sp)) + 4
        parent.sp /= 12
        parent.embedding = np.arange(0, parent.sp.shape[0]).astype(np.int64)[:,np.newaxis]
        parent.sorting = np.arange(0, parent.sp.shape[0]).astype(np.int64)
        
        parent.loaded = True
        parent.plot_activity()
        parent.show()
        parent.runRMAP.setEnabled(True)
        parent.loadBeh.setEnabled(True)
        print('done loading')