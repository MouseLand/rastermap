from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
from scipy.stats import zscore
from . import io

# ------ MENU BAR -----------------
def mainmenu(parent): 
    loadMat =  QtGui.QAction("&Load data matrix", parent)
    loadMat.setShortcut("Ctrl+L")
    loadMat.triggered.connect(lambda: io.load_mat(parent, name=None))
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
    parent.loadBeh.triggered.connect(io.load_run_data)
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

