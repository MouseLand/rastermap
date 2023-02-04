from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QAction
import pyqtgraph as pg
import numpy as np
from . import io, run

# ------ MENU BAR -----------------
def mainmenu(parent): 
    # make mainmenu!
    main_menu = parent.menuBar()
    
    file_menu = main_menu.addMenu("&File")
    
    loadMat =  QAction("&Load data matrix", parent)
    loadMat.setShortcut("Ctrl+L")
    loadMat.triggered.connect(lambda: io.load_mat(parent, name=None))
    parent.addAction(loadMat)
    file_menu.addAction(loadMat)
    
    parent.loadOne =  QAction("Load &one-d variable", parent)
    parent.loadOne.setShortcut("Ctrl+O")
    parent.loadOne.triggered.connect(lambda: io.load_oned_data(parent))
    parent.loadOne.setEnabled(False)
    parent.addAction(parent.loadOne)
    file_menu.addAction(parent.loadOne)
    
    parent.loadNd =  QAction("Load &n-d variable (times or cont.)", parent)
    parent.loadNd.setShortcut("Ctrl+N")
    parent.loadNd.triggered.connect(lambda: io.load_behav_file(parent))
    parent.loadNd.setEnabled(False)
    parent.addAction(parent.loadNd)
    file_menu.addAction(parent.loadNd)

    # load processed data
    parent.loadProc = QAction("&Load processed data", parent)
    parent.loadProc.setShortcut("Ctrl+P")
    parent.loadProc.triggered.connect(lambda: io.load_proc(parent, name=None))
    parent.addAction(parent.loadProc)
    file_menu.addAction(parent.loadProc)    
    
    # export figure
    exportFig = QAction("Export as image (svg)", parent)
    exportFig.triggered.connect(lambda: export_fig(parent))
    exportFig.setEnabled(True)
    parent.addAction(exportFig)

    run_menu = main_menu.addMenu("&Run")
    # Save processed data
    parent.runRmap = QAction("&Run rastermap", parent)
    parent.runRmap.setShortcut("Ctrl+R")
    parent.runRmap.triggered.connect(lambda: run.RunWindow(parent))
    parent.runRmap.setEnabled(False)
    parent.addAction(parent.runRmap)
    run_menu.addAction(parent.runRmap)

    save_menu = main_menu.addMenu("&Save")
    # Save processed data
    parent.saveProc = QAction("&Save processed data", parent)
    parent.saveProc.setShortcut("Ctrl+S")
    parent.saveProc.triggered.connect(lambda: io.save_proc(parent))
    parent.addAction(parent.saveProc)
    save_menu.addAction(parent.saveProc) 

def export_fig(parent):
    parent.win.scene().contextMenuItem = parent.p0
    parent.win.scene().showExportDialog()

