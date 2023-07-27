"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
from qtpy import QtGui, QtCore, QtWidgets
from qtpy.QtWidgets import QAction
import pyqtgraph as pg
import numpy as np
from . import io, run, gui, views


# ------ MENU BAR -----------------
def mainmenu(parent):
    # make mainmenu!
    main_menu = parent.menuBar()

    file_menu = main_menu.addMenu("&File")

    loadMat = QAction("&Load data matrix", parent)
    loadMat.setShortcut("Ctrl+L")
    loadMat.triggered.connect(lambda: io.load_mat(parent, name=None))
    parent.addAction(loadMat)
    file_menu.addAction(loadMat)

    parent.loadXY = QAction("&Load xy(z) positions of neurons", parent)
    parent.loadXY.setShortcut("Ctrl+X")
    parent.loadXY.triggered.connect(lambda: io.load_neuron_pos(parent))
    parent.addAction(parent.loadXY)
    file_menu.addAction(parent.loadXY)

    # load Z-stack
    parent.loadProc = QAction("&Load z-stack (mean images)", parent)
    parent.loadProc.setShortcut("Ctrl+Z")
    parent.loadProc.triggered.connect(lambda: io.load_zstack(parent, name=None))
    parent.addAction(parent.loadProc)
    file_menu.addAction(parent.loadProc)

    parent.loadNd = QAction("Load &n-d variable (times or cont.)", parent)
    parent.loadNd.setShortcut("Ctrl+N")
    parent.loadNd.triggered.connect(lambda: io.get_behav_data(parent))
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

    #view_menu = main_menu.addMenu("&Views")
    #parent.view3D = QAction("&View multi-plane data", parent)
    #parent.view3D.setShortcut("Ctrl+V")
    #parent.view3D.triggered.connect(lambda: plane_window(parent))
    #parent.view3D.setEnabled(False)
    #parent.addAction(parent.view3D)
    #view_menu.addAction(parent.view3D)

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
