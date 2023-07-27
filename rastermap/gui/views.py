"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
from qtpy import QtGui, QtCore, QtWidgets
from qtpy.QtWidgets import QMainWindow, QApplication, QDialog, QWidget, QScrollBar, QSlider, QComboBox, QGridLayout, QPushButton, QFrame, QCheckBox, QLabel, QProgressBar, QLineEdit, QMessageBox, QGroupBox, QStyle, QStyleOptionSlider
import pyqtgraph as pg
from pyqtgraph import functions as fn
from pyqtgraph import Point
import numpy as np

from . import colormaps

nclust_max = 100


class PlaneWindow(QMainWindow):

    def __init__(self, parent=None):
        super(PlaneWindow, self).__init__(parent)
        self.parent = parent
        self.setGeometry(50, 50, 800, 800)
        self.setWindowTitle("view neurons multi-plane")
        self.win = QWidget(self)
        self.layout = QGridLayout()
        self.layout.setHorizontalSpacing(25)
        self.win.setLayout(self.layout)

        self.cwidget = QWidget(self)
        self.setCentralWidget(self.cwidget)
        self.l0 = QGridLayout()
        self.cwidget.setLayout(self.l0)
        self.win = pg.GraphicsLayoutWidget()
        self.l0.addWidget(self.win, 0, 0, 1, 1)
        layout = self.win.ci.layout

        self.menuBar().clear()

        neuron_pos = self.parent.neuron_pos.copy()
        if neuron_pos.shape[-1] == 3:
            y, x, z = neuron_pos.T
        else:
            y, x = neuron_pos.T
            z = np.zeros_like(y)

        zgt, z = np.unique(z, return_inverse=True)
        zgt = zgt.astype("int")
        n_planes = z.max() + 1
        if n_planes > 30:
            bins = np.linspace(0, n_planes + 1, 30)
            bin_centers = bins[:-1] + np.diff(bins)[0] / 2
            zgt = zgt[bin_centers.astype(int)]
            z = np.digitize(z, bins)
            z = np.unique(z, return_inverse=True)[1].astype(int)
        n_planes = z.max() + 1

        Ly, Lx = y.max(), x.max()

        nX = np.ceil(np.sqrt(float(Ly) * float(Lx) * n_planes) / float(Lx))
        nX = int(nX)
        nY = n_planes // nX
        print(n_planes, nX, nY)
        self.x, self.y, self.z = x, y, z
        self.nX, self.nY = nX, nY
        self.plots = []
        self.scatter_plots = []
        self.imgs = []

        self.parent.PlaneWindow = self

        self.all_neurons_checkBox = self.parent.all_neurons_checkBox
        self.embedding = self.parent.embedding
        self.sorting = self.parent.sorting
        self.cluster_rois = self.parent.cluster_rois
        self.cluster_slices = self.parent.cluster_slices
        self.colors = self.parent.colors
        self.smooth_bin = self.parent.smooth_bin
        self.zstack = self.parent.zstack

        for ii in range(self.nY):
            for jj in range(self.nX):
                iplane = ii * self.nX + jj
                self.plots.append(
                    self.win.addPlot(title=f"z = {iplane}", row=ii, col=jj, rowspan=1,
                                     colspan=1))
                self.scatter_plots.append([])
                self.imgs.append(pg.ImageItem())
                self.plots[-1].addItem(self.imgs[-1])
                if self.zstack is not None:
                    self.imgs[-1].setImage(self.zstack[:, :, zgt[iplane]])
                for i in range(nclust_max + 1):
                    self.scatter_plots[-1].append(pg.ScatterPlotItem())
                    self.plots[-1].addItem(self.scatter_plots[-1][-1])

        self.update_plots()
        self.win.show()
        self.show()

    def update_plots(self, roi_id=None):
        for ii in range(self.nY):
            for jj in range(self.nX):
                iplane = ii * self.nX + jj
                ip = self.z == iplane
                self.plot_scatter(self.x[ip], self.y[ip], iplane=iplane, neurons=ip,
                                  roi_id=roi_id)
                self.plots[iplane].show()

    def neurons_selected(self, selected=None, neurons=None):
        selected = selected if selected is not None else self.selected
        neurons_select = np.zeros(len(self.sorting), "bool")
        neurons_select[self.sorting[selected.start * self.smooth_bin:selected.stop *
                                    self.smooth_bin]] = True
        if neurons is not None:
            neurons_select = neurons_select[neurons]
        return neurons_select

    def plot_scatter(self, x, y, roi_id=None, iplane=0, neurons=None):
        if self.all_neurons_checkBox.isChecked() and roi_id is None:
            colors = colormaps.gist_ncar[np.linspace(
                0, 254, len(x)).astype("int")][self.sorting]
            brushes = [pg.mkBrush(color=c) for c in colors]
            self.scatter_plots[iplane][0].setData(x, y, symbol="o", brush=brushes,
                                                  hoverable=True)
            for i in range(1, nclust_max + 1):
                self.scatter_plots[iplane][i].setData([], [])
        else:
            if roi_id is None:
                self.scatter_plots[iplane][0].setData(
                    x, y, symbol="o", brush=pg.mkBrush(color=(180, 180, 180)),
                    hoverable=True)
                for roi_id in range(nclust_max):
                    if roi_id < len(self.cluster_rois):
                        selected = self.neurons_selected(self.cluster_slices[roi_id],
                                                         neurons=neurons)
                        self.scatter_plots[iplane][roi_id + 1].setData(
                            x[selected], y[selected], symbol="o",
                            brush=pg.mkBrush(color=self.colors[roi_id][:3]),
                            hoverable=True)
                    else:
                        self.scatter_plots[iplane][roi_id + 1].setData([], [])
            else:
                selected = self.neurons_selected(self.cluster_slices[roi_id],
                                                 neurons=neurons)
                self.scatter_plots[iplane][roi_id + 1].setData(
                    x[selected], y[selected], symbol="o",
                    brush=pg.mkBrush(color=self.colors[roi_id][:3]), hoverable=True)
