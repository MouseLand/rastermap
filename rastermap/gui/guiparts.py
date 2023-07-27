"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
from qtpy import QtGui, QtCore, QtWidgets
from qtpy.QtWidgets import QMainWindow, QApplication, QWidget, QScrollBar, QSlider, QComboBox, QGridLayout, QPushButton, QFrame, QCheckBox, QLabel, QProgressBar, QLineEdit, QMessageBox, QGroupBox, QStyle, QStyleOptionSlider
import pyqtgraph as pg
from pyqtgraph import functions as fn
from pyqtgraph import Point
import numpy as np


class TimeROI(pg.LinearRegionItem):

    def __init__(self, parent=None, color=[128, 128, 255, 50], bounds=[0, 100],
                 roi_id=0):
        self.color = color
        self.parent = parent
        self.pen = pg.mkPen(pg.mkColor(*self.color), width=2, style=QtCore.Qt.SolidLine)
        self.brush = pg.mkBrush(pg.mkColor(*self.color[:3], 50))
        self.hover_brush = pg.mkBrush(pg.mkColor(*self.color))
        self.roi_id = roi_id
        self.bounds = bounds
        super().__init__(orientation="vertical", bounds=bounds, pen=self.pen,
                         brush=self.brush, hoverBrush=self.hover_brush)

    def time_set(self):
        region = self.getRegion()
        region = [int(region[0]), int(region[1])]
        region[0] = max(self.bounds[0], region[0])
        region[1] = min(self.bounds[1] - 1, region[1])
        x0, x1 = region[0], region[1] + 1

        self.parent.xrange = slice(x0, x1)
        # Update zoom in plot
        self.parent.imgROI.setImage(self.parent.sp_smoothed[:, self.parent.xrange])
        self.parent.imgROI.setLevels(
            [self.parent.sat[0] / 100., self.parent.sat[1] / 100.])
        self.parent.p2.setXRange(0, x1 - x0, padding=0)
        self.parent.p2.show()

        # update other plots
        self.parent.p3.setLimits(xMin=x0, xMax=x1)
        self.parent.p3.setXRange(x0, x1)
        self.parent.p3.show()
        self.parent.p4.setLimits(xMin=x0, xMax=x1)
        self.parent.p4.setXRange(x0, x1)
        self.parent.p4.show()


class ClusterROI(pg.LinearRegionItem):

    def __init__(self, parent=None, color=[128, 128, 255, 50], bounds=[0, 100],
                 roi_id=0):
        self.color = color
        self.parent = parent
        self.pen = pg.mkPen(pg.mkColor(*self.color), width=2, style=QtCore.Qt.SolidLine)
        self.brush = pg.mkBrush(pg.mkColor(*self.color[:3], 50))
        self.hover_brush = pg.mkBrush(pg.mkColor(*self.color))
        self.roi_id = roi_id
        self.bounds = bounds
        super().__init__(orientation="horizontal", bounds=bounds, pen=self.pen,
                         brush=self.brush, hoverBrush=self.hover_brush)
        self.sigRegionChanged.connect(self.cluster_set)

    def cluster_set(self):
        region = self.getRegion()
        region = [int(region[0]), int(region[1])]
        region[0] = max(self.bounds[0], region[0])
        region[1] = min(self.bounds[1], region[1])
        if len(self.parent.cluster_slices) > self.roi_id:
            self.parent.cluster_slices[self.roi_id] = slice(region[0], region[1])
            self.parent.selected = self.parent.cluster_slices[self.roi_id]
            self.parent.plot_traces(roi_id=self.roi_id)
            if self.parent.neuron_pos is not None or self.parent.behav_data is not None:
                self.parent.update_scatter(roi_id=self.roi_id)
            if hasattr(self.parent, "PlaneWindow"):
                self.parent.PlaneWindow.update_plots(roi_id=self.roi_id)
            self.parent.p3.show()

    def mouseClickEvent(self, ev):
        if self.moving and ev.button() == QtCore.Qt.RightButton:
            ev.accept()
            for i, l in enumerate(self.lines):
                l.setPos(self.startPositions[i])
            self.moving = False
            self.cluster_set()
            self.sigRegionChanged.emit(self)
            self.sigRegionChangeFinished.emit(self)
        elif ev.button() == QtCore.Qt.LeftButton and ev.modifiers(
        ) == QtCore.Qt.ControlModifier:
            if len(self.parent.cluster_rois) > 0:
                print("removing cluster roi")
                self.remove()

    def remove(self):
        # delete color and add to end of list
        del self.parent.colors[self.roi_id]
        self.parent.colors.append(self.color)

        # delete slice and ROI
        del self.parent.cluster_slices[self.roi_id]
        self.parent.p2.removeItem(self.parent.cluster_rois[self.roi_id])
        del self.parent.cluster_rois[self.roi_id]

        # remove scatter plots
        self.parent.p5.removeItem(self.parent.scatter_plots[0][self.roi_id + 1])
        del self.parent.scatter_plots[0][self.roi_id + 1]
        self.parent.scatter_plots[0].append(pg.ScatterPlotItem())
        self.parent.p5.addItem(self.parent.scatter_plots[0][-1])
        self.parent.p5.addItem(self.parent.cluster_plots[-1])

        # remove avg activity
        self.parent.p3.removeItem(self.parent.cluster_plots[self.roi_id])
        del self.parent.cluster_plots[self.roi_id]
        self.parent.cluster_plots.append(pg.PlotDataItem())
        self.parent.p3.addItem(self.parent.cluster_plots[-1])

        # reindex roi_id
        for i in range(len(self.parent.cluster_rois)):
            self.parent.cluster_rois[i].roi_id = i

        # update avg plot
        self.parent.plot_traces()


# custom vertical label
class VerticalLabel(QWidget):

    def __init__(self, text=None):
        super(self.__class__, self).__init__()
        self.text = text

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setPen(QtCore.Qt.white)
        painter.translate(0, 0)
        painter.rotate(90)
        if self.text:
            painter.drawText(0, 0, self.text)
        painter.end()
