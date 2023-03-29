from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QScrollBar, QSlider, QComboBox, QGridLayout, QPushButton, QFrame, QCheckBox, QLabel, QProgressBar, QLineEdit, QMessageBox, QGroupBox, QStyle, QStyleOptionSlider
import pyqtgraph as pg
from pyqtgraph import functions as fn
from pyqtgraph import Point
import numpy as np


class LinearRegionItem(pg.LinearRegionItem):
    def __init__(self, parent=None, color=[128, 128, 255, 50], 
                 bounds=[0,100], roi_id=0):
        self.color = color
        self.parent = parent
        self.pen = pg.mkPen(pg.mkColor(*self.color),
                            width=2,
                            style=QtCore.Qt.SolidLine)
        self.brush = pg.mkBrush(pg.mkColor(*self.color[:3], 50))
        self.hover_brush = pg.mkBrush(pg.mkColor(*self.color))
        self.roi_id = roi_id
        super().__init__(orientation="horizontal", bounds=bounds,
                         pen=self.pen, brush=self.brush, 
                         hoverBrush=self.hover_brush)
        self.sigRegionChanged.connect(self.cluster_set)
        
    def cluster_set(self):
        region = self.getRegion()
        region = (int(region[0]), int(region[1]))
        if len(self.parent.cluster_slices) > self.roi_id:
            self.parent.cluster_slices[self.roi_id] = slice(region[0], region[1])
            self.parent.selected = self.parent.cluster_slices[self.roi_id]
            self.parent.plot_avg_activity_trace(roi_id=self.roi_id)
            self.parent.update_scatter(roi_id=self.roi_id)
            if hasattr(self.parent, 'PlaneWindow'):
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
        elif ev.button() == QtCore.Qt.LeftButton and ev.modifiers() == QtCore.Qt.ControlModifier:
            if len(self.parent.cluster_rois)>0:
                print('removing cluster roi')
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
        self.parent.p5.removeItem(self.parent.scatter_plots[self.roi_id])
        del self.parent.scatter_plots[self.roi_id]
        self.parent.scatter_plots.append(pg.ScatterPlotItem())
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
        self.parent.plot_avg_activity_trace()

        
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


class LineRegionInit():
    '''
    draw a line segment which is the gradient over which to plot the points
    '''
    def __init__(self, pos, prect, color, parent=None):
        self.prect = prect
        self.pos = pos
        self.d = ((prect[0][0,:] - prect[0][1,:])**2).sum()**0.5 / 2
        self.color = color
        self.pen = pg.mkPen(pg.mkColor(self.color),
                                width=3,
                                style=QtCore.Qt.SolidLine)
        pts, pdist   = self.inROI(parent.embedding)
        inds  = np.argsort(pdist)
        self.selected = pts[inds[::-1]]

        self.ROIplot = pg.PlotDataItem(self.prect[0][:,0], self.prect[0][:,1], pen=self.pen)
        parent.p0.addItem(self.ROIplot)
        self.dotplot = pg.ScatterPlotItem(pos=self.pos[-1][1,:][np.newaxis], pen=self.pen, symbol='+')
        parent.p0.addItem(self.dotplot)

        parent.show()
