from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QScrollBar, QSlider, QComboBox, QGridLayout, QPushButton, QFrame, QCheckBox, QLabel, QProgressBar, QLineEdit, QMessageBox, QGroupBox, QStyle, QStyleOptionSlider
import pyqtgraph as pg
from pyqtgraph import functions as fn
from pyqtgraph import Point
import numpy as np
from pyqtgraph import ItemSample

class LinearRegionItem(pg.LinearRegionItem):
    def __init__(self, parent=None, color=[128, 128, 255, 150], 
                 bounds=[0,100], roi_id=0):
        self.color = color
        self.parent = parent
        self.pen = pg.mkPen(pg.mkColor(*self.color),
                            width=3,
                            style=QtCore.Qt.SolidLine)
        self.brush = pg.mkBrush(pg.mkColor(255,255,255,0))
        self.hover_brush = pg.mkBrush(pg.mkColor(*self.color))
        self.roi_id = roi_id
        super().__init__(orientation="horizontal", bounds=bounds,
                         pen=self.pen, brush=self.brush, 
                         hoverBrush=self.hover_brush)

        #self.sigRegionChanged.connect(self.cluster_set())
        
    def cluster_set(self):
        print('hi')
        region = self.getRegion()
        region = (int(region[0]), int(region[1]))
        self.parent.cluster_slices[self.roi_id] = slice(region[0], region[1])
        self.parent.selected = self.parent.cluster_slices[self.roi_id]
        self.parent.plot_avg_activity_trace()
        self.parent.update_scatter()
        self.parent.p3.show()

    def lineMoved(self):
        if self.blockLineSignal:
            return
        self.prepareGeometryChange()
        self.cluster_set()
        self.sigRegionChanged.emit(self)

    def lineMoveFinished(self):
        self.cluster_set()
        self.sigRegionChangeFinished.emit(self)

    def mouseDragEvent(self, ev):
        if not self.movable or int(ev.button() & QtCore.Qt.LeftButton) == 0:
            return
        ev.accept()
        
        if ev.isStart():
            bdp = ev.buttonDownPos()
            self.cursorOffsets = [l.pos() - bdp for l in self.lines]
            self.startPositions = [l.pos() for l in self.lines]
            self.moving = True
            
        if not self.moving:
            return
            
        #delta = ev.pos() - ev.lastPos()
        self.lines[0].blockSignals(True)  # only want to update once
        for i, l in enumerate(self.lines):
            l.setPos(self.cursorOffsets[i] + ev.pos())
            #l.setPos(l.pos()+delta)
            #l.mouseDragEvent(ev)
        self.lines[0].blockSignals(False)
        self.prepareGeometryChange()
        
        if ev.isFinish():
            self.moving = False
            self.cluster_set()
            self.sigRegionChangeFinished.emit(self)
        else:
            self.cluster_set()
            self.sigRegionChanged.emit(self)
            
    def mouseClickEvent(self, ev):
        if self.moving and ev.button() == QtCore.Qt.RightButton:
            ev.accept()
            for i, l in enumerate(self.lines):
                l.setPos(self.startPositions[i])
            self.moving = False
            self.cluster_set()
            self.sigRegionChanged.emit(self)
            self.sigRegionChangeFinished.emit(self)
        elif ev.button() == QtCore.Qt.RightButton:
            if len(self.parent.cluster_rois)>0:
                print('removing cluster roi')
        
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
