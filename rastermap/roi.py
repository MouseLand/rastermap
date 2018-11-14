import pyqtgraph as pg
import numpy as np
from PyQt5 import QtGui, QtCore

def triangle_area(p0, p1, p2):
    if p2.ndim < 2:
        p2 = p2[np.newaxis, :]
    '''p2 can be a vector'''
    area = 0.5 * np.abs(p0[0] * p1[1] - p0[0] * p2[:,1] +
           p1[0] * p2[:,1] - p1[0] * p0[1] +
           p2[:,0] * p0[1] - p2[:,0] * p1[1])
    return area

class gROI():
    '''
    draw a line segment which is the gradient over which to plot the points
    '''
    def __init__(self, pos, prect, color, parent=None):
        self.prect = prect
        self.pos = pos
        #self.slope = (pos[1,1] - pos[0,1]) / (pos[1,0] - pos[0,0])
        #self.yint  = pos[1,0] - self.slope * pos[0,0]
        self.color = color
        self.pen = pg.mkPen(pg.mkColor(self.color),
                                width=3,
                                style=QtCore.Qt.SolidLine)
        self.ROIplot = pg.PlotDataItem(self.prect[:,0], self.prect[:,1], pen=self.pen)
        self.dotplot = pg.ScatterPlotItem(pos=self.pos[0,:][np.newaxis], pen=self.pen, symbol='+')
        parent.p0.addItem(self.ROIplot)
        parent.p0.addItem(self.dotplot)
        pts = self.inROI(parent.embedding)
        pdist = self.orthproj(parent.embedding[pts,:])
        inds = np.argsort(pdist)
        self.selected = pts[inds[::-1]]
        parent.show()

    def orthproj(self, p):
        # center at origin
        vproj = self.pos[1,:] - self.pos[0,:]
        vproj = vproj[np.newaxis,:]
        p   = p - self.pos[0,:][np.newaxis,:]
        pproj = (vproj.T @ vproj) / (vproj**2).sum() @ p.T
        pdist = (pproj**2).sum(axis=0)
        return pdist

    def inROI(self, Y):
        '''which points are inside ROI'''
        if Y.ndim > 1:
            area = np.zeros((Y.shape[0],4))
        else:
            area = np.zeros((1,4))
        self.square_area = (triangle_area(self.prect[0,:], self.prect[1,:], self.prect[2,:]) +
                        triangle_area(self.prect[2,:], self.prect[3,:], self.prect[4,:]))
        for n in range(4):
            area[:,n] = triangle_area(self.prect[0+n,:], self.prect[1+n,:], Y)
        # points inside prect
        pts = np.array((area.sum(axis=1) <= self.square_area+1e-5).nonzero()).flatten().astype(int)
        return pts

    def remove(self, parent):
        '''remove ROI'''
        parent.p0.removeItem(self.ROIplot)
        parent.p0.removeItem(self.dotplot)
