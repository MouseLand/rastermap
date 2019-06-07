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
        self.d = ((prect[0][0,:] - prect[0][1,:])**2).sum()**0.5 / 2
        #self.slope = (pos[1,1] - pos[0,1]) / (pos[1,0] - pos[0,0])
        #self.yint  = pos[1,0] - self.slope * pos[0,0]
        np.save('groi.npy', {'prect': self.prect, 'pos': self.pos})
        self.color = color
        self.pen = pg.mkPen(pg.mkColor(self.color),
                                width=3,
                                style=QtCore.Qt.SolidLine)
        pts, pdist   = self.inROI(parent.embedding)
        inds  = np.argsort(pdist)
        self.selected = pts[inds[::-1]]

        # make a grid of points and find external points - use as hull
        if len(self.prect) > 1:
            pos2 = np.array(prect)
            xmin = pos2[:,:,0].min()
            xmax = pos2[:,:,0].max()
            ymin = pos2[:,:,1].min()
            ymax = pos2[:,:,1].max()
            dx = np.maximum((xmax-xmin)/100, (ymax-ymin)/100)
            grid = np.meshgrid(np.arange(xmin,xmax,dx), np.arange(ymin,ymax,dx))
            grid = np.concatenate((grid[0].flatten()[:,np.newaxis], grid[1].flatten()[:,np.newaxis]), axis=1)
            gridpts, _ = self.inROI(grid)
            grid = grid[gridpts, :]
            dists = ((grid[:,0][:,np.newaxis] - grid[:,0][np.newaxis,:])**2 +
                   (grid[:,1][:,np.newaxis] - grid[:,1][np.newaxis,:])**2)**0.5
            dists[np.arange(0, grid.shape[0]), np.arange(0, grid.shape[0])] = np.Inf
            nneigh = (dists<=dx+1e-5).sum(axis=1)
            exterior = grid[nneigh<4,:]

            # npts = exterior.shape[0]
            # distmat = dists[np.ix_(nneigh<4, nneigh<4)]<=1.5*dx
            # n0 = 0
            # norder = np.zeros((npts,), np.int32)
            # for n in range(npts-1):#distmat.shape[0]):
            #     d = distmat[n0]
            #     try:
            #         neigh = (d>0).nonzero()[0]
            #         ibad = np.isin(neigh, norder)
            #         neigh = neigh[~ibad]
            #         n0 = neigh[0]
            #         norder[n] = n0
            #     except:
            #         break
            # self.ROIplot = pg.PlotDataItem(exterior[norder,0], exterior[norder,1], pen=self.pen)
            #
            self.ROIplot = pg.ScatterPlotItem(pos=exterior, pen=self.pen, symbol='o', size=4)
        else:
            self.ROIplot = pg.PlotDataItem(self.prect[0][:,0], self.prect[0][:,1], pen=self.pen)
        parent.p0.addItem(self.ROIplot)
        self.dotplot = pg.ScatterPlotItem(pos=self.pos[-1][1,:][np.newaxis], pen=self.pen, symbol='+')
        parent.p0.addItem(self.dotplot)

        # theta = np.linspace(0, 2*np.pi, 50)
        # self.ROIplot = []
        # self.dotplot = []
        # self.circleplot = []
        # for k in range(len(self.prect)):
        #     self.ROIplot.append(pg.PlotDataItem(self.prect[k][:,0], self.prect[k][:,1], pen=self.pen))
        #     parent.p0.addItem(self.ROIplot[-1])
        #     if k == len(self.prect)-1:
        #         self.dotplot = pg.ScatterPlotItem(pos=self.pos[k][0,:][np.newaxis], pen=self.pen, symbol='+')
        #         parent.p0.addItem(self.dotplot)
        #     else:
        #         self.circleplot.append(pg.PlotDataItem(np.cos(theta)*self.d + self.pos[k][1,0],
        #                                           np.sin(theta)*self.d + self.pos[k][1,1], pen=self.pen))
        #         parent.p0.addItem(self.circleplot[-1])

        parent.show()

    def orthproj(self, p, k):
        # center at origin
        vproj = self.pos[k][1,:] - self.pos[k][0,:]
        vproj = vproj[np.newaxis,:]
        p   = p - self.pos[k][0,:][np.newaxis,:]
        pproj = (vproj.T @ vproj) / (vproj**2).sum() @ p.T
        pdist = (pproj**2).sum(axis=0)
        return pdist

    def inROI(self, Y):
        '''which points are inside ROI'''
        if Y.ndim > 1:
            area = np.zeros((Y.shape[0],4))
        else:
            area = np.zeros((1,4))
        pts = np.zeros((0,), int)
        pdist = np.zeros((0,), int)
        dist0 = 0
        for k in range(len(self.prect)):
            self.square_area = (triangle_area(self.prect[k][0,:], self.prect[k][1,:], self.prect[k][2,:]) +
                            triangle_area(self.prect[k][2,:], self.prect[k][3,:], self.prect[k][4,:]))
            for n in range(4):
                area[:,n] = triangle_area(self.prect[k][0+n,:], self.prect[k][1+n,:], Y)
            # points inside prect
            newpts = np.array((area.sum(axis=1) <= self.square_area+1e-5).nonzero()).flatten().astype(int)
            if newpts.size > 0:
                pts = np.concatenate((pts, newpts))
                newdists = self.orthproj(Y[newpts, :], k) + dist0
                pdist = np.concatenate((pdist, newdists))
            dist0 += (np.diff(self.pos[k], axis=0)[0,:]**2).sum()
            # check if in radius of circle
            if k < len(self.prect)-1:
                pcent = self.pos[k][1,:]
                dist = ((Y - pcent[np.newaxis,:])**2).sum(axis=1)**0.5
                newpts = np.array((dist<=self.d).nonzero()[0].astype(int))
                if newpts.size > 0:
                    pts = np.concatenate((pts, newpts))
                    newdists = dist0 * np.ones(newpts.shape)
                    pdist = np.concatenate((pdist, newdists))

        pts, inds = np.unique(pts, return_index=True)
        pdist = pdist[inds]
        return pts, pdist

    def remove(self, parent):
        '''remove ROI'''
        parent.p0.removeItem(self.ROIplot)
        parent.p0.removeItem(self.dotplot)
