import sys
import os
import shutil
import time
import numpy as np
from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph import GraphicsScene
from scipy.stats import zscore
from matplotlib import cm
from rastermap.roi import gROI, dbROI
import rastermap.run
from rastermap import Rastermap
from sklearn.cluster import DBSCAN

def triangle_area(p):
    area = 0.5 * np.abs(p[0,0] * p[1,1] - p[0,0] * p[2,1] +
           p[1,0] * p[2,1] - p[1,0] * p[0,1] +
           p[2,0] * p[0,1] - p[2,0] * p[1,1])
    return area

def dist_to_line(p):
    d = 2 * triangle_area(p)
    d /= ((p[1,0] - p[0,0])**2 + (p[1,1] - p[0,1])**2)**0.5
    return d

def rect_from_line(p,d):
    dline = ((p[1,0] - p[0,0])**2 + (p[1,1] - p[0,1])**2)**0.5
    theta = np.pi/2 - np.arctan((p[1,1] - p[0,1]) / (p[1,0] - p[0,0] + 1e-5))
    prect = np.zeros((5,2))
    prect[0,:] = [p[1,0] + d * np.cos(theta), p[1,1] - d * np.sin(theta)]
    prect[1,:] = [p[1,0] - d * np.cos(theta), p[1,1] + d * np.sin(theta)]
    #theta = np.pi/2 - theta
    prect[2,:] = [p[0,0] - d * np.cos(theta), p[0,1] + d * np.sin(theta)]
    prect[3,:] = [p[0,0] + d * np.cos(theta), p[0,1] - d * np.sin(theta)]
    prect[-1,:] = prect[0,:]
    return prect

class Slider(QtGui.QSlider):
    def __init__(self, bid, parent=None):
        super(self.__class__, self).__init__()
        self.bid = bid
        self.setMinimum(0)
        self.setMaximum(100)
        self.setValue(parent.sat[bid]*200)
        self.setTickPosition(QtGui.QSlider.TicksLeft)
        self.setTickInterval(10)
        self.valueChanged.connect(lambda: self.level_change(parent,bid))
        self.setTracking(False)
        if self.bid==0:
            self.setInvertedAppearance(True)

    def level_change(self, parent, bid):
        parent.sat[bid] = float(self.value())/200
        if bid==1:
            parent.sat[bid] += 0.5
        else:
            parent.sat[bid] = 0.5 - parent.sat[bid]
        parent.img.setLevels([parent.sat[0],parent.sat[1]])
        parent.imgfull.setLevels([parent.sat[0],parent.sat[1]])
        parent.win.show()

# custom vertical label
class VerticalLabel(QtGui.QWidget):
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

class MainW(QtGui.QMainWindow):
    def __init__(self):
        super(MainW, self).__init__()
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(25, 25, 1800, 1000)
        self.setWindowTitle("Rastermap")
        icon_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "logo.png"
        )
        app_icon = QtGui.QIcon()
        app_icon.addFile(icon_path, QtCore.QSize(16, 16))
        app_icon.addFile(icon_path, QtCore.QSize(24, 24))
        app_icon.addFile(icon_path, QtCore.QSize(32, 32))
        app_icon.addFile(icon_path, QtCore.QSize(48, 48))
        app_icon.addFile(icon_path, QtCore.QSize(96, 96))
        app_icon.addFile(icon_path, QtCore.QSize(256, 256))
        self.setWindowIcon(app_icon)
        self.setStyleSheet("QMainWindow {background: 'black';}")
        self.stylePressed = ("QPushButton { "
                             "background-color: rgb(100,50,100); "
                             "color:white;}")
        self.styleUnpressed = ("QPushButton { "
                               "background-color: rgb(50,50,50); "
                               "color:white;}")
        self.styleInactive = ("QPushButton { "
                              "background-color: rgb(50,50,50); "
                              "color:gray;}")
        self.loaded = False

        # ------ MENU BAR -----------------
        loadMat =  QtGui.QAction("&Load data matrix", self)
        loadMat.setShortcut("Ctrl+L")
        loadMat.triggered.connect(lambda: self.load_mat(name=None))
        self.addAction(loadMat)
        # run rastermap from scratch
        self.runRMAP = QtGui.QAction("&Run embedding algorithm", self)
        self.runRMAP.setShortcut("Ctrl+R")
        self.runRMAP.triggered.connect(self.run_RMAP)
        self.addAction(self.runRMAP)
        self.runRMAP.setEnabled(False)
        # load processed data
        loadProc = QtGui.QAction("&Load processed data", self)
        loadProc.setShortcut("Ctrl+P")
        loadProc.triggered.connect(lambda: self.load_proc(name=None))
        self.addAction(loadProc)
        # load a behavioral trace
        self.loadBeh = QtGui.QAction(
            "Load behavior or stim trace (1D only)", self
        )
        self.loadBeh.triggered.connect(self.load_behavior)
        self.loadBeh.setEnabled(False)
        self.addAction(self.loadBeh)
        # export figure
        exportFig = QtGui.QAction("Export as image (svg)", self)
        exportFig.triggered.connect(self.export_fig)
        exportFig.setEnabled(True)
        self.addAction(exportFig)

        # make mainmenu!
        main_menu = self.menuBar()
        file_menu = main_menu.addMenu("&File")
        file_menu.addAction(loadMat)
        file_menu.addAction(loadProc)
        file_menu.addAction(self.runRMAP)
        file_menu.addAction(self.loadBeh)
        file_menu.addAction(exportFig)

        #### --------- MAIN WIDGET LAYOUT --------- ####
        #pg.setConfigOption('background', 'w')
        #cwidget = EventWidget(self)
        cwidget = QtGui.QWidget()
        self.l0 = QtGui.QGridLayout()
        cwidget.setLayout(self.l0)
        self.setCentralWidget(cwidget)

        # -------- MAIN PLOTTING AREA ----------
        self.win = pg.GraphicsLayoutWidget()
        #self.win.move(600, 0)
        #self.win.resize(1000, 500)
        self.l0.addWidget(self.win, 0, 0, 50, 30)
        layout = self.win.ci.layout
        # --- full recording
        self.pfull = self.win.addPlot(title="FULL VIEW",row=0,col=2,rowspan=1,colspan=3)
        self.pfull.setMouseEnabled(x=False,y=False)
        self.imgfull = pg.ImageItem(autoDownsample=True)
        self.pfull.addItem(self.imgfull)
        self.pfull.hideAxis('left')
        #self.pfull.hideAxis('bottom')

        # --- embedding image
        self.p0 = self.win.addPlot(row=1, col=0, rowspan=2, colspan=1, lockAspect=True)
        self.p0.setAspectLocked(ratio=1)
        self.p0.scene().sigMouseMoved.connect(self.mouse_moved_embedding)

        # ---- colorbar
        self.p3 = self.win.addPlot(row=1, col=1, rowspan=3, colspan=1)
        self.p3.setMouseEnabled(x=False,y=False)
        self.p3.setMenuEnabled(False)
        self.colorimg = pg.ImageItem(autoDownsample=True)
        self.p3.addItem(self.colorimg)
        self.p3.scene().sigMouseMoved.connect(self.mouse_moved_bar)

        # --- activity image
        self.p1 = self.win.addPlot(row=1, col=2, colspan=3,
                                   rowspan=3, invertY=True, padding=0)
        self.p1.setMouseEnabled(x=False, y=False)
        self.img = pg.ImageItem(autoDownsample=True)
        self.p1.hideAxis('left')
        self.p1.setMenuEnabled(False)
        self.p1.scene().contextMenuItem = self.p1
        self.p1.addItem(self.img)
        self.p1.scene().sigMouseMoved.connect(self.mouse_moved_activity)

        # bottom row for buttons
        #self.p2 = self.win.addViewBox(row=2, col=0)
        #self.p2.setMouseEnabled(x=False,y=False)
        #self.p2.setMenuEnabled(False)

        self.win.scene().sigMouseClicked.connect(self.plot_clicked)

        #self.win.ci.layout.setRowStretchFactor(0, .5)
        self.win.ci.layout.setRowStretchFactor(1, 2)
        self.win.ci.layout.setRowStretchFactor(2, 2)
        #self.win.ci.layout.setColumnStretchFactor(0, .5)
        self.win.ci.layout.setColumnStretchFactor(1, .1)
        self.win.ci.layout.setColumnStretchFactor(3, 2)

        # self.key_on(self.win.scene().keyPressEvent)
        rs = 2
        addROI = QtGui.QLabel("<font color='white'>add an ROI by SHIFT click</font>")
        self.l0.addWidget(addROI, rs+0, 0, 1, 3)
        addROI = QtGui.QLabel("<font color='white'>delete an ROI by ALT click</font>")
        self.l0.addWidget(addROI, rs+1, 0, 1, 3)
        addROI = QtGui.QLabel("<font color='white'>delete last-drawn ROI by DELETE</font>")
        self.l0.addWidget(addROI, rs+2, 0, 1, 3)
        addROI = QtGui.QLabel("<font color='white'>delete all ROIs by ALT-DELETE</font>")
        self.l0.addWidget(addROI, rs+3, 0, 1, 3)
        self.updateROI = QtGui.QPushButton("update (SPACE)")
        self.updateROI.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.updateROI.clicked.connect(self.ROI_selection)
        self.updateROI.setStyleSheet(self.styleInactive)
        self.updateROI.setEnabled(False)
        self.updateROI.setFixedWidth(100)
        self.l0.addWidget(self.updateROI, rs+4, 0, 1, 1)
        self.saveROI = QtGui.QPushButton("save ROIs")
        self.saveROI.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.saveROI.clicked.connect(self.ROI_save)
        self.saveROI.setStyleSheet(self.styleInactive)
        self.saveROI.setEnabled(False)
        self.saveROI.setFixedWidth(100)
        self.l0.addWidget(self.saveROI, rs+5, 0, 1, 1)

        self.makegrid = QtGui.QPushButton("make ROI grid, n=")
        self.makegrid.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.makegrid.clicked.connect(self.make_grid)
        self.makegrid.setStyleSheet(self.styleInactive)
        self.makegrid.setEnabled(False)
        self.makegrid.setFixedWidth(200)
        self.l0.addWidget(self.makegrid, rs+7, 0, 1, 1)
        self.gridsize = QtGui.QLineEdit(self)
        self.gridsize.setValidator(QtGui.QIntValidator(2, 20))
        self.gridsize.setText("5")
        self.gridsize.setFixedWidth(45)
        self.gridsize.setAlignment(QtCore.Qt.AlignRight)
        self.gridsize.returnPressed.connect(self.make_grid)
        self.l0.addWidget(self.gridsize, rs+7, 1, 1, 1)

        self.dbbutton = QtGui.QPushButton("DBSCAN clusters, ms=")
        self.dbbutton.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.dbbutton.clicked.connect(self.dbscan)
        self.dbbutton.setStyleSheet(self.styleInactive)
        self.dbbutton.setEnabled(False)
        self.dbbutton.setFixedWidth(200)
        self.l0.addWidget(self.dbbutton, rs+8, 0, 1, 1)
        self.min_samples = QtGui.QLineEdit(self)
        self.min_samples.setValidator(QtGui.QIntValidator(5, 200))
        self.min_samples.setText("50")
        self.min_samples.setFixedWidth(45)
        self.min_samples.setAlignment(QtCore.Qt.AlignRight)
        self.min_samples.returnPressed.connect(self.dbscan)
        self.l0.addWidget(self.min_samples, rs+8, 1, 1, 1)

        ysm = QtGui.QLabel("<font color='white'>y-binning</font>")
        ysm.setFixedWidth(100)
        self.l0.addWidget(ysm, rs+6, 0, 1, 1)
        self.smooth = QtGui.QLineEdit(self)
        self.smooth.setValidator(QtGui.QIntValidator(0, 500))
        self.smooth.setText("10")
        self.smooth.setFixedWidth(45)
        self.smooth.setAlignment(QtCore.Qt.AlignRight)
        self.smooth.returnPressed.connect(self.plot_activity)
        self.l0.addWidget(self.smooth, rs+6, 1, 1, 1)

        # add slider for levels
        self.sl = []
        txt = ["lower saturation", 'upper saturation']
        self.sat = [0.3,0.7]
        for j in range(2):
            self.sl.append(Slider(j, self))
            self.l0.addWidget(self.sl[j],rs+4-4*j,3,4,1)
            qlabel = VerticalLabel(text=txt[j])
            #qlabel.setStyleSheet('color: white;')
            self.l0.addWidget(qlabel,rs+4-4*j,4,4,1)
        colormap = cm.get_cmap("gray_r")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        lut = lut[0:-3,:]
        # apply the colormap
        self.img.setLookupTable(lut)
        self.imgfull.setLookupTable(lut)
        self.img.setLevels([self.sat[0], self.sat[1]])
        self.imgfull.setLevels([self.sat[0], self.sat[1]])

        # ------ CHOOSE CELL-------
        #self.ROIedit = QtGui.QLineEdit(self)
        #self.ROIedit.setValidator(QtGui.QIntValidator(0, 10000))
        #self.ROIedit.setText("0")
        #self.ROIedit.setFixedWidth(45)
        #self.ROIedit.setAlignment(QtCore.Qt.AlignRight)
        #self.ROIedit.returnPressed.connect(self.number_chosen)

        self.startROI = False
        self.endROI = False
        self.posROI = np.zeros((3,2))
        self.prect = np.zeros((5,2))
        self.ROIs = []
        self.ROIorder = []
        self.Rselected = []
        self.Rcolors = []
        self.embedded = False
        self.posAll = []
        self.lp = []
        #self.fname = '/media/carsen/DATA1/BootCamp/mesoscope_cortex/spks.npy'
        # self.load_behavior('C:/Users/carse/github/TX4/beh.npy')
        self.file_iscell = None
        #self.fname = '/media/carsen/DATA2/grive/rastermap/DATA/embedding.npy'
        self.fname = 'D:/grive/cshl_suite2p/TX39/embedding.npy'
        self.load_proc(self.fname)

        self.show()
        self.win.show()


    def add_imgROI(self):
        if hasattr(self, 'imgROI'):
            self.pfull.removeItem(self.imgROI)
        nt = self.sp.shape[1]
        nn = self.sp.shape[0]

        redpen = pg.mkPen(pg.mkColor(255, 0, 0),
                                width=3,
                                style=QtCore.Qt.SolidLine)
        self.imgROI = pg.RectROI([nt*.25, -.5], [nt*.25, nn+.5],
                      maxBounds=QtCore.QRectF(-.5,-.5,nt+.5,nn+.5),
                      pen=redpen)
        self.yrange = np.arange(0, nn).astype(np.int32)
        self.imgROI.handleSize = 10
        self.imgROI.handlePen = redpen
        # Add top and right Handles
        self.imgROI.addScaleHandle([1, 0.5], [0., 0.5])
        self.imgROI.addScaleHandle([0.5, 0], [0.5, 1])
        self.imgROI.sigRegionChangeFinished.connect(self.imgROI_position)
        self.pfull.addItem(self.imgROI)
        self.imgROI.setZValue(10)  # make sure ROI is drawn above image

    def plot_embedding(self):
        self.se = pg.ScatterPlotItem(size=4, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 70))
        pos = self.embedding.T
        spots = [{'pos': pos[:,i], 'data': 1} for i in range(pos.shape[1])] + [{'pos': [0,0], 'data': 1}]
        self.se.addPoints(spots)
        self.p0.addItem(self.se)

    def smooth_activity(self):
        N = int(self.smooth.text())
        if N > 1:
            NN = self.selected.size
            nn = int(np.floor(NN/N))
            self.sp_smoothed = np.reshape(self.sp[self.selected][:nn*N].copy(), (nn, N, -1)).mean(axis=1)
            #cumsum = np.cumsum(np.concatenate((np.zeros((N,self.sp.shape[1])), self.sp[self.selected,:]), axis=0), axis=0)
            #self.sp_smoothed = (cumsum[N:, :] - cumsum[:-N, :]) / float(N)

            self.sp_smoothed = zscore(self.sp_smoothed, axis=1)
            self.sp_smoothed = np.maximum(-4, np.minimum(8, self.sp_smoothed)) + 4
            self.sp_smoothed /= 12

    def plot_activity(self):
        if self.embedded:
            self.smooth_activity()
            nn = self.sp_smoothed.shape[0]
            nt = self.sp_smoothed.shape[1]
            self.imgfull.setImage(self.sp_smoothed)
            self.imgfull.setLevels([self.sat[0],self.sat[1]])
            self.img.setImage(self.sp_smoothed)
            self.img.setLevels([self.sat[0],self.sat[1]])
            self.p1.setXRange(0, nt, padding=0)
            self.p1.setYRange(0, nn, padding=0)
            self.p1.setLimits(xMin=0,xMax=nt,yMin=0,yMax=nn)
            self.pfull.setXRange(0, nt, padding=0)
            self.pfull.setYRange(0, nn, padding=0)
            self.pfull.setLimits(xMin=-1,xMax=nt+1,yMin=-1,yMax=nn+1)
            self.imgROI.setPos(-.5,-.5)
            self.imgROI.setSize([nt+.5,nn+.5])
            self.imgROI.maxBounds = QtCore.QRectF(-1.,-1.,nt+1,nn+1)
        else:
            nn = self.sp.shape[0]
            nt = self.sp.shape[1]
            self.imgfull.setImage(self.sp)
            self.imgfull.setLevels([self.sat[0],self.sat[1]])
            self.img.setImage(self.sp)
            self.img.setLevels([self.sat[0],self.sat[1]])
            self.p1.setXRange(0, nt, padding=0)
            self.p1.setYRange(0, nn, padding=0)
            self.p1.setLimits(xMin=0,xMax=nt,yMin=0,yMax=nn)
            self.pfull.setXRange(0, nt, padding=0)
            self.pfull.setYRange(0, nn, padding=0)
            self.pfull.setLimits(xMin=-1,xMax=nt+1,yMin=-1,yMax=nn+1)
            self.imgROI.setPos(-.5,-.5)
            self.imgROI.setSize([nt+.5,nn+.5])
            self.imgROI.maxBounds = QtCore.QRectF(-1.,-1.,nt+1,nn+1)
            self.yrange = np.arange(0,nn,1,int)
        self.show()
        self.win.show()

    def plot_colorbar(self):
        nneur = self.colormat_plot.shape[0]
        self.colorimg.setImage(self.colormat_plot)
        if self.embedded:
            N = int(self.smooth.text())
        else:
            N = 1
        self.p3.setYRange(self.yrange[0]*N, self.yrange[-1]*N)
        self.p3.setXRange(0,10)
        self.p3.setLimits(yMin=self.yrange[0]*N,yMax=self.yrange[-1]*N,xMin=0,xMax=10)
        self.p3.getAxis('bottom').setTicks([[(0,'')]])
        self.win.show()

    def export_fig(self):
        self.win.scene().contextMenuItem = self.p0
        self.win.scene().showExportDialog()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            if self.updateROI.isEnabled:
                self.ROI_selection()
        elif event.key() == QtCore.Qt.Key_Delete:
            if len(self.ROIs) > 0:
                if event.modifiers() == QtCore.Qt.AltModifier:
                    for n in range(len(self.ROIs)):
                        self.ROI_delete()
                else:
                    self.ROI_delete()

    def imgROI_range(self):
        pos = self.imgROI.pos()
        posy = pos.y()
        posx = pos.x()
        sizex,sizey = self.imgROI.size()
        xrange = (np.arange(0,int(sizex)) + np.floor(posx)).astype(np.int32)
        yrange = (np.arange(0,int(sizey)) + np.floor(posy)).astype(np.int32)
        xrange = xrange[xrange>=0]
        yrange = yrange[yrange>=0]
        if self.embedded:
            xrange = xrange[xrange<self.sp_smoothed.shape[1]]
            yrange = yrange[yrange<self.sp_smoothed.shape[0]]
        else:
            xrange = xrange[xrange<self.sp.shape[1]]
            yrange = yrange[yrange<self.sp.shape[0]]
        return xrange,yrange

    def imgROI_position(self):
        xrange,yrange = self.imgROI_range()
        if self.embedded:
            self.img.setImage(self.sp_smoothed[np.ix_(yrange,xrange)])
        else:
            self.img.setImage(self.sp[np.ix_(yrange,xrange)])
        self.img.setLevels([self.sat[0],self.sat[1]])
        self.p1.setXRange(0,xrange.size)
        self.p1.setYRange(0,yrange.size)
        self.p1.setLimits(xMin=0,xMax=xrange.size,yMin=0,yMax=yrange.size)
        self.yrange = yrange
        axy = self.p3.getAxis('left')
        axx = self.p1.getAxis('bottom')
        self.plot_colorbar()
        if self.embedded:
            N = int(self.smooth.text())
        else:
            N = 1
        axy.setTicks([[(0,str(self.yrange[0])),(self.yrange[-1]*N,str(self.yrange[-1]*N))]])
        axx.setTicks([[(0.0,str(xrange[0])),(float(xrange.size),str(xrange[-1]))]])

    def ROI_selection(self, loaded=False):
        self.colormat = np.zeros((0,10,3), dtype=np.int64)
        lROI = len(self.Rselected)
        if lROI > 0:
            self.selected = np.array([item for sublist in self.Rselected for item in sublist])
            self.colormat = np.concatenate(self.Rcolors, axis=0)
            if not loaded:
                print('yo')
                if lROI > 4:
                    self.Ur = np.zeros((lROI, self.U.shape[1]), dtype=np.float32)
                    ugood = np.zeros((lROI,)).astype(np.int32)
                    for r,rc in enumerate(self.Rselected):
                        if len(rc) > 0:
                            self.Ur[r,:] = self.U[rc,:].mean(axis=0)
                            ugood[r] = 1
                    ugood = ugood.astype(bool)
                    if ugood.sum() > 4:
                        model = Rastermap(n_components=1, n_X=20, init=np.arange(0,ugood.sum()).astype(np.int32)[:,np.newaxis])
                        y     = model.fit_transform(self.Ur[ugood,:])
                        y     = y.flatten()
                        y2 = np.zeros((lROI,))
                        y2[(ugood).nonzero()[0]] = y
                        print(y2)
                        rsort = np.argsort(y2)
                        print(rsort)
                        roiorder = []
                        for r in self.ROIorder:
                            roiorder.append((rsort==r).nonzero()[0][0])
                        self.ROIorder = roiorder
                        self.ROIs = [self.ROIs[i] for i in rsort]
                        self.Rselected = [self.Rselected[i] for i in rsort]
                        self.Rcolors = [self.Rcolors[i] for i in rsort]
                        self.selected = np.array([item for sublist in self.Rselected for item in sublist])
                        self.colormat = np.concatenate(self.Rcolors, axis=0)
        else:
            self.selected = np.argsort(self.embedding[:,0])
            self.colormat = 255*np.ones((self.sp.shape[0],10,3), dtype=np.int32)

        self.colormat_plot = self.colormat.copy()
        self.plot_activity()
        print('plotted activity')
        self.plot_colorbar()
        print('plotted colorbar')
        self.win.show()

    def update_selected(self, ineur):
        # add bar to colorbar
        NN = self.colormat.shape[0]
        nrange = np.round(float(NN)/500)
        ineur_range = ineur
        if nrange > 0:
            ineur_range = ineur + np.arange(-1*nrange, nrange).astype(np.int32)
            ineur_range[(ineur_range < 0)] = 0
            ineur_range[(ineur_range > NN-1)] = NN-1
        self.colormat_plot = self.colormat.copy()
        self.colormat_plot[ineur_range,:,:] = np.zeros((10,3)).astype(int)
        self.plot_colorbar()
        # x point on embedding
        if self.embedded:
            ineur = self.selected[ineur]
            self.xp.setData(pos=self.embedding[ineur,:][np.newaxis,:])

    def dbscan(self):
        ms = int(self.min_samples.text())
        # remove previous ROIs
        if len(self.ROIs) > 0:
            for n in range(len(self.ROIs)):
                self.ROI_delete()

        db = DBSCAN(eps=0.8, min_samples=ms).fit(self.embedding)
        ilabels = np.unique(db.labels_)
        ilabels = ilabels[ilabels>=0]
        print(ilabels)
        #ilabels = ilabels[:1]
        for i in ilabels:
            self.dbROI_add((db.labels_==i).nonzero()[0])
        self.ROI_selection()

    def make_grid(self):
        ng = int(self.gridsize.text())
        if len(self.ROIs) > 0:
            for n in range(len(self.ROIs)):
                self.ROI_delete()
        sz = (self.embedding.max() - self.embedding.min()) / ng
        corners = np.linspace(self.embedding.min(), self.embedding.max(), ng+1)
        jet = cm.get_cmap('jet')
        jet = jet(np.linspace(0,1,ng**2))
        jet = jet[:,:3]
        for j in range(ng):
            for k in range(ng):
                prect = [np.array([[corners[j],corners[k]],
                                  [corners[j+1],corners[k]],
                                  [corners[j+1],corners[k+1]],
                                  [corners[j],corners[k+1]],
                                  [corners[j],corners[k]]])]
                pos = [np.array([[corners[j+1],corners[k]+sz/2],
                                [corners[j],corners[k]+sz/2]])]
                self.ROI_add(pos, prect, color=jet[j+k*ng]*255.0)
        self.ROI_selection()

    def dbROI_add(self, selected, color=None):
        if color is None:
            color =  np.random.randint(255,size=(3,))
        self.ROIs.append(dbROI(selected, color, self))
        self.Rselected.append(self.ROIs[-1].selected)
        self.Rcolors.append(np.reshape(np.tile(self.ROIs[-1].color, 10 * self.Rselected[-1].size),
                            (self.Rselected[-1].size, 10, 3)))
        self.ROIorder.append(len(self.ROIs)-1)

    def ROI_add(self, pos, prect, color=None):
        if color is None:
            color =  np.random.randint(255,size=(3,))
        self.ROIs.append(gROI(pos, prect, color, self))
        self.Rselected.append(self.ROIs[-1].selected)
        self.Rcolors.append(np.reshape(np.tile(self.ROIs[-1].color, 10 * self.Rselected[-1].size),
                            (self.Rselected[-1].size, 10, 3)))
        self.ROIorder.append(len(self.ROIs)-1)
        #self.ROI_selection()

    def ROI_delete(self):
        if len(self.ROIs) > 0:
            n = self.ROIorder[-1]
            self.delete(n)

    def delete(self, n):
        self.ROIs[n].remove(self)
        del self.ROIs[n]
        del self.Rselected[n]
        del self.Rcolors[n]
        for i,r in enumerate(self.ROIorder):
            if r > n:
                self.ROIorder[i] = self.ROIorder[i] - 1
        self.ROIorder.remove(n)

    def ROI_remove(self, p):
        if len(self.ROIs) > 0:
            if len(p) > 1:
                for n in range(len(self.ROIs)-1,-1,-1):
                    ptrue, pdist = self.ROIs[n].inROI(np.array(p)[np.newaxis,:])
                    if ptrue.shape[0] > 0:
                        self.delete(n)
                        break
            elif len(p)==1:
                p = int(p[0])
                for n in range(len(self.ROIs)-1,-1,-1):
                    if self.selected[p] in self.ROIs[n].selected:
                        self.delete(n)
                        break

    def ROI_save(self):
        name = QtGui.QFileDialog.getSaveFileName(self,'ROI name (*.npy)')
        name = name[0]
        self.proc['ROIs'] = []
        for r in self.ROIs:
            self.proc['ROIs'].append({'pos': r.pos, 'prect': r.prect, 'color': r.color, 'selected': r.selected})
        np.save(name, self.proc)

    def enable_loaded(self):
        self.runRMAP.setEnabled(True)

    def enable_embedded(self):
        self.updateROI.setEnabled(True)
        self.saveROI.setEnabled(True)
        self.makegrid.setEnabled(True)
        self.dbbutton.setEnabled(True)

        self.updateROI.setStyleSheet(self.styleUnpressed)
        self.saveROI.setStyleSheet(self.styleUnpressed)
        self.makegrid.setStyleSheet(self.styleUnpressed)
        self.dbbutton.setStyleSheet(self.styleUnpressed)

    def disable_embedded(self):
        self.updateROI.setEnabled(False)
        self.saveROI.setEnabled(False)
        self.makegrid.setEnabled(False)
        self.updateROI.setStyleSheet(self.styleInactive)
        self.saveROI.setStyleSheet(self.styleInactive)
        self.makegrid.setStyleSheet(self.styleInactive)

    def mouse_moved_embedding(self, pos):
        if self.embedded:
            if self.p0.sceneBoundingRect().contains(pos):
                x = self.p0.vb.mapSceneToView(pos).x()
                y = self.p0.vb.mapSceneToView(pos).y()
                if self.startROI or self.endROI:
                    if self.startROI:
                        self.p0.removeItem(self.l0)
                        self.posROI[1,:] = [x,y]
                        self.l0 = pg.PlotDataItem(self.posROI[:2,0],self.posROI[:2,1])
                        self.p0.addItem(self.l0)
                    else:
                        # compute the distance from the line to the point
                        self.posROI[2,:] = [x,y]
                        d = dist_to_line(self.posROI)
                        self.prect = []
                        for k in range(len(self.lp)):
                            self.p0.removeItem(self.lp[k])
                            self.prect.append(rect_from_line(self.posAll[k], d))
                            self.lp[k] = pg.PlotDataItem(self.prect[-1][:,0], self.prect[-1][:,1])
                            self.p0.addItem(self.lp[k])
                        #self.prect.append(rect_from_line(self.posROI, d))
                        #self.p0.removeItem(self.l0)
                        #self.l0 = pg.PlotDataItem(self.prect[0][:,0], self.prect[0][:,1])
                        #self.p0.addItem(self.l0)
                        self.p0.show()
                        self.show()
                else:
                    dists = (self.embedding[self.selected,0] - x)**2 + (self.embedding[self.selected,1] - y)**2
                    ineur = np.argmin(dists.flatten()).astype(int)
                    self.update_selected(ineur)

    def mouse_moved_activity(self, pos):
        if self.loaded:
            if self.p1.sceneBoundingRect().contains(pos):
                y = self.p1.vb.mapSceneToView(pos).y()
                #y += self.yrange[0]
                N = int(self.smooth.text())
                ineur = float(y) * float(N)
                ineur += (self.yrange[0]) * float(N)
                ineur = min((self.yrange[-1])*N, max((self.yrange[0])*N, int(np.floor(ineur))))
                #ineur = ineur + self.yrange[0]
                self.update_selected(ineur)

    def mouse_moved_bar(self, pos):
        if self.loaded:
            if self.p3.sceneBoundingRect().contains(pos):
                y = self.p3.vb.mapSceneToView(pos).y()
                ineur = min(self.colormat.shape[0]-1, max(0, int(np.floor(y))))
                self.update_selected(ineur)

    def plot_clicked(self, event):
        """left-click chooses a cell, right-click flips cell to other view"""
        flip = False
        choose = False
        zoom = False
        replot = False
        items = self.win.scene().items(event.scenePos())
        posx = 0
        posy = 0
        iplot = 0
        if self.loaded:
            # print(event.modifiers() == QtCore.Qt.ControlModifier)
            for x in items:
                if x == self.p0:
                    if self.embedded:
                        iplot = 0
                        vb = self.p0.vb
                        pos = vb.mapSceneToView(event.scenePos())
                        x = pos.x()
                        y = pos.y()
                        if event.double():
                            self.zoom_plot(iplot)
                        elif event.button() == 2:
                            # do nothing
                            nothing = True
                        elif event.modifiers() == QtCore.Qt.ShiftModifier:
                            if not self.startROI:
                                self.startROI = True
                                self.endROI = False
                                self.posROI[0,:] = [x,y]
                            else:
                                # plotting
                                self.startROI = True
                                self.endROI = False
                                self.posROI[1,:] = [x,y]
                                #print(self.)
                                self.posAll.append(self.posROI[:2,:].copy())
                                pos = self.posAll[-1]
                                self.lp.append(pg.PlotDataItem(pos[:, 0], pos[:, 1]))
                                self.posROI[0,:] = [x,y]
                                self.p0.addItem(self.lp[-1])
                                self.p0.show()
                        elif self.startROI:
                            self.posROI[1,:] = [x,y]
                            self.posAll.append(self.posROI[:2,:].copy())
                            self.p0.removeItem(self.l0)
                            pos = self.posAll[-1]
                            self.lp.append(pg.PlotDataItem(pos[:, 0], pos[:, 1]))
                            self.p0.addItem(self.lp[-1])
                            self.p0.show()
                            self.endROI = True
                            self.startROI = False
                        elif self.endROI:
                            self.posROI[2,:] = [x,y]
                            self.endROI = False
                            for lp in self.lp:
                                self.p0.removeItem(lp)
                            print(self.posAll, self.prect)
                            self.ROI_add(self.posAll, self.prect)
                            self.posAll = []
                            self.lp = []

                        elif event.modifiers() == QtCore.Qt.AltModifier:
                            self.ROI_remove([x,y])

                elif x == self.p1:
                    iplot = 1
                    y = self.p1.vb.mapSceneToView(event.scenePos()).y()
                    ineur = min(self.colormat.shape[0]-1, max(0, int(np.floor(y))))
                    ineur = ineur + self.yrange[0]
                    if event.double():
                        self.zoom_plot(iplot)
                    elif event.modifiers() == QtCore.Qt.AltModifier:
                        self.ROI_remove([ineur])
                elif x == self.p3:
                    iplot = 2
                    y = self.p3.vb.mapSceneToView(event.scenePos()).y()
                    ineur = min(self.colormat.shape[0]-1, max(0, int(np.floor(y))))
                    if event.modifiers() == QtCore.Qt.AltModifier:
                        self.ROI_remove([ineur])

    def zoom_plot(self, iplot):
        if iplot == 0:
            self.p0.setXRange(self.embedding[:,0].min(), self.embedding[:,0].max())
            self.p0.setYRange(self.embedding[:,1].min(), self.embedding[:,1].max())
        else:
            self.p1.setYRange(0, self.sp_smoothed.shape[0])
            self.p1.setXRange(0, self.sp_smoothed.shape[1])
        self.show()

    def run_RMAP(self):
        RW = rastermap.run.RunWindow(self)
        RW.show()

    def load_mat(self, name=None):
        if name is None:
            name = QtGui.QFileDialog.getOpenFileName(
                self, "Open *.npy", filter="*.npy"
            )
            self.fname = name[0]
            self.filebase = name[0]
        else:
            self.fname = name
            self.filebase = name
        try:
            X = np.load(self.fname)
            print(X.shape)
        except (ValueError, KeyError, OSError,
                RuntimeError, TypeError, NameError):
            print('ERROR: this is not a *.npy array :( ')
            X = None
        if X is not None and X.ndim > 1:
            self.startROI = False
            self.endROI = False
            self.posROI = np.zeros((3,2))
            self.prect = np.zeros((5,2))
            self.ROIs = []
            self.ROIorder = []
            self.Rselected = []
            self.Rcolors = []
            iscell, file_iscell = self.load_iscell()
            self.file_iscell = None
            if iscell is not None:
                if iscell.size == X.shape[0]:
                    X = X[iscell, :]
                    self.file_iscell = file_iscell
                    print('using iscell.npy in folder')
            if len(X.shape) > 2:
                X = X.mean(axis=-1)
            self.p0.clear()
            self.sp = zscore(X, axis=1)
            del X
            self.sp = np.maximum(-4, np.minimum(8, self.sp)) + 4
            self.sp /= 12
            self.selected = np.arange(0, self.sp.shape[0]).astype(np.int64)
            self.embedding = self.selected[:, np.newaxis]
            nn = self.sp.shape[0]
            #self.yrange = np.arange(0, nn).astype(np.int32)
            #self.colormat = 255*np.ones((self.sp.shape[0],10,3), dtype=np.int32)
            self.add_imgROI()
            self.ROI_selection()
            self.enable_loaded()
            self.show()
            print('done loading')
            self.loaded = True

    def load_iscell(self):
        basename,filename = os.path.split(self.filebase)
        try:
            iscell = np.load(basename + "/iscell.npy")
            probcell = iscell[:, 1]
            iscell = iscell[:, 0].astype(np.bool)
            file_iscell = basename + "/iscell.npy"
        except (ValueError, OSError, RuntimeError, TypeError, NameError):
            iscell = None
            file_iscell = None
        return iscell, file_iscell

    def load_proc(self, name=None):
        if name is None:
            name = QtGui.QFileDialog.getOpenFileName(
                self, "Open processed file", filter="*.npy"
                )
            self.fname = name[0]
            name = self.fname
            print(name)
        else:
            self.fname = name
        try:
            proc = np.load(name, allow_pickle=True)
            proc = proc.item()
            self.proc = proc
            # do not load X, use reconstruction
            #X    = np.load(self.proc['filename'])
            self.filebase = self.proc['filename']
            y    = self.proc['embedding']
            print(y.shape)
            u    = self.proc['uv'][0]
            v    = self.proc['uv'][1]
            ops  = self.proc['ops']
            X    = u @ v.T
        except (ValueError, KeyError, OSError,
                RuntimeError, TypeError, NameError):
            print('ERROR: this is not a *.npy file :( ')
            X = None
        if X is not None:
            self.filebase = self.proc['filename']
            iscell, file_iscell = self.load_iscell()

            # check if training set used
            if 'train_time' in self.proc:
                if self.proc['train_time'].sum() < self.proc['train_time'].size:
                    # not all training pts used
                    X    = np.load(self.proc['filename'])
                    # show only test timepoints
                    X    = X[:,~self.proc['train_time']]
                    if iscell is not None:
                        if iscell.size == X.shape[0]:
                            X = X[iscell, :]
                            print('using iscell.npy in folder')
                    if len(X.shape) > 2:
                        X = X.mean(axis=-1)
                    v = (u.T @ X).T
                    v /= ((v**2).sum(axis=0))**0.5
                    X = u @ v.T
                    print(X.shape)
            self.startROI = False
            self.endROI = False
            self.posROI = np.zeros((3,2))
            self.prect = np.zeros((5,2))
            self.ROIs = []
            self.ROIorder = []
            self.Rselected = []
            self.Rcolors = []
            self.p0.clear()
            self.sp = X#zscore(X, axis=1)
            del X
            self.sp += 1
            self.sp /= 9
            self.embedding = y
            if ops['n_components'] > 1:
                self.embedded = True
                self.enable_embedded()
            else:
                self.p0.clear()
                self.embedded=False
                self.disable_embedded()

            self.selected = np.argsort(self.embedding[:,0])
            self.enable_loaded()
            #self.usv = usv
            self.U   = u #@ np.diag(usv[1])
            ineur = 0
            # if ROIs saved
            if 'ROIs' in self.proc:
                for r,roi in enumerate(self.proc['ROIs']):
                    if 'color' in roi:
                        self.ROI_add(roi['pos'], roi['prect'], roi['color'])
                    else:
                        self.ROI_add(roi['pos'], roi['prect'],  np.random.randint(255,size=(3,)))

            self.add_imgROI()
            self.ROI_selection(loaded=True)
            if self.embedded:
                self.plot_embedding()
                self.xp = pg.ScatterPlotItem(pos=self.embedding[ineur,:][np.newaxis,:],
                                         symbol='x', pen=pg.mkPen(color=(255,0,0,255), width=3),
                                         size=12)#brush=pg.mkBrush(color=(255,0,0,255)), size=14)
                self.p0.addItem(self.xp)
            self.show()
            self.loaded = True

    def load_behavior(self):
        name = QtGui.QFileDialog.getOpenFileName(
            self, "Open *.npy", filter="*.npy"
        )
        name = name[0]
        bloaded = False
        try:
            beh = np.load(name)
            beh = beh.flatten()
            if beh.size == self.Fcell.shape[1]:
                self.bloaded = True
        except (ValueError, KeyError, OSError,
                RuntimeError, TypeError, NameError):
            print("ERROR: this is not a 1D array with length of data")
        if self.bloaded:
            beh -= beh.min()
            beh /= beh.max()
            self.beh = beh
            b = len(self.colors)
            self.colorbtns.button(b).setEnabled(True)
            self.colorbtns.button(b).setStyleSheet(self.styleUnpressed)
            fig.beh_masks(self)
            fig.plot_trace(self)
            self.show()
        else:
            print("ERROR: this is not a 1D array with length of data")

def run():
    # Always start by initializing Qt (only once per application)
    app = QtGui.QApplication(sys.argv)
    icon_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "logo.png"
    )
    app_icon = QtGui.QIcon()
    app_icon.addFile(icon_path, QtCore.QSize(16, 16))
    app_icon.addFile(icon_path, QtCore.QSize(24, 24))
    app_icon.addFile(icon_path, QtCore.QSize(32, 32))
    app_icon.addFile(icon_path, QtCore.QSize(48, 48))
    app_icon.addFile(icon_path, QtCore.QSize(96, 96))
    app_icon.addFile(icon_path, QtCore.QSize(256, 256))
    app.setWindowIcon(app_icon)
    GUI = MainW()
    ret = app.exec_()
    # GUI.save_gui_data()
    sys.exit(ret)


# run()
