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
from rastermap.roi import gROI
import rastermap.run
from rastermap import Rastermap

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
        self.setGeometry(25, 25, 1600, 1000)
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
        self.stylePressed = ("QPushButton {Text-align: left; "
                             "background-color: rgb(100,50,100); "
                             "color:white;}")
        self.styleUnpressed = ("QPushButton {Text-align: left; "
                               "background-color: rgb(50,50,50); "
                               "color:white;}")
        self.styleInactive = ("QPushButton {Text-align: left; "
                              "background-color: rgb(50,50,50); "
                              "color:gray;}")
        self.loaded = False

        # ------ MENU BAR -----------------
        loadMat =  QtGui.QAction("&Load data matrix", self)
        loadMat.setShortcut("Ctrl+L")
        loadMat.triggered.connect(self.load_mat)
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
        self.l0.addWidget(self.win, 0, 0, 38, 30)
        layout = self.win.ci.layout
        # --- embedding image
        self.p0 = self.win.addPlot(row=0, col=0, lockAspect=True)
        self.p0.setAspectLocked(ratio=1)
        self.p0.scene().sigMouseMoved.connect(self.mouse_moved_embedding)
        self.win.ci.layout.setRowStretchFactor(1, .1)

        # ---- colorbar
        self.p3 = self.win.addPlot(row=0, col=1, rowspan=2)
        self.p3.setMouseEnabled(x=False,y=False)
        self.p3.setMenuEnabled(False)
        self.colorimg = pg.ImageItem(autoDownsample=True)
        self.p3.addItem(self.colorimg)
        self.p3.scene().sigMouseMoved.connect(self.mouse_moved_bar)
        #self.p3.hideAxis('bottom')
        # --- activity image
        self.p1 = self.win.addPlot(row=0, col=2,
                                   rowspan=2, invertY=True, padding=0)
        self.p1.setMouseEnabled(x=True, y=False)
        self.img = pg.ImageItem(autoDownsample=False)
        colormap = cm.get_cmap("viridis")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        lut = lut[0:-3,:]
        # apply the colormap
        #self.img.setLookupTable(lut)
        self.img.setLevels([0,1])
        self.p1.setMenuEnabled(False)
        self.p1.scene().contextMenuItem = self.p1
        self.p1.addItem(self.img)
        self.p1.scene().sigMouseMoved.connect(self.mouse_moved_activity)

        # bottom row for buttons
        self.p2 = self.win.addViewBox(row=1, col=0)
        self.p2.setMouseEnabled(x=False,y=False)
        self.p2.setMenuEnabled(False)

        self.win.scene().sigMouseClicked.connect(self.plot_clicked)

        self.win.ci.layout.setColumnStretchFactor(0, 2)
        self.win.ci.layout.setColumnStretchFactor(1, .1)
        self.win.ci.layout.setColumnStretchFactor(2, 3)

        # self.key_on(self.win.scene().keyPressEvent)

        addROI = QtGui.QLabel("<font color='white'>add an ROI by SHIFT click</font>")
        self.l0.addWidget(addROI, 18, 0, 1, 1)
        addROI = QtGui.QLabel("<font color='white'>delete an ROI by ALT click</font>")
        self.l0.addWidget(addROI, 19, 0, 1, 1)
        self.updateROI = QtGui.QPushButton("update plot (ENTER)")
        self.updateROI.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.updateROI.clicked.connect(self.ROI_selection)
        self.updateROI.setStyleSheet(self.styleInactive)
        self.updateROI.setEnabled(False)
        self.l0.addWidget(self.updateROI, 20, 0, 1, 1)
        self.saveROI = QtGui.QPushButton("save ROIs")
        self.saveROI.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.saveROI.clicked.connect(self.ROI_save)
        self.saveROI.setStyleSheet(self.styleInactive)
        self.saveROI.setEnabled(False)
        self.l0.addWidget(self.saveROI, 21, 0, 1, 1)

        # add slider for levels
        self.sl = QtGui.QSlider(QtCore.Qt.Vertical)
        self.sl.setMinimum(0)
        self.sl.setMaximum(100)
        self.sl.setValue(100)
        self.sl.setTickPosition(QtGui.QSlider.TicksLeft)
        self.sl.setTickInterval(10)
        self.sl.valueChanged.connect(self.levelchange)
        self.sl.setTracking(False)
        self.sat = 1.0
        self.l0.addWidget(self.sl,24,2,5,1)
        qlabel = VerticalLabel(text='saturation')
        #qlabel.setStyleSheet('color: white;')
        self.l0.addWidget(qlabel,25,3,3,1)

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
        self.Rselected = []
        self.Rcolors = []
        self.embedded = False

        #self.fname = '/media/carsen/DATA1/BootCamp/mesoscope_cortex/embedding.npy'
        # self.load_behavior('C:/Users/carse/github/TX4/beh.npy')
        #self.fname = 'C:/Users/carse/github/TX4/spks.npy'
        #self.load_proc(self.fname)

        self.show()
        self.win.show()

    def levelchange(self):
        self.sat = float(self.sl.value())/100
        self.img.setLevels([0,self.sat])
        self.show()
        self.win.show()

    def plot_embedding(self):
        self.se = pg.ScatterPlotItem(size=4, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
        pos = self.embedding.T
        spots = [{'pos': pos[:,i], 'data': 1} for i in range(pos.shape[1])] + [{'pos': [0,0], 'data': 1}]
        self.se.addPoints(spots)
        self.p0.addItem(self.se)

    def smooth_activity(self):
        if self.sp.shape[0] == self.selected.size:
            sp_smoothed = self.sp
        else:
            cumsum = np.cumsum(np.concatenate((np.zeros((1,self.sp.shape[1])), self.sp[self.selected,:]), axis=0), axis=0)
            N = 10
            sp_smoothed = (cumsum[N:, :] - cumsum[:-N, :]) / float(N)
            sp_smoothed = zscore(sp_smoothed, axis=1)
            sp_smoothed += 1
            sp_smoothed /= 9
        return sp_smoothed

    def plot_activity(self):
        sp_smoothed = self.smooth_activity()
        self.img.setImage(sp_smoothed)
        self.img.setLevels([0,self.sat])
        self.p1.setXRange(0, self.sp.shape[1], padding=0)
        self.p1.setYRange(0, self.selected.size, padding=0)
        self.p1.setLimits(xMin=0,xMax=self.sp.shape[1],yMin=0,yMax=self.selected.size)
        self.show()
        self.win.show()

    def plot_colorbar(self):
        nneur = self.colormat_plot.shape[0]
        self.colorimg.setImage(self.colormat_plot)
        self.p3.setYRange(0,nneur)
        self.p3.setXRange(0,10)
        self.p3.setLimits(yMin=0,yMax=nneur,xMin=0,xMax=10)
        self.p3.getAxis('bottom').setTicks([[(0,'')]])
        self.win.show()

    def export_fig(self):
        self.win.scene().contextMenuItem = self.p0
        self.win.scene().showExportDialog()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return:
            if self.updateROI.isEnabled:
                self.ROI_selection()

    def ROI_selection(self):
        self.colormat = np.zeros((0,10,3), dtype=np.int64)
        lROI = len(self.Rselected)
        if lROI > 0:
            if lROI > 4:
                self.Ur = np.zeros((lROI, self.U.shape[1]), dtype=np.float32)
                for r,rc in enumerate(self.Rselected):
                    self.Ur[r,:] = self.U[rc,:].mean(axis=0)
                model = Rastermap(n_components=1, n_X=20, init=np.arange(0,lROI).astype(np.int32))
                y     = model.fit_transform(self.Ur)
                y     = y.flatten()
                rsort = np.argsort(y)
                print(rsort)
                self.ROIs = [self.ROIs[i] for i in rsort]
                self.Rselected = [self.Rselected[i] for i in rsort]
                self.Rcolors = [self.Rcolors[i] for i in rsort]
                #self.Rcolors = [y for x,y in sorted(zip(self.Rcolors, rsort))]
                self.selected = np.array([item for sublist in self.Rselected for item in sublist])
                #for r in rsort:
                #    for d in self.Rselected[r]:
                #        self.selected.append(d)
                #self.selected = np.array(self.selected).astype(np.int32)
                #print(self.selected)
            else:
                self.selected = np.array([item for sublist in self.Rselected for item in sublist])
            self.colormat = np.concatenate(self.Rcolors, axis=0)
        else:
            self.selected = np.arange(0, self.X.shape[0]).astype(np.int64)
            self.colormat = 255*np.ones((self.X.shape[0],10,3), dtype=np.int32)
        self.colormat[:,-1,:] = 0
        self.colormat_plot = self.colormat.copy()
        self.plot_activity()
        self.plot_colorbar()

    def ROI_add(self, pos, prect):
        self.ROIs.append(gROI(pos, prect, self))
        self.Rselected.append(self.ROIs[-1].selected)
        self.Rcolors.append(np.reshape(np.tile(self.ROIs[-1].color, 10 * self.Rselected[-1].size),
                            (self.Rselected[-1].size, 10, 3)))
        #self.ROI_selection()

    def ROI_remove(self, p):
        p = np.array(p)
        for n in range(len(self.ROIs)-1,-1,-1):
            ptrue = self.ROIs[n].inROI(p)
            if ptrue.shape[0] > 0:
                self.ROIs[n].remove(self)
                del self.ROIs[n]
                del self.Rselected[n]
                del self.Rcolors[n]
                break
        #self.ROI_selection()

    def ROI_save(self):
        name = QtGui.QFileDialog.getSaveFileName(self,'ROI name (*.npy)')
        name = name[0]
        self.proc['ROIs'] = []
        for r in self.ROIs:
            self.proc['ROIs'].append({'pos': r.pos, 'prect': r.prect})
        np.save(name, self.proc)

    def enable_loaded(self):
        self.runRMAP.setEnabled(True)

    def enable_embedded(self):
        self.updateROI.setEnabled(True)
        self.saveROI.setEnabled(True)
        self.updateROI.setStyleSheet(self.styleUnpressed)
        self.saveROI.setStyleSheet(self.styleUnpressed)


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
                        self.prect = rect_from_line(self.posROI, d)
                        self.p0.removeItem(self.l0)
                        self.l0 = pg.PlotDataItem(self.prect[:,0], self.prect[:,1])
                        self.p0.addItem(self.l0)
                else:
                    dists = (self.embedding[self.selected,0] - x)**2 + (self.embedding[self.selected,1] - y)**2
                    ineur = np.argmin(dists.flatten()).astype(int)
                    self.update_selected(ineur)


    def mouse_moved_activity(self, pos):
        if self.loaded:
            if self.p1.sceneBoundingRect().contains(pos):
                y = self.p1.vb.mapSceneToView(pos).y()
                ineur = min(self.colormat.shape[0]-1, max(0, int(np.floor(y))))
                self.update_selected(ineur)

    def mouse_moved_bar(self, pos):
        if self.loaded:
            if self.p3.sceneBoundingRect().contains(pos):
                y = self.p3.vb.mapSceneToView(pos).y()
                ineur = min(self.colormat.shape[0]-1, max(0, int(np.floor(y))))
                self.update_selected(ineur)

    def update_selected(self, ineur):
        # add bar to colorbar
        self.colormat_plot = self.colormat.copy()
        self.colormat_plot[ineur,:,:] = np.zeros((10,3)).astype(int)
        self.plot_colorbar()
        # x point on embedding
        if self.embedded:
            ineur = self.selected[ineur]
            self.xp.setData(pos=self.embedding[ineur,:][np.newaxis,:])

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
                        elif self.startROI:
                            self.posROI[1,:] = [x,y]
                            self.endROI = True
                            self.startROI = False
                        elif self.endROI:
                            self.posROI[2,:] = [x,y]
                            self.endROI = False
                            self.p0.removeItem(self.l0)
                            self.ROI_add(self.posROI, self.prect)
                        elif event.modifiers() == QtCore.Qt.ShiftModifier:
                            self.startROI = True
                            self.posROI[0,:] = [x,y]
                        elif event.modifiers() == QtCore.Qt.AltModifier:
                            self.ROI_remove([x,y])

                elif x == self.p1:
                    iplot = 1
                    if event.double():
                        self.zoom_plot(iplot)

    def zoom_plot(self, iplot):
        if iplot == 0:
            self.p0.setXRange(self.embedding[:,0].min(), self.embedding[:,0].max())
            self.p0.setYRange(self.embedding[:,1].min(), self.embedding[:,1].max())
        else:
            self.p1.setYRange(0, self.X.shape[0])
            self.p1.setXRange(0, self.X.shape[1])
        self.show()

    def run_RMAP(self):
        RW = rastermap.run.RunWindow(self)
        RW.show()

    def load_mat(self):
        name = QtGui.QFileDialog.getOpenFileName(
            self, "Open *.npy", filter="*.npy"
            )
        self.fname = name[0]
        self.filebase = name[0]
        try:
            X = np.load(self.fname)
            print(X.shape)
        except (ValueError, KeyError, OSError,
                RuntimeError, TypeError, NameError):
            print('ERROR: this is not a *.npy array :( ')
            X = None
        if X is not None and X.ndim > 1:
            iscell, file_iscell = self.load_iscell()
            self.file_iscell = None
            self.X = X
            if iscell is not None:
                if iscell.size == self.X.shape[0]:
                    self.X = self.X[iscell, :]
                    self.file_iscell = file_iscell
                    print('using iscell.npy in folder')

            self.p0.clear()
            self.sp = zscore(self.X, axis=1)
            self.sp += 1
            self.sp /= 9
            self.selected = np.arange(0, self.X.shape[0]).astype(np.int64)
            self.ROI_selection()
            self.enable_loaded()
            self.show()
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
            proc = np.load(name)
            proc = proc.item()
            self.proc = proc
            X    = np.load(self.proc['filename'])
            self.filebase = self.proc['filename']
            y    = self.proc['embedding']
            usv  = self.proc['usv']
        except (ValueError, KeyError, OSError,
                RuntimeError, TypeError, NameError):
            print('ERROR: this is not a *.npy file :( ')
            X = None
        if X is not None:
            iscell, file_iscell = self.load_iscell()
            self.X = X
            self.file_iscell = None
            if iscell is not None:
                if iscell.size == self.X.shape[0]:
                    self.X = self.X[iscell, :]
                    self.file_iscell = file_iscell
                    print('using iscell.npy in folder')
            self.p0.clear()
            self.sp = zscore(self.X, axis=1)
            self.sp += 1
            self.sp /= 9
            self.selected = np.arange(0, self.X.shape[0]).astype(np.int64)
            self.embedding = y
            self.embedded = True
            self.enable_loaded()
            self.enable_embedded()
            self.usv = usv
            self.U   = usv[0] @ np.diag(usv[1])
            ineur = 0
            self.xp = pg.ScatterPlotItem(pos=self.embedding[ineur,:][np.newaxis,:],
                                         symbol='x', size=14, pen=pg.mkPen(color=(255,255,255)))
            self.p0.addItem(self.xp)
            # if ROIs saved
            if 'ROIs' in self.proc:
                for r in self.proc['ROIs']:
                    self.ROI_add(r['pos'], r['prect'])
            self.plot_embedding()
            self.plot_activity()
            self.ROI_selection()
            self.plot_colorbar()
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
