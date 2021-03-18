import sys, time, os

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore
from matplotlib import cm
from scipy.stats import zscore

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

class RangeSlider(QtGui.QSlider):
    """ A slider for ranges.

        This class provides a dual-slider for ranges, where there is a defined
        maximum and minimum, as is a normal slider, but instead of having a
        single slider value, there are 2 slider values.

        This class emits the same signals as the QSlider base class, with the
        exception of valueChanged

        Found this slider here: https://www.mail-archive.com/pyqt@riverbankcomputing.com/msg22889.html
        and modified it
    """
    def __init__(self, parent=None, *args):
        super(RangeSlider, self).__init__(*args)

        self._low = self.minimum()
        self._high = self.maximum()

        self.pressed_control = QtGui.QStyle.SC_None
        self.hover_control = QtGui.QStyle.SC_None
        self.click_offset = 0

        self.setOrientation(QtCore.Qt.Vertical)
        self.setTickPosition(QtGui.QSlider.TicksRight)
        self.setStyleSheet(\
                "QSlider::handle:horizontal {\
                background-color: white;\
                border: 1px solid #5c5c5c;\
                border-radius: 0px;\
                border-color: black;\
                height: 8px;\
                width: 6px;\
                margin: -8px 2; \
                }")
        # 0 for the low, 1 for the high, -1 for both
        self.active_slider = 0
        self.parent = parent

    def level_change(self):
        self.saturation = [self._low, self._high]

    def low(self):
        return self._low

    def setLow(self, low):
        self._low = low
        self.update()

    def high(self):
        return self._high

    def setHigh(self, high):
        self._high = high
        self.update()

    def paintEvent(self, event):
        # based on http://qt.gitorious.org/qt/qt/blobs/master/src/gui/widgets/qslider.cpp
        painter = QtGui.QPainter(self)
        style = QtGui.QApplication.style()
        for i, value in enumerate([self._low, self._high]):
            opt = QtGui.QStyleOptionSlider()
            self.initStyleOption(opt)
            # Only draw the groove for the first slider so it doesn't get drawn
            # on top of the existing ones every time
            if i == 0:
                opt.subControls = QtGui.QStyle.SC_SliderHandle#QtGui.QStyle.SC_SliderGroove | QtGui.QStyle.SC_SliderHandle
            else:
                opt.subControls = QtGui.QStyle.SC_SliderHandle
            if self.tickPosition() != self.NoTicks:
                opt.subControls |= QtGui.QStyle.SC_SliderTickmarks
            if self.pressed_control:
                opt.activeSubControls = self.pressed_control
                opt.state |= QtGui.QStyle.State_Sunken
            else:
                opt.activeSubControls = self.hover_control
            opt.sliderPosition = value
            opt.sliderValue = value
            style.drawComplexControl(QtGui.QStyle.CC_Slider, opt, painter, self)

    def mousePressEvent(self, event):
        event.accept()
        style = QtGui.QApplication.style()
        button = event.button()
        if button:
            opt = QtGui.QStyleOptionSlider()
            self.initStyleOption(opt)
            self.active_slider = -1
            for i, value in enumerate([self._low, self._high]):
                opt.sliderPosition = value
                hit = style.hitTestComplexControl(style.CC_Slider, opt, event.pos(), self)
                if hit == style.SC_SliderHandle:
                    self.active_slider = i
                    self.pressed_control = hit
                    self.triggerAction(self.SliderMove)
                    self.setRepeatAction(self.SliderNoAction)
                    self.setSliderDown(True)
                    break
            if self.active_slider < 0:
                self.pressed_control = QtGui.QStyle.SC_SliderHandle
                self.click_offset = self.__pixelPosToRangeValue(self.__pick(event.pos()))
                self.triggerAction(self.SliderMove)
                self.setRepeatAction(self.SliderNoAction)
        else:
            event.ignore()
    def mouseMoveEvent(self, event):
        if self.pressed_control != QtGui.QStyle.SC_SliderHandle:
            event.ignore()
            return
        event.accept()
        new_pos = self.__pixelPosToRangeValue(self.__pick(event.pos()))
        opt = QtGui.QStyleOptionSlider()
        self.initStyleOption(opt)
        if self.active_slider < 0:
            offset = new_pos - self.click_offset
            self._high += offset
            self._low += offset
            if self._low < self.minimum():
                diff = self.minimum() - self._low
                self._low += diff
                self._high += diff
            if self._high > self.maximum():
                diff = self.maximum() - self._high
                self._low += diff
                self._high += diff
        elif self.active_slider == 0:
            if new_pos >= self._high:
                new_pos = self._high - 1
            self._low = new_pos
        else:
            if new_pos <= self._low:
                new_pos = self._low + 1
            self._high = new_pos
        self.click_offset = new_pos
        self.update()
    def mouseReleaseEvent(self, event):
        self.level_change()
    def __pick(self, pt):
        if self.orientation() == QtCore.Qt.Horizontal:
            return pt.x()
        else:
            return pt.y()
    def __pixelPosToRangeValue(self, pos):
        opt = QtGui.QStyleOptionSlider()
        self.initStyleOption(opt)
        style = QtGui.QApplication.style()

        gr = style.subControlRect(style.CC_Slider, opt, style.SC_SliderGroove, self)
        sr = style.subControlRect(style.CC_Slider, opt, style.SC_SliderHandle, self)

        if self.orientation() == QtCore.Qt.Horizontal:
            slider_length = sr.width()
            slider_min = gr.x()
            slider_max = gr.right() - slider_length + 1
        else:
            slider_length = sr.height()
            slider_min = gr.y()
            slider_max = gr.bottom() - slider_length + 1

        return style.sliderValueFromPosition(self.minimum(), self.maximum(),
                                             pos-slider_min, slider_max-slider_min,
                                             opt.upsideDown)

class SatSlider(RangeSlider):
    def __init__(self, parent=None):
        super(SatSlider, self).__init__(parent)
        self.parent = parent
        self.setMinimum(0)
        self.setMaximum(100)
        self.setLow(30)
        self.setHigh(70)

    def level_change(self):
        self.parent.sat[0] = float(self._low)/100
        self.parent.sat[1] = float(self._high)/100
        self.parent.img.setLevels([self.parent.sat[0],self.parent.sat[1]])
        self.parent.imgROI.setLevels([self.parent.sat[0],self.parent.sat[1]])
        self.parent.win.show()

class NeuronSlider(RangeSlider):
    def __init__(self, parent=None):
        super(SatSlider, self).__init__(parent)
        self.parent = parent
        self.setMinimum(0)
        self.setMaximum(100)
        self.setLow(30)
        self.setHigh(70)

    def level_change(self):
        self.parent.sat[0] = float(self._low)/100
        self.parent.sat[1] = float(self._high)/100
        self.parent.img.setLevels([self.parent.sat[0],self.parent.sat[1]])
        self.parent.imgROI.setLevels([self.parent.sat[0],self.parent.sat[1]])
        self.parent.win.show()

class Slider(QtGui.QSlider):
    def __init__(self, bid, parent=None):
        super(self.__class__, self).__init__()
        self.bid = bid
        self.setMinimum(0)
        self.setMaximum(100)
        self.setValue(parent.sat[bid]*100)
        self.setTickPosition(QtGui.QSlider.TicksLeft)
        self.setTickInterval(10)
        self.valueChanged.connect(lambda: self.level_change(parent,bid))
        self.setTracking(False)

    def level_change(self, parent, bid):
        parent.sat[bid] = float(self.value())/100
        parent.img.setLevels([parent.sat[0],parent.sat[1]])
        parent.imgROI.setLevels([parent.sat[0],parent.sat[1]])
        parent.win.show()



class MainW(QtGui.QMainWindow):
    def __init__(self):
        super(MainW, self).__init__()
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(25, 25, 1800, 1000)
        self.setWindowTitle("Rastermap - neural data visualization")
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
        #file_menu.addAction(loadProc)
        file_menu.addAction(self.runRMAP)
        file_menu.addAction(self.loadBeh)
        file_menu.addAction(exportFig)

        self.cwidget = QtGui.QWidget(self)
        self.setCentralWidget(self.cwidget)
        self.l0 = QtGui.QGridLayout()
        #layout = QtGui.QFormLayout()
        self.cwidget.setLayout(self.l0)
        #self.p0 = pg.ViewBox(lockAspect=False,name='plot1',border=[100,100,100],invertY=True)
        self.win = pg.GraphicsLayoutWidget()
        
        sp = np.zeros((100,100), np.float32)
        self.sp = sp
        nt = sp.shape[1]
        nn = sp.shape[0]

        # --- cells image
        self.win = pg.GraphicsLayoutWidget()
        self.win.move(600,0)
        self.win.resize(1000,500)
        self.l0.addWidget(self.win,0,0,14,14)
        layout = self.win.ci.layout
        # A plot area (ViewBox + axes) for displaying the image
        self.p0 = self.win.addViewBox(row=0,col=0)
        self.p0.setMenuEnabled(False)
        self.p1 = self.win.addPlot(title="FULL VIEW",row=0,col=1)
        self.p1.setMouseEnabled(x=False,y=False)
        self.img = pg.ImageItem(autoDownsample=True)
        self.p1.addItem(self.img)
        self.p1.setXRange(0,nt)
        self.p1.setYRange(0,nn)
        self.p1.setLabel('left', 'neurons')
        self.p1.setLabel('bottom', 'time')
        # zoom in on a selected image region
        self.selected = np.arange(0,nn,1,int)
        self.p2 = self.win.addPlot(title='ZOOM IN',row=1,col=0,colspan=2)
        self.imgROI = pg.ImageItem(autoDownsample=True)
        self.p2.addItem(self.imgROI)
        self.p2.setMouseEnabled(x=False,y=False)
        self.p2.hideAxis('bottom')
        self.p3 = self.win.addPlot(title='',row=2,col=0,colspan=2)
        self.p3.setMouseEnabled(x=False,y=False)
        self.p3.setLabel('bottom', 'time')
        # set colormap to viridis
        colormap = cm.get_cmap("gray_r")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        lut = lut[0:-3,:]
        # apply the colormap
        self.img.setLookupTable(lut)
        self.imgROI.setLookupTable(lut)
        layout.setColumnStretchFactor(1,3)
        layout.setRowStretchFactor(1,3)

        # add bin size across neurons
        ysm = QtGui.QLabel("<font color='white'>y-binning</font>")
        ysm.setFixedWidth(60)
        self.l0.addWidget(ysm, 0, 0, 1, 1)
        self.smooth = QtGui.QLineEdit(self)
        self.smooth.setValidator(QtGui.QIntValidator(0, 500))
        self.smooth.setText("10")
        self.smooth.setFixedWidth(45)
        self.smooth.setAlignment(QtCore.Qt.AlignRight)
        self.smooth.returnPressed.connect(self.plot_activity)
        self.l0.addWidget(self.smooth, 0, 1, 1, 1)

        # add slider for levels
        self.sat = [0.3,0.7]
        slider = SatSlider(self)
        slider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.l0.addWidget(slider, 0,2,3,1)
        qlabel = VerticalLabel(text='saturation')
        qlabel.setStyleSheet('color: white;')
        self.img.setLevels([self.sat[0], self.sat[1]])
        self.imgROI.setLevels([self.sat[0], self.sat[1]])
        self.l0.addWidget(qlabel,3,1,3,2)

        # ROI on main plot
        redpen = pg.mkPen(pg.mkColor(255, 0, 0),
                                width=3,
                                style=QtCore.Qt.SolidLine)
        self.ROI = pg.RectROI([nt*.25, -1], [nt*.25, nn+1],
                      maxBounds=QtCore.QRectF(-1.,-1.,nt+1,nn+1),
                      pen=redpen)
        self.xrange = np.arange(nt*.25, nt*.5,1,int)
        self.ROI.handleSize = 10
        self.ROI.handlePen = redpen
        # Add right Handle
        self.ROI.handles = []
        self.ROI.addScaleHandle([1, 0.5], [0., 0.5])
        self.ROI.addScaleHandle([0., 0.5], [1., 0.5])
        self.ROI.sigRegionChangeFinished.connect(self.ROI_position)
        self.p1.addItem(self.ROI)
        self.ROI.setZValue(10)  # make sure ROI is drawn above image

        self.LINE = pg.RectROI([-1, nn*.4], [nt*.25, nn*.2],
                      maxBounds=QtCore.QRectF(-1,-1.,nt*.25,nn+1),
                      pen=redpen)
        self.selected = np.arange(nn*.4, nn*.6, 1, int)
        self.LINE.handleSize = 10
        self.LINE.handlePen = redpen
        # Add top handle
        self.LINE.handles = []
        self.LINE.addScaleHandle([0.5, 1], [0.5, 0])
        self.LINE.addScaleHandle([0.5, 0], [0.5, 1])
        self.LINE.sigRegionChangeFinished.connect(self.LINE_position)
        self.p2.addItem(self.LINE)
        self.LINE.setZValue(10)  # make sure ROI is drawn above image


        greenpen = pg.mkPen(pg.mkColor(0, 255, 0),
                                width=3,
                                style=QtCore.Qt.SolidLine)
        
        self.tpos = -0.5
        self.tsize = 1
        self.bloaded = False 
        self.loaded = False

        self.win.show()
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        self.show()

    def export_fig(self):
        self.win.scene().contextMenuItem = self.p0
        self.win.scene().showExportDialog()


    def plot_clicked(self,event):
        items = self.win.scene().items(event.scenePos())
        for x in items:
            if x==self.p1:
                if event.button()==1:
                    if event.double():
                        self.ROI.setPos([-1,-1])
                        self.ROI.setSize([self.sp.shape[1]+1, self.sp.shape[0]+1])

    def keyPressEvent(self, event):
        bid = -1
        move = False
        nn,nt = self.sp.shape
        if event.modifiers() !=  QtCore.Qt.ShiftModifier:
            if event.key() == QtCore.Qt.Key_Down:
                bid = 0
            elif event.key() == QtCore.Qt.Key_Up:
                bid=1
            elif event.key() == QtCore.Qt.Key_Left:
                bid=2
            elif event.key() == QtCore.Qt.Key_Right:
                bid=3
            if bid==2 or bid==3:
                xrange,yrange = self.roi_range(self.ROI)
                if xrange.size < nt:
                    # can move
                    if bid==2:
                        move = True
                        xrange = xrange - np.minimum(xrange.min()+1,nt*0.05)
                    else:
                        move = True
                        xrange = xrange + np.minimum(nt-xrange.max()-1,nt*0.05)
                    if move:
                        self.ROI.setPos([xrange.min()-1, -1])
                        self.ROI.setSize([xrange.size+1, nn+1])
            if bid==0 or bid==1:
                xrange,yrange = self.roi_range(self.LINE)
                if yrange.size < nn:
                    # can move
                    if bid==0:
                        move = True
                        yrange = yrange - np.minimum(yrange.min(),nn*0.05)
                    else:
                        move = True
                        yrange = yrange + np.minimum(nn-yrange.max()-1,nn*0.05)
                    if move:
                        self.LINE.setPos([-1, yrange.min()])
                        self.LINE.setSize([self.xrange.size+1,  yrange.size])
        else:
            if event.key() == QtCore.Qt.Key_Down:
                bid = 0
            elif event.key() == QtCore.Qt.Key_Up:
                bid=1
            elif event.key() == QtCore.Qt.Key_Left:
                bid=2
            elif event.key() == QtCore.Qt.Key_Right:
                bid=3
            if bid==2 or bid==3:
                xrange,_ = self.roi_range(self.ROI)
                dx = nt*0.05 / (nt/xrange.size)
                if bid==2:
                    if xrange.size > dx:
                        # can move
                        move = True
                        xmax = xrange.size - dx
                        xrange = xrange.min() + np.arange(0,xmax).astype(np.int32)
                else:
                    if xrange.size < nt-dx + 1:
                        move = True
                        xmax = xrange.size + dx
                        xrange = xrange.min() + np.arange(0,xmax).astype(np.int32)
                if move:
                    self.ROI.setPos([xrange.min()-1, -1])
                    self.ROI.setSize([xrange.size+1, nn+1])

            elif bid>=0:
                _,yrange = self.roi_range(self.LINE)
                dy = nn*0.05 / (nn/yrange.size)
                if bid==0:
                    if yrange.size > dy:
                        # can move
                        move = True
                        ymax = yrange.size - dy
                        yrange = yrange.min() + np.arange(0,ymax).astype(np.int32)
                else:
                    if yrange.size < nn-dy + 1:
                        move = True
                        ymax = yrange.size + dy
                        yrange = yrange.min() + np.arange(0,ymax).astype(np.int32)
                if move:
                    self.LINE.setPos([-1, yrange.min()])
                    self.LINE.setSize([self.xrange.size+1,  yrange.size])

    
    def roi_range(self, roi):
        pos = roi.pos()
        posy = pos.y()
        posx = pos.x()
        sizex,sizey = roi.size()
        xrange = (np.arange(0,int(sizex)) + int(posx)).astype(np.int32)
        yrange = (np.arange(0,int(sizey)) + int(posy)).astype(np.int32)
        xrange = xrange[xrange>=0]
        xrange = xrange[xrange<self.sp.shape[1]]
        yrange = yrange[yrange>=0]
        yrange = yrange[yrange<self.sp.shape[0]]
        return xrange,yrange

    def plot_traces(self):
        avg = self.sp_smoothed[np.ix_(self.selected,self.xrange)].mean(axis=0)
        avg -= avg.min()
        avg /= avg.max()
        self.p3.clear()
        self.p3.plot(self.xrange,avg,pen=(255,0,0))
        if self.bloaded:
            self.p3.plot(self.parent.beh_time,self.parent.beh,pen='w')
        self.p3.setXRange(self.xrange[0],self.xrange[-1])
        self.p3.show()
        
    def LINE_position(self):
        _,yrange = self.roi_range(self.LINE)
        self.selected = yrange.astype('int')
        self.plot_traces()


    def ROI_position(self):
        xrange,_ = self.roi_range(self.ROI)
        self.xrange = xrange
        self.imgROI.setImage(self.sp_smoothed[:, self.xrange])
        self.p2.setXRange(0,self.xrange.size)

        self.plot_traces()

        # reset ROIs
        self.LINE.maxBounds = QtCore.QRectF(-1,-1.,
                                xrange.size+1,self.sp.shape[0]+1)
        self.LINE.setSize([xrange.size+1, self.selected.size])
        self.LINE.setZValue(10)

        axy = self.p2.getAxis('left')
        axx = self.p2.getAxis('bottom')
        #axy.setTicks([[(0.0,str(yrange[0])),(float(yrange.size),str(yrange[-1]))]])
        self.imgROI.setLevels([self.sat[0], self.sat[1]])


    def smooth_activity(self):
        N = int(self.smooth.text())
        if N > 1:
            NN = self.sp.shape[0]
            nn = int(np.floor(NN/N))
            self.sp_smoothed = np.reshape(self.sp[self.sorting][:nn*N].copy(), (nn, N, -1)).mean(axis=1)
            self.sp_smoothed = zscore(self.sp_smoothed, axis=1)
            self.sp_smoothed = np.maximum(-4, np.minimum(8, self.sp_smoothed)) + 4
            self.sp_smoothed /= 12

    def plot_activity(self):
        if self.loaded:
            self.smooth_activity()
            nn = self.sp_smoothed.shape[0]
            nt = self.sp_smoothed.shape[1]
            print(nn, nt)
            self.img.setImage(self.sp_smoothed)
            self.img.setLevels([self.sat[0],self.sat[1]])
            self.p1.setXRange(-nt*0.01, nt*1.01, padding=0)
            self.p1.setYRange(-nn*0.01, nn*1.01, padding=0)
            self.p1.show()
            self.p2.setXRange(0, nt, padding=0)
            self.p2.setYRange(0, nn, padding=0)
            self.p2.show()
            self.ROI.setPos(-.5,-.5)
            self.ROI.setSize([nt+.5,nn+.5])
            self.ROI.maxBounds = QtCore.QRectF(-1.,-1.,nt+1,nn+1)
            self.ROI_position()
        self.show()
        self.win.show()

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
            self.embedding = np.arange(0, self.sp.shape[0]).astype(np.int64)[:,np.newaxis]
            self.sorting = np.arange(0, self.sp.shape[0]).astype(np.int64)
            
            self.loaded = True
            self.plot_activity()
            self.show()
            self.runRMAP.setEnabled(True)
            self.loadBeh.setEnabled(True)
            print('done loading')

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

            self.enable_loaded()
            #self.usv = usv
            self.U   = u #@ np.diag(usv[1])
            ineur = 0

            self.ROI_position()
            if self.embedded:
                self.plot_embedding()
                self.xp = pg.ScatterPlotItem(pos=self.embedding[ineur,:][np.newaxis,:],
                                         symbol='x', pen=pg.mkPen(color=(255,0,0,255), width=3),
                                         size=12)#brush=pg.mkBrush(color=(255,0,0,255)), size=14)
                self.p0.addItem(self.xp)
            self.show()
            self.loaded = True

    def run_RMAP(self):
        RW = RunWindow(self)
        RW.show()

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
