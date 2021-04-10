import sys, time, os
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore
from matplotlib import cm
from scipy.stats import zscore
from .mapping import Rastermap
from . import menus, guiparts, io

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

        menus.mainmenu(self)

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
        self.p1.setLabel('left', 'Neurons')
        self.p1.setLabel('bottom', 'Time')
        # zoom in on a selected image region
        self.selected = np.arange(0,nn,1,int)
        self.p2 = self.win.addPlot(title='ZOOM IN',row=1,col=0,colspan=2)
        self.imgROI = pg.ImageItem(autoDownsample=True)
        self.p2.addItem(self.imgROI)
        self.p2.setMouseEnabled(x=False,y=False)
        self.p2.hideAxis('bottom')
        self.p3 = self.win.addPlot(title='Avg. activity (zoomed in neurons)',row=2,col=0,
                                    colspan=2)
        self.p3.setMouseEnabled(x=False,y=False)
        self.p3.setLabel('bottom', 'Time')
        self.p4 = self.win.addPlot(title='Heatmap',row=3,col=0,colspan=2)
        self.p4.setMouseEnabled(x=False,y=False)
        self.p4.setLabel('bottom', 'Time')
        self.p5 = self.win.addPlot(title='Scatter plot',row=1,col=2)
        self.p5.setMouseEnabled(x=False,y=False)
        self.p5.setLabel('bottom', 'time')
        
        self.win.removeItem(self.p4)
        self.win.removeItem(self.p5)

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
        ysm = QtGui.QLabel("<font color='gray'>Bin neurons:</font>")
        self.smooth = QtGui.QLineEdit(self)
        self.smooth.setValidator(QtGui.QIntValidator(0, 500))
        self.smooth.setText("10")
        self.smooth.setFixedWidth(45)
        self.smooth.setAlignment(QtCore.Qt.AlignRight)
        self.smooth.returnPressed.connect(self.plot_activity)

        params = QtGui.QLabel("<font color='gray'>Rastermap parameters</font>")
        params.setAlignment(QtCore.Qt.AlignCenter)
        self.run_embedding_button = QtGui.QPushButton('Run embedding')
        self.run_embedding_button.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.run_embedding_button.clicked.connect(self.run_RMAP)
        self.run_embedding_button.setEnabled(False)

        self.RadioGroup = QtGui.QButtonGroup()
        self.default_param_radiobutton = QtGui.QRadioButton("Default")
        self.default_param_radiobutton.setStyleSheet("color: gray;")
        self.default_param_radiobutton.setChecked(True)
        self.default_param_radiobutton.toggled.connect(lambda: io.set_params(self))
        self.RadioGroup.addButton(self.default_param_radiobutton)
        self.custom_param_radiobutton = QtGui.QRadioButton("Custom")
        self.custom_param_radiobutton.setStyleSheet("color: gray;")
        self.custom_param_radiobutton.toggled.connect(lambda: io.get_params(self))
        self.RadioGroup.addButton(self.custom_param_radiobutton)

        self.heatmap_checkBox = QtGui.QCheckBox("Behaviour")
        self.heatmap_checkBox.setStyleSheet("color: gray;")
        self.heatmap_checkBox.stateChanged.connect(self.update_plot_p4)
        self.upload_behav_button = QtGui.QPushButton('Upload behavior')
        self.upload_behav_button.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.upload_behav_button.clicked.connect(lambda: io.load_behav_data(self))
        self.upload_behav_button.setEnabled(False)
        self.scatterplot_checkBox = QtGui.QCheckBox("Scatter plot")
        self.scatterplot_checkBox.setStyleSheet("color: gray;")
        self.scatterplot_checkBox.stateChanged.connect(self.update_plot_p5)
        self.upload_run_button = QtGui.QPushButton('Upload run')
        self.upload_run_button.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.upload_run_button.clicked.connect(lambda: io.load_run_data(self))
        self.upload_run_button.setEnabled(False)

        # add slider for levels
        self.sat = [0.3,0.7]
        slider = guiparts.SatSlider(self)
        slider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.l0.addWidget(slider, 0,2,3,1)
        qlabel = guiparts.VerticalLabel(text='saturation')
        qlabel.setStyleSheet('color: white;')
        self.img.setLevels([self.sat[0], self.sat[1]])
        self.imgROI.setLevels([self.sat[0], self.sat[1]])

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
        self.loaded = False
        self.behav_loaded = False
        self.run_loaded = False 

        # Add features to window
        ops_row_pos = 0
        self.l0.addWidget(ysm, ops_row_pos, 0, 1, 1)
        self.l0.addWidget(self.smooth, ops_row_pos, 1, 1, 1)
        self.l0.addWidget(params, ops_row_pos+1, 0, 1, 2)
        self.l0.addWidget(self.default_param_radiobutton, ops_row_pos+2, 0, 1, 1)
        self.l0.addWidget(self.custom_param_radiobutton, ops_row_pos+2, 1, 1, 1)
        self.l0.addWidget(self.run_embedding_button, ops_row_pos+3, 0, 1, 2)
        self.l0.addWidget(self.upload_behav_button, ops_row_pos+4, 0, 1, 1)
        self.l0.addWidget(self.heatmap_checkBox, ops_row_pos+4, 1, 1, 1)
        self.l0.addWidget(self.upload_run_button, ops_row_pos+5, 0, 1, 1)
        self.l0.addWidget(self.scatterplot_checkBox, ops_row_pos+5, 1, 1, 1)
        self.l0.addWidget(qlabel,ops_row_pos+3,1,3,2)

        self.win.show()
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        self.show()

    def reset(self):
        self.run_embedding_button.setEnabled(False)
        self.heatmap_checkBox.setEnabled(False)
        self.scatterplot_checkBox.setEnabled(False)
        self.p1.clear()
        self.p2.clear()
        self.p3.clear()
        self.p4.clear()
        self.p5.clear()
        self.loaded = False
        self.run_loaded = False 
        self.behav_loaded = False

    def update_plot_p4(self):
        if self.heatmap_checkBox.isChecked() and self.behav_loaded:
            self.win.addItem(self.p4, row=3, col=0, colspan=2)
        else:
            print("Please upload a behav file")
            self.win.removeItem(self.p4)
        
    def update_plot_p5(self):
        if self.scatterplot_checkBox.isChecked() and self.run_loaded:
            self.win.addItem(self.p5, row=1, col=2)
        else:
            print("Please upload a behav file")
            self.win.removeItem(self.p5)

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
        if self.loaded:
            avg = self.sp_smoothed[np.ix_(self.selected,self.xrange)].mean(axis=0)
            avg -= avg.min()
            avg /= avg.max()
            self.p3.clear()
            self.p3.plot(self.xrange,avg,pen=(255,0,0))
            if self.run_loaded:
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
            self.upload_behav_button.setEnabled(True)
            self.upload_run_button.setEnabled(True)

    def run_RMAP(self):
        if self.default_param_radiobutton.isChecked():
            io.set_params(self)
        model = Rastermap(smoothness=1, 
                                       n_clusters=self.n_clusters, 
                                       n_PCs=200, 
                                       n_splits=self.n_splits,
                                       grid_upsample=self.grid_upsample).fit(self.sp)

        self.embedding = model.embedding
        self.sorting = model.isort
        self.plot_activity()

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
