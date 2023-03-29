import sys, time, os
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore
from PyQt5 import QtWidgets as QtW
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QScrollBar, QSlider, QComboBox, QGridLayout, QPushButton, QFrame, QCheckBox, QLabel, QProgressBar, QLineEdit, QMessageBox, QGroupBox, QButtonGroup, QRadioButton, QStatusBar
from scipy.stats import zscore, pearsonr
# patch for Qt 5.15 on macos >= 12
os.environ["USE_MAC_SLIDER_PATCH"] = "1"
from superqt import QRangeSlider  # noqa
Horizontal = QtCore.Qt.Orientation.Horizontal
Vertical = QtCore.Qt.Orientation.Vertical

from . import menus, guiparts, io, colormaps, views

nclust_max = 100

class MainW(QMainWindow):
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

        self.cwidget = QWidget(self)
        self.setCentralWidget(self.cwidget)
        self.l0 = QGridLayout()
        self.cwidget.setLayout(self.l0)
        self.win = pg.GraphicsLayoutWidget()
        self.win.move(600,0)
        self.win.resize(1000,500)
        self.l0.addWidget(self.win,0,0,22,14)
        layout = self.win.ci.layout
        
        # Neural activity/spike dataset set as self.sps
        sp = np.zeros((100,100), np.float32)
        self.sp = sp
        nt = sp.shape[1]
        nn = sp.shape[0]

        # --- cells image        
        # A plot area (ViewBox + axes) for displaying the image
        self.p0 = self.win.addViewBox(lockAspect=True, row=0,col=0)
        self.p0.setMenuEnabled(False)

        # Plot entire neural activity dataset
        self.p1 = self.win.addPlot(title="FULL VIEW",row=0,col=1,colspan=2)
        self.p1.setMouseEnabled(x=False,y=False)
        self.img = pg.ImageItem(autoDownsample=True)
        self.p1.addItem(self.img)
        self.p1.setXRange(0,nt)
        self.p1.setYRange(0,nn)
        self.p1.setLabel('left', 'binned neurons')
        self.p1.setLabel('bottom', 'time')
        self.p1.invertY(True)

        # Plot a zoomed in region from full view (changes across time axis)
        self.selected = slice(0, nn)
        self.p2 = self.win.addPlot(title='ZOOM IN',row=1,col=0,colspan=2,rowspan=1)
        self.p2.setMenuEnabled(False)
        self.imgROI = pg.ImageItem(autoDownsample=True)
        self.p2.addItem(self.imgROI)
        self.p2.setMouseEnabled(x=False, y=False)
        self.p2.setLabel('bottom', 'time')
        self.p2.setLabel('left', 'binned neurons')
        ax = self.p2.getAxis('bottom')
        ticks = [0]
        ax.setTicks([[(v, '.') for v in ticks ]])
        self.p2.invertY(True)
        self.p2.scene().sigMouseMoved.connect(self.mouse_moved)

        # Plot avg. activity of neurons selected in ROI of zoomed in view
        self.p3 = self.win.addPlot(row=2, col=0, rowspan=1,
                                   colspan=2, padding=0)
        self.p3.setMouseEnabled(x=False,y=False)
        self.p3.setLabel('bottom', 'time')
        self.p3.setLabel('left', 'selected')
        self.cluster_plots = []
        for i in range(nclust_max):
            self.cluster_plots.append(pg.PlotDataItem())
        for i in range(nclust_max):
            self.p3.addItem(self.cluster_plots[i])
        
        # Plot behavioral dataset as heatmap
        self.p4 = self.win.addPlot(row=3,col=0,colspan=2,rowspan=1)
        self.p4.setMouseEnabled(x=False,y=False)
        self.p4.setLabel('bottom', 'time')
        self.p4.setLabel('left', 'behavior')

        # align plots
        self.p2.getAxis('left').setWidth(int(40))
        self.p3.getAxis('left').setWidth(int(40))
        self.p4.getAxis('left').setWidth(int(40))

        # Scatter plot for oned correlation, neuron position, and depth (ephys) information
        self.p5 = self.win.addPlot(title='scatter plot',row=1,col=2)
        self.p5.setMouseEnabled(x=False,y=False)
        self.scatter_plots = [[]]
        for i in range(nclust_max+1):
            self.scatter_plots[0].append(pg.ScatterPlotItem())
        for i in range(nclust_max+1):
            self.p5.addItem(self.scatter_plots[0][i])
                
        
        # Set colormap to deafult of gray_r. ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Future: add option to change cmap ~~~~~~~~~~~~~~
        lut = colormaps.gray[::-1]
        # apply the colormap
        self.img.setLookupTable(lut)
        self.imgROI.setLookupTable(lut)
        layout.setColumnStretchFactor(1,3)
        layout.setRowStretchFactor(1,4)
        layout.setRowStretchFactor(2,2)
        layout.setRowStretchFactor(3,0.5)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Options on top left of GUI ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # add bin size across neurons
        ysm = QLabel("<font color='gray'>bin neurons:</font>")
        self.smooth = QLineEdit(self)
        self.smooth.setValidator(QtGui.QIntValidator(0, 500))
        self.smooth.setText("10")
        self.smooth.setFixedWidth(45)
        self.smooth.setAlignment(QtCore.Qt.AlignRight)
        self.smooth.returnPressed.connect(self.plot_activity)

        # Add slider for saturation  
        self.sat = [30.,70.]
        self.sat_slider = QRangeSlider(Horizontal)
        self.sat_slider.setRange(0., 200.)
        self.sat_slider.setTickPosition(QtW.QSlider.TickPosition.TicksAbove)
        self.sat_slider.valueChanged.connect(self.sat_changed)
        self.sat_slider.setValue((self.sat[0], self.sat[1]))
        sat_label = QLabel("Saturation")
        sat_label.setStyleSheet('color: white;')
        
        # Add drop down options for scatter plot
        self.scatter_comboBox = QComboBox(self)
        self.scatter_comboBox.setFixedWidth(120)
        scatter_comboBox_ops = ["-- Select --", "neuron position", "1D correlation"]
        self.scatter_comboBox.setEditable(True)
        self.scatter_comboBox.addItems(scatter_comboBox_ops)
        self.scatter_comboBox.setCurrentIndex(0)
        self.scatter_comboBox.setCurrentIndex(0)
        self.all_neurons_checkBox = QCheckBox("color all neurons")
        self.all_neurons_checkBox.setStyleSheet("color: gray;")
        self.scatterplot_button = QPushButton('plot')
        self.scatterplot_button.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.scatterplot_button.clicked.connect(self.plot_scatter_pressed)
        self.scatterplot_button_3D = QPushButton('view 3D')
        self.scatterplot_button_3D.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.scatterplot_button_3D.clicked.connect(self.plane_window)
        

        # ROI on main plot
        bluepen = pg.mkPen(pg.mkColor(0, 0, 255),
                                width=3,
                                style=QtCore.Qt.SolidLine)
        redpen = pg.mkPen(pg.mkColor(255, 0, 0),
                                width=3,
                                style=QtCore.Qt.SolidLine)
        self.ROI = pg.RectROI([nt*.25, -1], [nt*.25, nn+1],
                      maxBounds=QtCore.QRectF(-1.,-1.,nt+1,nn+1),
                      pen=bluepen)
        self.xrange = np.arange(nt*.25, nt*.5,1,int)
        self.ROI.handles = []
        self.ROI.sigRegionChangeFinished.connect(self.ROI_position)
        self.p1.addItem(self.ROI)
        self.ROI.setZValue(10)  # make sure ROI is drawn above image

        self.cluster_rois = []
        
        # Status bar
        self.statusBar = QStatusBar()
        
        # Default variables
        self.tpos = -0.5
        self.tsize = 1
        self.reset_variables()

        # Add features to window
        ops_row_pos = 0
        self.l0.addWidget(ysm, ops_row_pos, 0, 1, 1)
        self.l0.addWidget(self.smooth, ops_row_pos, 1, 1, 1)
        self.l0.addWidget(sat_label,ops_row_pos+1,0,1,2)
        self.l0.addWidget(self.sat_slider, ops_row_pos+2,0,1,2)

        self.l0.addWidget(self.scatterplot_button,16,12,1,1)
        self.l0.addWidget(self.scatterplot_button_3D,16,13,1,1)
        self.l0.addWidget(self.scatter_comboBox,17,12,1,1)
        self.l0.addWidget(self.all_neurons_checkBox,17,13,1,1)

        
        self.win.show()
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)

        #io.load_mat(self, '/media/carsen/ssd2/TX60/suite2p/plane0/spks.npy')
        self.show()

    def randomize_colors(self, random=False):
        np.random.seed(0 if not random else np.random.randint(500))
        rperm = np.random.permutation(nclust_max)
        self.colors = colormaps.gist_rainbow[np.linspace(0, 254, nclust_max).astype('int')][rperm]
        self.colors[:,-1] = 50
        self.colors = list(self.colors)

    def sat_changed(self):
        self.sat = self.sat_slider.value()
        self.img.setLevels([self.sat[0]/100., self.sat[1]/100.])
        self.imgROI.setLevels([self.sat[0]/100., self.sat[1]/100.])
        self.show()

    def reset(self): 
        self.run_embedding_button.setEnabled(False)
        self.p1.clear()
        self.p2.clear()
        self.p3.clear()
        self.p4.clear()
        self.p5.clear()
    
    def reset_variables(self):
        for i in range(len(self.cluster_rois)):
            self.p2.removeItem(self.cluster_rois[i])
        for i in range(nclust_max+1):
            self.scatter_plots[0][i].setData([], [])
            if i < nclust_max:
                self.cluster_plots[i].setData([], [])
        self.p2.show()
        self.p5.show()
        self.randomize_colors()
        self.startROI = False 
        self.posROI = np.zeros((2,2))
        self.cluster_rois, self.cluster_slices = [], []
        self.loaded = False
        self.oned_loaded = False 
        self.embedded = False
        self.behav_data = None
        self.behav_binary_data = None
        self.behav_bin_plot_list = []
        self.behav_labels = []
        self.behav_loaded = False
        self.behav_labels_loaded = False
        self.behav_binary_labels = []
        self.behav_binary_labels_loaded = False
        self.oned_corr_all = None
        self.oned_corr_selected = None
        self.xrange = None
        self.file_iscell = None
        self.iscell = None
        self.neuron_pos = None
        self.save_path = None  # Set default to current folder
        self.embedding = None
        self.heatmap = None
        self.heatmap_chkbxs = []
        self.sat_slider.setValue([30.,70.])
        self.line = pg.PlotDataItem()
        self.oned_trace_plot = pg.PlotDataItem()
        self.oned_trace_plot_added = False
        self.oned_legend = pg.LegendItem(labelTextSize='12pt', horSpacing=30)
        self.symbol_list = ['star', 'd', 'x', 'o', 't', 't1', 't2', 'p', '+', 's', 't3', 'h']
        self.embed_time_range = -1
        self.params_set = False

    def plane_window(self):
        self.PlaneWindow = views.PlaneWindow(self)
        self.PlaneWindow.show()

    def update_status_bar(self, message, update_progress=False):
        if update_progress:
            self.progressBar.show()
            progressBar_value = [int(s) for s in message.split("%")[0].split() if s.isdigit()]
            self.progressBar.setValue(progressBar_value[0])
            frames_processed = np.floor((progressBar_value[0]/100)*float(self.totalFrameNumber.text()))
            self.setFrame.setText(str(frames_processed))
            self.statusBar.showMessage(message.split("|")[0])
        else: 
            self.statusBar.showMessage(message)
            print(message)
        self.show()

    def plot_clicked(self,event):
        items = self.win.scene().items(event.scenePos())
        for x in items:
            if x==self.p1 and event.button()==1 and event.double():
                self.ROI.setPos([-1,-1])
                self.ROI.setSize([self.sp.shape[1]+1, self.sp.shape[0]+1])
            elif x==self.p2 and event.button() == QtCore.Qt.RightButton:
                pos = self.p2.vb.mapSceneToView(event.scenePos())
                x,y = pos.x(), pos.y()
                if not self.startROI:
                    self.startROI = True
                    self.posROI[0,:] = [x,y]
                else:
                    # plotting
                    self.startROI = False 
                    self.posROI[1,:] = [x,y]
                    self.p2.removeItem(self.line)
                    y0, y1 = self.posROI[:,1].min(), self.posROI[:,1].max()
                    y0, y1 = int(y0), int(y1)
                    y1 = y1 if y1 > y0 else y0+1
                    self.selected = slice(y0, y1)
                    self.add_cluster()
                self.posAll = []
                self.lp = []

    
    def mouse_moved(self, pos):
        if self.p2.sceneBoundingRect().contains(pos):
            x = self.p2.vb.mapSceneToView(pos).x()
            y = self.p2.vb.mapSceneToView(pos).y()
            if self.startROI:
                self.posROI[1,:] = [x,y]
                self.p2.removeItem(self.line)
                pen = pg.mkPen(color='yellow', width=3)
                self.line = pg.PlotDataItem(self.posROI[:,0], self.posROI[:,1], pen=pen)
                self.p2.addItem(self.line)
            #else:
            #    dists = (self.embedding[self.selected,0] - x)**2 + (self.embedding[self.selected,1] - y)**2
            #    ineur = np.argmin(dists.flatten()).astype(int)
            #    self.update_selected(ineur)

    def behav_chkbx_toggled(self):
        if self.heatmap_chkbxs[0].isChecked():
            self.plot_behav_data()
            for k in np.arange(1, len(self.heatmap_chkbxs)):
                self.heatmap_chkbxs[k].setEnabled(False)
        else:
            for k in np.arange(1, len(self.heatmap_chkbxs)):
                self.heatmap_chkbxs[k].setEnabled(True)
            disp_ind = []
            for k in np.arange(1, len(self.heatmap_chkbxs)):
                if self.heatmap_chkbxs[k].isChecked():
                    disp_ind.append(np.where(self.heatmap_chkbxs[k].text() == self.behav_labels)[0][0])
            if len(disp_ind) > 0:
                self.plot_behav_data(np.array(disp_ind))

    def keyPressEvent(self, event):
        bid = -1
        move = False
        if self.loaded:
            xrange = self.roi_range(self.ROI)[0]
            if event.key() == QtCore.Qt.Key_Down:
                bid = 0
            elif event.key() == QtCore.Qt.Key_Up:
                bid=1
            elif event.key() == QtCore.Qt.Key_Left:
                bid=2
            elif event.key() == QtCore.Qt.Key_Right:
                bid=3

            if event.modifiers() != QtCore.Qt.ShiftModifier:
                move_time = True if bid==2 or bid==3 and (xrange.stop - xrange.start < self.ntime) else False
                if move_time:
                    ### move in time in increments of 1/2 size of window
                    twin = xrange.stop - xrange.start    
                    if bid==2:
                        if xrange.start > 0:
                            move = True
                            x0 = max(0, xrange.start - twin//2)
                            x1 = xrange.stop - xrange.start + x0
                    elif bid==3:
                        if xrange.stop < self.ntime:
                            move = True
                            x1 = min(self.ntime, xrange.stop + twin//2)
                            x0 = x1 - (xrange.stop - xrange.start)
                    if move:
                        self.set_ROI_position(xrange = slice(x0, x1))
            else:
                if bid==2 or bid==3:
                    twin = xrange.stop - xrange.start    
                    tbin = 50
                    zoom_in = True if bid==2 and twin > tbin else False
                    zoom_out = True if bid==3 and twin < self.ntime else False
                    move_time = zoom_in or zoom_out
                    if move_time:
                        if zoom_in:
                            x0 = xrange.start + tbin//2
                            x1 = max(x0 + tbin, xrange.stop - tbin//2)
                        elif zoom_out:
                            x0 = max(0, xrange.start - tbin//2)
                            x1 = min(self.ntime, x0 + (xrange.stop - xrange.start) + tbin)
                        self.set_ROI_position(xrange = slice(x0, x1))
                
    def roi_range(self, roi):
        pos = roi.pos()
        posy = pos.y()
        posx = pos.x()
        sizex,sizey = roi.size()
        xrange = (np.arange(0,int(sizex)) + int(posx)).astype(np.int32)
        yrange = (np.arange(0,int(sizey)) + int(posy)).astype(np.int32)
        xrange = xrange[xrange>=0]
        xrange = xrange[xrange<self.ntime]
        yrange = yrange[yrange>=0]
        yrange = yrange[yrange<self.nsmooth]
        yrange = slice(yrange[0], yrange[-1]+1)
        xrange = slice(xrange[0], xrange[-1]+1)
        return xrange, yrange

    #def update_avg_activity_trace(self):
        
    def plot_avg_activity_trace(self, roi_id=None):
        if self.loaded:
            kspace = 0.5
            x = np.arange(self.xrange.start, self.xrange.stop)
            if roi_id is None:
                for roi_id in range(nclust_max):
                    if roi_id < len(self.cluster_rois):
                        selected = self.cluster_slices[roi_id]
                        y = self.sp_smoothed[selected].mean(axis=0)
                        y -= y.min()
                        y /= y.max()
                        y += kspace * roi_id
                        self.cluster_plots[roi_id].setData(x, y[self.xrange],
                                                           pen=pg.mkPen(color=self.colors[roi_id][:3]))
                    else:
                        self.cluster_plots[roi_id].setData([], [])
            else:
                selected = self.cluster_slices[roi_id]
                y = self.sp_smoothed[selected].mean(axis=0)
                y -= y.min()
                y /= y.max()
                y += kspace * roi_id
                self.cluster_plots[roi_id].setData(x, y[self.xrange],
                                                    pen=pg.mkPen(color=self.colors[roi_id][:3]))
            self.p3.setXRange(self.xrange.start, self.xrange.stop-1)
            self.p3.setLimits(xMin=self.xrange.start, xMax=self.xrange.stop-1)
            if self.oned_loaded: 
                self.plot_oned_trace()
            self.p3.show()

    def set_ROI_position(self, xrange):
        self.xrange = xrange
        self.ROI.setPos([xrange.start, -1])
        self.ROI.setSize([xrange.stop - xrange.start, self.nneurons + 1])

    def ROI_position(self):
        xrange,_ = self.roi_range(self.ROI)    
        self.xrange = xrange
        # Update zoom in plot
        self.imgROI.setImage(self.sp_smoothed[:, self.xrange])
        self.p2.setXRange(0, self.xrange.stop - self.xrange.start,padding=0)

        if self.behav_loaded:
            self.behav_ROI_update()
        if self.behav_binary_data is not None:
            self.plot_behav_binary_data()

        self.plot_avg_activity_trace()

        axy = self.p2.getAxis('left')
        axx = self.p2.getAxis('bottom')
        self.imgROI.setLevels([self.sat[0]/100., self.sat[1]/100.])

    def smooth_activity(self):
        N = int(self.smooth.text())
        self.smooth_bin = N
        NN = self.sp.shape[0]
        nn = int(np.floor(NN/N))
        if N > 1:    
            self.sp_smoothed = np.reshape(self.sp[self.sorting][:nn*N].copy(), (nn, N, -1)).mean(axis=1)
            self.sp_smoothed = zscore(self.sp_smoothed, axis=1)
            self.sp_smoothed = np.maximum(-4, np.minimum(8, self.sp_smoothed)) + 4
            self.sp_smoothed /= 12
        else:
            self.sp_smoothed = self.sp.copy() 
        self.nsmooth = self.sp_smoothed.shape[0]
        yr0 = min(4, self.nsmooth//4)
        ym = self.nsmooth//2
        self.selected = slice(ym - yr0, ym + yr0)
        if len(self.cluster_rois) > 0:
            for i in range(len(self.cluster_rois)):
                self.p2.removeItem(self.cluster_rois[i])
        self.cluster_rois, self.cluster_slices = [], []
        self.add_cluster()
        self.plot_avg_activity_trace()
        self.update_scatter()
        self.p2.show()
        self.p3.show()
        if self.oned_loaded: 
            self.oned_corr_all = None
            self.plot_1d_corr()

    def add_cluster(self):
        roi_id = len(self.cluster_rois)
        self.cluster_rois.append(guiparts.LinearRegionItem(self, color=self.colors[roi_id], 
                                bounds=(0,self.sp_smoothed.shape[0]), roi_id=roi_id))
        self.cluster_slices.append(self.selected)
        self.cluster_rois[-1].setRegion((self.selected.start, self.selected.stop))
        self.p2.addItem(self.cluster_rois[-1])
        self.cluster_rois[-1].setZValue(10)  # make sure ROI is drawn above image
        self.cluster_rois[-1].cluster_set()
            
    def plot_activity(self):
        if self.loaded:
            self.nneurons, self.ntime = self.sp.shape
            if self.xrange is None:
                self.xrange = slice(0, min(500, ((self.ntime//10)//4)*4))
            self.smooth_activity()
            nn, nt = self.sp_smoothed.shape
            self.img.setImage(self.sp_smoothed)
            self.img.setLevels([self.sat[0]/100.,self.sat[1]/100.])
            self.p1.setXRange(-nt*0.01, nt*1.01, padding=0)
            self.p1.setYRange(-nn*0.01, nn*1.01, padding=0)
            self.p1.show()
            self.p2.setXRange(0, nt, padding=0)
            self.p2.setYRange(0, nn, padding=0)
            self.p2.show()
            self.ROI.maxBounds = QtCore.QRectF(-1.,-1.,nt+1,nn+1)
            self.set_ROI_position(xrange = self.xrange)
            self.plot_avg_activity_trace()
        self.show()
        self.win.show()

    def plot_oned_trace(self):
        avg = self.oned_data
        avg -= avg.min()
        avg /= avg.max()
        avg = avg[self.xrange]
        self.oned_trace_plot.setData(np.arange(self.xrange.start, self.xrange.stop), avg, pen=(0,255,0))
        if self.oned_trace_plot_added:
            self.p3.removeItem(self.oned_trace_plot)
        else:
            self.oned_legend.addItem(self.oned_trace_plot, name='1D variable')
            self.oned_legend.setPos(self.oned_trace_plot.x()+70, self.oned_trace_plot.y())
            self.oned_legend.setParentItem(self.p3)
        self.p3.addItem(self.oned_trace_plot, pen=(0,255,0))
        self.oned_trace_plot_added = True
        self.p3.setXRange(self.xrange.start, self.xrange.stop-1)
        self.p3.setLimits(xMin=self.xrange.start, xMax=self.xrange.stop-1)
        try:
            self.oned_legend.sigClicked.connect(self.mouseClickEvent)
        except Exception as e:
            return

    def plot_behav_binary_data(self):
        for i in range(len(self.behav_bin_plot_list)):
            self.p4.removeItem(self.behav_bin_plot_list[i])
            dat = self.behav_binary_data[i][self.xrange]
            xdat, ydat = np.arange(self.xrange.start, self.xrange.stop)[dat>0], dat[dat>0]
            self.behav_bin_plot_list[i].setData(xdat, ydat, pen=None, symbol=self.symbol_list[i], symbolSize=12)
            self.p4.addItem(self.behav_bin_plot_list[i])
            try:
                self.behav_bin_legend.sigClicked.connect(self.mouseClickEvent)
            except Exception as e:
                return
        self.p4.setLabel('left', '.')

    def plot_behav_data(self, selected=None):
        if self.heatmap is not None:
            if len(self.heatmap) > 1:
                for i in range(len(self.heatmap)):
                    self.p4.removeItem(self.heatmap[i])
            else:
                self.p4.removeItem(self.heatmap)
        if selected is None:
            beh = self.behav_data
        else:
            beh = self.behav_data[selected]
        if beh.shape[0] > 10:
            vmin, vmax = -np.percentile(self.behav_data, 95), np.percentile(self.behav_data, 95)
            self.heatmap = pg.ImageItem(beh, autoDownsample=True, levels=(vmin,vmax))
            lut = colormaps.viridis
            # apply the colormap
            self.heatmap.setLookupTable(lut)        
            self.p4.addItem(self.heatmap)
            self.p4.setLabel('left', 'index')
        else:
            self.heatmap = []
            cmap = colormaps.gist_rainbow[np.linspace(10, 254, beh.shape[0]).astype('int')]
            for i in range(beh.shape[0]):
                self.heatmap.append(pg.PlotCurveItem())
                self.heatmap[-1].setData(np.arange(0, beh.shape[-1]), 
                                         zscore(beh[i]))
                self.heatmap[-1].setPen({'color': cmap[i], 'width': 1})
                self.p4.addItem(self.heatmap[-1])
            self.p4.setLabel('left', 'z-scored')
        self.behav_ROI_update()

    def behav_ROI_update(self):
        self.p4.setLimits(xMin=self.xrange.start, xMax=self.xrange.stop)
        self.p4.setXRange(self.xrange.start, self.xrange.stop)

    def plot_scatter_pressed(self):
        request = self.scatter_comboBox.currentIndex()
        self.p5.setLabel('left', "")
        self.p5.setLabel('bottom', "")
        self.p5.invertY(False)
        if request == 2:
            if self.oned_loaded:
                self.plot_1d_corr(init=True)
            else:
                self.update_status_bar("ERROR: please upload 1D data")
        elif request == 1:
            if self.neuron_pos is not None:
                self.plot_neuron_pos(init=True)
            else:
                self.update_status_bar("ERROR: please upload neuron position data")
        else:
            return

    def update_scatter(self, roi_id=None):
        request = self.scatter_comboBox.currentIndex()
        if request == 2:
            self.plot_1d_corr(roi_id=roi_id)
        elif request == 1:
            self.plot_neuron_pos(roi_id=roi_id)
        else:
            return

    def get_oned_corr(self):
        if self.oned_corr_all is None:
            self.oned_corr_all = (self.sp_smoothed * zscore(self.oned_data)).mean(axis=-1)
            return self.oned_corr_all
        else:
            return self.oned_corr_all
        
    def neurons_selected(self, selected=None):
        selected = selected if selected is not None else self.selected
        if self.embedded:
            neurons_select = self.sorting[selected.start * self.smooth_bin : selected.stop * self.smooth_bin]
        else:
            neurons_select = slice(selected.start * self.smooth_bin, 
                                   selected.stop * self.smooth_bin)  
        return neurons_select      

    def plot_scatter(self, x, y, roi_id=None, iplane=0):
        if self.all_neurons_checkBox.isChecked() and roi_id is None:
            colors = colormaps.gist_ncar[np.linspace(0, 254, len(x)).astype('int')][self.sorting]
            brushes = [pg.mkBrush(color=c) for c in colors]
            self.scatter_plots[iplane][0].setData(x, y, symbol='o', 
                                      brush=brushes, hoverable=True)
            for i in range(1, nclust_max+1):
                self.scatter_plots[iplane][i].setData([], [])
        else:
            if roi_id is None:
                self.scatter_plots[iplane][0].setData(x, y, symbol='o', 
                                              brush=pg.mkBrush(color=(180,180,180)), hoverable=True)
                for roi_id in range(nclust_max):
                    if roi_id < len(self.cluster_rois):
                        selected = self.neurons_selected(self.cluster_slices[roi_id])
                        self.scatter_plots[iplane][roi_id+1].setData(x[selected], 
                                                    y[selected], symbol='o', 
                                                    brush=pg.mkBrush(color=self.colors[roi_id][:3]), 
                                                    hoverable=True)
                    else:
                        self.scatter_plots[iplane][roi_id+1].setData([],[])
            else:
                selected = self.neurons_selected(self.cluster_slices[roi_id])
                self.scatter_plots[iplane][roi_id+1].setData(x[selected], 
                                            y[selected], symbol='o', 
                                            brush=pg.mkBrush(color=self.colors[roi_id][:3]), 
                                            hoverable=True)

    def plot_1d_corr(self, init=False, roi_id=None):
        if self.oned_loaded:
            r = self.get_oned_corr()
            self.plot_scatter(r, np.arange(0, self.nsmooth), roi_id=roi_id)
            self.p5.setYRange(0, self.nsmooth)
            if init:
                self.p5.invertY(True)
                self.p5.setLabel('left', "position")
                self.p5.setLabel('bottom', "correlation")

    def plot_neuron_pos(self, init=False, roi_id=None):
        if self.neuron_pos is not None:
            ypos, xpos = self.neuron_pos[:,0], self.neuron_pos[:,1]
            self.plot_scatter(ypos, xpos, roi_id=roi_id)
            if init:
                self.p5.setLabel('left', "y position")
                self.p5.setLabel('bottom', "x position") 

    def update_plot_p4(self):
        if self.behav_loaded:
            print('loaded')
        else:
            self.update_status_bar("Please upload a behav file")
            return

def run():
    # Always start by initializing Qt (only once per application)
    app = QApplication(sys.argv)
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
