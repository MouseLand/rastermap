import sys, time, os
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QScrollBar, QSlider, QComboBox, QGridLayout, QPushButton, QFrame, QCheckBox, QLabel, QProgressBar, QLineEdit, QMessageBox, QGroupBox, QButtonGroup, QRadioButton, QStatusBar
from scipy.stats import zscore, pearsonr
from . import menus, guiparts, io, colormaps

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
        self.p1 = self.win.addPlot(title="FULL VIEW",row=0,col=1)
        self.p1.setMouseEnabled(x=False,y=False)
        self.img = pg.ImageItem(autoDownsample=True)
        self.p1.addItem(self.img)
        self.p1.setXRange(0,nt)
        self.p1.setYRange(0,nn)
        self.p1.setLabel('left', 'binned neurons')
        self.p1.setLabel('bottom', 'time')

        # Plot a zoomed in region from full view (changes across time axis)
        self.selected = slice(0, nn)
        self.p2 = self.win.addPlot(title='ZOOM IN',row=1,col=0,colspan=2)
        self.imgROI = pg.ImageItem(autoDownsample=True)
        self.p2.addItem(self.imgROI)
        self.p2.setMouseEnabled(x=False, y=False)
        self.p2.setLabel('bottom', 'time')
        self.p2.setLabel('left', 'binned neurons')
        ax = self.p2.getAxis('bottom')
        ticks = [0]
        ax.setTicks([[(v, '.') for v in ticks ]])
        self.p2.invertY(True)

        # Plot avg. activity of neurons selected in ROI of zoomed in view
        self.p3 = self.win.addPlot(title='selected neurons', 
                                   row=2, col=0,
                                   colspan=2, padding=0)
        self.p3.setMouseEnabled(x=False,y=False)
        self.p3.setLabel('bottom', 'time')
        self.p3.setLabel('left', 'avg activity')

        # Plot behavioral dataset as heatmap
        self.p4 = self.win.addPlot(title='behavior',row=3,col=0,colspan=2)
        self.p4.setMouseEnabled(x=False,y=False)
        self.p4.setLabel('bottom', 'time')

        # Scatter plot for oned correlation, neuron position, and depth (ephys) information
        self.p5 = self.win.addPlot(title='scatter plot',row=1,col=2)
        self.p5.setMouseEnabled(x=False,y=False)
        self.scatter_plot = pg.ScatterPlotItem()
        self.scatter_plot_selected = pg.ScatterPlotItem()
        
        # Optional plots toggled w/ checkboxes
        self.win.removeItem(self.p4)
        self.win.removeItem(self.p5)

        # Set colormap to deafult of gray_r. ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Future: add option to change cmap ~~~~~~~~~~~~~~
        lut = colormaps.gray[::-1]
        # apply the colormap
        self.img.setLookupTable(lut)
        self.imgROI.setLookupTable(lut)
        layout.setColumnStretchFactor(1,3)
        layout.setRowStretchFactor(1,3)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Options on top left of GUI ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # add bin size across neurons
        ysm = QLabel("<font color='gray'>bin neurons:</font>")
        self.smooth = QLineEdit(self)
        self.smooth.setValidator(QtGui.QIntValidator(0, 500))
        self.smooth.setText("10")
        self.smooth.setFixedWidth(45)
        self.smooth.setAlignment(QtCore.Qt.AlignRight)
        self.smooth.returnPressed.connect(self.plot_activity)

        self.heatmap_checkBox = QCheckBox("show n-d array")
        self.heatmap_checkBox.setStyleSheet("color: gray;")
        self.heatmap_checkBox.stateChanged.connect(self.update_plot_p4)
        self.heatmap_checkBox.setEnabled(False)
        self.scatterplot_checkBox = QCheckBox("scatter pos / corr")
        self.scatterplot_checkBox.setStyleSheet("color: gray;")
        self.scatterplot_checkBox.stateChanged.connect(self.update_plot_p5)
        
        # Add slider for levels  
        self.sat = [0.3,0.7]
        slider = guiparts.SatSlider(self)
        sat_label = QLabel("Saturation")
        sat_label.setStyleSheet('color: white;')
        self.img.setLevels([self.sat[0], self.sat[1]])
        self.imgROI.setLevels([self.sat[0], self.sat[1]])

        # Add drop down options for scatter plot
        self.scatter_comboBox = QComboBox(self)
        self.scatter_comboBox.setFixedWidth(120)
        scatter_comboBox_ops = ["-- Select --", "1D correlation", "neuron position"]
        self.scatter_comboBox.setEditable(True)
        self.scatter_comboBox.addItems(scatter_comboBox_ops)
        self.scatter_comboBox.setCurrentIndex(0)
        line_edit = self.scatter_comboBox.lineEdit()
        line_edit.setAlignment(QtCore.Qt.AlignCenter)  
        line_edit.setReadOnly(True)
        self.scatter_comboBox.setCurrentIndex(0)
        self.scatter_comboBox.hide()
        self.all_neurons_checkBox = QCheckBox("color all neurons")
        self.all_neurons_checkBox.setStyleSheet("color: gray;")
        self.all_neurons_checkBox.hide()
        self.scatterplot_button = QPushButton('Plot')
        self.scatterplot_button.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.scatterplot_button.clicked.connect(self.plot_scatter_pressed)
        self.scatterplot_button.hide()

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
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.progressBar = QProgressBar()
        self.statusBar.addPermanentWidget(self.progressBar)
        self.progressBar.setGeometry(0, 0, 300, 25)
        self.progressBar.setMaximum(100)
        self.progressBar.hide()

        # Default variables
        self.tpos = -0.5
        self.tsize = 1
        self.reset_variables()
        
        # Add features to window
        ops_row_pos = 0
        self.l0.addWidget(ysm, ops_row_pos, 0, 1, 1)
        self.l0.addWidget(self.smooth, ops_row_pos, 1, 1, 1)
        self.l0.addWidget(sat_label,ops_row_pos+1,0,1,2)
        self.l0.addWidget(slider, ops_row_pos+2,0,1,2)
        self.l0.addWidget(self.heatmap_checkBox, ops_row_pos+3, 0, 1, 2)
        self.l0.addWidget(self.scatterplot_checkBox, ops_row_pos+4, 0, 1, 2)
        self.l0.addWidget(self.scatter_comboBox,ops_row_pos+16,12,1,1)
        
        self.win.show()
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        self.show()

    def reset(self): 
        self.run_embedding_button.setEnabled(False)
        self.heatmap_checkBox.setEnabled(False)
        self.p1.clear()
        self.p2.clear()
        self.p3.clear()
        self.p4.clear()
        self.p5.clear()
    
    def reset_variables(self):
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
        self.xpos_dat = None
        self.ypos_dat = None
        self.depth_dat = None
        self.save_path = None  # Set default to current folder
        self.embedding = None
        self.heatmap = None
        self.heatmap_chkbxs = []
        self.oned_trace_plot = pg.PlotDataItem()
        self.oned_trace_plot_added = False
        self.oned_legend = pg.LegendItem(labelTextSize='12pt', horSpacing=30)
        self.symbol_list = ['star', 'd', 'x', 'o', 't', 't1', 't2', 'p', '+', 's', 't3', 'h']
        self.embed_time_range = -1
        self.params_set = False

    def update_status_bar(self, message, update_progress=False):
        if update_progress:
            self.progressBar.show()
            progressBar_value = [int(s) for s in message.split("%")[0].split() if s.isdigit()]
            self.progressBar.setValue(progressBar_value[0])
            frames_processed = np.floor((progressBar_value[0]/100)*float(self.totalFrameNumber.text()))
            self.setFrame.setText(str(frames_processed))
            self.statusBar.showMessage(message.split("|")[0])
        else: 
            self.progressBar.hide()
            self.statusBar.showMessage(message)
            print(message)
        self.show()

    def plot_clicked(self,event):
        items = self.win.scene().items(event.scenePos())
        for x in items:
            if x==self.p1 and event.button()==1 and event.double():
                self.ROI.setPos([-1,-1])
                self.ROI.setSize([self.sp.shape[1]+1, self.sp.shape[0]+1])

    def update_scatter_ops_pos(self):
        self.l0.removeWidget(self.scatter_comboBox)
        self.l0.removeWidget(self.all_neurons_checkBox)
        self.l0.removeWidget(self.scatterplot_button)
        if self.heatmap_checkBox.isChecked() and self.behav_loaded:
            if len(self.heatmap_chkbxs) <= 3:
                k = 1
            elif len(self.heatmap_chkbxs) >= 5:
                k = -1
            else:
                k = 0
            if self.behav_labels_loaded:
                self.l0.addWidget(self.scatterplot_button,13+k,12,1,2)
                self.l0.addWidget(self.scatter_comboBox,14+k,12,1,1)
                self.l0.addWidget(self.all_neurons_checkBox,14+k,13,1,1)
            else:
                self.l0.addWidget(self.scatterplot_button,16,12,1,2)
                self.l0.addWidget(self.scatter_comboBox,17,12,1,1)
                self.l0.addWidget(self.all_neurons_checkBox,17,13,1,1)
        else:
            self.l0.addWidget(self.scatterplot_button,18,12,1,2)
            self.l0.addWidget(self.scatter_comboBox,19,12,1,1)
            self.l0.addWidget(self.all_neurons_checkBox,19,13,1,1)

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

    def show_heatmap_ops(self):
        for k in range(len(self.heatmap_chkbxs)):
            self.heatmap_chkbxs[k].show()
    
    def hide_heatmap_ops(self):
        for k in range(len(self.heatmap_chkbxs)):
            self.heatmap_chkbxs[k].hide()

    def keyPressEvent(self, event):
        bid = -1
        move = False
        if self.loaded:
            xrange = self.roi_range(self.ROI)[0]
            yrange = self.roi_range(self.LINE)[1]
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
                move_neurons = True if bid==0 or bid==1 and (yrange.stop - yrange.start < self.nsmooth) else False
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
                elif move_neurons:
                    ### move in neurons in increments of 1/2 size of window
                    nwin = yrange.stop - yrange.start    
                    if bid==1:
                        if yrange.start > 0:
                            move = True
                            x0 = max(0, yrange.start - nwin//2)
                            x1 = yrange.stop - yrange.start + x0
                    elif bid==0:
                        if yrange.stop < self.nsmooth:
                            move = True
                            x1 = min(self.nsmooth, yrange.stop + nwin//2)
                            x0 = x1 - (yrange.stop - yrange.start)
                    if move:
                        self.LINE.setPos([-0.5, x0])
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
                else:
                    nwin = yrange.stop - yrange.start    
                    nbin = 10
                    zoom_in = True if bid==0 and nwin > nbin else False
                    zoom_out = True if bid==1 and nwin < self.nsmooth else False
                    move_neurons = zoom_in or zoom_out
                    if move_neurons:
                        if zoom_in:
                            x0 = yrange.start + nbin//2
                            x1 = max(x0 + nbin, yrange.stop - nbin//2)
                        elif zoom_out:
                            x0 = max(0, yrange.start - nbin//2)
                            x1 = min(self.nsmooth, x0 + (yrange.stop - yrange.start) + nbin)
                        self.LINE.setPos([-0.5, x0])
                        self.LINE.setSize([xrange.stop - xrange.start + 1, x1 - x0])

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

    def plot_avg_activity_trace(self):
        if self.loaded:
            avg = self.sp_smoothed[self.selected, self.xrange].mean(axis=0)
            avg -= avg.min()
            avg /= avg.max()
            self.p3.clear()
            self.p3.plot(np.arange(self.xrange.start, self.xrange.stop), avg, pen=(255,0,0))
            self.p3.setXRange(self.xrange.start, self.xrange.stop-1)
            self.p3.setLimits(xMin=self.xrange.start, xMax=self.xrange.stop-1)
            if self.oned_loaded: 
                self.plot_oned_trace()
            self.p3.show()
        
    def LINE_position(self):
        _,yrange = self.roi_range(self.LINE)
        self.selected = yrange
        #for i in range(len(self.LINE.handles)):
        #    self.LINE.handles[i]['item'].setSelectable(True)
        self.plot_avg_activity_trace()
        self.LINE.setZValue(10)
        self.update_scatter()

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
        
        # reset LINE ROI
        self.LINE.maxBounds = QtCore.QRectF(-1,-1.,
                                self.xrange.stop - self.xrange.start + 1, self.nsmooth+1)
        self.LINE.setPos(-.5, self.selected.start)
        self.LINE.setSize([self.xrange.stop - self.xrange.start + 1,
                           self.selected.stop - self.selected.start
                           ])
        self.LINE.setZValue(10)

        if self.behav_loaded:
            self.behav_ROI_update()
        if self.behav_binary_data is not None:
            self.plot_behav_binary_data()

        axy = self.p2.getAxis('left')
        axx = self.p2.getAxis('bottom')
        self.imgROI.setLevels([self.sat[0], self.sat[1]])

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
        if self.oned_loaded: 
            self.oned_corr_all = None
            self.plot_1d_corr()
            
    def plot_activity(self):
        if self.loaded:
            self.nneurons, self.ntime = self.sp.shape
            if self.xrange is None:
                self.xrange = slice(0, min(500, ((self.ntime//10)//4)*4))
            self.smooth_activity()
            nn, nt = self.sp_smoothed.shape
            self.img.setImage(self.sp_smoothed)
            self.img.setLevels([self.sat[0],self.sat[1]])
            self.p1.setXRange(-nt*0.01, nt*1.01, padding=0)
            self.p1.setYRange(-nn*0.01, nn*1.01, padding=0)
            self.p1.show()
            self.p2.setXRange(0, nt, padding=0)
            self.p2.setYRange(0, nn, padding=0)
            self.p2.show()
            self.ROI.maxBounds = QtCore.QRectF(-1.,-1.,nt+1,nn+1)
            self.set_ROI_position(xrange = self.xrange)
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
        self.p5.clear()
        self.p5.removeItem(self.scatter_plot)
        self.p5.removeItem(self.scatter_plot_selected)
        self.p5.setLabel('left', "")
        self.p5.setLabel('bottom', "")
        self.p5.invertY(False)
        if request == 1:
            if self.oned_loaded:
                self.plot_1d_corr(init=True)
            else:
                self.update_status_bar("ERROR: please upload 1D data")
        elif request == 2:
            if self.xpos_dat is None or self.ypos_dat is None:
                io.get_neuron_pos_data(self)
            self.plot_neuron_pos(init=True)
        else:
            return

    def update_scatter(self):
        request = self.scatter_comboBox.currentIndex()
        if request == 1:
            self.plot_1d_corr()
        elif request == 2:
            self.plot_neuron_pos()
        else:
            return

    def get_oned_corr(self):
        if self.oned_corr_all is None:
            self.oned_corr_all = (self.sp_smoothed * zscore(self.oned_data)).mean(axis=-1)
            return self.oned_corr_all
        else:
            return self.oned_corr_all
        
    def neurons_selected(self):
        if self.embedded:
            selected = self.sorting[self.selected.start * self.smooth_bin : self.selected.stop * self.smooth_bin]
        else:
            selected = slice(self.selected.start * self.smooth_bin, 
                             self.selected.stop * self.smooth_bin)  
        return selected      

    def plot_1d_corr(self, init=False):
        if self.oned_loaded:
            r = self.get_oned_corr()
            if self.all_neurons_checkBox.isChecked():
                colors = colormaps.gist_ncar[np.linspace(0, 254, self.nsmooth).astype('int')]
                brushes = [pg.mkBrush(color=c) for c in colors]
                self.scatter_plot.setData(r, np.arange(0, self.nsmooth), symbol='o', 
                                          brush=brushes, hoverable=True)
                self.scatter_plot_selected.setData([], [])
            else:
                self.scatter_plot.setData(r, np.arange(0, self.nsmooth), symbol='o', 
                                          brush=pg.mkBrush(color=(180,180,180)), hoverable=True)
                self.scatter_plot_selected.setData(r[self.selected], 
                                                   np.arange(self.selected.start, self.selected.stop), 
                                                   symbol='o', 
                                                   brush=pg.mkBrush(color=(255,0,0)), 
                                                   hoverable=True)
            self.p5.setYRange(0, self.nsmooth)
            if init:
                self.p5.addItem(self.scatter_plot)
                self.p5.addItem(self.scatter_plot_selected)
                self.p5.invertY(True)
                self.p5.setLabel('left', "position")
                self.p5.setLabel('bottom', "correlation")

    def plot_neuron_pos(self, init=False):
        if self.xpos_dat is not None or self.ypos_dat is not None:
            xpos, ypos = self.xpos_dat, -self.ypos_dat
            selected = self.neurons_selected()
            xpos_selected, ypos_selected = self.xpos_dat[selected], -self.ypos_dat[selected]
            if self.all_neurons_checkBox.isChecked():
                colors = colormaps.gist_ncar[np.linspace(0, 254, len(xpos)).astype('int')]
                brushes = [pg.mkBrush(color=c) for c in colors]
                self.scatter_plot.setData(xpos, ypos, symbol='o', 
                                        brush=brushes, hoverable=True)
                self.scatter_plot_selected.setData([], [])
            else:
                self.scatter_plot.setData(xpos, ypos, symbol='o', 
                                            brush=pg.mkBrush(color=(180,180,180)), hoverable=True)
                self.scatter_plot_selected.setData(xpos_selected, ypos_selected, symbol='o', 
                                                brush=pg.mkBrush(color=(255,0,0)), hoverable=True)
            if init:
                self.p5.addItem(self.scatter_plot)
                self.p5.addItem(self.scatter_plot_selected)
                self.p5.setLabel('left', "y position")
                self.p5.setLabel('bottom', "x position") 
            
    def get_colors(self, data):
        num_classes = len(np.unique(data))
        colors = gist_ncar[np.linspace(0, 254, nd).astype('int')]
        brushes = [pg.mkBrush(color=c) for c in colors]
        cmap = np.asarray([brushes[np.unique(data).tolist().index(i)] for i in data])
        return cmap

    def update_plot_p4(self):
        self.update_scatter_ops_pos()
        if self.heatmap_checkBox.isChecked() and self.behav_loaded:
            self.win.addItem(self.p4, row=3, col=0, colspan=2)
            if self.scatterplot_checkBox.isChecked():
                self.show_heatmap_ops()
        elif not self.heatmap_checkBox.isChecked() and self.behav_loaded:
            try:
                self.win.removeItem(self.p4)
                self.hide_heatmap_ops()
            except Exception as e:
                return
        else:
            self.update_status_bar("Please upload a behav file")
            return
        
    def update_plot_p5(self):
        self.update_scatter_ops_pos()
        if self.scatterplot_checkBox.isChecked():
            self.win.addItem(self.p5, row=1, col=2)
            self.win.removeItem(self.p1)
            self.win.addItem(self.p1, row=0, col=1, colspan=2)
            self.scatter_comboBox.show()
            self.all_neurons_checkBox.show()
            self.scatterplot_button.show()
            if self.heatmap_checkBox.isChecked():
                self.show_heatmap_ops()
        else:
            try:
                self.win.removeItem(self.p5)
                self.win.removeItem(self.p1)
                self.win.addItem(self.p1, row=0, col=1)
                self.scatter_comboBox.hide()
                self.all_neurons_checkBox.hide()
                self.scatterplot_button.hide()
                self.hide_heatmap_ops()
            except Exception as e:
                return
        
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
