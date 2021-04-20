import sys, time, os
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore
from matplotlib import cm
from scipy.stats import zscore, pearsonr
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
        self.p1.setLabel('left', 'Neurons')
        self.p1.setLabel('bottom', 'Time')

        # Plot a zoomed in region from full view (changes across time axis
        self.selected = np.arange(0,nn,1,int)
        self.p2 = self.win.addPlot(title='ZOOM IN',row=1,col=0,colspan=2)
        self.imgROI = pg.ImageItem(autoDownsample=True)
        self.p2.addItem(self.imgROI)
        self.p2.setMouseEnabled(x=False,y=False)
        self.p2.hideAxis('bottom')
        self.p2.invertY(True)

        # Plot avg. activity of neurons selected in ROI of zoomed in view
        self.p3 = self.win.addPlot(title='Zoom in ROI neural activity trace',row=2,col=0,
                                    colspan=2, padding=0)
        self.p3.setMouseEnabled(x=False,y=False)
        self.p3.setLabel('bottom', 'Time')
        self.p3.setLabel('left', 'Avg. activity')

        # Plot behavioral dataset as heatmap
        self.p4 = self.win.addPlot(title='Heatmap',row=3,col=0,colspan=2)
        self.p4.setMouseEnabled(x=False,y=False)
        self.p4.setLabel('bottom', 'Time')

        # Scatter plot for run correlation, neuron position, and depth (ephys) information
        self.p5 = self.win.addPlot(title='Scatter plot',row=1,col=2)
        self.p5.setMouseEnabled(x=False,y=False)
        self.scatter_plot = pg.ScatterPlotItem()
        self.neuron_pos_scatter = pg.ScatterPlotItem()
        
        # Optional plots toggled w/ checkboxes
        self.win.removeItem(self.p4)
        self.win.removeItem(self.p5)

        # Set colormap to deafult of gray_r. ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Future: add option to change cmap ~~~~~~~~~~~~~~
        colormap = cm.get_cmap("gray_r")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        lut = lut[0:-3,:]
        # apply the colormap
        self.img.setLookupTable(lut)
        self.imgROI.setLookupTable(lut)
        layout.setColumnStretchFactor(1,3)
        layout.setRowStretchFactor(1,3)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Options on top left of GUI ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        self.default_param_radiobutton.toggled.connect(lambda: io.set_rastermap_params(self))
        self.RadioGroup.addButton(self.default_param_radiobutton)
        self.custom_param_radiobutton = QtGui.QRadioButton("Custom")
        self.custom_param_radiobutton.setStyleSheet("color: gray;")
        self.custom_param_radiobutton.toggled.connect(lambda: io.get_rastermap_params(self))
        self.RadioGroup.addButton(self.custom_param_radiobutton)

        self.heatmap_checkBox = QtGui.QCheckBox("Behaviour")
        self.heatmap_checkBox.setStyleSheet("color: gray;")
        self.heatmap_checkBox.stateChanged.connect(self.update_plot_p4)
        self.heatmap_checkBox.setEnabled(False)
        self.upload_behav_button = QtGui.QPushButton('Upload behavior')
        self.upload_behav_button.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.upload_behav_button.clicked.connect(lambda: io.get_behav_data(self))
        self.upload_behav_button.setEnabled(False)
        self.scatterplot_checkBox = QtGui.QCheckBox("Scatter plot")
        self.scatterplot_checkBox.setStyleSheet("color: gray;")
        self.scatterplot_checkBox.stateChanged.connect(self.update_plot_p5)
        self.upload_run_button = QtGui.QPushButton('Upload run')
        self.upload_run_button.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.upload_run_button.clicked.connect(lambda: io.load_run_data(self))
        self.upload_run_button.setEnabled(False)

        # Add slider for levels  
        self.sat = [0.3,0.7]
        slider = guiparts.SatSlider(self)
        sat_label = QtGui.QLabel("Saturation")
        sat_label.setStyleSheet('color: white;')
        self.img.setLevels([self.sat[0], self.sat[1]])
        self.imgROI.setLevels([self.sat[0], self.sat[1]])

        # Add drop down options for scatter plot
        self.scatter_comboBox = QtGui.QComboBox(self)
        self.scatter_comboBox.setFixedWidth(120)
        scatter_comboBox_ops = ["-- Select --", "Run correlation", "Neuron position", "Neuron depth"]
        self.scatter_comboBox.setEditable(True)
        self.scatter_comboBox.addItems(scatter_comboBox_ops)
        self.scatter_comboBox.setCurrentIndex(0)
        line_edit = self.scatter_comboBox.lineEdit()
        line_edit.setAlignment(QtCore.Qt.AlignCenter)  
        line_edit.setReadOnly(True)
        self.scatter_comboBox.setCurrentIndex(0)
        self.scatter_comboBox.hide()
        self.all_neurons_checkBox = QtGui.QCheckBox("All neurons")
        self.all_neurons_checkBox.setStyleSheet("color: gray;")
        self.all_neurons_checkBox.hide()
        self.scatterplot_button = QtGui.QPushButton('Plot')
        self.scatterplot_button.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.scatterplot_button.clicked.connect(self.plot_scatter_pressed)
        self.scatterplot_button.hide()

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
        
        # Default variables
        self.tpos = -0.5
        self.tsize = 1
        self.reset_variables()
        
        # Add features to window
        ops_row_pos = 0
        self.l0.addWidget(ysm, ops_row_pos, 0, 1, 1)
        self.l0.addWidget(self.smooth, ops_row_pos, 1, 1, 1)
        self.l0.addWidget(params, ops_row_pos+1, 0, 1, 2)
        self.l0.addWidget(self.default_param_radiobutton, ops_row_pos+2, 0, 1, 1)
        self.l0.addWidget(self.custom_param_radiobutton, ops_row_pos+2, 1, 1, 1)
        self.l0.addWidget(self.run_embedding_button, ops_row_pos+3, 0, 1, 2)
        self.l0.addWidget(self.upload_behav_button, ops_row_pos+4, 0, 1, 1)
        self.l0.addWidget(self.upload_run_button, ops_row_pos+4, 1, 1, 1)
        self.l0.addWidget(self.heatmap_checkBox, ops_row_pos+5, 0, 1, 1)
        self.l0.addWidget(self.scatterplot_checkBox, ops_row_pos+5, 1, 1, 1)
        self.l0.addWidget(sat_label,ops_row_pos,2,1,1)
        self.l0.addWidget(slider, ops_row_pos,3,1,2)
        self.l0.addWidget(self.scatter_comboBox,ops_row_pos+16,12,1,1)
        self.l0.addWidget(self.all_neurons_checkBox,ops_row_pos+18,13,1,1)
        self.l0.addWidget(self.scatterplot_button,ops_row_pos+19,12,1,1)
        
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
        self.run_loaded = False 
        self.embedded = False
        self.behav_data = None
        self.behav_binary_data = None
        self.behav_bin_plot_list = []
        self.behav_labels = []
        self.behav_loaded = False
        self.behav_labels_loaded = False
        self.behav_binary_labels = []
        self.behav_binary_labels_loaded = False
        self.run_corr_all = None
        self.run_corr_selected = None
        self.xpos_dat = None
        self.ypos_dat = None
        self.depth_dat = None
        self.save_path = None  # Set default to current folder
        self.embedding = None
        self.heatmap = None
        self.heatmap_chkbxs = []
        self.run_trace_plot = pg.PlotDataItem()
        self.run_trace_plot_added = False
        self.run_legend = pg.LegendItem(labelTextSize='12pt', horSpacing=30)
        self.symbol_list = ['star', 'd', 'x', 'o', 't', 't1', 't2', 'p', '+', 's', 't3', 'h']

    def plot_scatter_pressed(self):
        request = self.scatter_comboBox.currentIndex()
        self.p5.clear()
        self.p5.removeItem(self.scatter_plot)
        self.p5.setLabel('left', "")
        self.p5.setLabel('bottom', "")
        self.p5.invertY(False)
        if request == 1:
            if self.run_loaded and self.embedded:
                self.plot_run_corr()
            else:
                print("Please run embedding or upload run data")
        elif request == 2:
            if self.xpos_dat is None or self.ypos_dat is None:
                io.get_neuron_pos_data(self)
            self.plot_neuron_pos()
        elif request == 3:
            if self.depth_dat is None:
                io.get_neuron_depth_data(self)
            self.plot_neuron_depth()
        else:
            return

    def get_run_corr(self):
        if self.all_neurons_checkBox.isChecked() and self.run_corr_all is None:
            self.run_corr_all = np.zeros(self.sp.shape[0])
            for i in range(len(self.run_corr_all)):
                self.run_corr_all[i], _ = pearsonr(self.sp[i], self.run_data)
            return self.run_corr_all
        elif self.all_neurons_checkBox.isChecked():
            return self.run_corr_all
        else:
            self.run_corr_selected = np.zeros(len(self.selected))
            for i, neuron in enumerate(self.selected):
                self.run_corr_selected[i], _ = pearsonr(self.sp_smoothed[neuron], self.run_data)
            return self.run_corr_selected

    def plot_run_corr(self):
        r = self.get_run_corr()
        if self.all_neurons_checkBox.isChecked(): # all neurons
            embed = self.embedding[:,0].squeeze()
        else:                                  # super neuron positions
            embed = self.selected#self.embedding[self.selected].squeeze()
        self.scatter_plot.setData(r, embed, symbol='o', brush=(1,1,1,1),
                                    hoverable=True, hoverSize=15)
        self.p5.addItem(self.scatter_plot)
        self.p5.invertY(True)
        self.p5.setLabel('left', "Embedding position")
        self.p5.setLabel('bottom', "Pearson's correlation")

    def plot_neuron_pos(self):
        if self.embedded:
            if self.all_neurons_checkBox.isChecked():
                embed = self.embedding[:,0].squeeze()
                xpos, ypos = self.xpos_dat, -self.ypos_dat
            else:
                embed = self.embedding[self.selected].squeeze()
                xpos, ypos = self.xpos_dat[self.selected], -self.ypos_dat[self.selected]
            brushes = self.get_colors(embed)
            self.scatter_plot.setData(xpos, ypos, symbol='o', brush=brushes,
                                 hoverable=True, hoverSize=15)
            self.p5.addItem(self.scatter_plot)
            self.p5.setLabel('left', "y position")
            self.p5.setLabel('bottom', "x position")
        else:
            print("Please run embedding")

    def get_colors(self, data):
        num_classes = len(np.unique(data))+1
        colors = cm.get_cmap('gist_ncar')(np.linspace(0, 1., num_classes))
        colors *= 255
        colors = colors.astype(int)
        colors[:,-1] = 127
        brushes = [pg.mkBrush(color=c) for c in colors]
        cmap = np.asarray([brushes[np.unique(data).tolist().index(i)] for i in data])
        return cmap

    def plot_neuron_depth(self):
        if self.embedded:
            if self.all_neurons_checkBox.isChecked():
                embed = self.embedding[:,0].squeeze()
                depth = self.depth_dat
            else:
                embed = self.embedding[self.selected].squeeze()
                depth = self.depth_dat[self.selected]
            self.scatter_plot.setData(depth, embed, symbol='o', 
                                    brush=(1,1,1,1), hoverable=True, hoverSize=15)
            self.p5.addItem(self.scatter_plot)
            self.p5.setLabel('left', "Embedding position")
            self.p5.setLabel('bottom', "Neuron depth")
        else:
            print("Please run embedding")

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
            print("Please upload a behav file")
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

    def plot_avg_activity_trace(self):
        if self.loaded:
            avg = self.sp_smoothed[np.ix_(self.selected,self.xrange)].mean(axis=0)
            avg -= avg.min()
            avg /= avg.max()
            self.p3.clear()
            self.p3.plot(self.xrange,avg,pen=(255,0,0))
            self.p3.setXRange(self.xrange[0],self.xrange[-1])
            self.p3.setLimits(xMin=self.xrange[0],xMax=self.xrange[-1])
            self.p3.show()
        
    def LINE_position(self):
        _,yrange = self.roi_range(self.LINE)
        self.selected = yrange.astype('int')
        self.plot_avg_activity_trace()
        if self.run_loaded: 
            self.plot_run_trace()
        if self.behav_loaded:
            self.behav_ROI_update()
        if self.behav_binary_data is not None:
            self.plot_behav_binary_data()

    def ROI_position(self):
        xrange,_ = self.roi_range(self.ROI)
        self.xrange = xrange
        # Update zoom in plot
        self.imgROI.setImage(self.sp_smoothed[:, self.xrange])
        self.p2.setXRange(0,self.xrange.size,padding=0)
        # Update avg. activity and other data loaded (behaviour and running)
        self.plot_avg_activity_trace()
        if self.run_loaded: 
            self.plot_run_trace()    
        if self.behav_loaded:
            self.behav_ROI_update()
        if self.behav_binary_data is not None:
            self.plot_behav_binary_data()

        # reset ROIs
        self.LINE.maxBounds = QtCore.QRectF(-1,-1.,
                                xrange.size+1,self.sp.shape[0]+1)
        self.LINE.setSize([xrange.size+1, self.selected.size])
        self.LINE.setZValue(10)

        axy = self.p2.getAxis('left')
        axx = self.p2.getAxis('bottom')
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

    def plot_run_trace(self):
        avg = self.run_data
        avg -= avg.min()
        avg /= avg.max()
        avg = avg[self.xrange]
        self.run_trace_plot.setData(self.xrange, avg, pen=(0,255,0))
        if self.run_trace_plot_added:
            self.p3.removeItem(self.run_trace_plot)
        else:
            self.run_legend.addItem(self.run_trace_plot, name='Run')
            self.run_legend.setPos(self.run_trace_plot.x()+70, self.run_trace_plot.y())
            self.run_legend.setParentItem(self.p3)
        self.p3.addItem(self.run_trace_plot, pen=(0,255,0))
        self.run_trace_plot_added = True
        self.p3.setXRange(self.xrange[0],self.xrange[-1])
        self.p3.setLimits(xMin=self.xrange[0],xMax=self.xrange[-1])
        try:
            self.run_legend.sigClicked.connect(self.mouseClickEvent)
        except Exception as e:
            return

    def plot_behav_binary_data(self):
        for i in range(len(self.behav_bin_plot_list)):
            self.p3.removeItem(self.behav_bin_plot_list[i])
            dat = self.behav_binary_data[i][self.xrange]
            xdat, ydat = self.xrange[dat>0], dat[dat>0]
            self.behav_bin_plot_list[i].setData(xdat, ydat, pen=None, symbol=self.symbol_list[i], symbolSize=12)
            self.p3.addItem(self.behav_bin_plot_list[i])
            try:
                self.behav_bin_legend.sigClicked.connect(self.mouseClickEvent)
            except Exception as e:
                return

    def plot_behav_data(self, selected=None):
        if self.heatmap is not None:
            self.p4.removeItem(self.heatmap)
        if selected is None:
            beh = self.behav_data
        else:
            beh = self.behav_data[selected]
        vmin, vmax = -np.percentile(self.behav_data, 95), np.percentile(self.behav_data, 95)
        self.heatmap = pg.ImageItem(beh, autoDownsample=True, levels=(vmin,vmax))
        colormap = cm.get_cmap("coolwarm")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        lut = lut[0:-3,:]
        # apply the colormap
        self.heatmap.setLookupTable(lut)        
        self.p4.addItem(self.heatmap)
        self.behav_ROI_update()

    def behav_ROI_update(self):
        self.p4.setXRange(self.xrange[0],self.xrange[-1])
        self.p4.setLimits(xMin=self.xrange[0],xMax=self.xrange[-1])

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

    def run_RMAP(self):
        if self.default_param_radiobutton.isChecked():
            io.set_rastermap_params(self)
            print("Using default params")
        else:
            print("Using custom rastermap params")
        model = Rastermap(smoothness=1, 
                        n_clusters=self.n_clusters, 
                        n_PCs=200, 
                        n_splits=self.n_splits,
                        grid_upsample=self.grid_upsample).fit(self.sp)

        self.embedding = model.embedding
        self.embedded = True
        self.sorting = model.isort
        self.U = model.U
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
