"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import sys, os
import numpy as np
import pyqtgraph as pg
from qtpy import QtGui, QtCore
from qtpy import QtWidgets as QtW
from qtpy.QtCore import QEvent
from qtpy.QtWidgets import QMainWindow, QApplication, QWidget, QScrollBar, QSlider, QComboBox, QGridLayout, QPushButton, QFrame, QCheckBox, QLabel, QProgressBar, QLineEdit, QMessageBox, QGroupBox, QButtonGroup, QRadioButton, QStatusBar
from scipy.stats import zscore
# patch for Qt 5.15 on macos >= 12
os.environ["USE_MAC_SLIDER_PATCH"] = "1"
from superqt import QRangeSlider  # noqa

Horizontal = QtCore.Qt.Orientation.Horizontal
Vertical = QtCore.Qt.Orientation.Vertical

from . import menus, guiparts, io, colormaps, views

nclust_max = 100


class MainW(QMainWindow):

    def __init__(self, filename=None, proc=False):
        super(MainW, self).__init__()
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(25, 25, 1600, 800)
        self.setWindowTitle("Rastermap - neural data visualization")
        icon_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "logo.png")
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
        self.win.move(600, 0)
        self.win.resize(1000, 500)
        self.l0.addWidget(self.win, 0, 0, 22, 14)
        layout = self.win.ci.layout

        
        # viewbox placeholder for buttons
        self.p0 = self.win.addViewBox(lockAspect=True, row=0, col=0)
        self.p0.setMenuEnabled(False)

        # Plot entire neural activity dataset
        self.p1 = self.win.addPlot(title="FULL VIEW", row=0, col=1, colspan=2)
        self.p1.setMouseEnabled(x=False, y=False)
        self.img = pg.ImageItem(autoDownsample=True)
        self.p1.addItem(self.img)
        self.p1.setLabel("left", "binned neurons")
        self.p1.setLabel("bottom", "time")
        self.p1.invertY(True)

        # Plot a zoomed in region from full view (changes across time axis)
        self.p2 = self.win.addPlot(title="ZOOM IN", row=1, col=0, colspan=2, rowspan=1)
        self.p2.setMenuEnabled(False)
        self.imgROI = pg.ImageItem(autoDownsample=True)
        self.p2.addItem(self.imgROI)
        self.p2.setMouseEnabled(x=False, y=False)
        self.p2.setLabel("bottom", "time")
        self.p2.setLabel("left", "binned neurons")
        ax = self.p2.getAxis("bottom")
        ticks = [0]
        ax.setTicks([[(v, ".") for v in ticks]])
        self.p2.invertY(True)
        self.p2.scene().sigMouseMoved.connect(self.mouse_moved)

        # Plot avg. activity of neurons selected in ROI of zoomed in view
        self.p3 = self.win.addPlot(row=2, col=0, rowspan=1, colspan=2, padding=0)
        self.p3.setMouseEnabled(x=False, y=False)
        self.p3.setLabel("bottom", "time")
        self.p3.setLabel("left", "selected")
        self.cluster_plots = []
        self.cluster_rois = []
        for i in range(nclust_max):
            self.cluster_plots.append(pg.PlotDataItem())
        for i in range(nclust_max):
            self.p3.addItem(self.cluster_plots[i])

        # Plot behavioral dataset as heatmap
        self.p4 = self.win.addPlot(row=3, col=0, colspan=2, rowspan=1)
        self.p4.setMouseEnabled(x=False, y=False)
        self.p4.setLabel("bottom", "time")
        self.p4.setLabel("left", "behavior")

        # align plots
        self.p2.getAxis("left").setWidth(int(40))
        self.p3.getAxis("left").setWidth(int(40))
        self.p4.getAxis("left").setWidth(int(40))

        # Scatter plot for oned correlation, neuron position, and depth (ephys) information
        self.p5 = self.win.addPlot(title="scatter plot", row=1, col=2)
        self.p5.setMouseEnabled(x=False, y=False)
        self.scatter_plots = [[]]
        for i in range(nclust_max + 1):
            self.scatter_plots[0].append(pg.ScatterPlotItem())
        for i in range(nclust_max + 1):
            self.p5.addItem(self.scatter_plots[0][i])

        # Set colormap to deafult of gray_r. ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Future: add option to change cmap ~~~~~~~~~~~~~~
        lut = colormaps.gray[::-1]
        # apply the colormap
        self.img.setLookupTable(lut)
        self.imgROI.setLookupTable(lut)
        layout.setColumnStretchFactor(1, 3)
        layout.setRowStretchFactor(1, 4)
        layout.setRowStretchFactor(2, 2)
        layout.setRowStretchFactor(3, 0)

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
        self.sat = [30., 70.]
        self.sat_slider = QRangeSlider(Horizontal)
        self.sat_slider.setRange(0., 100.)
        self.sat_slider.setTickPosition(QtW.QSlider.TickPosition.TicksAbove)
        self.sat_slider.valueChanged.connect(self.sat_changed)
        self.sat_slider.setValue((self.sat[0], self.sat[1]))
        self.sat_slider.setFixedWidth(130)
        sat_label = QLabel("Saturation")
        sat_label.setStyleSheet("color: white;")

        # Add drop down options for scatter plot
        self.scatter_comboBox = QComboBox(self)
        self.scatter_comboBox.setFixedWidth(120)
        scatter_comboBox_ops = ["neuron position"]
        self.scatter_comboBox.addItems(scatter_comboBox_ops)
        self.scatter_comboBox.setCurrentIndex(0)
        self.all_neurons_checkBox = QCheckBox("color all neurons")
        self.all_neurons_checkBox.setStyleSheet("color: gray;")
        self.scatterplot_button = QPushButton("plot")
        self.scatterplot_button.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.scatterplot_button.clicked.connect(self.plot_scatter_pressed)
        self.scatterplot_button_3D = QPushButton("view 3D")
        self.scatterplot_button_3D.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.scatterplot_button_3D.clicked.connect(self.plane_window)

        self.setAcceptDrops(True)

        # Status bar
        self.statusBar = QStatusBar()
        self.statusBar.setStyleSheet("color: white;")
        #self.l0.addWidget(self.statusBar, 23,0,1,14)

        # Default variables
        self.tpos = -0.5
        self.tsize = 1
        self.reset_variables()

        self.init_time_roi()

        # Add features to window
        ops_row_pos = 0
        self.l0.addWidget(ysm, ops_row_pos, 0, 1, 1)
        self.l0.addWidget(self.smooth, ops_row_pos, 1, 1, 1)
        self.l0.addWidget(sat_label, ops_row_pos + 1, 0, 1, 2)
        self.l0.addWidget(self.sat_slider, ops_row_pos + 2, 0, 1, 2)

        self.l0.addWidget(self.scatterplot_button, 16, 12, 1, 1)
        self.l0.addWidget(self.scatterplot_button_3D, 16, 13, 1, 1)
        self.l0.addWidget(self.scatter_comboBox, 17, 12, 1, 1)
        self.l0.addWidget(self.all_neurons_checkBox, 17, 13, 1, 1)

        self.win.show()
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)

        if filename is not None:
            if not proc:
                io.load_mat(self, filename)
            else:
                io.load_proc(self, name=filename)
        self.show()

    def init_time_roi(self):
        self.TimeROI = guiparts.TimeROI(parent=self, color=[0, 0, 255, 25],
                                        bounds=[0, self.n_time])
        self.xrange = slice(0, min(500, ((self.n_time // 10) // 4) * 4))
        self.TimeROI.setRegion([self.xrange.start, self.xrange.stop - 1])
        self.TimeROI.sigRegionChangeFinished.connect(self.TimeROI.time_set)
        self.p1.addItem(self.TimeROI)
        self.TimeROI.setZValue(10)  # make sure ROI is drawn above image
        self.TimeROI.time_set()

    def randomize_colors(self, random=False):
        np.random.seed(0 if not random else np.random.randint(500))
        rperm = np.random.permutation(nclust_max)
        self.colors = colormaps.gist_rainbow[np.linspace(
            0, 254, nclust_max).astype("int")][rperm]
        self.colors[:, -1] = 50
        self.colors = list(self.colors)

    def sat_changed(self):
        self.sat = self.sat_slider.value()
        self.img.setLevels([self.sat[0] / 100., self.sat[1] / 100.])
        self.imgROI.setLevels([self.sat[0] / 100., self.sat[1] / 100.])
        self.show()

    def reset(self):
        self.p1.clear()
        self.p2.clear()
        self.p3.clear()
        self.p4.clear()
        self.p5.clear()

    def reset_variables(self):
        # Neural activity/spike dataset set as self.sps
        nn = 100
        sp = np.zeros((nn, 100), np.float32)
        self.sp = sp
        self.sp_smoothed = self.sp.copy()
        self.n_time = self.sp.shape[1]
        self.smooth_limit = None
        self.selected = slice(0, nn)
        
        for i in range(len(self.cluster_rois)):
            self.p2.removeItem(self.cluster_rois[i])
        for i in range(nclust_max + 1):
            self.scatter_plots[0][i].setData([], [])
            if i < nclust_max:
                self.cluster_plots[i].setData([], [])
        self.p2.show()
        self.p5.show()
        self.randomize_colors()
        self.startROI = False
        self.posROI = np.zeros((2, 2))
        self.cluster_rois, self.cluster_slices = [], []
        self.user_clusters = None
        self.loaded = False
        self.behav_data = None
        self.behav_binary_data = None
        self.behav_bin_plot_list = []
        self.behav_labels = []
        self.behav_binary_labels = []
        self.behav_corr_all = None
        self.zstack = None
        self.xrange = None
        self.file_iscell = None
        self.iscell = None
        self.neuron_pos = None
        self.save_path = None  # Set default to current folder
        self.embedding = None
        self.heatmap = None
        self.sat_slider.setValue([30., 70.])
        self.line = pg.PlotDataItem()
        self.symbol_list = [
            "star", "d", "x", "o", "t", "t1", "t2", "p", "+", "s", "t3", "h"
        ]
        self.embed_time_range = -1
        self.params_set = False

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        file = files[0]
        file0, ext = os.path.splitext(file)
        proc_file = file0 + "_embedding.npy"
        if ext == ".npy" or ext == ".mat" or ext==".npz" or ext==".nwb":
            if file[-14:] == "_embedding.npy":
                io.load_proc(self, name=files[0])
            elif os.path.exists(proc_file):
                io.load_proc(self, name=proc_file)
            else:
                io.load_mat(self, name=files[0])
        else:
            print("ERROR: must drag and drop *.npy, *.npz, *.nwb or *.mat files")

    def plane_window(self):
        self.PlaneWindow = views.PlaneWindow(self)
        self.PlaneWindow.show()

    def update_status_bar(self, message, update_progress=False):
        if update_progress:
            self.progressBar.show()
            progressBar_value = [
                int(s) for s in message.split("%")[0].split() if s.isdigit()
            ]
            self.progressBar.setValue(progressBar_value[0])
            frames_processed = np.floor(
                (progressBar_value[0] / 100) * float(self.totalFrameNumber.text()))
            self.setFrame.setText(str(frames_processed))
            self.statusBar.showMessage(message.split("|")[0])
        else:
            print(message)
            #self.statusBar.showMessage(message)
        self.show()

    #def wheelEvent(self, event):
    #    return
        #if event.type() == QEvent.Wheel:
        #print(event)
        #items = self.win.scene().items(event.scenePos())
        #for x in items:
        #if x==self.p1 or x==self.p2 or x==self.p3 or x==self.p4:
        #self.changed.emit(event.angleDelta().y() > 0)
        #        print(event)
        #s = 0.9
        #zoom = (s, s) if in_or_out == "in" else (1/s, 1/s)
        #self.plot.vb.scaleBy(zoom)

    def plot_clicked(self, event):
        items = self.win.scene().items(event.scenePos())
        for x in items:
            if x == self.p1 and event.button() == 1 and event.double():
                self.TimeROI.setRegion([0, self.n_time])
            elif x == self.p2 and event.button() == QtCore.Qt.RightButton:
                pos = self.p2.vb.mapSceneToView(event.scenePos())
                x, y = pos.x(), pos.y()
                if not self.startROI:
                    self.startROI = True
                    self.posROI[0, :] = [x, y]
                else:
                    # plotting
                    self.startROI = False
                    self.posROI[1, :] = [x, y]
                    self.p2.removeItem(self.line)
                    y0, y1 = self.posROI[:, 1].min(), self.posROI[:, 1].max()
                    y0, y1 = int(y0), int(y1)
                    y1 = y1 if y1 > y0 else y0 + 1
                    self.selected = slice(y0, y1)
                    self.add_cluster()
                self.posAll = []
                self.lp = []

    def mouse_moved(self, pos):
        if self.p2.sceneBoundingRect().contains(pos):
            x = self.p2.vb.mapSceneToView(pos).x()
            y = self.p2.vb.mapSceneToView(pos).y()
            if self.startROI:
                self.posROI[1, :] = [x, y]
                self.p2.removeItem(self.line)
                pen = pg.mkPen(color="yellow", width=3)
                self.line = pg.PlotDataItem(self.posROI[:, 0], self.posROI[:, 1],
                                            pen=pen)
                self.p2.addItem(self.line)

    def keyPressEvent(self, event):
        bid = -1
        move_time = False
        if self.loaded:
            xrange = self.xrange
            twin = xrange.stop - xrange.start
            # todo: zoom and move across neurons
            #if event.key() == QtCore.Qt.Key_Down or event.key() == QtCore.Qt.Key_Up:
            # zoom and move in time
            if event.key() == QtCore.Qt.Key_Left or event.key() == QtCore.Qt.Key_Right:
                if event.modifiers() != QtCore.Qt.ShiftModifier:
                    move_time = True if (xrange.stop -
                                         xrange.start < self.n_time) else False
                    if move_time:
                        ### move in time in increments of 1/2 size of window
                        if xrange.start > 0 and event.key() == QtCore.Qt.Key_Left:
                            x0 = max(0, xrange.start - twin // 2)
                            x1 = x0 + (xrange.stop - xrange.start)
                        elif xrange.stop < self.n_time and event.key(
                        ) == QtCore.Qt.Key_Right:
                            x1 = min(self.n_time, xrange.stop + twin // 2)
                            x0 = x1 - (xrange.stop - xrange.start)
                        else:
                            move_time = False
                else:
                    tbin = 50
                    move_time = True
                    if twin > tbin and event.key() == QtCore.Qt.Key_Left:  # zoom in
                        x0 = xrange.start + tbin // 2
                        x1 = max(x0 + tbin, xrange.stop - tbin // 2)
                    elif twin < self.n_time and event.key(
                    ) == QtCore.Qt.Key_Right:  # zoom out
                        x0 = max(0, xrange.start - tbin // 2)
                        x1 = min(self.n_time, x0 + (xrange.stop - xrange.start) + tbin)
                    else:
                        move_time = False
                if move_time:
                    self.TimeROI.setRegion([x0, x1 - 1])

    def plot_traces(self, roi_id=None):
        if self.loaded:
            if roi_id is None:
                for roi_id in range(nclust_max):
                    if roi_id < len(self.cluster_rois):
                        self.plot_roi_trace(roi_id)
                    else:
                        self.cluster_plots[roi_id].setData([], [])
            else:
                self.plot_roi_trace(roi_id)
            self.p3.setXRange(self.xrange.start, self.xrange.stop - 1)
            self.p3.setLimits(xMin=self.xrange.start, xMax=self.xrange.stop - 1)
            if self.behav_data is not None:
                self.plot_behav_data()
            elif self.behav_binary_data is not None:
                self.plot_behav_binary_data()
            self.p3.show()

    def plot_roi_trace(self, roi_id):
        x = np.arange(0, self.n_time)
        kspace = 0.5
        selected = self.cluster_slices[roi_id]
        y = self.sp_smoothed[selected].mean(axis=0)
        y = (y - y.min()) / (y.max() - y.min())
        y += kspace * roi_id
        self.cluster_plots[roi_id].setData(x, y,
                                           pen=pg.mkPen(color=self.colors[roi_id][:3]))

    def smooth_activity(self):
        N = int(self.smooth.text())
        if self.smooth_limit is not None and N < self.smooth_limit:
            self.update_status_bar("cannot show matrix > 10GB, increasing smoothing")
            N = self.smooth_limit 
            self.smooth.setText(str(N))

        self.smooth_bin = N
        NN = self.n_samples
        nn = int(np.floor(NN / N))
        if N > 1:
            if self.sp is not None:
                self.sp_smoothed = np.reshape(self.sp[self.sorting][:nn * N],
                                              (nn, N, -1)).mean(axis=1)
            else:
                Usv_ds = np.reshape(self.Usv[self.sorting][:nn * N],
                                              (nn, N, -1)).mean(axis=1)
                self.sp_smoothed = (Usv_ds / self.sv) @ self.Vsv.T
            self.sp_smoothed = zscore(self.sp_smoothed, axis=1)
            self.sp_smoothed = np.maximum(-2, np.minimum(5, self.sp_smoothed)) + 2
            self.sp_smoothed /= 7
        else:
            self.sp_smoothed = self.sp[self.sorting].copy()
        self.nsmooth = self.sp_smoothed.shape[0]
        yr0 = min(4, self.nsmooth // 4)
        ym = self.nsmooth // 2
        self.selected = slice(ym - yr0, ym + yr0)
        if len(self.cluster_rois) > 0:
            for i in range(len(self.cluster_rois)):
                self.p2.removeItem(self.cluster_rois[i])
        self.cluster_rois, self.cluster_slices = [], []
        if self.user_clusters is None:
            self.add_cluster()
        self.get_behav_corr() if self.behav_data else None
        if self.neuron_pos is not None or self.behav_data is not None:
            self.update_scatter(init=True)
        elif self.neuron_pos is None and self.scatter_comboBox.currentIndex()==0:
            self.p5.clear()
        self.p2.show()
        self.p3.show()

    def add_cluster(self):
        roi_id = len(self.cluster_rois)
        self.cluster_rois.append(
            guiparts.ClusterROI(self, color=self.colors[roi_id],
                                bounds=(0, self.sp_smoothed.shape[0]), roi_id=roi_id))
        self.cluster_slices.append(self.selected)
        self.cluster_rois[-1].setRegion((self.selected.start, self.selected.stop))
        self.p2.addItem(self.cluster_rois[-1])
        self.cluster_rois[-1].setZValue(10)  # make sure ROI is drawn above image
        self.cluster_rois[-1].cluster_set()

    def plot_activity(self, init=False):
        if self.loaded:
            self.smooth_activity()
            nn, nt = self.sp_smoothed.shape
            self.img.setImage(self.sp_smoothed)
            self.img.setLevels([self.sat[0] / 100., self.sat[1] / 100.])
            self.p1.setXRange(0, nt, padding=0)
            self.p1.setYRange(0, nn, padding=0)
            self.p1.show()
            self.p2.setXRange(0, nt, padding=0)
            self.p2.setYRange(0, nn, padding=0)
            self.p2.show()
            if init:
                self.p1.removeItem(self.TimeROI)
                self.init_time_roi()
            else:
                self.TimeROI.time_set()
            self.plot_traces()
        self.show()
        self.win.show()

    def plot_behav_binary_data(self):
        for i in range(len(self.behav_bin_plot_list)):
            self.p4.removeItem(self.behav_bin_plot_list[i])
            dat = self.behav_binary_data[i][self.xrange]
            xdat, ydat = np.arange(self.xrange.start,
                                   self.xrange.stop)[dat > 0], dat[dat > 0]
            self.behav_bin_plot_list[i].setData(xdat, ydat, pen=None,
                                                symbol=self.symbol_list[i],
                                                symbolSize=12)
            self.p4.addItem(self.behav_bin_plot_list[i])
        self.p4.setLabel("left", ".")

    def plot_behav_data(self, selected=None):
        if self.heatmap is not None:
            if len(self.heatmap) > 1:
                for i in range(len(self.heatmap)):
                    self.p4.removeItem(self.heatmap[i])
            else:
                self.p4.removeItem(self.heatmap)
        beh = self.behav_data
        if beh.shape[0] > 10:
            vmin, vmax = np.percentile(beh, 5), np.percentile(beh, 95)
            self.heatmap = pg.ImageItem(beh, autoDownsample=True, levels=(vmin, vmax))
            # apply the colormap
            self.heatmap.setLookupTable(colormaps.viridis)
            self.p4.addItem(self.heatmap)
            self.p4.setLabel("left", "index")
        else:
            self.heatmap = []
            cmap = colormaps.gist_rainbow[np.linspace(10, 254,
                                                      beh.shape[0]).astype("int")]
            for i in range(beh.shape[0]):
                self.heatmap.append(pg.PlotCurveItem())
                self.heatmap[-1].setData(np.arange(0, beh.shape[-1]), zscore(beh[i]))
                self.heatmap[-1].setPen({"color": cmap[i], "width": 1})
                self.p4.addItem(self.heatmap[-1])
            self.p4.setLabel("left", "z-scored")

    def plot_scatter_pressed(self):
        self.update_scatter(init=True)

    def update_scatter(self, init=False, roi_id=None):
        if init:
            self.p5.setLabel("left", "")
            self.p5.setLabel("bottom", "")
            self.p5.invertY(False)
        request = self.scatter_comboBox.currentIndex()
        if request > 0:
            self.plot_behav_corr(roi_id=roi_id, init=init)
        else:
            self.plot_neuron_pos(roi_id=roi_id, init=init)

    def get_behav_corr(self):
        beh = self.behav_data
        self.behav_corr_all = (self.sp_smoothed @ beh.T) / self.n_time

    def neurons_selected(self, selected=None):
        selected = selected if selected is not None else self.selected
        select_slice = slice(selected.start * self.smooth_bin,
                             selected.stop * self.smooth_bin)
        neurons_select = self.sorting[
            select_slice] if self.embedding is not None else select_slice
        return neurons_select

    def plot_behav_corr(self, init=False, roi_id=None):
        if self.behav_data is not None:
            r = self.behav_corr_all()[:, self.scatter_comboBox.currentIndex() - 1]
            self.plot_scatter(r, np.arange(0, self.nsmooth), roi_id=roi_id)
            self.p5.setYRange(0, self.nsmooth)
            if init:
                self.p5.invertY(True)
                self.p5.setLabel("left", "position")
                self.p5.setLabel("bottom", "correlation")
        else:
            self.update_status_bar("ERROR: please upload behavioral data")

    def plot_neuron_pos(self, init=False, roi_id=None):
        if self.neuron_pos is not None:
            ypos, xpos = self.neuron_pos[:, 0], self.neuron_pos[:, 1]
            self.plot_scatter(ypos, xpos, roi_id=roi_id)
            if init:
                self.p5.setLabel("left", "y position")
                self.p5.setLabel("bottom", "x position")
        else:
            self.update_status_bar("ERROR: please upload neuron position data")

    def plot_scatter(self, x, y, roi_id=None, iplane=0):
        subsample = max(1, int(len(x)/5000))
        n_pts = len(x) // subsample
        marker_size = int(3 * max(1, 800 / n_pts))
        if self.all_neurons_checkBox.isChecked() and roi_id is None:
            colors = colormaps.gist_ncar[np.linspace(
                0, 254, len(x)).astype("int")][self.sorting]
            brushes = [pg.mkBrush(color=c) for c in colors[::subsample]]
            self.scatter_plots[iplane][0].setData(x[::subsample], y[::subsample], 
                                                  symbol="o", size=marker_size,
                                                  brush=brushes,
                                                  hoverable=True)
            for i in range(1, nclust_max + 1):
                self.scatter_plots[iplane][i].setData([], [])
        else:
            if roi_id is None:
                self.scatter_plots[iplane][0].setData(
                    x, y, symbol="o", size=marker_size,
                    brush=pg.mkBrush(color=(180, 180, 180)),
                    hoverable=True)
                for roi_id in range(nclust_max):
                    if roi_id < len(self.cluster_rois):
                        selected = self.neurons_selected(self.cluster_slices[roi_id])
                        self.scatter_plots[iplane][roi_id + 1].setData(
                            x[selected][::subsample], y[selected][::subsample], 
                            symbol="o", size=marker_size,
                            brush=pg.mkBrush(color=self.colors[roi_id][:3]),
                            hoverable=True)
                    else:
                        self.scatter_plots[iplane][roi_id + 1].setData([], [])
            else:
                selected = self.neurons_selected(self.cluster_slices[roi_id])
                self.scatter_plots[iplane][roi_id + 1].setData(
                    x[selected], y[selected], symbol="o", size=marker_size,
                    brush=pg.mkBrush(color=self.colors[roi_id][:3]), hoverable=True)


def run(filename=None, proc=False):
    # Always start by initializing Qt (only once per application)
    app = QApplication(sys.argv)
    icon_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logo.png")
    app_icon = QtGui.QIcon()
    app_icon.addFile(icon_path, QtCore.QSize(16, 16))
    app_icon.addFile(icon_path, QtCore.QSize(24, 24))
    app_icon.addFile(icon_path, QtCore.QSize(32, 32))
    app_icon.addFile(icon_path, QtCore.QSize(48, 48))
    app_icon.addFile(icon_path, QtCore.QSize(96, 96))
    app_icon.addFile(icon_path, QtCore.QSize(256, 256))
    app.setWindowIcon(app_icon)
    GUI = MainW(filename=filename, proc=proc)
    ret = app.exec_()
    # GUI.save_gui_data()
    sys.exit(ret)
