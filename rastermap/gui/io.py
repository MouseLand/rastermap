import os, glob
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication, QWidget, QScrollBar, QSlider, QComboBox, QGridLayout, QPushButton, QFrame, QCheckBox, QLabel, QProgressBar, QLineEdit, QMessageBox, QGroupBox
import pyqtgraph as pg
from scipy.stats import zscore
import scipy.io as sio
from . import guiparts

def load_mat(parent, name=None):
    """ load data matrix of neurons by time (*.npy or *.mat)
    
    Note: can only load mat files containing one key assigned to data matrix
    
    """
    #name = 'C:/Users/carse/DATA/gt1/suite2p/plane0/spks.npy'
    try:
        if name is None:
            name = QFileDialog.getOpenFileName(
                parent, "Open *.npy or *.mat", filter="*.npy *.mat")
            parent.fname = name[0]
            parent.filebase = name[0]
        else:
            parent.fname = name
            parent.filebase = name
        ext = os.path.splitext(parent.fname)[-1]
        parent.update_status_bar("Loading "+ parent.fname)
        if ext == '.mat':    
            try:   
                X = sio.loadmat(parent.fname)
                if isinstance(X, dict):
                    for i, key in enumerate(X.keys()):
                        if key not in ['__header__', '__version__', '__globals__']:
                            X = X[key]
            except NotImplementedError:
                try:
                    import mat73
                except ImportError:
                    print('please "pip install mat73"')
                X = mat73.loadmat(parent.fname)
                if isinstance(X, dict):
                    for i, key in enumerate(X.keys()):
                        if key not in ['__header__', '__version__', '__globals__']:
                            X = X[key]
        elif ext == '.npy':
            X = np.load(parent.fname) # allow_pickle=True
        else:
            raise Exception("Invalid file type")
    except Exception as e:
        parent.update_status_bar(e)
        X = None
        return

    if X is None:
        return 
    if X.ndim == 1:
        parent.update_status_bar('ERROR: 1D array provided, but rastermap requires 2D array')
        return
    elif X.ndim > 3:
        parent.update_status_bar('ERROR: nD array provided (n>3), but rastermap requires 2D array')
        return
    elif X.ndim == 3:
        parent.update_status_bar('WARNING: 3D array provided (n>3), rastermap requires 2D array, will flatten to 2D')
        return

    if X.shape[0] < 10:
        parent.update_status_bar('ERROR: matrix with fewer than 10 neurons provided')
    
    parent.update_status_bar(f'activity loaded: {X.shape[0]} neurons by {X.shape[1]} timepoints')

    if len(X.shape) == 3:
        parent.update_status_bar(f'activity matrix has third dimension of size {X.shape[-1]}, flattening matrix to size ({X.shape[0]}, {X.shape[1] * X.shape[-1]}')
        X = X.reshape(X.shape[0], -1)
    parent.update_status_bar(f'z-scoring activity matrix')
    parent.sp = zscore(X, axis=1)
    del X

    load_iscell_stat(parent)

    parent.p0.clear()
    parent.embedding = np.arange(0, parent.sp.shape[0]).astype(np.int64)[:,np.newaxis]
    parent.sorting = np.arange(0, parent.sp.shape[0]).astype(np.int64)
    load_sp(parent)


def load_proc(parent, name=None):
    if name is None:
        name = QFileDialog.getOpenFileName(
            parent, "Open processed file", filter="*.npy"
            )
        parent.fname = name[0]
        name = parent.fname
    else:
        parent.fname = name
    try:
        proc = np.load(name, allow_pickle=True).item()
        parent.proc = proc
        X    = np.load(parent.proc['filename'])
        parent.filebase = parent.proc['filename']
        isort = parent.proc['isort']
        y     = parent.proc['embedding']
        u     = parent.proc['uv'][0] 
        ops   = parent.proc['ops']
    except Exception as e:
        parent.update_status_bar(e)
        X = None

    if X is not None:
        parent.filebase = parent.proc['filename']
        iscell, file_iscell = load_iscell(parent)
        xy, file_stat = load_stat(parent)

        parent.neuron_pos = xy 
        parent.iscell = iscell
        parent.startROI = False
        parent.posROI = np.zeros((2,2))
        parent.p0.clear()

        parent.update_status_bar(f'z-scoring activity matrix')
        parent.sp = zscore(X, axis=1)
        del X
        
        parent.embedding = y
        parent.sorting = isort
        parent.U = u
        parent.embedded = True

        parent.update_status_bar("Loaded: "+ parent.proc['filename'])
        load_iscell_stat(parent)
        load_sp(parent)
        parent.show()
    if 0:
        #elif X is not None:
        parent.filebase = parent.proc['filename']
        parent.embedding = y
        parent.sorting = isort
        parent.U = u
        parent.embedded = True
        parent.plot_activity() 
        parent.show()


def load_sp(parent):
    if parent.sp.shape[0] < 100:
        smooth = 1
    elif parent.sp.shape[0] < 1000:
        smooth = 5
    else:
        smooth = 10
    parent.update_status_bar(f'setting neuron bin size to {smooth} for visualization')
    parent.smooth.setText(str(smooth))
    
    parent.loaded = True
    parent.plot_activity()
    if parent.neuron_pos is not None:
        parent.scatter_comboBox.setCurrentIndex(1)
        parent.plot_scatter_pressed() 
            
    parent.show()
    parent.loadOne.setEnabled(True)
    parent.loadNd.setEnabled(True)
    parent.loadXY.setEnabled(True)
    parent.runRmap.setEnabled(True)

def load_iscell_stat(parent):
    iscell, file_iscell = load_iscell(parent)
    xy, file_stat = load_stat(parent)
    
    if iscell is not None:
        if len(iscell) == parent.sp.shape[0]:
            parent.sp = parent.sp[iscell, :]
        parent.iscell = iscell 
        parent.file_iscell = file_iscell
        parent.update_status_bar(f'using iscell.npy in folder, {parent.sp.shape[0]} neurons labeled as cells')
    
    if xy is not None:
        if iscell is not None and len(xy)==len(iscell): 
            xy = xy[iscell]
        if len(xy) == parent.sp.shape[0]:
            parent.neuron_pos = xy
            parent.update_status_bar(f'using stat.npy in folder for xy positions of neurons')
            parent.file_stat = file_stat


def enable_time_range(dialog):
    if dialog.time_checkbox.isChecked():
        dialog.slider.setEnabled(True)
    else:
        dialog.slider.setEnabled(False)

def get_behav_data(parent):
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Upload behavior files")
    dialog.verticalLayout = QtWidgets.QVBoxLayout(dialog)

    # Param options
    dialog.behav_data_label = QtWidgets.QLabel(dialog)
    dialog.behav_data_label.setTextFormat(QtCore.Qt.RichText)
    dialog.behav_data_label.setText("Behavior matrix (*.npy, *.mat):")
    dialog.behav_data_button = QPushButton('Upload')
    dialog.behav_data_button.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    dialog.behav_data_button.clicked.connect(lambda: load_behav_file(parent, dialog.behav_data_button))

    dialog.behav_comps_label = QtWidgets.QLabel(dialog)
    dialog.behav_comps_label.setTextFormat(QtCore.Qt.RichText)
    dialog.behav_comps_label.setText("(Optional) Behavior labels file (*.npy):")
    dialog.behav_comps_button = QPushButton('Upload')
    dialog.behav_comps_button.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    dialog.behav_comps_button.clicked.connect(lambda: load_behav_comps_file(parent, dialog.behav_comps_button))

    dialog.ok_button = QPushButton('Done')
    dialog.ok_button.setDefault(True)
    dialog.ok_button.clicked.connect(dialog.close)

    # Set layout of options
    dialog.widget = QtWidgets.QWidget(dialog)
    dialog.horizontalLayout = QtWidgets.QHBoxLayout(dialog.widget)
    dialog.horizontalLayout.setContentsMargins(-1, -1, -1, 0)
    dialog.horizontalLayout.setObjectName("horizontalLayout")
    dialog.horizontalLayout.addWidget(dialog.behav_data_label)
    dialog.horizontalLayout.addWidget(dialog.behav_data_button)

    dialog.widget1 = QtWidgets.QWidget(dialog)
    dialog.horizontalLayout = QtWidgets.QHBoxLayout(dialog.widget1)
    dialog.horizontalLayout.setContentsMargins(-1, -1, -1, 0)
    dialog.horizontalLayout.setObjectName("horizontalLayout")
    dialog.horizontalLayout.addWidget(dialog.behav_comps_label)
    dialog.horizontalLayout.addWidget(dialog.behav_comps_button)

    dialog.widget2 = QtWidgets.QWidget(dialog)
    dialog.horizontalLayout = QtWidgets.QHBoxLayout(dialog.widget2)
    dialog.horizontalLayout.addWidget(dialog.ok_button)

    # Add options to dialog box
    dialog.verticalLayout.addWidget(dialog.widget)
    dialog.verticalLayout.addWidget(dialog.widget1)
    dialog.verticalLayout.addWidget(dialog.widget2)

    dialog.adjustSize()
    dialog.exec_()

def load_iscell(parent):
    basename,filename = os.path.split(parent.filebase)
    try:
        file_iscell = os.path.join(basename, "iscell.npy")
        iscell = np.load(file_iscell)
        probcell = iscell[:, 1]
        iscell = iscell[:, 0].astype(np.bool)
    except (ValueError, OSError, RuntimeError, TypeError, NameError):
        iscell = None
        file_iscell = None
    return iscell, file_iscell

def load_stat(parent):
    basename,filename = os.path.split(parent.filebase)
    try:
        file_stat = os.path.join(basename, "stat.npy")
        stat = np.load(file_stat, allow_pickle=True)
        xy = np.array([s['med'] for s in stat])
    except (ValueError, OSError, RuntimeError, TypeError, NameError):
        xy = None
        file_stat = None
    return xy, file_stat

def load_behav_comps_file(parent, button):
    name = QFileDialog.getOpenFileName(
        parent, "Open *.npy", filter="*.npy"
    )
    name = name[0]
    parent.behav_labels_loaded = False
    try:
        if parent.behav_loaded:
            beh = np.load(name, allow_pickle=True) # Load file (behav_comps x time)
            if beh.ndim == 1 and beh.shape[0] == parent.behav_data.shape[0]:
                parent.behav_labels_loaded = True
                parent.behav_labels = beh
                parent.update_status_bar("Behav labels file loaded")
                button.setText("Uploaded!")
                parent.heatmap_checkBox.setEnabled(True)
            else:
                raise Exception("File contains incorrect dataset. Dimensions mismatch",
                            beh.shape, "not same as", parent.behav_data.shape[0])
        else:
            raise Exception("Please upload behav data (matrix) first")
    except Exception as e:
        print(e)
    add_behav_checkboxes(parent)

def add_behav_checkboxes(parent):
    # Add checkboxes for behav comps to display on heatmap and/or avg trace
    if parent.behav_labels_loaded:
        parent.behav_labels_selected = np.arange(0, len(parent.behav_labels)+1)
        if len(parent.behav_labels) > 5:
            prompt_behav_comps_ind(parent)
        clear_old_behav_checkboxes(parent)
        parent.heatmap_chkbxs.append(QCheckBox("All"))
        parent.heatmap_chkbxs[0].setStyleSheet("color: gray;")
        parent.heatmap_chkbxs[0].setChecked(True)
        parent.heatmap_chkbxs[0].toggled.connect(parent.behav_chkbx_toggled)
        parent.l0.addWidget(parent.heatmap_chkbxs[-1], 15, 12, 1, 2)
        for i, comp_ind in enumerate(parent.behav_labels_selected):
            parent.heatmap_chkbxs.append(QCheckBox(parent.behav_labels[comp_ind]))
            parent.heatmap_chkbxs[-1].setStyleSheet("color: gray;")
            parent.heatmap_chkbxs[-1].toggled.connect(parent.behav_chkbx_toggled)
            parent.heatmap_chkbxs[-1].setEnabled(False)
            parent.l0.addWidget(parent.heatmap_chkbxs[-1], 16+i, 12, 1, 2)
        parent.show_heatmap_ops()
        parent.update_scatter_ops_pos()
        parent.scatterplot_checkBox.setChecked(True)
    else:
        return

def clear_old_behav_checkboxes(parent):
    for k in range(len(parent.heatmap_chkbxs)):
        parent.l0.removeWidget(parent.heatmap_chkbxs[k])
    parent.heatmap_chkbxs = []

def prompt_behav_comps_ind(parent):
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Select max 5")
    dialog.verticalLayout = QtWidgets.QVBoxLayout(dialog)

    dialog.chkbxs = [] 
    for k in range(len(parent.behav_labels)):
        dialog.chkbxs.append(QCheckBox(parent.behav_labels[k]))
        dialog.chkbxs[k].setStyleSheet("color: black;")
        dialog.chkbxs[k].toggled.connect(lambda: restrict_behav_comps_selection(dialog, parent))

    dialog.ok_button = QPushButton('Done')
    dialog.ok_button.setDefault(True)
    dialog.ok_button.clicked.connect(lambda: get_behav_comps_ind(dialog, parent))

    # Add options to dialog box
    for k in range(len(dialog.chkbxs)):
        dialog.verticalLayout.addWidget(dialog.chkbxs[k])
    dialog.verticalLayout.addWidget(dialog.ok_button)
    
    dialog.adjustSize()
    dialog.exec_()

def get_behav_comps_ind(dialog, parent):
    parent.behav_labels_selected = []
    for k in range(len(parent.behav_labels)):
        if dialog.chkbxs[k].isChecked():
            parent.behav_labels_selected.append(k)
    dialog.close()

def restrict_behav_comps_selection(dialog, parent):
    chkbxs_count = 0
    for k in range(len(dialog.chkbxs)):
        if dialog.chkbxs[k].isChecked():
            chkbxs_count += 1
    if chkbxs_count > 5:
        for k in range(len(dialog.chkbxs)):
            dialog.chkbxs[k].setChecked(False)

def load_behav_file(parent, button):
    name = QFileDialog.getOpenFileName(
        parent, "Load behavior data", filter="*.npy *.mat"
    )
    name = name[0]
    parent.behav_loaded = False
    try:  # Load file (behav_comps x time)
        ext = name.split(".")[-1]
        if ext == "mat":
            beh = sio.loadmat(name)
            load_behav_dict(parent, beh)
            del beh
        elif ext == "npy":
            beh = np.load(name, allow_pickle=True) 
            dict_item = False
            if beh.size == 1:
                beh = beh.item()
                dict_item = True
            if dict_item:
                load_behav_dict(parent, beh)
            else:  # load matrix w/o labels and set default labels
                if beh.ndim==1:
                    beh = beh[np.newaxis,:]
                elif beh.ndim==3:
                    parent.update_status_bar('WARNING: 3D array provided (n>3), rastermap requires 2D array, will flatten to 2D')
                    beh = beh.reshape(beh.shape[0], -1)
                if parent.embedded and parent.embed_time_range != -1:
                    beh = beh[:,parent.embed_time_range[0]:parent.embed_time_range[-1]]
                if beh.shape[1] == parent.sp.shape[1]:
                    parent.behav_data = beh
                    clear_old_behav_checkboxes(parent)
                    parent.behav_loaded = True
                else:
                    raise Exception("File contains incorrect dataset. Dimensions mismatch",
                                beh.shape[1], "not same as", parent.sp.shape[1])
            del beh
    except Exception as e:
        parent.update_status_bar(f'ERROR: {e}')
    if parent.behav_loaded:
        button.setText("Uploaded!")
        parent.behav_data = zscore(parent.behav_data, axis=1)
        parent.plot_behav_data()
        parent.heatmap_checkBox.setEnabled(True)
        parent.heatmap_checkBox.setChecked(True)
    else:
        return

def load_behav_dict(parent, beh):
    for i, key in enumerate(beh.keys()):
        if key not in ['__header__', '__version__', '__globals__']:
            if np.array(beh[key]).size == parent.sp.shape[1]:
                parent.behav_labels.append(key)
                parent.behav_labels_loaded = True
            else:
                parent.behav_binary_labels.append(key)
                parent.behav_binary_labels_loaded = True
    if parent.behav_labels_loaded:
        parent.behav_data = np.zeros((len(parent.behav_labels), parent.sp.shape[1]))
        for j, key in enumerate(parent.behav_labels):
            parent.behav_data[j] = beh[key]
        parent.behav_data = np.array(parent.behav_data)
        parent.behav_labels = np.array(parent.behav_labels)
        parent.behav_loaded = True
        add_behav_checkboxes(parent)
    if parent.behav_binary_labels_loaded:
        parent.behav_binary_data = np.zeros((len(parent.behav_binary_labels), parent.sp.shape[1]))
        parent.behav_bin_legend = pg.LegendItem(labelTextSize='12pt', horSpacing=30, colCount=len(parent.behav_binary_labels))
        for i, key in enumerate(parent.behav_binary_labels):
            dat = np.zeros(parent.sp.shape[1]) 
            dat[beh[key]] = 1                   # Convert to binary for stim/lick time
            parent.behav_binary_data[i] = dat
            parent.behav_bin_plot_list.append(pg.PlotDataItem(symbol=parent.symbol_list[i]))
            parent.behav_bin_legend.addItem(parent.behav_bin_plot_list[i], name=parent.behav_binary_labels[i])
            parent.behav_bin_legend.setPos(parent.oned_trace_plot.x()+(20*i), parent.oned_trace_plot.y())
            parent.behav_bin_legend.setParentItem(parent.p3)
            parent.p3.addItem(parent.behav_bin_plot_list[-1])
    if parent.behav_binary_data is not None:
        parent.plot_behav_binary_data()

def load_oned_data(parent):
    name = QFileDialog.getOpenFileName(
        parent, "Open *.npy", filter="*.npy"
    )
    name = name[0]
    parent.oned_loaded = False
    try:
        oned = np.load(name)
        oned = oned.flatten()
        if parent.embedded and parent.embed_time_range != -1:
            oned = oned[parent.embed_time_range[0]:parent.embed_time_range[-1]]
        if oned.size == parent.sp.shape[1]:
            parent.oned_loaded = True
    except (ValueError, KeyError, OSError,
            RuntimeError, TypeError, NameError):
        parent.update_status_bar("ERROR: this is not a 1D array with length of data")
    if parent.oned_loaded:
        parent.oned_data = oned
        parent.plot_oned_trace()
        if parent.scatterplot_checkBox.isChecked():
            parent.scatterplot_checkBox.setChecked(True)
    else:
        return

def get_neuron_depth_data(parent):
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Upload file")
    dialog.verticalLayout = QtWidgets.QVBoxLayout(dialog)

    dialog.depth_label = QtWidgets.QLabel(dialog)
    dialog.depth_label.setTextFormat(QtCore.Qt.RichText)
    dialog.depth_label.setText("Depth (Ephys):")
    dialog.depth_button = QPushButton('Upload')
    dialog.depth_button.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    dialog.depth_button.clicked.connect(lambda: load_neuron_pos(parent, dialog.depth_button, depth=True))

    dialog.ok_button = QPushButton('Done')
    dialog.ok_button.setDefault(True)
    dialog.ok_button.clicked.connect(dialog.close)

    dialog.widget = QtWidgets.QWidget(dialog)
    dialog.horizontalLayout = QtWidgets.QHBoxLayout(dialog.widget)
    dialog.horizontalLayout.setContentsMargins(-1, -1, -1, 0)
    dialog.horizontalLayout.setObjectName("horizontalLayout")
    dialog.horizontalLayout.addWidget(dialog.depth_label)
    dialog.horizontalLayout.addWidget(dialog.depth_button)

    dialog.verticalLayout.addWidget(dialog.widget)
    dialog.verticalLayout.addWidget(dialog.ok_button)
    dialog.adjustSize()
    dialog.exec_()

def load_neuron_pos(parent):
    try:
        file_name = QFileDialog.getOpenFileName(
                    parent, "Open *.npy (array or stat.npy)", filter="*.npy")
        data = np.load(file_name[0], allow_pickle=True)
        
        if len(data) != parent.sp.shape[0] and hasattr(parent, 'iscell') and parent.iscell is not None:
            data = np.array(data)[parent.iscell]
        elif len(data) != parent.sp.shape[0]:
            parent.update_status_bar('ERROR: npy array is not the same length as data ')
            
        if isinstance(data[0], np.ndarray):
            parent.neuron_pos = data
        elif isinstance(data[0], dict):
            parent.neuron_pos = np.array([s['med'] for s in data])

        parent.scatter_comboBox.setCurrentIndex(1)
        parent.plot_neuron_pos(init=True)
        
    except Exception as e:
        parent.update_status_bar('ERROR: this is not a *.npy array :( ')

def save_proc(parent): # Save embedding output
    try:
        if parent.embedded:
            if parent.save_path is None:
                folderName = QFileDialog.getExistingDirectory(parent,
                                    "Choose save folder")
                parent.save_path = folderName
                    
            else:
                raise Exception("Incorrect folder. Please select a folder")
            if parent.save_path:
                filename = parent.fname.split("/")[-1]
                filename, ext = os.path.splitext(filename)
                savename = os.path.join(parent.save_path, ("%s_rastermap_proc.npy"%filename))
                # Rastermap embedding parameters
                ops = {'n_components'       : parent.n_components, 
                        'n_clusters'        : parent.n_clusters,
                        'n_neurons'         : parent.n_neurons, 
                        'grid_upsample'     : parent.grid_upsample,
                        'n_splits'          : parent.n_splits,
                        'embed_time_range'  : parent.embed_time_range}
                proc = {'filename': parent.fname, 'save_path': parent.save_path,
                        'isort' : parent.sorting, 'embedding' : parent.embedding,
                        'U' : parent.U, 'ops' : ops}
                
                np.save(savename, proc, allow_pickle=True)
                parent.update_status_bar("File saved: "+ savename)
        else:
            raise Exception("Please run embedding to save output")
    except Exception as e:
        #parent.update_status_bar(e)
        return
