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
    parent.plot_activity(init=True)
            
    parent.show()
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
    try:
        if parent.behav_data is not None:
            beh = np.load(name, allow_pickle=True) # Load file (behav_comps x time)
            if beh.ndim == 1 and beh.shape[0] == parent.behav_data.shape[0]:
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

def load_behav_file(parent, button):
    name = QFileDialog.getOpenFileName(
        parent, "Load behavior data", filter="*.npy *.mat"
    )
    name = name[0]
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
                beh = beh[np.newaxis,:] if beh.ndim==1 else beh
                if beh.ndim==3:
                    parent.update_status_bar('WARNING: 3D array provided (n>3), rastermap requires 2D array, will flatten to 2D')
                    beh = beh.reshape(beh.shape[0], -1)
                beh = beh.T.copy() if beh.shape[0] == parent.sp.shape[1] else beh
                if beh.shape[1] == parent.sp.shape[1]:
                    parent.behav_data = beh
                else:
                    raise Exception("File contains incorrect dataset. Dimensions mismatch",
                                beh.shape[1], "not same as", parent.sp.shape[1])
            del beh
        button.setText("Uploaded!")
        parent.behav_data = zscore(parent.behav_data, axis=1)
        parent.get_behav_corr()
        parent.plot_behav_data()    
    except Exception as e:
        parent.update_status_bar(f'ERROR: {e}')
        return

def load_behav_dict(parent, beh):
    binary = False
    for i, key in enumerate(beh.keys()):
        if key not in ['__header__', '__version__', '__globals__']:
            if np.array(beh[key]).size == parent.sp.shape[1]:
                if j==0:
                    parent.behav_labels = []
                parent.behav_labels.append(key)
                binary = False
            else:
                if j==0:
                    parent.behav_binary_labels = []
                parent.behav_binary_labels.append(key)
                binary = True
            j+=1
    if j>0:
        if not binary:
            parent.behav_data = np.zeros((len(parent.behav_labels), parent.sp.shape[1]))
            for j, key in enumerate(parent.behav_labels):
                parent.behav_data[j] = beh[key]
            parent.behav_data = np.array(parent.behav_data)
            parent.behav_data = zscore(parent.behav_data, axis=1)
            parent.behav_labels = np.array(parent.behav_labels)
            parent.get_behav_corr()
            parent.plot_behav_data()    
        else:
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
            parent.plot_behav_binary_data()

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

        parent.scatter_comboBox.setCurrentIndex(0)
        parent.plot_neuron_pos(init=True)
        
    except Exception as e:
        parent.update_status_bar('ERROR: this is not a *.npy array :( ')

def load_zstack(parent, name=None):
    try:
        file_name = QFileDialog.getOpenFileName(
                    parent, "Open *.npy (array or ops.npy)", filter="*.npy")
        data = np.load(file_name[0], allow_pickle=True)
        
        if isinstance(data[0], np.ndarray):
            parent.zstack = data
            
        elif isinstance(data[0], dict):
            parent.zstack = np.array(data['meanImg'])
        if parent.zstack.ndim!=3:
            parent.update_status_bar('ERROR: zstack must be a 3D array with Z axis first')
            
    except Exception as e:
        parent.update_status_bar('ERROR: this is not a *.npy array :( ')


def load_proc(parent, name=None):
    if name is None:
        name = QFileDialog.getOpenFileName(
            parent, "Open processed file", filter="*.npy"
            )
    try:
        proc = np.load(name, allow_pickle=True).item()
        parent.proc = proc
        foldername = os.path.split(name)[0]
        parent.save_path = foldername
        filename = os.path.split(parent.proc['filename'])[-1]

        # check if file exists in original location or in current folder
        if os.path.exists(parent.proc['filename']):
            X    = np.load(parent.proc['filename'])
            parent.fname = parent.proc['filename']
        elif os.path.exists(os.path.join(foldername, filename)):
            X = np.load(os.path.join(foldername, filename))
            parent.fname = os.path.join(foldername, filename)
        else:
            print(f"ERROR: {parent.proc['filename']} not found")
            return 
        
        isort = parent.proc['isort']
        y     = parent.proc['embedding']
        u     = parent.proc['U']
        ops   = parent.proc['ops']
        user_clusters = parent.proc.get('user_clusters', None)
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

        if user_clusters is not None:
            parent.smooth_bin = user_clusters[0]['binsize']
            parent.smooth.setText(str(int(parent.smooth_bin)))
            
        parent.embedding = y
        parent.sorting = isort
        parent.U = u
        parent.ops = ops
        parent.user_clusters = user_clusters

        parent.update_status_bar(f"loaded:  {parent.proc['filename']}")
        load_iscell_stat(parent)
        load_sp(parent) 

        if user_clusters is not None:
            for uc in user_clusters:
                parent.selected = uc['slice']
                parent.add_cluster()
        parent.user_clusters = None # remove after loading

        parent.show()
    
def save_proc(parent): # Save embedding output
    try:
        if parent.embedding is not None:
            if parent.save_path is None:
                folderName = QFileDialog.getExistingDirectory(parent,
                                    "Choose save folder")
                parent.save_path = folderName  
            if parent.save_path:
                filename = parent.fname.split("/")[-1]
                filename, ext = os.path.splitext(filename)
                savename = os.path.join(parent.save_path, ("%s_embedding.npy"%filename))
                # save user clusters
                if len(parent.cluster_slices) > 0:
                    user_clusters = []
                    for roi_id, cs in enumerate(parent.cluster_slices):
                        user_clusters.append(
                            {'ids'      : parent.neurons_selected(cs),
                            'slice'    : cs, 
                            'binsize'  : parent.smooth_bin,
                            'color'    : parent.colors[roi_id]}
                        )
                else:
                    user_clusters = None
                # Rastermap embedding parameters
                ops = parent.ops
                proc = {'filename'      : parent.fname, 
                        'save_path'     : parent.save_path,
                        'isort'         : parent.sorting, 
                        'embedding'     : parent.embedding,
                        'U'             : parent.U, 
                        'user_clusters' : user_clusters,
                        'ops'           : ops}
                
                np.save(savename, proc, allow_pickle=True)
                parent.update_status_bar(f"processed file saved: {savename}")
        else:
            raise Exception("Please run embedding to save output")
    except Exception as e:
        print(e)
        #parent.update_status_bar(e)
        return
