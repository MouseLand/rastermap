import os, glob
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from scipy.stats import zscore

def load_mat(parent, name=None):
    try:
        if name is None:
            name = QtGui.QFileDialog.getOpenFileName(
                parent, "Open *.npy", filter="*.npy")
            parent.fname = name[0]
            parent.filebase = name[0]
        else:
            parent.fname = name
            parent.filebase = name
        print("Loading", parent.fname)
        X = np.load(parent.fname)
        print("Data loaded:", X.shape)
    except Exception as e:
                print('ERROR: this is not a *.npy array :( ')
                X = None
    if X is not None and X.ndim > 1:
        iscell, file_iscell = parent.load_iscell()
        parent.file_iscell = None
        if iscell is not None:
            if iscell.size == X.shape[0]:
                X = X[iscell, :]
                parent.file_iscell = file_iscell
                print('using iscell.npy in folder')
        if len(X.shape) > 2:
            X = X.mean(axis=-1)
        parent.p0.clear()
        parent.sp = zscore(X, axis=1)
        del X
        parent.sp = np.maximum(-4, np.minimum(8, parent.sp)) + 4
        parent.sp /= 12
        parent.embedding = np.arange(0, parent.sp.shape[0]).astype(np.int64)[:,np.newaxis]
        parent.sorting = np.arange(0, parent.sp.shape[0]).astype(np.int64)
        
        parent.loaded = True
        parent.plot_activity()
        parent.show()
        parent.run_embedding_button.setEnabled(True)
        parent.upload_behav_button.setEnabled(True)
        parent.upload_run_button.setEnabled(True)

def get_rastermap_params(parent):
    if parent.custom_param_radiobutton.isChecked():
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Set rastermap parameters")
        dialog.verticalLayout = QtWidgets.QVBoxLayout(dialog)

        # Param options
        dialog.n_clusters_label = QtWidgets.QLabel(dialog)
        dialog.n_clusters_label.setTextFormat(QtCore.Qt.RichText)
        dialog.n_clusters_label.setText("n_clusters:")
        dialog.n_clusters = QtGui.QLineEdit()
        dialog.n_clusters.setText(str(50))

        dialog.n_neurons_label = QtWidgets.QLabel(dialog)
        dialog.n_neurons_label.setTextFormat(QtCore.Qt.RichText)
        dialog.n_neurons_label.setText("n_neurons:")
        dialog.n_neurons = QtGui.QLineEdit()
        dialog.n_neurons.setText(str(parent.sp.shape[0]))

        dialog.grid_upsample_label = QtWidgets.QLabel(dialog)
        dialog.grid_upsample_label.setTextFormat(QtCore.Qt.RichText)
        dialog.grid_upsample_label.setText("grid_upsample:")
        dialog.grid_upsample = QtGui.QLineEdit()
        dialog.grid_upsample.setText(str(10))

        dialog.n_splits_label = QtWidgets.QLabel(dialog)
        dialog.n_splits_label.setTextFormat(QtCore.Qt.RichText)
        dialog.n_splits_label.setText("n_splits:")
        dialog.n_splits = QtGui.QLineEdit()
        dialog.n_splits.setText(str(4))

        dialog.ok_button = QtGui.QPushButton('Ok')
        dialog.ok_button.setDefault(True)
        dialog.ok_button.clicked.connect(lambda: custom_set_params(parent, dialog))
        dialog.cancel_button = QtGui.QPushButton('Cancel')
        dialog.cancel_button.clicked.connect(dialog.close)

        # Set layout of options
        dialog.widget = QtWidgets.QWidget(dialog)
        dialog.horizontalLayout = QtWidgets.QHBoxLayout(dialog.widget)
        dialog.horizontalLayout.setContentsMargins(-1, -1, -1, 0)
        dialog.horizontalLayout.setObjectName("horizontalLayout")
        dialog.horizontalLayout.addWidget(dialog.n_clusters_label)
        dialog.horizontalLayout.addWidget(dialog.n_clusters)

        dialog.widget2 = QtWidgets.QWidget(dialog)
        dialog.horizontalLayout = QtWidgets.QHBoxLayout(dialog.widget2)
        dialog.horizontalLayout.setContentsMargins(-1, -1, -1, 0)
        dialog.horizontalLayout.setObjectName("horizontalLayout")
        dialog.horizontalLayout.addWidget(dialog.n_neurons_label)
        dialog.horizontalLayout.addWidget(dialog.n_neurons)

        dialog.widget3 = QtWidgets.QWidget(dialog)
        dialog.horizontalLayout = QtWidgets.QHBoxLayout(dialog.widget3)
        dialog.horizontalLayout.setContentsMargins(-1, -1, -1, 0)
        dialog.horizontalLayout.setObjectName("horizontalLayout")
        dialog.horizontalLayout.addWidget(dialog.grid_upsample_label)
        dialog.horizontalLayout.addWidget(dialog.grid_upsample)

        dialog.widget4 = QtWidgets.QWidget(dialog)
        dialog.horizontalLayout = QtWidgets.QHBoxLayout(dialog.widget4)
        dialog.horizontalLayout.setContentsMargins(-1, -1, -1, 0)
        dialog.horizontalLayout.setObjectName("horizontalLayout")
        dialog.horizontalLayout.addWidget(dialog.n_splits_label)
        dialog.horizontalLayout.addWidget(dialog.n_splits)

        dialog.widget5 = QtWidgets.QWidget(dialog)
        dialog.horizontalLayout = QtWidgets.QHBoxLayout(dialog.widget5)
        dialog.horizontalLayout.addWidget(dialog.cancel_button)
        dialog.horizontalLayout.addWidget(dialog.ok_button)

        # Add options to dialog box
        dialog.verticalLayout.addWidget(dialog.widget)
        dialog.verticalLayout.addWidget(dialog.widget2)
        dialog.verticalLayout.addWidget(dialog.widget3)
        dialog.verticalLayout.addWidget(dialog.widget4)
        dialog.verticalLayout.addWidget(dialog.widget5)

        dialog.adjustSize()
        dialog.exec_()

def set_rastermap_params(parent):
    if parent.default_param_radiobutton.isChecked():
        print("Using default params")
        parent.n_clusters = 50
        parent.n_neurons = parent.sp.shape[0]
        if parent.n_neurons > 1000:
            parent.n_splits = min(4, parent.n_neurons//1000)
        else:
            parent.n_splits = 4
        parent.grid_upsample = min(10, parent.n_neurons // (parent.n_splits * (parent.n_clusters+1)))
    
def custom_set_params(parent, dialogBox):
    try:
        parent.n_clusters = int(dialogBox.n_clusters.text())
        parent.n_neurons = int(dialogBox.n_neurons.text())
        parent.grid_upsample = int(dialogBox.grid_upsample.text())
        parent.n_splits = int(dialogBox.n_splits.text())
        print("Using custom rastermap params")
    except Exception as e:
        QtGui.QMessageBox.about(parent, 'Error','Invalid input entered')
        print(e)
        pass
    dialogBox.close()

def load_behav_data(parent):
    name = QtGui.QFileDialog.getOpenFileName(
        parent, "Open *.npy", filter="*.npy"
    )
    name = name[0]
    parent.behav_loaded = False
    try:
        beh = np.load(name)
        if beh.ndim == 2 and beh.shape[1] == parent.sp.shape[1]:
            parent.behav_loaded = True
            print("Behav file loaded")
    except (ValueError, KeyError, OSError,
            RuntimeError, TypeError, NameError):
        print("ERROR: this is not a 2D array with length of data")
    if parent.behav_loaded:
        parent.behav_data = beh
        parent.plot_behav_data()
        parent.heatmap_checkBox.setChecked(True)
    else:
        return

def load_run_data(parent):
    name = QtGui.QFileDialog.getOpenFileName(
        parent, "Open *.npy", filter="*.npy"
    )
    name = name[0]
    parent.run_loaded = False
    try:
        run = np.load(name)
        run = run.flatten()
        if run.size == parent.sp.shape[1]:
            parent.run_loaded = True
    except (ValueError, KeyError, OSError,
            RuntimeError, TypeError, NameError):
        print("ERROR: this is not a 1D array with length of data")
    if parent.run_loaded:
        parent.run_data = run
        parent.plot_run_trace()
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
    dialog.depth_button = QtGui.QPushButton('Upload')
    dialog.depth_button.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    dialog.depth_button.clicked.connect(lambda: load_neuron_pos(parent, dialog.depth_button, depth=True))

    dialog.ok_button = QtGui.QPushButton('Done')
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

def get_neuron_pos_data(parent):
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Upload files")
    dialog.verticalLayout = QtWidgets.QVBoxLayout(dialog)

    # Param options
    dialog.xpos_label = QtWidgets.QLabel(dialog)
    dialog.xpos_label.setTextFormat(QtCore.Qt.RichText)
    dialog.xpos_label.setText("x position:")
    dialog.xpos_button = QtGui.QPushButton('Upload')
    dialog.xpos_button.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    dialog.xpos_button.clicked.connect(lambda: load_neuron_pos(parent, dialog.xpos_button, xpos=True))

    dialog.ypos_label = QtWidgets.QLabel(dialog)
    dialog.ypos_label.setTextFormat(QtCore.Qt.RichText)
    dialog.ypos_label.setText("y position:")
    dialog.ypos_button = QtGui.QPushButton('Upload')
    dialog.ypos_button.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    dialog.ypos_button.clicked.connect(lambda: load_neuron_pos(parent, dialog.ypos_button, ypos=True))

    dialog.ok_button = QtGui.QPushButton('Done')
    dialog.ok_button.setDefault(True)
    dialog.ok_button.clicked.connect(dialog.close)

    dialog.widget = QtWidgets.QWidget(dialog)
    dialog.horizontalLayout = QtWidgets.QHBoxLayout(dialog.widget)
    dialog.horizontalLayout.setContentsMargins(-1, -1, -1, 0)
    dialog.horizontalLayout.setObjectName("horizontalLayout")
    dialog.horizontalLayout.addWidget(dialog.xpos_label)
    dialog.horizontalLayout.addWidget(dialog.xpos_button)

    dialog.widget2 = QtWidgets.QWidget(dialog)
    dialog.horizontalLayout = QtWidgets.QHBoxLayout(dialog.widget2)
    dialog.horizontalLayout.setContentsMargins(-1, -1, -1, 0)
    dialog.horizontalLayout.setObjectName("horizontalLayout")
    dialog.horizontalLayout.addWidget(dialog.ypos_label)
    dialog.horizontalLayout.addWidget(dialog.ypos_button)

    # Add options to dialog box
    dialog.verticalLayout.addWidget(dialog.widget)
    dialog.verticalLayout.addWidget(dialog.widget2)
    dialog.verticalLayout.addWidget(dialog.ok_button)

    dialog.adjustSize()
    dialog.exec_()

def load_neuron_pos(parent, button, xpos=False, ypos=False, depth=False):
    try:
        file_name = QtGui.QFileDialog.getOpenFileName(
                    parent, "Open *.npy", filter="*.npy")
        data = np.load(file_name[0])
        if xpos and data.size == parent.sp.shape[0]:
            parent.xpos_dat = data
            button.setText("Uploaded!")
            print("xpos data loaded")
        elif ypos and data.size == parent.sp.shape[0]: 
            parent.ypos_dat = data
            button.setText("Uploaded!")
            print("ypos data loaded")
        elif depth and data.size == parent.sp.shape[0]:
            parent.depth_dat = data
            button.setText("Uploaded!")
            print("depth data loaded")
        else:
            print("incorrect data uploaded")
            return
    except Exception as e:
        print('ERROR: this is not a *.npy array :( ')

def save_proc(parent): # Save embedding output
    try:
        if parent.embedded:
            if parent.save_path is None:
                folderName = QtGui.QFileDialog.getExistingDirectory(parent,
                                    "Choose save folder")
                parent.save_path = folderName
                    
            else:
                raise Exception("Incorrect folder. Please select a folder")
            if parent.save_path:
                filename = parent.fname.split("/")[-1]
                filename, ext = os.path.splitext(filename)
                savename = os.path.join(parent.save_path, ("%s_rastermap_proc.npy"%filename))

                proc = {'filenames': parent.fname, 'save_path': savepath,
                        'isort' : parent.sorting, 'embedding' : parent.embedding}
                
                np.save(savename, proc)
                print("File saved:", savename)
        else:
            raise Exception("Please run embedding to save output")
    except Exception as e:
        print(e)
        return

def load_proc(parent, name=None):
    if name is None:
        name = QtGui.QFileDialog.getOpenFileName(
            parent, "Open processed file", filter="*.npy"
            )
        parent.fname = name[0]
        name = parent.fname
    else:
        parent.fname = name
    try:
        proc = np.load(name, allow_pickle=True)
        proc = proc.item()
        parent.proc = proc
        # do not load X, use reconstruction
        #X    = np.load(parent.proc['filename'])
        parent.filebase = parent.proc['filename']
        y    = parent.proc['embedding']
        u    = parent.proc['uv'][0]
        v    = parent.proc['uv'][1]
        ops  = parent.proc['ops']
        X    = u @ v.T
    except (ValueError, KeyError, OSError,
            RuntimeError, TypeError, NameError):
        print('ERROR: this is not a *.npy file :( ')
        X = None
    if X is not None:
        parent.filebase = parent.proc['filename']
        iscell, file_iscell = parent.load_iscell()

        # check if training set used
        if 'train_time' in parent.proc:
            if parent.proc['train_time'].sum() < parent.proc['train_time'].size:
                # not all training pts used
                X    = np.load(parent.proc['filename'])
                # show only test timepoints
                X    = X[:,~parent.proc['train_time']]
                if iscell is not None:
                    if iscell.size == X.shape[0]:
                        X = X[iscell, :]
                        print('using iscell.npy in folder')
                if len(X.shape) > 2:
                    X = X.mean(axis=-1)
                v = (u.T @ X).T
                v /= ((v**2).sum(axis=0))**0.5
                X = u @ v.T
        parent.startROI = False
        parent.endROI = False
        parent.posROI = np.zeros((3,2))
        parent.prect = np.zeros((5,2))
        parent.ROIs = []
        parent.ROIorder = []
        parent.Rselected = []
        parent.Rcolors = []
        parent.p0.clear()
        parent.sp = X#zscore(X, axis=1)
        del X
        parent.sp += 1
        parent.sp /= 9
        parent.embedding = y
        if ops['n_components'] > 1:
            parent.embedded = True
            parent.enable_embedded()
        else:
            parent.p0.clear()
            parent.embedded=False
            parent.disable_embedded()

        parent.U   = u #@ np.diag(usv[1])
        ineur = 0
        parent.loaded = True
        parent.embedded = True
        parent.run_embedding_button.setEnabled(True)
        parent.upload_behav_button.setEnabled(True)
        parent.upload_run_button.setEnabled(True)
        parent.ROI_position()
        if parent.embedded:
            parent.plot_embedding()
            parent.xp = pg.ScatterPlotItem(pos=parent.embedding[ineur,:][np.newaxis,:],
                                        symbol='x', pen=pg.mkPen(color=(255,0,0,255), width=3),
                                        size=12)#brush=pg.mkBrush(color=(255,0,0,255)), size=14)
            parent.p0.addItem(parent.xp)
        parent.show()
