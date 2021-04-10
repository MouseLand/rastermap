import os, glob
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg

def load_mat(parent, name=None):
    if name is None:
        name = QtGui.QFileDialog.getOpenFileName(
            parent, "Open *.npy", filter="*.npy"
        )
        parent.fname = name[0]
        parent.filebase = name[0]
    else:
        parent.fname = name
        parent.filebase = name
    try:
        print("Loading", parent.fname)
        X = np.load(parent.fname)
        print(X.shape)
    except (ValueError, KeyError, OSError,
            RuntimeError, TypeError, NameError):
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
        parent.runRMAP.setEnabled(True)
        parent.upload_behav_button.setEnabled(True)
        parent.upload_run_button.setEnabled(True)
        print('done loading')

def get_params(parent):
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

def set_params(parent):
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
        beh = beh.flatten()
        if len(beh.shape) == 2:
            parent.behav_loaded = True
    except (ValueError, KeyError, OSError,
            RuntimeError, TypeError, NameError):
        print("ERROR: this is not a 2D array with length of data")
    if parent.behav_loaded:
        print("behav loaded")
    else:
        print("ERROR: this is not a 2D array with length of data")

def load_run_data(parent):
    name = QtGui.QFileDialog.getOpenFileName(
        parent, "Open *.npy", filter="*.npy"
    )
    name = name[0]
    parent.run_loaded = False
    try:
        beh = np.load(name)
        beh = beh.flatten()
        if beh.size == parent.sp.shape[1]:
            parent.run_loaded = True
    except (ValueError, KeyError, OSError,
            RuntimeError, TypeError, NameError):
        print("ERROR: this is not a 1D array with length of data")
    if parent.run_loaded:
        beh -= beh.min()
        beh /= beh.max()
        parent.beh = beh
        b = len(self.colors)
        parent.colorbtns.button(b).setEnabled(True)
        parent.colorbtns.button(b).setStyleSheet(self.styleUnpressed)
        fig.beh_masks(parent)
        fig.plot_trace(parent)
        parent.show()
    else:
        print("ERROR: this is not a 1D array with length of data")
