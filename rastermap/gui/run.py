"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
import os, sys
from qtpy import QtGui, QtCore
from qtpy.QtWidgets import QMainWindow, QApplication, QSizePolicy, QDialog, QWidget, QScrollBar, QSlider, QComboBox, QGridLayout, QPushButton, QFrame, QCheckBox, QLabel, QProgressBar, QLineEdit, QMessageBox, QGroupBox, QButtonGroup, QRadioButton, QStatusBar, QTextEdit
from . import io


### custom QDialog which allows user to fill in ops and run rastermap
class RunWindow(QDialog):

    def __init__(self, parent=None):
        super(RunWindow, self).__init__(parent)
        self.setGeometry(50, 50, 600, 600)
        self.setWindowTitle("Choose rastermap run options")
        self.win = QWidget(self)
        self.layout = QGridLayout()
        self.layout.setHorizontalSpacing(25)
        self.win.setLayout(self.layout)

        print(
            ">>> importing rastermap functions (will be slow if you haven't run rastermap before) <<<"
        )
        from rastermap import default_settings, settings_info, Rastermap
        # default ops
        self.ops = default_settings()
        info = settings_info()
        keys = [
            "n_clusters", "n_PCs", "time_lag_window", "locality", "grid_upsample",
            "time_bin", "n_splits"
        ]
        tooltips = [info[key] for key in keys]
        bigfont = QtGui.QFont("Arial", 10, QtGui.QFont.Bold)
        l = 0
        self.keylist = []
        self.editlist = []
        k = 0
        for key in keys:
            qedit = LineEdit(k, key, self)
            qlabel = QLabel(key)
            qlabel.setToolTip(tooltips[k])
            qedit.set_text(self.ops)
            qedit.setFixedWidth(90)
            self.layout.addWidget(qlabel, k, 0, 1, 1)
            self.layout.addWidget(qedit, k, 1, 1, 1)
            self.keylist.append(key)
            self.editlist.append(qedit)
            k += 1

        #for j in range(10):
        #    self.layout.addWidget(QLabel("."),19,4+j,1,1)

        self.layout.setColumnStretch(4, 10)
        self.runButton = QPushButton("RUN")
        self.runButton.clicked.connect(lambda: self.run_RMAP(parent))
        self.layout.addWidget(self.runButton, 19, 0, 1, 1)
        #self.runButton.setEnabled(False)
        self.textEdit = QTextEdit()
        self.textEdit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(self.textEdit, 20, 0, 30, 14)
        self.process = QtCore.QProcess(self)
        self.process.readyReadStandardOutput.connect(self.stdout_write)
        self.process.readyReadStandardError.connect(self.stderr_write)
        # disable the button when running the rastermap process
        self.process.started.connect(self.started)
        self.process.finished.connect(lambda: self.finished(parent))
        self.process.errorOccurred.connect(self.errored)
        # stop process
        self.stopButton = QPushButton("STOP")
        self.stopButton.setEnabled(False)
        self.layout.addWidget(self.stopButton, 19, 1, 1, 1)
        self.stopButton.clicked.connect(self.stop)

        self.show()

    def run_RMAP(self, parent):
        self.finish = True
        self.error = False
        self.save_text()
        ops_path = os.path.join(os.getcwd(), "rmap_ops.npy")
        np.save(ops_path, self.ops)
        print("Running rastermap with command:")
        cmd = f"-u -W ignore -m rastermap --ops {ops_path} --S {parent.fname}"
        if parent.file_iscell is not None:
            cmd += f"--iscell {parent.file_iscell}"
        print("python " + cmd)
        self.process.start(sys.executable, cmd.split(" "))

    def stop(self):
        self.finish = False
        self.process.kill()

    def errored(self, error):
        print("ERROR")
        process = self.process
        print("error: ", error, "-", " ".join([process.program()] + process.arguments()))

    def started(self):
        self.runButton.setEnabled(False)
        self.stopButton.setEnabled(True)

    def finished(self, parent):
        self.runButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        if self.finish and not self.error:
            cursor = self.textEdit.textCursor()
            cursor.movePosition(cursor.End)
            cursor.insertText("Opening in GUI (can close this window)\n")
            basename, fname = os.path.split(parent.fname)
            fname = os.path.splitext(fname)[0]
            if os.path.isfile(os.path.join(basename, f"{fname}_embedding.npy")):
                parent.fname = os.path.join(basename, f"{fname}_embedding.npy")
            else:
                parent.fname = f"{fname}_embedding.npy"
            io.load_proc(parent, name=parent.fname)
        elif not self.error:
            cursor = self.textEdit.textCursor()
            cursor.movePosition(cursor.End)
            cursor.insertText("Interrupted by user (not finished)\n")
        else:
            cursor = self.textEdit.textCursor()
            cursor.movePosition(cursor.End)
            cursor.insertText("Interrupted by error (not finished)\n")

    def save_text(self):
        for k in range(len(self.editlist)):
            key = self.keylist[k]
            self.ops[key] = self.editlist[k].get_text(self.ops[key])

    def stdout_write(self):
        cursor = self.textEdit.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(str(self.process.readAllStandardOutput(), "utf-8"))
        self.textEdit.ensureCursorVisible()

    def stderr_write(self):
        cursor = self.textEdit.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(">>>ERROR<<<\n")
        cursor.insertText(str(self.process.readAllStandardError(), "utf-8"))
        self.textEdit.ensureCursorVisible()
        self.error = True


class LineEdit(QLineEdit):

    def __init__(self, k, key, parent=None):
        super(LineEdit, self).__init__(parent)
        self.key = key
        #self.textEdited.connect(lambda: self.edit_changed(parent.ops, k))

    def get_text(self, okey):
        key = self.key
        if key == "diameter" or key == "block_size":
            diams = self.text().replace(" ", "").split(",")
            if len(diams) > 1:
                okey = [int(diams[0]), int(diams[1])]
            else:
                okey = int(diams[0])
        else:
            if type(okey) is float:
                okey = float(self.text())
            elif type(okey) is str:
                okey = self.text()
            elif type(okey) is int or bool:
                okey = int(self.text())

        return okey

    def set_text(self, ops):
        key = self.key
        if key == "diameter" or key == "block_size":
            if (type(ops[key]) is not int) and (len(ops[key]) > 1):
                dstr = str(int(ops[key][0])) + ", " + str(int(ops[key][1]))
            else:
                dstr = str(int(ops[key]))
        else:
            if type(ops[key]) is bool:
                dstr = str(int(ops[key]))
            elif type(ops[key]) is str:
                dstr = ops[key]
            else:
                dstr = str(ops[key])
        self.setText(dstr)
