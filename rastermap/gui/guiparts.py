from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QScrollBar, QSlider, QComboBox, QGridLayout, QPushButton, QFrame, QCheckBox, QLabel, QProgressBar, QLineEdit, QMessageBox, QGroupBox, QStyle, QStyleOptionSlider
import pyqtgraph as pg
from pyqtgraph import functions as fn
from pyqtgraph import Point
import numpy as np
from pyqtgraph import ItemSample

# custom vertical label
class VerticalLabel(QWidget):
    def __init__(self, text=None):
        super(self.__class__, self).__init__()
        self.text = text

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setPen(QtCore.Qt.white)
        painter.translate(0, 0)
        painter.rotate(90)
        if self.text:
            painter.drawText(0, 0, self.text)
        painter.end()

class TimeRangeSlider(QWidget):
    def __init__(self, parent=None):
        super(TimeRangeSlider, self).__init__(parent)
        self.init_custom(parent)

    def init_custom(self, parent):
        
        self.slider = RangeSlider(parent) 

        slider_vbox = QtGui.QVBoxLayout()
        slider_hbox = QtGui.QHBoxLayout()
        slider_hbox.setContentsMargins(0, 0, 0, 0)
        slider_vbox.setContentsMargins(0, 0, 0, 0)
        slider_vbox.setSpacing(0)

        label_minimum = QLabel(alignment=QtCore.Qt.AlignLeft)
        self.slider.minimumChanged.connect(label_minimum.setNum)

        label_maximum = QLabel(alignment=QtCore.Qt.AlignRight)
        self.slider.maximumChanged.connect(label_maximum.setNum)

        slider_vbox.addWidget(self.slider)
        slider_vbox.addLayout(slider_hbox)
        slider_hbox.addWidget(label_minimum, QtCore.Qt.AlignLeft)
        slider_hbox.addWidget(label_maximum, QtCore.Qt.AlignRight)
        slider_vbox.addStretch()

        min_time, max_time = 0, parent.sp.shape[1]
        self.slider.setMinimum(min_time)
        self.slider.setMaximum(max_time)
        self.slider.setLow(parent.xrange[0])
        self.slider.setHigh(parent.xrange[-1])
        self.slider.setTickInterval((self.slider.maximum()-self.slider.minimum())//100)

        vbox = QtGui.QVBoxLayout(self)
        vbox.addLayout(slider_vbox)
        self.setGeometry(300, 300, 300, 150)
        self.show()
    
    def get_slider_values(self):
        return self.slider._low, self.slider._high
        
class RangeSlider(QSlider):
    """ A slider for ranges.

        This class provides a dual-slider for ranges, where there is a defined
        maximum and minimum, as is a normal slider, but instead of having a
        single slider value, there are 2 slider values.

        This class emits the same signals as the QSlider base class, with the
        exception of valueChanged

        Found this slider here: https://www.mail-archive.com/pyqt@riverbankcomputing.com/msg22889.html
        and modified it
    """
    minimumChanged = QtCore.pyqtSignal(int)
    maximumChanged = QtCore.pyqtSignal(int)

    def __init__(self, parent=None, *args):
        super(RangeSlider, self).__init__(*args)

        self._low = self.minimum()
        self._high = self.maximum()
        self.setOrientation(QtCore.Qt.Horizontal)

        self.pressed_control = QStyle.SC_None
        self.hover_control = QStyle.SC_None
        self.click_offset = 0

        self.setTickPosition(QSlider.TicksRight)
        self.setStyleSheet(\
                "QSlider::handle:horizontal {\
                background-color: white;\
                border: 1px solid #5c5c5c;\
                border-radius: 0px;\
                border-color: black;\
                height: 8px;\
                width: 6px;\
                margin: -8px 2; \
                }")
        # 0 for the low, 1 for the high, -1 for both
        self.active_slider = 0
        self.parent = parent

    def low(self):
        return self._low

    def setLow(self, low):
        self._low = low
        self.update()

    def high(self):
        return self._high

    def setHigh(self, high):
        self._high = high
        self.update()

    def level_change(self):
        self.maximumChanged.emit(self._high)
        self.minimumChanged.emit(self._low)
        return self._low, self._high
        
    def paintEvent(self, event):
        # based on http://qt.gitorious.org/qt/qt/blobs/master/src/gui/widgets/qslider.cpp
        painter = QtGui.QPainter(self)
        style = QApplication.style()
        for i, value in enumerate([self._low, self._high]):
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            # Only draw the groove for the first slider so it doesn't get drawn
            # on top of the existing ones every time
            if i == 0:
                opt.subControls = QStyle.SC_SliderHandle#QStyle.SC_SliderGroove | QStyle.SC_SliderHandle
            else:
                opt.subControls = QStyle.SC_SliderHandle
            if self.tickPosition() != self.NoTicks:
                opt.subControls |= QStyle.SC_SliderTickmarks
            if self.pressed_control:
                opt.activeSubControls = self.pressed_control
                opt.state |= QStyle.State_Sunken
            else:
                opt.activeSubControls = self.hover_control
            opt.sliderPosition = value
            opt.sliderValue = value
            style.drawComplexControl(QStyle.CC_Slider, opt, painter, self)

    def mousePressEvent(self, event):
        event.accept()
        style = QtGui.QApplication.style()
        button = event.button()
        if button:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            self.active_slider = -1
            for i, value in enumerate([self._low, self._high]):
                opt.sliderPosition = value
                hit = style.hitTestComplexControl(style.CC_Slider, opt, event.pos(), self)
                if hit == style.SC_SliderHandle:
                    self.active_slider = i
                    self.pressed_control = hit
                    self.triggerAction(self.SliderMove)
                    self.setRepeatAction(self.SliderNoAction)
                    self.setSliderDown(True)
                    break
            if self.active_slider < 0:
                self.pressed_control = QStyle.SC_SliderHandle
                self.click_offset = self.__pixelPosToRangeValue(self.__pick(event.pos()))
                self.triggerAction(self.SliderMove)
                self.setRepeatAction(self.SliderNoAction)
        else:
            event.ignore()
    
    def mouseMoveEvent(self, event):
        if self.pressed_control != QStyle.SC_SliderHandle:
            event.ignore()
            return
        event.accept()
        new_pos = self.__pixelPosToRangeValue(self.__pick(event.pos()))
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        if self.active_slider < 0:
            offset = new_pos - self.click_offset
            self._high += offset
            self._low += offset
            if self._low < self.minimum():
                diff = self.minimum() - self._low
                self._low += diff
                self._high += diff
            if self._high > self.maximum():
                diff = self.maximum() - self._high
                self._low += diff
                self._high += diff
        elif self.active_slider == 0:
            if new_pos >= self._high:
                new_pos = self._high - 1
            self._low = new_pos
        else:
            if new_pos <= self._low:
                new_pos = self._low + 1
            self._high = new_pos
        self.click_offset = new_pos
        self.update()
    
    def mouseReleaseEvent(self, event):
        self.level_change()
    
    def __pick(self, pt):
        if self.orientation() == QtCore.Qt.Horizontal:
            return pt.x()
        else:
            return pt.y()
    
    def __pixelPosToRangeValue(self, pos):
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        style = QtGui.QApplication.style()

        gr = style.subControlRect(style.CC_Slider, opt, style.SC_SliderGroove, self)
        sr = style.subControlRect(style.CC_Slider, opt, style.SC_SliderHandle, self)

        if self.orientation() == QtCore.Qt.Horizontal:
            slider_length = sr.width()
            slider_min = gr.x()
            slider_max = gr.right() - slider_length + 1
        else:
            slider_length = sr.height()
            slider_min = gr.y()
            slider_max = gr.bottom() - slider_length + 1

        return style.sliderValueFromPosition(self.minimum(), self.maximum(),
                                             pos-slider_min, slider_max-slider_min,
                                             opt.upsideDown)

class SatSlider(RangeSlider):
    def __init__(self, parent=None):
        super(SatSlider, self).__init__(parent)
        #super(self.__class__, self).__init__()
        self.parent = parent
        self.setMinimum(0)
        self.setMaximum(100)
        self.setOrientation(QtCore.Qt.Horizontal)
        #self.setLow(30)
        #self.setHigh(70)
        self.setTickInterval(10)
        self.valueChanged.connect(lambda: self.level_change(parent))
        self.setTracking(False)

    def level_change(self):
        self.parent.sat[0] = float(self._low)/100
        self.parent.sat[1] = float(self._high)/100
        self.parent.img.setLevels([self.parent.sat[0],self.parent.sat[1]])
        self.parent.imgROI.setLevels([self.parent.sat[0],self.parent.sat[1]])
        self.parent.win.show()

class NeuronSlider(RangeSlider):
    def __init__(self, parent=None):
        super(SatSlider, self).__init__(parent)
        self.parent = parent
        self.setMinimum(0)
        self.setMaximum(100)
        self.setLow(30)
        self.setHigh(70)

    def level_change(self):
        self.parent.sat[0] = float(self._low)/100
        self.parent.sat[1] = float(self._high)/100
        self.parent.img.setLevels([self.parent.sat[0],self.parent.sat[1]])
        self.parent.imgROI.setLevels([self.parent.sat[0],self.parent.sat[1]])
        self.parent.win.show()

class Slider(QSlider):
    def __init__(self, bid, parent=None):
        super(self.__class__, self).__init__()
        self.bid = bid
        self.setMinimum(0)
        self.setMaximum(100)
        self.setValue(parent.sat[bid]*100)
        self.setTickPosition(QSlider)
        self.setTickInterval(10)
        self.valueChanged.connect(lambda: self.level_change(parent,bid))
        self.setTracking(False)

    def level_change(self, parent, bid):
        parent.sat[bid] = float(self.value())/100
        parent.img.setLevels([parent.sat[0],parent.sat[1]])
        parent.imgROI.setLevels([parent.sat[0],parent.sat[1]])
        parent.win.show()

