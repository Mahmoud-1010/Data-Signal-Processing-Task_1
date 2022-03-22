from PyQt5 import QtWidgets, QtCore, QtGui
from mainwindow import Ui_MainWindow
from newwindow import Ui_MainWindow2
from scipy import signal
from scipy.signal import butter, lfilter
import sys
import scipy
from scipy.io import wavfile
import os
from fpdf import FPDF
import shutil
import pyqtgraph.exporters
import numpy as np
from scipy import fftpack
from scipy.fftpack import fft
import sounddevice as sd
from pyqtgraph import PlotWidget, PlotItem
import pyqtgraph as pg
from collections import OrderedDict


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.c = 0

        self.ui.slider_1.valueChanged.connect(self.update)
        self.ui.slider_2.valueChanged.connect(self.update)
        self.ui.slider_3.valueChanged.connect(self.update)
        self.ui.slider_4.valueChanged.connect(self.update)
        self.ui.slider_5.valueChanged.connect(self.update)
        self.ui.slider_6.valueChanged.connect(self.update)
        self.ui.slider_7.valueChanged.connect(self.update)
        self.ui.slider_8.valueChanged.connect(self.update)
        self.ui.slider_9.valueChanged.connect(self.update)
        self.ui.slider_10.valueChanged.connect(self.update)
        self.gain = [self.ui.slider_1.value(), self.ui.slider_2.value(), self.ui.slider_3.value(), self.ui.slider_4.value(), self.ui.slider_5.value(
        ), self.ui.slider_6.value(), self.ui.slider_7.value(), self.ui.slider_8.value(), self.ui.slider_9.value(), self.ui.slider_10.value()]
        self.color1 = pg.mkPen(color=(255, 255, 0))

        self.ui.actionopen_file.triggered.connect(lambda: self.loadFile())

        self.ui.r1.toggled.connect(self.onclicked)
        self.ui.r2.toggled.connect(self.onclicked)
        self.ui.r3.toggled.connect(self.onclicked)
        self.ui.r4.toggled.connect(self.onclicked)

        self.ui.pause.clicked.connect(lambda: self.pause_fn())
        self.ui.play.clicked.connect(lambda: self.play_fn())
        self.ui.zoomx.clicked.connect(lambda: self.zoom_x())
        self.ui.zoomy.clicked.connect(lambda: self.zoom_y())
        self.ui.actionZoom_In.triggered.connect(lambda: self.zoom_i())
        self.ui.actionZoom_Out.triggered.connect(lambda: self.zoom_o())
        self.ui.actionZoom_X.triggered.connect(lambda: self.zoom_x())
        self.ui.actionZoom_Y.triggered.connect(lambda: self.zoom_y())
        self.ui.Export.clicked.connect(lambda: self.printPDF())
        self.ui.zoom_in.clicked.connect(lambda: self.zoom_i())
        self.ui.zoom_out.clicked.connect(lambda: self.zoom_o())
        self.ui.clear_Button.clicked.connect(lambda: self.clear())
        self.ui.ExitButton.clicked.connect(exit)
        self.ui.actionnew_window.triggered.connect(lambda: self.new_win())
        # self.ui.scroll_x.valueChanged.connect(self.scrollx)
        # self.ui.scroll_y.valueChanged.connect(self.scrolly)

    def loadFile(self):
        fname = QtGui.QFileDialog.getOpenFileName(
            self, 'choose the signal', os.getenv('HOME'), "wav(*.wav)")
        self.path = fname[0]
        if self.path == "":
            return
        self.fs, self.data = wavfile.read(self.path)
        samples_count = self.data.shape[0]
        self.time = np.arange(samples_count)/self.fs
        amplitude = self.data
        self.ui.Channel_1.plot(self.time, amplitude, pen=self.color1)
        print(amplitude)
        self.ui.Channel_1.plotItem.setXRange(min(self.time), max(self.time)/50)
        self.ui.Channel_1.plotItem.setYRange(min(amplitude), max(amplitude))
        self.generate_spectrogram(self.ui.Spectrogram_1, amplitude)

        # fourier
        self.DataFourier = np.fft.fft(self.data)
        self.phase = np.angle(self.DataFourier)
        self.freqs = np.fft.fftfreq(len(self.data), 1/self.fs)
        self.Data_amplitude = np.abs(self.DataFourier)
        sd.play(self.data, self.fs)

    def update(self):
        self.gain = [self.ui.slider_1.value(), self.ui.slider_2.value(), self.ui.slider_3.value(), self.ui.slider_4.value(), self.ui.slider_5.value(
        ), self.ui.slider_6.value(), self.ui.slider_7.value(), self.ui.slider_8.value(), self.ui.slider_9.value(), self.ui.slider_10.value()]

        bandlength = int(len(self.freqs)/20)
        print(bandlength)

        length_frequency = int(len(self.freqs)/2)
        print(length_frequency)
        self.Data_update = self.Data_amplitude.copy()
        for i in range(0, 10):

            self.Data_update[length_frequency + i*bandlength: length_frequency + (
                i+1) * bandlength] = self.Data_amplitude[length_frequency + i*bandlength: length_frequency + (i+1) * bandlength]*np.float64(self.gain[i])
            self.Data_update[length_frequency - (i+1)*bandlength: length_frequency - i*bandlength] = self.Data_amplitude[length_frequency - (
                i+1)*bandlength: length_frequency - i*bandlength]*np.float64(self.gain[i])

        # self.ui.Spectrogram_1.clear()
        # self.ui.Spectrogram_1.plot(self.freqs , self.Data_update ,pen=self.color1)
        datafourier_modified = np.multiply(
            self.Data_update, np.exp(1j*self.phase))
        self.data_modified = np.real(np.fft.ifft(datafourier_modified))

        self.ui.Channel_2.clear()
        self.ui.Channel_2.plot(self.time, self.data_modified, pen=self.color1)
        self.ui.Channel_2.plotItem.setXRange(min(self.time), max(self.time))
        self.ui.Channel_2.plotItem.setYRange(
            min(self.data_modified), max(self.data_modified))

        self.ui.Spectrogram_2.clear()
        self.generate_spectrogram(self.ui.Spectrogram_2, self.data_modified)

    # spectrogram function:
    def generate_spectrogram(self, widget, data):
        x = data
        fs = 16000
        f, t, Sxx = signal.spectrogram(x, fs)
        pg.setConfigOptions(imageAxisOrder='row-major')
        p1 = widget.addPlot()
        # Item for displaying image data
        img = pg.ImageItem()
        p1.addItem(img)
        # Add a histogram with which to control the gradient of the image
        hist = pg.HistogramLUTItem()
        # Link the histogram to the image
        hist.setImageItem(img)
        widget.addItem(hist)
        # Fit the min and max levels of the histogram to the data available
        hist.setLevels(np.min(Sxx), np.max(Sxx))

        hist.gradient.restoreState(self.Gradients[self.text])
        img.setImage(Sxx)
        # Scale the X and Y Axis to time and frequency (standard is pixels)
        img.scale(t[-1] / np.size(Sxx, axis=1), f[-1] / np.size(Sxx, axis=0))
        # Limit panning/zooming to the spectrogram
        p1.setLimits(xMin=0, xMax=t[-1], yMin=0, yMax=f[-1])
        # Add labels to the axis
        p1.setLabel('bottom', "Time", units='s')
        p1.setLabel('left', "Frequency", units='Hz')

    def colorpallet(self):
        self.Gradients = OrderedDict([
            ('thermal', {'ticks': [(0.3333, (185, 0, 0, 255)), (0.6666, (255, 220, 0, 255)), (
                1, (255, 255, 255, 255)), (0, (0, 0, 0, 255))], 'mode': 'rgb'}),
            ('flame', {'ticks': [(0.2, (7, 0, 220, 255)), (0.5, (236, 0, 134, 255)), (0.8, (
                246, 246, 0, 255)), (1.0, (255, 255, 255, 255)), (0.0, (0, 0, 0, 255))], 'mode': 'rgb'}),
            ('yellowy', {'ticks': [(0.0, (0, 0, 0, 255)), (0.2328863796753704, (32, 0, 129, 255)), (0.8362738179251941, (
                255, 255, 0, 255)), (0.5257586450247, (115, 15, 255, 255)), (1.0, (255, 255, 255, 255))], 'mode': 'rgb'}),
            ('bipolar', {'ticks': [(0.0, (0, 255, 255, 255)), (1.0, (255, 255, 0, 255)), (0.5, (
                0, 0, 0, 255)), (0.25, (0, 0, 255, 255)), (0.75, (255, 0, 0, 255))], 'mode': 'rgb'}),
            ('spectrum', {
             'ticks': [(1.0, (255, 0, 255, 255)), (0.0, (255, 0, 0, 255))], 'mode': 'hsv'}),
            ('cyclic', {'ticks': [(0.0, (255, 0, 4, 255)),
                                  (1.0, (255, 0, 0, 255))], 'mode': 'hsv'}),
            ('greyclip', {'ticks': [(0.0, (0, 0, 0, 255)), (0.99, (255,
                                                                   255, 255, 255)), (1.0, (255, 0, 0, 255))], 'mode': 'rgb'}),
            ('grey', {'ticks': [(0.0, (0, 0, 0, 255)),
                                (1.0, (255, 255, 255, 255))], 'mode': 'rgb'}),
            # Perceptually uniform sequential colormaps from Matplotlib 2.0
            ('viridis', {'ticks': [(0.0, (68, 1, 84, 255)), (0.25, (58, 82, 139, 255)), (0.5, (
                32, 144, 140, 255)), (0.75, (94, 201, 97, 255)), (1.0, (253, 231, 36, 255))], 'mode': 'rgb'}),
            ('inferno', {'ticks': [(0.0, (0, 0, 3, 255)), (0.25, (87, 15, 109, 255)), (0.5, (
                187, 55, 84, 255)), (0.75, (249, 142, 8, 255)), (1.0, (252, 254, 164, 255))], 'mode': 'rgb'}),
            ('plasma', {'ticks': [(0.0, (12, 7, 134, 255)), (0.25, (126, 3, 167, 255)), (0.5, (
                203, 71, 119, 255)), (0.75, (248, 149, 64, 255)), (1.0, (239, 248, 33, 255))], 'mode': 'rgb'}),
            ('magma', {'ticks': [(0.0, (0, 0, 3, 255)), (0.25, (80, 18, 123, 255)), (0.5, (
                182, 54, 121, 255)), (0.75, (251, 136, 97, 255)), (1.0, (251, 252, 191, 255))], 'mode': 'rgb'}),
        ])
        self.text = self.ui.color_pallets.currentText()

    def onclicked(self):
        r = self.sender()
        if r.isChecked():
            self.c = r.value

    def clear(self):
        if self.c == 1:
            self.ui.Channel_1.clear()
        elif self.c == 2:
            self.ui.Channel_2.clear()
        elif self.c == 3:
            self.ui.Spectrogram_1.clear()
        elif self.c == 4:
            self.ui.Spectrogram_2.clear()

    # pause function

    def pause_fn(self):
        if self.c == 1:
            sd.stop()
        elif self.c == 2:
            sd.stop()

    # play function :

    def play_fn(self):
        if self.c == 1:
            sd.play(self.data, self.fs)
        elif self.c == 2:
            sd.play(self.data_modified, self.fs)

    ##zoom in function

    def zoom_i(self):
        if self.c == 1:
            self.ui.Channel_1.plotItem.getViewBox().scaleBy((0.75, 0.75))

        elif self.c == 2:
            self.ui.Channel_2.plotItem.getViewBox().scaleBy((0.75, 0.75))

    # zoom out function
    def zoom_o(self):
        if self.c == 1:
            self.ui.Channel_1.plotItem.getViewBox().scaleBy((1.25, 1.25))

        elif self.c == 2:
            self.ui.Channel_2.plotItem.getViewBox().scaleBy((1.25, 1.25))

    # zoom x function
    def zoom_x(self):
        if self.c == 1:
            self.ui.Channel_1.plotItem.getViewBox().scaleBy((0.75, 1))

        elif self.c == 2:
            self.ui.Channel_2.plotItem.getViewBox().scaleBy((0.75, 1))

    # zoom y function

    def zoom_y(self):
        if self.c == 1:
            self.ui.Channel_1.plotItem.getViewBox().scaleBy((1, 0.75))

        elif self.c == 2:
            self.ui.Channel_2.plotItem.getViewBox().scaleBy((1, 0.75))

    def new_win(self):
        self.window = Ui_MainWindow2()
        self.window.show()
    # pdf function

    def printPDF(self):

        self.WIDTH = 210
        self.HEIGHT = 297
        pdf = FPDF()
        try:
            shutil.rmtree('plots')
            os.mkdir('plots')
        except FileNotFoundError:
            os.mkdir('plots')
        if self.time.any():
            exporter = pg.exporters.ImageExporter(self.ui.Channel_1.plotItem)
            exporter.parameters()['width'] = self.ui.Channel_1.plotItem.width()
            exporter.export('plots/plot-1.png')
            pdf.add_page()
            pdf.image('plots/plot-1.png', 15, 30, self.WIDTH - 30)
            exporter = pg.exporters.ImageExporter(
                self.ui.Spectrogram_1.scene())
            exporter.export('plots/spec-1.png')
            if self.ui.Channel_1.plotItem.height() > 600 or self.ui.Spectrogram_1.height() > 600:
                pdf.add_page()
                pdf.image('plots/spec-1.png', 15, 30, self.WIDTH - 30)
            else:
                pdf.image('plots/spec-1.png', 15,
                          self.WIDTH / 2 + 50, self.WIDTH - 30)

        if self.data_modified.any():
            exporter = pg.exporters.ImageExporter(self.ui.Channel_2.plotItem)
            exporter.parameters()['width'] = self.ui.Channel_2.plotItem.width()
            exporter.export('plots/plot-2.png')
            pdf.add_page()
            pdf.image('plots/plot-2.png', 15, 30, self.WIDTH - 30)

            exporter = pg.exporters.ImageExporter(
                self.ui.Spectrogram_2.scene())
            exporter.export('plots/spec-2.png')
            if self.ui.Channel_2.plotItem.height() > 600 or self.ui.Spectrogram_2.height() > 600:
                pdf.add_page()
                pdf.image('plots/spec-2.png', 15, 30, self.WIDTH - 30)
            else:
                pdf.image('plots/spec-2.png', 15,
                          self.WIDTH / 2 + 50, self.WIDTH - 30)

        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export PDF", None, "PDF files (.pdf)")
        if fn:
            if QtCore.QFileInfo(fn).suffix() == "":
                fn += ".pdf"
        # now =datetime.now()
        # now = f'{now:%Y-%m-%d %H-%M-%S.%f %p}'
        try:
            os.mkdir('pdf')
        except:
            pass
        # pdf.output(f'pdf/{fn}.pdf', 'F')
        pdf.output(f'{fn}', 'F')
        try:
            shutil.rmtree('plots')
        except:
            pass


def main():
    # to load the directory folder
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
