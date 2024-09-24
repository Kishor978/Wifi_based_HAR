# visualization server that listens to any incoming packets, plot them and store
import argparse 
import csv
import os
import io
from copy import deepcopy
import cv2
import numpy as np
from PyQt6 import QtNetwork,QtWidgets,uic
from pyqtgraph.Qt import QtCore

from csi_data_extraction import calc_phase_angle,csi_phase_calibration, Csi_amplitude_calibration,unpack_csi_struct

class UI(QtWidgets.QWidget):
    def __init__(self,app:QtWidgets.QApplication,parent:QtWidgets.QWidget=None,is_5ghz:bool=True):
        super(UI,self).__init__(parent)
        self.app=app
        self.is_5ghz=is_5ghz
        # load ui
        uic.load_ui('window.ui',self)
        self.setWindowTitle(f"{'5GHz' if is_5ghz else '2.4GHz'} CSI Data Visualization.")
        self.antenna_pairs,self.carrier,self.amplitude,self.phase=[],[],[],[]
        amp=self.box_amp
        amp.setBackground('w')
        amp.setWindowTitle('Amplitude')
        amp.setLabel('bottom','carrier',units='')
        amp.setLabel('left','Amplitude',units='')
        amp.setYRange(0,1,padding=0)
        amp.setXRange(0,112 if is_5ghz else 57,padding=0)
        self.penAmp0_0 = amp.plot(pen={'color': (200, 0, 0), 'width': 3})
        self.penAmp0_1 = amp.plot(pen={'color': (200, 200, 0), 'width': 3})
        self.penAmp1_0 = amp.plot(pen={'color': (0, 0, 200), 'width': 3})
        self.penAmp1_1 = amp.plot(pen={'color': (0, 200, 200), 'width': 3})

        phase = self.box_phase
        phase.setBackground('w')
        phase.setWindowTitle('Phase')
        phase.setLabel('bottom', 'Carrier', units='')
        phase.setLabel('left', 'Phase', units='')
        phase.setYRange(-np.pi, np.pi, padding=0)
        phase.setXRange(0, 114 if is_5ghz else 57, padding=0)
        self.penPhase0_0 = phase.plot(pen={'color': (200, 0, 0), 'width': 3})
        self.penPhase0_1 = phase.plot(pen={'color': (200, 200, 0), 'width': 3})
        self.penPhase1_0 = phase.plot(pen={'color': (0, 0, 200), 'width': 3})
        self.penPhase1_1 = phase.plot(pen={'color': (0, 200, 200), 'width': 3})

        self.amp = amp
        self.box_phase = phase
        
    @QtCore.pyqtSlot()
    def update_plots(self):
        if len(self.amplitude) and len(self.amplitude[0]):
            if len(self.antenna_pairs) > 0:
                self.penAmp0_0.setData(self.carrier, self.amplitude[0])
                self.penPhase0_0.setData(self.carrier, self.phase[0])

            if len(self.antenna_pairs) > 1:
                self.penAmp0_1.setData(self.carrier, self.amplitude[1])
                self.penPhase0_1.setData(self.carrier, self.phase[1])

            if len(self.antenna_pairs) > 2:
                self.penAmp1_0.setData(self.carrier, self.amplitude[2])
                self.penPhase1_0.setData(self.carrier, self.phase[2])

            if len(self.antenna_pairs) > 3:
                self.penAmp1_1.setData(self.carrier, self.amplitude[3])
                self.penPhase1_1.setData(self.carrier, self.phase[3])

        self.process_events()  # force complete redraw for every plot
    def process_events(self):
        self.app.processEvents()
        
