# -*- coding: utf-8 -*-
"""
One port RF device characterization app
"""
import os
import datetime
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QMessageBox, QButtonGroup
import pyqtgraph as pg
import skrf as rf
import numpy as np
from scipy import optimize
import visa


class OnePortExtended(rf.OnePort):
    def RunSlidingLoad(self, short_meas, short_ideal, offset_meas, offset_ideal, sliding_data):
        # error box
        SLe00 = np.zeros((len(sliding_data[0])), dtype=complex)
        SLe10 = np.zeros((len(sliding_data[0])), dtype=complex)
        SLe11 = np.zeros((len(sliding_data[0])), dtype=complex)

        for fpoint in range(len(sliding_data[0])):
            # Coordinates of the 2D points
            x = []
            y = []

            def f_leastsq(c):
                # calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc)
                Ri = np.sqrt((x-c[0])**2 + (y-c[1])**2)
                return Ri - Ri.mean()

            for sliding_position in sliding_data:   # diffrent slide positions
                x.append(sliding_position[fpoint].s[0, 0, 0].real)
                y.append(sliding_position[fpoint].s[0, 0, 0].imag)

            # coordinates of the barycenter
            x_m = np.mean(x)
            y_m = np.mean(y)
            center_estimate = x_m, y_m
            center_2, ier = optimize.leastsq(f_leastsq, center_estimate)
            xc_2, yc_2 = center_2
            
            # solving for e00
            SLe00[fpoint] = complex(xc_2, yc_2)

        # solving for e11, e10e01
        for fpoint in range(len(SLe00)):
            SLe10[fpoint] = ((offset_ideal.s[fpoint, 0, 0] - short_ideal.s[fpoint, 0, 0]) * \
                            (offset_meas.s[fpoint, 0, 0] - SLe00[fpoint]) * \
                            (short_meas.s[fpoint,0, 0] - SLe00[fpoint])) / \
                            (offset_ideal.s[fpoint, 0, 0] * short_ideal.s[fpoint, 0, 0] * \
                            (offset_meas.s[fpoint, 0, 0] - short_meas.s[fpoint, 0, 0]))
            
            SLe11[fpoint] = (short_ideal.s[fpoint, 0, 0] * (offset_meas.s[fpoint, 0, 0] - SLe00[fpoint]) - \
                            (offset_ideal.s[fpoint, 0, 0] * (short_meas.s[fpoint, 0, 0] - SLe00[fpoint]))) / \
                            (offset_ideal.s[fpoint, 0, 0] * short_ideal.s[fpoint, 0, 0] * \
                            (offset_meas.s[fpoint, 0, 0] - short_meas.s[fpoint, 0, 0]))
        
        # write calculated error box
        self._coefs = {
            'directivity': SLe00,
            'reflection tracking': SLe10,
            'source match': SLe11}

    def RunOffsetLoad(self,waveguide,short_meas,short_ideal,offset_meas,offset_ideal,load_meas,offset_load_meas,offset_length):
        # Offset Load data
        DataOL = [load_meas, offset_load_meas]
        # error box
        OLe00 = np.zeros((len(DataOL[0])), dtype=complex)
        OLe10 = np.zeros((len(DataOL[0])), dtype=complex)
        OLe11 = np.zeros((len(DataOL[0])), dtype=complex)

        for fpoint in range(len(DataOL[0])):
            # Coordinates of the 2D points
            x1 = DataOL[0][fpoint].s[0, 0, 0].real
            y1 = DataOL[0][fpoint].s[0, 0, 0].imag
            x2 = DataOL[1][fpoint].s[0, 0, 0].real
            y2 = DataOL[1][fpoint].s[0, 0, 0].imag

            angle = waveguide.line(offset_length * rf.milli, 'm')[fpoint].s_rad[0, 1, 0]

            q = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            y3 = (y1+y2)/2
            x3 = (x1+x2)/2

            r = (0.5*q)/np.sin(abs(angle))

            # we got two circles, and one of them with center at e00
            xc1 = x3 + np.sqrt(r**2-(q/2)**2)*(y1-y2)/q
            yc1 = y3 + np.sqrt(r**2-(q/2)**2)*(x2-x1)/q
            xc2 = x3 - np.sqrt(r**2-(q/2)**2)*(y1-y2)/q
            yc2 = y3 - np.sqrt(r**2-(q/2)**2)*(x2-x1)/q

            # check first circle
            det1 = (x2 - xc1) * (y1 - yc1) - (x1 - xc1) * (y2 - yc1)

            # calculate phase change (wave go through offset twice, forward and backward)
            angleOff = waveguide.line(2 * offset_length * rf.milli, 'm')[fpoint].s_rad[0, 1, 0]

            if (angleOff < 0):  # we need Right side
                if (det1 > 0):
                    # Right
                    xc = xc1
                    yc = yc1
                else:
                    # Left
                    xc = xc2
                    yc = yc2
            else:  # we need Left side
                if (det1 > 0):
                    # Right
                    xc = xc2
                    yc = yc2
                else:
                    # Left
                    xc = xc1
                    yc = yc1

            OLe00[fpoint] = complex(xc, yc)

            if (angleOff < 0):
                angleOff = abs(angleOff * 180 / np.pi)
            else:
                angleOff = abs((angleOff * 180 / np.pi) - 360)

        # Solving for e11, e10e01
        for fpoint in range(len(OLe00)):
            OLe10[fpoint] = ((offset_ideal.s[fpoint, 0, 0] - short_ideal.s[fpoint, 0, 0]) * \
                            (offset_meas.s[fpoint, 0, 0] - OLe00[fpoint]) * \
                            (short_meas.s[fpoint, 0, 0] - OLe00[fpoint])) / \
                            (offset_ideal.s[fpoint, 0, 0] * short_ideal.s[fpoint, 0, 0] * \
                            (offset_meas.s[fpoint, 0, 0] - short_meas.s[fpoint, 0, 0]))
            OLe11[fpoint] = (short_ideal.s[fpoint, 0, 0] * (offset_meas.s[fpoint, 0, 0] - OLe00[fpoint]) - \
                            (offset_ideal.s[fpoint, 0, 0] * (short_meas.s[fpoint, 0, 0] - OLe00[fpoint]))) / \
                            (offset_ideal.s[fpoint, 0, 0] * short_ideal.s[fpoint, 0, 0] * \
                            (offset_meas.s[fpoint, 0, 0] - short_meas.s[fpoint, 0, 0]))
        
        # write calculated error box
        self._coefs = {
            'directivity': OLe00,
            'reflection tracking': OLe10,
            'source match': OLe11}


class VNA():
    def __init__(self):
        self.rm = visa.ResourceManager()
    
    def connect(self,visa_name,freq_start,freq_stop,freq_step):
        try:
            self.device = self.rm.open_resource(visa_name)
            self.device.write("*RST;*CLS")
            self.device.write("SYST:DISP:UPD ON")
            self.device.write("FREQ:CONV:DEV:NAME 'NONE'")
            self.device.write("SENS1:FREQ:STAR " + freq_start)
            self.device.write("SENS1:FREQ:STOP " + freq_stop)
            self.device.write("SENS1:SWE:STEP " + freq_step)
            self.device.write("SENS1:BAND 100 HZ")
            self.device.write("CALC1:PARAMETER:SDEFINE 'Trc1', 'S11'")
            self.device.write("DISPLAY:WINDOW1:TRACE1:FEED 'Trc1'")
            self.device.write(":INITIATE:CONTINUOUS OFF")
        except:
            return False
        else:
            return True

    def measureSParam(self, result):
        self.device.write(":INITIATE:IMMEDIATE")
        self.device.query("*OPC?")
        asc = self.device.query_ascii_values("CALC1:DATA? SDAT")
        stm = self.device.query_ascii_values("CALC1:DATA:STIM?")
        self.device.write("DISP:WINDOW1:TRACE1:Y:AUTO ONCE")
        reSparam = asc[::2]
        imSparam = asc[1::2]
        result['freq'] = rf.Frequency(stm[0], stm[-1], len(stm), 'hz')
        result['Sparam'] = [x + y*1j for x, y in zip(reSparam, imSparam)]


pg.mkQApp()

# Define main window class from template
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'oneportmeas.ui')
WindowTemplate, TemplateBaseClass = pg.Qt.loadUiType(uiFile)


class MainWindow(TemplateBaseClass):

    def init_ui(self):
        self.style_btn_set = 'background-color: #a6dbac'
        self.style_btn_unset = 'background-color: None'

        self.ui = WindowTemplate()
        self.ui.setupUi(self)

        # VNA connection
        self.ui.btn_vna_connect.clicked.connect(self.VNAConnect)

        # Coaxial calibration
        self.ui.btn_coax_short_vna.clicked.connect(self.btn_coax_short_vna)
        self.ui.btn_coax_open_vna.clicked.connect(self.btn_coax_open_vna)
        self.ui.btn_coax_load_vna.clicked.connect(self.btn_coax_load_vna)

        self.ui.btn_coax_short_file.clicked.connect(self.btn_coax_short_file)
        self.ui.btn_coax_open_file.clicked.connect(self.btn_coax_open_file)
        self.ui.btn_coax_load_file.clicked.connect(self.btn_coax_load_file)

        self.ui.btn_coax_short_ch_file.clicked.connect(
            self.btn_coax_short_ch_file)
        self.ui.btn_coax_open_ch_file.clicked.connect(
            self.btn_coax_open_ch_file)
        self.ui.btn_coax_load_ch_file.clicked.connect(
            self.btn_coax_load_ch_file)

        self.ui.btn_coax_short_ch_ideal.clicked.connect(
            self.btn_coax_short_ch_ideal)
        self.ui.btn_coax_open_ch_ideal.clicked.connect(
            self.btn_coax_open_ch_ideal)
        self.ui.btn_coax_load_ch_ideal.clicked.connect(
            self.btn_coax_load_ch_ideal)

        self.ui.btn_coax_run.clicked.connect(self.btn_coax_run)
        self.ui.btn_coax_clear.clicked.connect(self.btn_coax_clear)

        # Waveguide calibration
        self.ui.btn_wg_short_vna.clicked.connect(self.btn_wg_short_vna)
        self.ui.btn_wg_offset_vna.clicked.connect(self.btn_wg_offset_vna)
        self.ui.btn_wg_load1_vna.clicked.connect(self.btn_wg_load1_vna)
        self.ui.btn_wg_load2_vna.clicked.connect(self.btn_wg_load2_vna)
        self.ui.btn_wg_load3_vna.clicked.connect(self.btn_wg_load3_vna)
        self.ui.btn_wg_load4_vna.clicked.connect(self.btn_wg_load4_vna)
        self.ui.btn_wg_load5_vna.clicked.connect(self.btn_wg_load5_vna)
        self.ui.btn_wg_load6_vna.clicked.connect(self.btn_wg_load6_vna)
        self.ui.btn_wg_load7_vna.clicked.connect(self.btn_wg_load7_vna)
        self.ui.btn_wg_load_l4_vna.clicked.connect(self.btn_wg_load_l4_vna)

        self.ui.btn_wg_short_file.clicked.connect(self.btn_wg_short_file)
        self.ui.btn_wg_offset_file.clicked.connect(self.btn_wg_offset_file)
        self.ui.btn_wg_load1_file.clicked.connect(self.btn_wg_load1_file)
        self.ui.btn_wg_load2_file.clicked.connect(self.btn_wg_load2_file)
        self.ui.btn_wg_load3_file.clicked.connect(self.btn_wg_load3_file)
        self.ui.btn_wg_load4_file.clicked.connect(self.btn_wg_load4_file)
        self.ui.btn_wg_load5_file.clicked.connect(self.btn_wg_load5_file)
        self.ui.btn_wg_load6_file.clicked.connect(self.btn_wg_load6_file)
        self.ui.btn_wg_load7_file.clicked.connect(self.btn_wg_load7_file)
        self.ui.btn_wg_load_l4_file.clicked.connect(self.btn_wg_load_l4_file)

        self.ui.btn_wg_short_file.setEnabled(True)
        self.ui.btn_wg_offset_file.setEnabled(True)
        self.ui.btn_wg_load1_file.setEnabled(True)
        self.ui.btn_wg_load2_file.setEnabled(True)
        self.ui.btn_wg_load3_file.setEnabled(True)
        self.ui.btn_wg_load4_file.setEnabled(True)
        self.ui.btn_wg_load5_file.setEnabled(True)
        self.ui.btn_wg_load6_file.setEnabled(True)
        self.ui.btn_wg_load7_file.setEnabled(True)
        self.ui.btn_wg_load_l4_file.setEnabled(True)

        self.ui.btn_wg_run_sol.clicked.connect(self.btn_wg_run_sol)
        self.ui.btn_wg_run_sliding.clicked.connect(self.btn_wg_run_sliding)
        self.ui.btn_wg_run_offset.clicked.connect(self.btn_wg_run_offset)
        self.ui.btn_wg_clear.clicked.connect(self.btn_wg_clear)

        # DUT measurements
        self.ui.btn_dut_file.clicked.connect(self.btn_dut_file)
        self.ui.btn_caldut_file.clicked.connect(self.btn_caldut_file)

        self.ui.btn_adapter_sol.clicked.connect(self.btn_adapter_sol)
        self.ui.btn_adapter_sliding.clicked.connect(self.btn_adapter_sliding)
        self.ui.btn_adapter_offset.clicked.connect(self.btn_adapter_offset)

        # Plotting
        self.ui.btn_save_s1p.clicked.connect(self.btn_save_s1p)

        self.button_group_spar = QButtonGroup()
        self.button_group_spar.addButton(self.ui.radio_s11)
        self.button_group_spar.addButton(self.ui.radio_s21)
        self.button_group_spar.addButton(self.ui.radio_s12)
        self.button_group_spar.addButton(self.ui.radio_s22)
        self.button_group_spar.buttonClicked.connect(
            self._on_radio_button_spar_clicked)

        self.button_group_format = QButtonGroup()
        self.button_group_format.addButton(self.ui.radio_db)
        self.button_group_format.addButton(self.ui.radio_phase)
        self.button_group_format.addButton(self.ui.radio_vswr)
        self.button_group_format.buttonClicked.connect(
            self._on_radio_button_format_clicked)

        self.show()
        self.clear_plot()
    
    def __init__(self):
        TemplateBaseClass.__init__(self)
        self.init_ui()
        self.DATA = {}
        self.APP = {'coax_short_ideal': False,
            'coax_open_ideal': False,
            'coax_load_ideal': False,
            'waveguide': False,
            'sliding_count': 0}
        self.Cal = {'coax_short_m': 0, 'coax_open_m': 0, 'coax_load_m': 0, 'coax_short_ch': 0, 'coax_open_ch': 0, 'coax_load_ch': 0,
           'wg_short_m': 0, 'wg_offset_m': 0, 'wg_load1_m': 0, 'wg_load2_m': 0,
           'wg_load3_m': 0, 'wg_load4_m': 0, 'wg_load5_m': 0, 'wg_load6_m': 0,
           'wg_load7_m': 0, 'wg_load_l4_m': 0}
        self.vna = VNA()
        datadir = os.path.abspath("data")
        if not os.path.isdir(datadir):
            os.makedirs(datadir)

    def btn_coax_short_vna(self):
        self.VNAMeasure('coax_short_m')
        self.ui.btn_coax_short_vna.setStyleSheet(self.style_btn_set)
        self.ui.btn_coax_short_file.setStyleSheet(self.style_btn_unset)
        self.Cal['coax_short_m'] = True
        self.check_coax_cal_btn()

    def btn_coax_open_vna(self):
        self.VNAMeasure('coax_open_m')
        self.ui.btn_coax_open_vna.setStyleSheet(self.style_btn_set)
        self.ui.btn_coax_open_file.setStyleSheet(self.style_btn_unset)
        self.Cal['coax_open_m'] = True
        self.check_coax_cal_btn()

    def btn_coax_load_vna(self):
        self.VNAMeasure('coax_load_m')
        self.ui.btn_coax_load_vna.setStyleSheet(self.style_btn_set)
        self.ui.btn_coax_load_file.setStyleSheet(self.style_btn_unset)
        self.Cal['coax_load_m'] = True
        self.check_coax_cal_btn()

    def btn_coax_short_file(self):
        file, _ = QtGui.QFileDialog.getOpenFileName(
            None, 'Open file', '', 's1p(*.s1p)')
        if not file:
            return
        self.DATA['coax_short_m'] = rf.Network(file)
        self.DATA['dut'] = self.DATA['coax_short_m']
        self.ui.plot.plot(x=self.DATA['coax_short_m'].frequency.f,
                          y=self.DATA['coax_short_m'].s_db[:, 0, 0], clear=True, pen='g')
        self.ui.btn_coax_short_file.setStyleSheet(self.style_btn_set)
        self.ui.btn_coax_short_vna.setStyleSheet(self.style_btn_unset)
        self.Cal['coax_short_m'] = True
        self.check_coax_cal_btn()
        self.update_plot()

    def btn_coax_open_file(self):
        file, _ = QtGui.QFileDialog.getOpenFileName(
            None, 'Open file', '', 's1p(*.s1p)')
        if not file:
            return
        self.DATA['coax_open_m'] = rf.Network(file)
        self.DATA['dut'] = self.DATA['coax_open_m']
        self.ui.plot.plot(x=self.DATA['coax_open_m'].frequency.f,
                          y=self.DATA['coax_open_m'].s_db[:, 0, 0], clear=True, pen='g')
        self.ui.btn_coax_open_file.setStyleSheet(self.style_btn_set)
        self.ui.btn_coax_open_vna.setStyleSheet(self.style_btn_unset)
        self.Cal['coax_open_m'] = True
        self.check_coax_cal_btn()
        self.update_plot()

    def btn_coax_load_file(self):
        file, _ = QtGui.QFileDialog.getOpenFileName(
            None, 'Open file', '', 's1p(*.s1p)')
        if not file:
            return
        self.DATA['coax_load_m'] = rf.Network(file)
        self.DATA['dut'] = self.DATA['coax_load_m']
        self.ui.btn_coax_load_file.setStyleSheet(self.style_btn_set)
        self.ui.btn_coax_load_vna.setStyleSheet(self.style_btn_unset)
        self.Cal['coax_load_m'] = True
        self.check_coax_cal_btn()
        self.update_plot()

    def btn_coax_short_ch_file(self):
        file, _ = QtGui.QFileDialog.getOpenFileName(
            None, 'Open file', '', 's1p(*.s1p)')
        if not file:
            return
        self.DATA['coax_short_ch'] = rf.Network(file)
        self.DATA['dut'] = self.DATA['coax_short_ch']
        self.APP['coax_short_ideal'] = False
        self.ui.btn_coax_short_ch_file.setStyleSheet(self.style_btn_set)
        self.ui.btn_coax_short_ch_ideal.setStyleSheet(self.style_btn_unset)
        self.Cal['coax_short_ch'] = True
        self.check_coax_cal_btn()
        self.update_plot()

    def btn_coax_open_ch_file(self):
        file, _ = QtGui.QFileDialog.getOpenFileName(
            None, 'Open file', '', 's1p(*.s1p)')
        if not file:
            return
        self.DATA['coax_open_ch'] = rf.Network(file)
        self.DATA['dut'] = self.DATA['coax_open_ch']
        self.APP['coax_open_ideal'] = False
        self.ui.btn_coax_open_ch_file.setStyleSheet(self.style_btn_set)
        self.ui.btn_coax_open_ch_ideal.setStyleSheet(self.style_btn_unset)
        self.Cal['coax_open_ch'] = True
        self.check_coax_cal_btn()
        self.update_plot()

    def btn_coax_load_ch_file(self):
        file, _ = QtGui.QFileDialog.getOpenFileName(
            None, 'Open file', '', 's1p(*.s1p)')
        if not file:
            return
        self.DATA['coax_load_ch'] = rf.Network(file)
        self.DATA['dut'] = self.DATA['coax_load_ch']
        self.APP['coax_load_ideal'] = False
        self.ui.btn_coax_load_ch_file.setStyleSheet(self.style_btn_set)
        self.ui.btn_coax_load_ch_ideal.setStyleSheet(self.style_btn_unset)
        self.Cal['coax_load_ch'] = True
        self.check_coax_cal_btn()
        self.update_plot()

    def btn_coax_short_ch_ideal(self):
        self.APP['coax_short_ideal'] = True
        self.clear_plot()
        self.ui.btn_coax_short_ch_file.setStyleSheet(self.style_btn_unset)
        self.ui.btn_coax_short_ch_ideal.setStyleSheet(self.style_btn_set)
        self.Cal['coax_short_ch'] = True
        self.check_coax_cal_btn()

    def btn_coax_open_ch_ideal(self):
        self.APP['coax_open_ideal'] = True
        self.clear_plot()
        self.ui.btn_coax_open_ch_file.setStyleSheet(self.style_btn_unset)
        self.ui.btn_coax_open_ch_ideal.setStyleSheet(self.style_btn_set)
        self.Cal['coax_open_ch'] = True
        self.check_coax_cal_btn()

    def btn_coax_load_ch_ideal(self):
        self.APP['coax_load_ideal'] = True
        self.clear_plot()
        self.ui.btn_coax_load_ch_file.setStyleSheet(self.style_btn_unset)
        self.ui.btn_coax_load_ch_ideal.setStyleSheet(self.style_btn_set)
        self.Cal['coax_load_ch'] = True
        self.check_coax_cal_btn()

    def btn_wg_short_vna(self):
        self.VNAMeasure('wg_short_m')
        self.ui.btn_wg_short_vna.setStyleSheet(self.style_btn_set)
        self.Cal['wg_short_m'] = True
        self.check_wg_cal_btn()

    def btn_wg_offset_vna(self):
        self.VNAMeasure('wg_offset_m')
        self.ui.btn_wg_offset_vna.setStyleSheet(self.style_btn_set)
        self.Cal['wg_offset_m'] = True
        self.check_wg_cal_btn()

    def btn_wg_load1_vna(self):
        self.VNAMeasure('wg_load1_m')
        self.APP['sliding_count'] = 1
        self.ui.btn_wg_load1_vna.setStyleSheet(self.style_btn_set)
        self.Cal['wg_load1_m'] = True
        self.check_wg_cal_btn()

    def btn_wg_load2_vna(self):
        self.VNAMeasure('wg_load2_m')
        self.APP['sliding_count'] = 2
        self.ui.btn_wg_load2_vna.setStyleSheet(self.style_btn_set)
        self.Cal['wg_load2_m'] = True
        self.check_wg_cal_btn()

    def btn_wg_load3_vna(self):
        self.VNAMeasure('wg_load3_m')
        self.APP['sliding_count'] = 3
        self.ui.btn_wg_load3_vna.setStyleSheet(self.style_btn_set)
        self.Cal['wg_load3_m'] = True
        self.check_wg_cal_btn()

    def btn_wg_load4_vna(self):
        self.VNAMeasure('wg_load4_m')
        self.APP['sliding_count'] = 4
        self.ui.btn_wg_load4_vna.setStyleSheet(self.style_btn_set)
        self.Cal['wg_load4_m'] = True
        self.check_wg_cal_btn()

    def btn_wg_load5_vna(self):
        self.VNAMeasure('wg_load5_m')
        self.APP['sliding_count'] = 5
        self.ui.btn_wg_load5_vna.setStyleSheet(self.style_btn_set)
        self.Cal['wg_load5_m'] = True
        self.check_wg_cal_btn()

    def btn_wg_load6_vna(self):
        self.VNAMeasure('wg_load6_m')
        self.APP['sliding_count'] = 6
        self.ui.btn_wg_load6_vna.setStyleSheet(self.style_btn_set)
        self.Cal['wg_load6_m'] = True
        self.check_wg_cal_btn()

    def btn_wg_load7_vna(self):
        self.VNAMeasure('wg_load7_m')
        self.APP['sliding_count'] = 7
        self.ui.btn_wg_load7_vna.setStyleSheet(self.style_btn_set)
        self.Cal['wg_load7_m'] = True
        self.check_wg_cal_btn()

    def btn_wg_load_l4_vna(self):
        self.VNAMeasure('wg_load_l4_m')
        self.ui.btn_wg_load_l4_vna.setStyleSheet(self.style_btn_set)
        self.Cal['wg_load_l4_m'] = True
        self.check_wg_cal_btn()

    def btn_wg_short_file(self):
        file, _ = QtGui.QFileDialog.getOpenFileName(
            None, 'Open file', '', 's1p(*.s1p)')
        if not file:
            return
        self.DATA['wg_short_m'] = rf.Network(file)
        self.ui.btn_wg_short_file.setStyleSheet(self.style_btn_set)
        self.Cal['wg_short_m'] = True
        self.check_wg_cal_btn()

    def btn_wg_offset_file(self):
        file, _ = QtGui.QFileDialog.getOpenFileName(
            None, 'Open file', '', 's1p(*.s1p)')
        if not file:
            return
        self.DATA['wg_offset_m'] = rf.Network(file)
        self.ui.btn_wg_offset_file.setStyleSheet(self.style_btn_set)
        self.Cal['wg_offset_m'] = True
        self.check_wg_cal_btn()

    def btn_wg_load1_file(self):
        file, _ = QtGui.QFileDialog.getOpenFileName(
            None, 'Open file', '', 's1p(*.s1p)')
        if not file:
            return
        self.DATA['wg_load1_m'] = rf.Network(file)
        self.APP['sliding_count'] = 1
        self.ui.btn_wg_load1_file.setStyleSheet(self.style_btn_set)
        self.Cal['wg_load1_m'] = True
        self.check_wg_cal_btn()

    def btn_wg_load2_file(self):
        file, _ = QtGui.QFileDialog.getOpenFileName(
            None, 'Open file', '', 's1p(*.s1p)')
        if not file:
            return
        self.DATA['wg_load2_m'] = rf.Network(file)
        self.APP['sliding_count'] = 2
        self.ui.btn_wg_load2_file.setStyleSheet(self.style_btn_set)
        self.Cal['wg_load2_m'] = True
        self.check_wg_cal_btn()

    def btn_wg_load3_file(self):
        file, _ = QtGui.QFileDialog.getOpenFileName(
            None, 'Open file', '', 's1p(*.s1p)')
        if not file:
            return
        self.DATA['wg_load3_m'] = rf.Network(file)
        self.APP['sliding_count'] = 3
        self.ui.btn_wg_load3_file.setStyleSheet(self.style_btn_set)
        self.Cal['wg_load3_m'] = True
        self.check_wg_cal_btn()

    def btn_wg_load4_file(self):
        file, _ = QtGui.QFileDialog.getOpenFileName(
            None, 'Open file', '', 's1p(*.s1p)')
        if not file:
            return
        self.DATA['wg_load4_m'] = rf.Network(file)
        self.APP['sliding_count'] = 4
        self.ui.btn_wg_load4_file.setStyleSheet(self.style_btn_set)
        self.Cal['wg_load4_m'] = True
        self.check_wg_cal_btn()

    def btn_wg_load5_file(self):
        file, _ = QtGui.QFileDialog.getOpenFileName(
            None, 'Open file', '', 's1p(*.s1p)')
        if not file:
            return
        self.DATA['wg_load5_m'] = rf.Network(file)
        self.APP['sliding_count'] = 5
        self.ui.btn_wg_load5_file.setStyleSheet(self.style_btn_set)
        self.Cal['wg_load5_m'] = True
        self.check_wg_cal_btn()

    def btn_wg_load6_file(self):
        file, _ = QtGui.QFileDialog.getOpenFileName(
            None, 'Open file', '', 's1p(*.s1p)')
        if not file:
            return
        self.DATA['wg_load6_m'] = rf.Network(file)
        self.APP['sliding_count'] = 6
        self.ui.btn_wg_load6_file.setStyleSheet(self.style_btn_set)
        self.Cal['wg_load6_m'] = True
        self.check_wg_cal_btn()

    def btn_wg_load7_file(self):
        file, _ = QtGui.QFileDialog.getOpenFileName(
            None, 'Open file', '', 's1p(*.s1p)')
        if not file:
            return
        self.DATA['wg_load7_m'] = rf.Network(file)
        self.APP['sliding_count'] = 7
        self.ui.btn_wg_load7_file.setStyleSheet(self.style_btn_set)
        self.Cal['wg_load7_m'] = True
        self.check_wg_cal_btn()

    def btn_wg_load_l4_file(self):
        file, _ = QtGui.QFileDialog.getOpenFileName(
            None, 'Open file', '', 's1p(*.s1p)')
        if not file:
            return
        self.DATA['wg_load_l4_m'] = rf.Network(file)
        self.ui.btn_wg_load_l4_file.setStyleSheet(self.style_btn_set)
        self.Cal['wg_load_l4_m'] = True
        self.check_wg_cal_btn()

    def check_coax_cal_btn(self):
        if ((self.Cal['coax_short_m']
             and self.Cal['coax_open_m']
             and self.Cal['coax_load_m']
             and self.Cal['coax_short_ch']
             and self.Cal['coax_open_ch']
             and self.Cal['coax_load_ch'])):
            self.ui.btn_coax_run.setEnabled(True)

    def check_wg_cal_btn(self):
        # SOLT
        if ((not self.ui.btn_wg_run_sol.isEnabled()) and self.Cal['wg_short_m'] and self.Cal['wg_offset_m'] and self.Cal['wg_load1_m']):
            self.ui.btn_wg_run_sol.setEnabled(True)
        # Sliding Load
        if ((not self.ui.btn_wg_run_sliding.isEnabled()) and self.Cal['wg_short_m'] and self.Cal['wg_offset_m'] and self.Cal['wg_load1_m']
            and self.Cal['wg_load2_m'] and self.Cal['wg_load3_m'] and self.Cal['wg_load4_m']
                and self.Cal['wg_load5_m']):
            self.ui.btn_wg_run_sliding.setEnabled(True)
        # Offset Load
        if ((not self.ui.btn_wg_run_offset.isEnabled()) and self.Cal['wg_short_m'] and self.Cal['wg_offset_m'] and self.Cal['wg_load1_m']
                and self.Cal['wg_load_l4_m']):
            self.ui.btn_wg_run_offset.setEnabled(True)

    def wg_make_stds(self):
        self.DATA['wg'] = rf.RectangularWaveguide(frequency=self.DATA['wg_short_m'].frequency, a=float(
            self.ui.edit_wg_a.text())*rf.milli, b=float(self.ui.edit_wg_b.text())*rf.milli, z0=50)
        self.DATA['wg_short'] = self.DATA['wg'].short()
        self.DATA['wg_offset'] = self.DATA['wg'].delay_short(
            d=float(self.ui.edit_wg_l4_length.text())*rf.milli, unit='m')
        self.DATA['wg_match'] = self.DATA['wg'].match()
        self.APP['waveguide'] = True
        self.ui.edit_wg_a.setEnabled(False)
        self.ui.edit_wg_b.setEnabled(False)
        self.ui.edit_wg_l4_length.setEnabled(False)

    def btn_wg_run_sol(self):
        if not self.APP['waveguide']:
            self.wg_make_stds()
        self.DATA['wg_cal_sol'] = OnePortExtended(ideals=[
            self.DATA['wg_short'], self.DATA['wg_offset'], self.DATA['wg_match']],
            measured=[
            self.DATA['wg_short_m'], self.DATA['wg_offset_m'], self.DATA['wg_load1_m']])
        self.DATA['wg_cal_sol'].run()
        self.ui.btn_wg_run_sol.setStyleSheet(self.style_btn_set)
        self.clear_plot()
        self.ui.btn_adapter_sol.setEnabled(True)

    def btn_wg_run_sliding(self):
        if not self.APP['waveguide']:
            self.wg_make_stds()

        sliding_data = []
        for i in range(self.APP['sliding_count']):
            sliding_data.append(self.DATA['wg_load' + str(i+1) + '_m'])

        self.DATA['wg_cal_sliding'] = OnePortExtended(measured=[self.DATA['wg_short_m']],ideals=[self.DATA['wg_short_m']])
        self.DATA['wg_cal_sliding'].RunSlidingLoad(self.DATA['wg_short'],self.DATA['wg_short_m'],\
            self.DATA['wg_offset'],self.DATA['wg_offset_m'],sliding_data)
        
        self.ui.btn_wg_run_sliding.setStyleSheet(self.style_btn_set)
        self.clear_plot()
        self.ui.btn_adapter_sliding.setEnabled(True)

    def btn_wg_run_offset(self):
        if (not self.APP['waveguide']):
            self.wg_make_stds()
        
        self.DATA['wg_cal_offset'] = OnePortExtended(measured=[self.DATA['wg_short_m']], ideals=[self.DATA['wg_short_m']])
        self.DATA['wg_cal_offset'].RunOffsetLoad(self.DATA['wg'], self.DATA['wg_short_m'], self.DATA['wg_short'], \
            self.DATA['wg_offset_m'], self.DATA['wg_offset'], self.DATA['wg_load1_m'], \
            self.DATA['wg_load_l4_m'], float(self.ui.edit_wg_l4_length.text()))

        self.ui.btn_wg_run_offset.setStyleSheet(self.style_btn_set)
        self.clear_plot()
        self.ui.btn_adapter_offset.setEnabled(True)

    def btn_adapter_sol(self):
        self.DATA['adapter_sol'] = self.DATA['coax_cal'].error_ntwk.inv ** self.DATA['wg_cal_sol'].error_ntwk
        self.DATA['dut'] = self.DATA['adapter_sol']
        self.update_plot()

    def btn_adapter_sliding(self):
        self.DATA['adapter_sliding'] = self.DATA['coax_cal'].error_ntwk.inv ** self.DATA['wg_cal_sliding'].error_ntwk
        self.DATA['dut'] = self.DATA['adapter_sliding']
        self.update_plot()

    def btn_adapter_offset(self):
        self.DATA['adapter_offset'] = self.DATA['coax_cal'].error_ntwk.inv ** self.DATA['wg_cal_offset'].error_ntwk
        self.DATA['dut'] = self.DATA['adapter_offset']
        self.update_plot()

    def _on_radio_button_format_clicked(self):
        self.update_plot()

    def _on_radio_button_spar_clicked(self):
        self.update_plot()

    def btn_coax_run(self):
        # SHORT
        if self.APP['coax_short_ideal']:
            self.DATA['coax_short_ch'] = rf.Network(frequency=self.DATA['coax_short_m'].frequency, s=[
                                               -1 for x in range(self.DATA['coax_short_m'].frequency.npoints)], z0=50, name='coax_short_ch')
        else:
            self.DATA['coax_short_ch'] = self.DATA['coax_short_ch'].interpolate_from_f(
                self.DATA['coax_short_m'].frequency)
        # OPEN
        if self.APP['coax_open_ideal']:
            self.DATA['coax_open_ch'] = rf.Network(frequency=self.DATA['coax_open_m'].frequency, s=[
                                              1 for x in range(self.DATA['coax_open_m'].frequency.npoints)], z0=50, name='coax_open_ch')
        else:
            self.DATA['coax_open_ch'] = self.DATA['coax_open_ch'].interpolate_from_f(
                self.DATA['coax_short_m'].frequency)
        # LOAD
        if self.APP['coax_load_ideal']:
            self.DATA['coax_load_ch'] = rf.Network(frequency=self.DATA['coax_load_m'].frequency, s=[
                                              0 for x in range(self.DATA['coax_load_m'].frequency.npoints)], z0=50, name='coax_load_ch')
        else:
            self.DATA['coax_load_ch'] = self.DATA['coax_load_ch'].interpolate_from_f(
                self.DATA['coax_short_m'].frequency)

        self.DATA['coax_cal'] = rf.OnePort(ideals=[
            self.DATA['coax_short_ch'], self.DATA['coax_open_ch'], self.DATA['coax_load_ch']],
            measured=[self.DATA['coax_short_m'], self.DATA['coax_open_m'], self.DATA['coax_load_m']])
        self.DATA['coax_cal'].run()
        self.ui.frameSOL.setEnabled(False)
        self.ui.btn_coax_run.setStyleSheet(self.style_btn_set)
        self.ui.btn_coax_run.setEnabled(False)
        self.ui.btn_coax_clear.setEnabled(True)
        self.clear_plot()

    def btn_coax_clear(self):
        self.Cal = {'coax_short_m': 0, 'coax_open_m': 0, 'coax_load_m': 0,
                    'coax_short_ch': 0, 'coax_open_ch': 0, 'coax_load_ch': 0}
        self.APP['coax_short_ideal'] = False
        self.APP['coax_open_ideal'] = False
        self.APP['coax_load_ideal'] = False
        self.ui.radio_raw.setChecked(True)
        self.ui.frameSOL.setEnabled(True)
        self.ui.btn_coax_run.setEnabled(False)
        self.ui.btn_coax_run.setStyleSheet(self.style_btn_unset)
        self.ui.btn_coax_clear.setEnabled(False)
        self.ui.btn_coax_short_ch_file.setStyleSheet(self.style_btn_unset)
        self.ui.btn_coax_open_ch_file.setStyleSheet(self.style_btn_unset)
        self.ui.btn_coax_load_ch_file.setStyleSheet(self.style_btn_unset)
        self.ui.btn_coax_short_file.setStyleSheet(self.style_btn_unset)
        self.ui.btn_coax_open_file.setStyleSheet(self.style_btn_unset)
        self.ui.btn_coax_load_file.setStyleSheet(self.style_btn_unset)
        self.ui.btn_coax_short_ch_ideal.setStyleSheet(self.style_btn_unset)
        self.ui.btn_coax_open_ch_ideal.setStyleSheet(self.style_btn_unset)
        self.ui.btn_coax_load_ch_ideal.setStyleSheet(self.style_btn_unset)
        self.clear_plot()

    def btn_wg_clear(self):
        self.clear_plot()

    def btn_dut_file(self):
        file, _ = QtGui.QFileDialog.getOpenFileName(
            None, 'Open file', '', 'sNp(*.s?p)')
        if not file:
            return
        self.DATA['dut'] = rf.Network(file)
        self.update_plot()

    def btn_caldut_file(self):
        file, _ = QtGui.QFileDialog.getOpenFileName(
            None, 'Open file', '', 'sNp(*.s?p)')
        if not file:
            return
        self.DATA['dut'] = self.DATA['coax_cal'].apply_cal(rf.Network(file))
        self.update_plot()

    def clear_plot(self):
        self.ui.plot.clear()
        for button in self.button_group_spar.buttons():
            button.setEnabled(False)
        for button in self.button_group_format.buttons():
            button.setEnabled(False)
        self.ui.btn_save_s1p.setEnabled(False)

    def update_plot(self):
        self.ui.plot.showGrid(True, True)
        self.ui.btn_save_s1p.setEnabled(True)
        i1 = i2 = -1
        if (self.DATA['dut'].nports > 1):
            for i, button in enumerate(self.button_group_spar.buttons()):
                button.setEnabled(True)
                if button.isChecked():
                    if i == 0:
                        i1 = i2 = 0
                    if i == 1:
                        i1, i2 = 1, 0
                    if i == 2:
                        i1, i2 = 0, 1
                    if i == 3:
                        i1 = i2 = 1
            if i1 == -1:
                self.ui.radio_s11.setChecked(True)
                i1 = i2 = 0
        else:
            self.ui.radio_s11.setEnabled(True)
            self.ui.radio_s21.setEnabled(False)
            self.ui.radio_s12.setEnabled(False)
            self.ui.radio_s22.setEnabled(False)
            self.ui.radio_s11.setChecked(True)
            i1 = i2 = 0

        checked = -1
        for i, button in enumerate(self.button_group_format.buttons()):
            button.setEnabled(True)
            if button.isChecked():
                checked = i
        if checked == -1:
            self.ui.radio_db.setChecked(True)
            checked = 0
        if checked == 0:  # dB
            self.ui.plot.plot(x=self.DATA['dut'].frequency.f,
                              y=self.DATA['dut'].s_db[:, i1, i2],
                              clear=True, pen='g')
        if checked == 1:  # Phase
            self.ui.plot.plot(x=self.DATA['dut'].frequency.f,
                              y=self.DATA['dut'].s_deg[:, i1, i2],
                              clear=True, pen='g')
        if checked == 2:  # VSWR
            self.ui.plot.plot(x=self.DATA['dut'].frequency.f,
                              y=self.DATA['dut'].s_vswr[:, i1, i2],
                              clear=True, pen='g')

    def VNAConnect(self):
        if self.vna.connect(self.ui.edit_tcpip.text(), self.ui.edit_start.text(), self.ui.edit_stop.text(), self.ui.edit_step.text()):
            self.ui.btn_coax_short_vna.setEnabled(True)
            self.ui.btn_coax_open_vna.setEnabled(True)
            self.ui.btn_coax_load_vna.setEnabled(True)
            self.ui.btn_wg_short_vna.setEnabled(True)
            self.ui.btn_wg_offset_vna.setEnabled(True)
            self.ui.btn_wg_load1_vna.setEnabled(True)
            self.ui.btn_wg_load2_vna.setEnabled(True)
            self.ui.btn_wg_load3_vna.setEnabled(True)
            self.ui.btn_wg_load4_vna.setEnabled(True)
            self.ui.btn_wg_load5_vna.setEnabled(True)
            self.ui.btn_wg_load6_vna.setEnabled(True)
            self.ui.btn_wg_load7_vna.setEnabled(True)
            self.ui.btn_wg_load_l4_vna.setEnabled(True)
            self.ui.btn_vna_connect.setEnabled(False)
            self.ui.btn_vna_close.setEnabled(True)
            self.ui.btn_vna_connect.setStyleSheet(self.style_btn_set)
        else:
            self.ui.btn_vna_connect.setEnabled(True)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Can't establish connection to VNA!")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

    def VNAMeasure(self, ch):
        Sparam = {}
        self.vna.measureSParam(Sparam)
        ntwk = rf.Network(frequency=Sparam['freq'], s=Sparam['Sparam'], z0=50, name='vna_measure')
        self.DATA['dut'] = ntwk
        self.DATA[ch] = ntwk
        filename = "data/" + \
            ch[0:len(ch)-2] + \
            '_{date:%Y-%m-%d_%H%M%S}.s1p'.format(date=datetime.datetime.now())
        filename = os.path.abspath(filename)
        self.DATA['dut'].write_touchstone(filename)
        self.update_plot()

    def btn_save_s1p(self):
        if (self.DATA['dut'].nports == 1):
            filename = QtGui.QFileDialog.getSaveFileName(
                self, 'Save File', "", "s1p (*.s1p)")
        else:
            filename = QtGui.QFileDialog.getSaveFileName(
                self, 'Save File', "", "s2p (*.s2p)")
        if not filename:
            return
        self.DATA['dut'].write_touchstone(filename=filename[0])


win = MainWindow()

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
