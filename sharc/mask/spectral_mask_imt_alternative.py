# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:06:56 2017

@author: Calil
"""

from sharc.support.enumerations import StationType
from sharc.mask.spectral_mask import SpectralMask

import numpy as np
import matplotlib.pyplot as plt

class SpectralMaskImtAlternative(SpectralMask):
    """
    Implements spectral masks for IMT-2020 outdoor BS's when freq < 26GHz,
        according to Documents ITU-R SM.1541-6, ITU-R SM.1539-1 and ETSI TS 138 104 V16.6.0. 
    Reference tables are:
        - Table 1 in ITU-R SM.1541-6 (not sure what for, actually)
        - Table 2 in ITU-R SM.1539-1
        - Table 6.6.4.2.1-2   in ETSI TS 138 104 V16.6.0
        - Table 6.6.4.2.2.1-2 in ETSI TS 138 104 V16.6.0
        - Table 6.6.5.2.1-1   in ETSI TS 138 104 V16.6.0
        - Table 6.6.5.2.1-2   in ETSI TS 138 104 V16.6.0
    
    Attributes:
        spurious_emissions (float): level of power emissions at spurious
            domain [dBm/MHz]. 
        delta_f_lin (np.array): mask delta f breaking limits in MHz. Delta f 
            values for which the spectral mask changes value. Hard coded as
            [0, 20, 400]. In this context, delta f is the frequency distance to
            the transmission's edge frequencies
        freq_lim (no.array): frequency values for which the spectral mask
            changes emission value
        sta_type (StationType): type of station to which consider the spectral
            mask. Possible values are StationType.IMT_BS and StationType.IMT_UE
        freq_mhz (float): center frequency of station in MHz
        band_mhs (float): transmitting bandwidth of station in MHz
        scenario (str): INDOOR or OUTDOOR scenario
        p_tx (float): station's transmit power in dBm/MHz
        mask_dbm (np.array): spectral mask emission values in dBm
    """
    OOB_VERTICAL_LINE_SAMPLESIZE: int = 100 # this is needed to sample the curve present in oob emission

    def __init__(self, 
                 sta_type: StationType, 
                 freq_mhz: float, 
                 band_mhz: float, 
                 spurious_emissions : float, 
                 scenario = "OUTDOOR"):
        """
        Class constructor.
        
        Parameters:
            sta_type (StationType): type of station to which consider the spectral
                mask. Currently, only possible value is StationType.IMT_BS
            freq_mhz (float): center frequency of station in MHz
            band_mhs (float): transmitting bandwidth of station in MHz
            spurious_emissions (float): level of spurious emissions [dBm/MHz]. 
            scenario (str): only supports OUTDOOR scenario
        """
        # Spurious domain limits [dBm/MHz]
        self.spurious_emissions = spurious_emissions

        is_within_etsi_ts_138_reference_requirements = sta_type is StationType.IMT_BS and scenario == "OUTDOOR"
        if not is_within_etsi_ts_138_reference_requirements:
            raise ValueError("Can only use SpectralMaskImt2 with IMT Base Station and on outdoor scenario")

        # Document 5-1/36-E already better documents frequencies higher than 26GHz
        is_better_masked_by_document_5_1_36_E = freq_mhz > 26000
        if is_better_masked_by_document_5_1_36_E:
            raise ValueError("When using higher freq than 26000MHz should use SpectralMaskImt instead of SpectralMaskImt2")

        if (freq_mhz > 0.009 and freq_mhz < 0.15):
            B_L = 0.00025
            B_L_separation = 0.000625
            B_U = 0.01
            B_U_separation = 1.5 * band_mhz + 0.01
        elif (freq_mhz > 0.15 and freq_mhz < 30):
            B_L = 0.004
            B_L_separation = 0.01
            B_U = 0.1
            B_U_separation = 1.5 * band_mhz + 0.1
        elif (freq_mhz > 30 and freq_mhz < 1000):
            B_L = 0.025
            B_L_separation = 0.0625
            B_U = 10
            B_U_separation = 1.5 * band_mhz + 10
        elif (freq_mhz > 1000 and freq_mhz < 3000):
            B_L = 0.1
            B_L_separation = 0.25
            B_U = 50
            B_U_separation = 1.5 * band_mhz + 50
        elif (freq_mhz > 3000 and freq_mhz < 10000):
            B_L = 0.1
            B_L_separation = 0.25
            B_U = 100
            B_U_separation = 1.5 * band_mhz + 100
        elif (freq_mhz > 10000 and freq_mhz < 15000):
            B_L = 0.3
            B_L_separation = 0.75
            B_U = 250
            B_U_separation = 1.5 * band_mhz + 250
        elif (freq_mhz > 15000 and freq_mhz < 26000):
            B_L = 0.5
            B_L_separation = 1.25
            B_U = 500
            B_U_separation = 1.5 * band_mhz + 500

        if band_mhz < B_L:
            delta_f_spurious = B_L_separation
        elif band_mhz > B_U:
            delta_f_spurious = B_U_separation
        else:
            delta_f_spurious = 2.5 * band_mhz

        arr = np.array([i * 5 / self.OOB_VERTICAL_LINE_SAMPLESIZE for i in range(self.OOB_VERTICAL_LINE_SAMPLESIZE)])
        arr2 = np.array([5, 10.0, delta_f_spurious - band_mhz/2])
        self.delta_f_lim = np.concatenate((arr, arr2))

        # Attributes
        self.sta_type = sta_type
        self.scenario = scenario
        self.band_mhz = band_mhz
        self.freq_mhz = freq_mhz

        self.freq_lim = np.concatenate(((freq_mhz - band_mhz/2)-self.delta_f_lim[::-1],
                                        (freq_mhz + band_mhz/2)+self.delta_f_lim))


    def set_mask(self, power = 0):
        """
        Sets the spectral mask (mask_dbm attribute) based on station type, 
        operating frequency and transmit power.
        
        Parameters:
            power (float): station transmit power. Default = 0
        """
        self.p_tx = power - 10*np.log10(self.band_mhz)
        
        if self.spurious_emissions == -13:
            # use document ETSI TS 138 104 V16.6.0 Table 6.6.4.2.1-2
            def func(f_mhz):# return dBm
                return -7 -7/5 * f_mhz
            arr = np.array([func(i * 5 / self.OOB_VERTICAL_LINE_SAMPLESIZE) for i in range(self.OOB_VERTICAL_LINE_SAMPLESIZE)])
            arr2 = np.array([-14, -13, self.spurious_emissions])
            mask_dbm = np.concatenate((arr, arr2))
        elif self.spurious_emissions == -30:
            # use document ETSI TS 138 104 V16.6.0 Table 6.6.4.2.2.1-2
            def func(f_mhz):# return dBm
                return -7 -7/5 * f_mhz
            arr = np.array([func(i * 5 / self.OOB_VERTICAL_LINE_SAMPLESIZE) for i in range(self.OOB_VERTICAL_LINE_SAMPLESIZE)])
            arr2 = np.array([-14, -15, self.spurious_emissions])
            mask_dbm = np.concatenate((arr, arr2))
        else:
            # Dummy spectral mask, for testing only
            mask_dbm = np.array([-10, -20, -50])
                 
        self.mask_dbm = np.concatenate((mask_dbm[::-1],np.array([self.p_tx]),
                                        mask_dbm))
        
if __name__ == '__main__':
    # Initialize variables
    sta_type = StationType.IMT_BS
    p_tx = 34.061799739838875
    freq = 9000
    band = 100
    spurious_emissions_dbm_mhz = -30

    # Create mask
    msk = SpectralMaskImtAlternative(sta_type,freq,band, spurious_emissions_dbm_mhz)
    msk.set_mask(p_tx)
    
    # Frequencies
    freqs = np.linspace(-600,600,num=1000)+freq
    
    # Mask values
    mask_val = np.ones_like(freqs)*msk.mask_dbm[0]
    for k in range(len(msk.freq_lim)-1,-1,-1):
        mask_val[np.where(freqs < msk.freq_lim[k])] = msk.mask_dbm[k]
        
    # Plot
    plt.plot(freqs,mask_val)
    plt.xlim([freqs[0],freqs[-1]])
    plt.xlabel("$\Delta$f [MHz]")
    plt.ylabel("Spectral Mask [dBc]")
    plt.grid()
    plt.show()
