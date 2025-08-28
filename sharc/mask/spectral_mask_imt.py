# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:06:56 2017

@author: Calil
"""

from sharc.support.enumerations import StationType
from sharc.mask.spectral_mask import SpectralMask

from warnings import warn
import numpy as np
import matplotlib.pyplot as plt


class SpectralMaskImt(SpectralMask):
    """
    Implements spectral masks for IMT-2020 according to Document 5-1/36-E.
        The masks are in the document's tables 1 to 8.

    Attributes:
        spurious_emissions (float): level of power emissions at spurious
            domain [dBm/MHz].
        delta_f_lin (np.array): mask delta f breaking limits in MHz. Delta f
            values for which the spectral mask changes value. In this context, delta f is the frequency distance to
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
    def __init__(
        self,
        sta_type: StationType,
        freq_mhz: float,
        band_mhz: float,
        spurious_emissions: float,
        scenario="OUTDOOR",
    ):
        """
        Class constructor.

        Parameters:
            sta_type (StationType): type of station to which consider the spectral
                mask. Possible values are StationType.IMT_BS and StationType.
                IMT_UE
            freq_mhz (float): center frequency of station in MHz
            band_mhs (float): transmitting bandwidth of station in MHz
            spurious_emissions (float): level of spurious emissions [dBm/MHz].
            scenario (str): INDOOR or OUTDOOR scenario
        """
        # Spurious domain limits [dBm/MHz]
        self.spurious_emissions = spurious_emissions

        # Mask delta f breaking limits [MHz]
        # use value from 5-1/36-E
        self.delta_f_lim = np.array([0, 20, 400])

        # Attributes
        self.sta_type = sta_type
        self.scenario = scenario
        self.band_mhz = band_mhz
        self.freq_mhz = freq_mhz

        self.freq_lim = np.concatenate((
            (freq_mhz - band_mhz / 2) - self.delta_f_lim[::-1],
            (freq_mhz + band_mhz / 2) + self.delta_f_lim,
        ))

    def set_mask(self, p_tx=0):
        """
        Sets the spectral mask (mask_dbm attribute) based on station type,
        operating frequency and transmit power.

        Parameters:
            p_tx (float): station transmit power. Default = 0
        """
        self.p_tx = p_tx - 10 * np.log10(self.band_mhz)

        # Set new transmit power value
        if self.sta_type is StationType.IMT_UE:
            # Table 8
            mask_dbm = np.array([-5, -13, self.spurious_emissions])

        elif self.sta_type is StationType.IMT_BS and self.scenario == "INDOOR":
            # Table 1
            mask_dbm = np.array([-5, -13, self.spurious_emissions])

        else:

            if (self.freq_mhz > 24250 and self.freq_mhz < 33400):
                if p_tx >= 34.5:
                    # Table 2
                    mask_dbm = np.array([-5, -13, self.spurious_emissions])
                else:
                    # Table 3
                    mask_dbm = np.array([
                        -5, np.max((p_tx - 47.5, -20)),
                        self.spurious_emissions,
                    ])
            elif (self.freq_mhz > 37000 and self.freq_mhz < 52600):
                if p_tx >= 32.5:
                    # Table 4
                    mask_dbm = np.array([-5, -13, self.spurious_emissions])
                else:
                    # Table 5
                    mask_dbm = np.array([
                        -5, np.max((p_tx - 45.5, -20)),
                        self.spurious_emissions,
                    ])
            elif (self.freq_mhz > 66000 and self.freq_mhz < 86000):
                if p_tx >= 30.5:
                    # Table 6
                    mask_dbm = np.array([-5, -13, self.spurious_emissions])
                else:
                    # Table 7
                    mask_dbm = np.array([
                        -5, np.max((p_tx - 43.5, -20)),
                        self.spurious_emissions,
                    ])
            else:
                mask_dbm = None
                # this will only be reached when spurious emission has been manually set to something invalid and
                warn(
                    "\nSpectralMaskIMT cannot be used with current parameters.\n"
                    "\tYou may:\n\t\t- Have set spurious emission to a value not in [-13,-30]"
                    "\n\t\t- Be trying to use the mask for IMT BS outdoor but freq not in (24.25, 86) GHz range"
                )

        if mask_dbm is not None:
            self.mask_dbm = np.concatenate((
                mask_dbm[::-1], np.array([self.p_tx]),
                mask_dbm,
            ))


if __name__ == '__main__':
    # Initialize variables
    sta_type = StationType.IMT_BS
    p_tx = 34.061799739838875
    freq = 9000
    band = 100
    spurious_emissions_dbm_mhz = -30

    # Create mask
    msk = SpectralMaskImt(sta_type, freq, band, spurious_emissions_dbm_mhz)
    msk.set_mask(p_tx)

    # Frequencies
    freqs = np.linspace(-600, 600, num=1000) + freq

    # Mask values
    mask_val = np.ones_like(freqs) * msk.mask_dbm[0]
    for k in range(len(msk.freq_lim) - 1, -1, -1):
        mask_val[np.where(freqs < msk.freq_lim[k])] = msk.mask_dbm[k]

    # Plot
    plt.plot(freqs, mask_val)
    plt.xlim([freqs[0], freqs[-1]])
    plt.xlabel(r"$\Delta$f [MHz]")
    plt.ylabel("Spectral Mask [dBc]")
    plt.grid()
    plt.show()
