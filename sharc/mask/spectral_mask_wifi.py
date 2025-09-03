import numpy as np
from sharc.mask.spectral_mask import SpectralMask
from sharc.support.enumerations import StationType

class SpectralMaskWifi(SpectralMask):
    """
    Ref: IEEE Std 802.11-2020, Sec. 17.3.9.3–4.
    """

    MASK_TABLE = {
        5: {
            "offsets": [2.75, 5, 7.5],
            "levels": [0, -20, -28, -40],  # relativo (dBr)
            "abs_floor": -47
        },
        10: {
            "offsets": [5.5, 10, 15],
            "levels": [0, -20, -28, -40],
            "abs_floor": -50
        },
        20: {
            "offsets": [11, 20, 30],
            "levels": [0, -20, -28, -40],
            "abs_floor": -53
        }
    }

    def __init__(self, freq_mhz: float, band_mhz: float, station_type: StationType, spurious_emissions: float = None):

        self.freq_mhz = freq_mhz
        self.band_mhz = band_mhz
        self.spurious_emissions = spurious_emissions

        delta_f_lim = self.get_frequency_limits(band_mhz)
        delta_f_lim_flipped = delta_f_lim[::-1]

        self.freq_lim = np.concatenate((
            (self.freq_mhz - self.band_mhz / 2) - delta_f_lim_flipped,
            (self.freq_mhz + self.band_mhz / 2) + delta_f_lim,
        ))

    def get_frequency_limits(self, bandwidth: float) -> np.array:
    
        if bandwidth in self.MASK_TABLE:
            return np.array([0] + self.MASK_TABLE[bandwidth]["offsets"])
        
        elif bandwidth in [40, 80, 160]:
            scale = bandwidth / 20
            base = self.MASK_TABLE[20]["offsets"]
            return np.array([0] + [o * scale for o in base])
        else:
            raise ValueError(f"Largura de banda {bandwidth} MHz não suportada")

    def get_emission_limits(self, bandwidth: float) -> np.array:
        """
        Retorna limites de emissão em dBm/MHz (relativos + spurious).
        """
        if bandwidth in self.MASK_TABLE:
            levels = self.MASK_TABLE[bandwidth]["levels"]
            abs_floor = self.MASK_TABLE[bandwidth]["abs_floor"]
        elif bandwidth in [40, 80, 160]:
            levels = self.MASK_TABLE[20]["levels"]  # mesmos dBr
            abs_floor = self.MASK_TABLE[20]["abs_floor"]
        else:
            raise ValueError(f"Largura de banda {bandwidth} MHz não suportada")

        # Se usuário passou um spurious manual, sobrescreve
        if self.spurious_emissions is not None:
            abs_floor = self.spurious_emissions

        return np.array(levels[:-1] + [max(levels[-1], abs_floor)])

    def set_mask(self, p_tx=0):
        """
        Define a máscara (mask_dbm) em dBm/MHz.
        """
        # Potência média por MHz
        self.p_tx = p_tx - 10 * np.log10(self.band_mhz)

        # Limites relativos convertidos para absolutos
        emission_limits = self.get_emission_limits(self.band_mhz) + self.p_tx

        # Monta a máscara simétrica
        emission_limits_flipped = emission_limits[::-1]
        self.mask_dbm = np.concatenate((
            emission_limits_flipped,
            np.array([self.p_tx]),
            emission_limits
        ))
