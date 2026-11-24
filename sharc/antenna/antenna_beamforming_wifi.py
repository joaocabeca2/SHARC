# -*- coding: utf-8 -*-
"""
Beamforming antenna for the Wi-Fi access point.

Reuses AntennaBeamformingImt (array of N_rows x N_columns elements, ITU-R
M.2101 array factor) with two Wi-Fi specific additions:

- Numerical normalization: the gain is shifted by a constant so that its mean
  over the sphere is 0 dBi (total radiated power equals the conducted power).
  The SHARC normalization files are built for IMT arrays and are not used.
- Peak gain (after normalization), used to apply an e.i.r.p. limit to the AP
  transmit power (Simulation.power_control_wifi).

The AP has a single beam (index 0), pointed at its associated STA every
snapshot (SystemWifi.point_ap_beams).
"""
import numpy as np

from sharc.antenna.antenna_beamforming_imt import AntennaBeamformingImt


class AntennaBeamformingWifi(AntennaBeamformingImt):
    """Wi-Fi AP array antenna with one beam and 0 dBi mean gain."""

    # (configuration key) -> (normalization offset [dB], peak gain [dBi])
    _cache = {}

    def __init__(self, par, azimuth: float, elevation: float):
        super().__init__(par, azimuth, elevation)
        self.norm_offset_db, self.peak_gain = self._normalization(par)

    @staticmethod
    def _key(par):
        sub = par.subarray
        return (
            str(par.element_pattern).upper(), float(par.element_max_g),
            float(par.element_phi_3db), float(par.element_theta_3db),
            float(par.element_am), float(par.element_sla_v),
            int(par.multiplication_factor), int(par.n_rows), int(par.n_columns),
            float(par.element_horiz_spacing), float(par.element_vert_spacing),
            bool(sub.is_enabled), int(sub.n_rows), float(sub.element_vert_spacing),
            float(sub.eletrical_downtilt),
        )

    def _normalization(self, par):
        """Mean gain over the sphere (beam at the local boresight) and peak
        gain, computed once per configuration. The mean gain of a steered
        beam differs little from the boresight one for small arrays."""
        key = self._key(par)
        if key not in self._cache:
            ref = AntennaBeamformingImt(par, 0.0, 0.0)
            ref.add_beam(0.0, 90.0)
            step = 1.0
            th = np.arange(step / 2, 180.0, step)
            ph = np.arange(-180.0 + step / 2, 180.0, step)
            TH, PH = np.meshgrid(th, ph, indexing="ij")
            g = ref._beam_gain_vec(PH.ravel(), TH.ravel(), np.zeros(TH.size, dtype=int))
            w = np.sin(np.radians(TH.ravel()))
            mean_db = 10 * np.log10(np.sum(w * 10 ** (g / 10)) / np.sum(w))
            peak_db = float(ref._beam_gain_vec(np.array([0.0]), np.array([90.0]), np.array([0]))[0])
            self._cache[key] = (float(mean_db), peak_db - float(mean_db))
        return self._cache[key]

    def calculate_gain(self, *args, **kwargs) -> np.array:
        return super().calculate_gain(*args, **kwargs) - self.norm_offset_db
