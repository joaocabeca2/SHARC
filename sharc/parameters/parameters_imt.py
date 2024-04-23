# -*- coding: utf-8 -*-
"""Parameters definitions for IMT systems
"""
import configparser
from dataclasses import dataclass

@dataclass
class ParametersImt:
    """Dataclass containing the IMT system parameters
    """
    topology: str = "MACROCELL"
    wrap_around: bool = False
    num_clusters: int = 1
    intersite_distance: int = 50
    minimum_separation_distance_bs_ue: int = 0
    interfered_with: bool = False
    frequency: float = 24350
    bandwidth: float = 200
    rb_bandwidth: float = 0.180
    spectral_mask: str = "IMT-2020"
    spurious_emissions: float = -13
    guard_band_ratio: float =  0.1
    bs_load_probability: float = .2
    bs_conducted_power: int = 10
    bs_height: float = 6
    bs_noise_figure: float = 10
    bs_noise_temperature: float = 290
    bs_ohmic_loss: float = 3
    ul_attenuation_factor: float = 0.4
    ul_sinr_min: float = -10
    ul_sinr_max: float = 22
    ue_k: int = 3
    ue_k_m: int = 1
    ue_indoor_percent: int = 5
    ue_distribution_type: str = "ANGLE_AND_DISTANCE"
    ue_distribution_distance: str = "RAYLEIGH"
    ue_distribution_azimuth: str = "NORMAL"
    ue_tx_power_control: bool = True
    ue_p_o_pusch: float = -95








