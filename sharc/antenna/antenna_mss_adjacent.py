# -*- coding: utf-8 -*-
from sharc.antenna.antenna import Antenna

import numpy as np


class AntennaMSSAdjacent(Antenna):
    """
    Implements part of EIRP mask for MSS-DC systems given in document WPGC
    as defined in the WP4C Working Document 4C/356-E
    You can choose the adjacent channel by choosing the tx power
    You need to also make sure ACLR_db = 0, otherwise SHARC's implementation will
    mess the EIRP up.
    """

    def __init__(self, frequency_MHz: float):
        super().__init__()
        self.frequency = frequency_MHz

    def calculate_gain(self, *args, **kwargs) -> np.array:
        theta = np.absolute(kwargs["off_axis_angle_vec"])

        return 20 * np.log10(self.frequency / 2e3) + 10 * np.log10(np.cos(np.deg2rad(theta)))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    frequency = 2170
    theta = np.linspace(0.01, 90, num=100000)

    antenna = AntennaMSSAdjacent(frequency)
    gain = antenna.calculate_gain(off_axis_angle_vec=theta)

    def create_plot_adj_channel(frequency, theta, chn, ax=None):
        tranls = {
            -1: 0,
            1: -55.6,
            2: -73.6,
            3: -83.6,
        }
        if chn == -1:
            ylabel = "Gain [dB]"
        else:
            ylabel = f"EIRP_{chn}"

        if ax is None:
            fig = plt.figure(facecolor='w', edgecolor='k')
            ax1 = fig.add_subplot()
        else:
            ax1 = ax

        ax1.plot(theta, gain + tranls[chn])
        ax1.grid(True)
        ax1.set_xlabel(r"Off-axis angle $\theta$ [deg]")
        ax1.set_ylabel(ylabel)

        label = f"$f = {frequency / 1e3: .3f}$ GHz"
        if chn != -1:
            label += f", channel {chn}"
        # ax1.semilogx(
        ax1.plot(
            theta, gain,
            label=label,
        )
        ax1.legend(loc="lower left")
        ax1.set_xlim((theta[0], theta[-1]))
        ax1.set_ylim((-80, 10))

        return ax1

    create_plot_adj_channel(frequency, theta, -1)
    plt.show()
    ax = create_plot_adj_channel(frequency, theta, 1)
    create_plot_adj_channel(frequency, theta, 2, ax)
    create_plot_adj_channel(frequency, theta, 3, ax)
    plt.show()
