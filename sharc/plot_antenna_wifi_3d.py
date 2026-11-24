"""Diagrama de radiação 3D da antena do AP Wi-Fi com beamforming (Grupo 3), só ilustrativo.

Usa a própria classe do simulador (AntennaBeamformingWifi) com os parâmetros de
campaigns/WIFI_ITU_BF_BEL: arranjo 1 x 4, elementos isotrópicos (FIXED, 0 dBi),
espaçamento 0,5 lambda, sem downtilt, normalizado para ganho médio de 0 dBi na esfera.

Uso: python plot_antenna_wifi_3d.py [--out-dir scenario_figs]
"""
import argparse
import os
import sys

import numpy as np

SHARC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SHARC_DIR))

from sharc.antenna.antenna_beamforming_wifi import AntennaBeamformingWifi  # noqa: E402
from sharc.parameters.wifi.parameters_antenna_wifi import ParametersAntennaWifi  # noqa: E402

PISO = -20.0      # dBi: ganhos abaixo disso são desenhados na origem
GMAX_COR = 8.0


def num(v, d=0):
    return f"{v:.{d}f}".replace(".", ",").replace("-", "−")


def antena(az_feixe, theta_feixe):
    par = ParametersAntennaWifi(n_rows=1, n_columns=4, element_pattern="FIXED", element_max_g=0.0,
                                beamforming=True, downtilt=0.0)
    ant = AntennaBeamformingWifi(par, 0.0, 0.0)
    ant.add_beam(az_feixe, theta_feixe)
    return ant


def ganho(ant, PH, TH):
    g = ant.calculate_gain(phi_vec=PH.ravel(), theta_vec=TH.ravel(),
                           beams_l=np.zeros(PH.size, dtype=int))
    return np.asarray(g).reshape(PH.shape)


def painel(ax, ant, titulo, cmap, norm):
    th = np.linspace(0.0, 180.0, 181)
    ph = np.linspace(-180.0, 180.0, 361)
    PH, TH = np.meshgrid(ph, th)
    G = ganho(ant, PH, TH)
    r = np.maximum(G, PISO) - PISO
    t, p = np.radians(TH), np.radians(PH)
    X, Y, Z = r * np.sin(t) * np.cos(p), r * np.sin(t) * np.sin(p), r * np.cos(t)
    ax.plot_surface(X, Y, Z, facecolors=cmap(norm(G)), rstride=2, cstride=2,
                    linewidth=0, antialiased=True, shade=False, alpha=0.75)
    L = GMAX_COR - PISO + 2
    for v, c, lab in (((1, 0, 0), "k", "x"), ((0, 1, 0), "k", "y"), ((0, 0, 1), "k", "z")):
        ax.plot([0, L * v[0]], [0, L * v[1]], [0, L * v[2]], color=c, lw=0.8)
        ax.text(1.06 * L * v[0], 1.06 * L * v[1], 1.06 * L * v[2], lab, fontsize=9)
    # elementos do arranjo (ao longo de y), fora de escala
    ye = (np.arange(4) - 1.5) * 1.6
    ax.scatter(np.zeros(4), ye, np.zeros(4), color="black", s=10, depthshade=False)
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(-L, L)
    ax.set_box_aspect((1, 1, 1), zoom=1.3)
    ax.set_axis_off()
    ax.view_init(elev=28, azim=-30)
    ax.text2D(0.5, 0.98, titulo, transform=ax.transAxes, ha="center", va="top", fontsize=10)
    return G


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=os.path.join(SHARC_DIR, "scenario_figs"))
    a = p.parse_args()

    cmap = plt.get_cmap("viridis")
    norm = colors.Normalize(vmin=PISO, vmax=GMAX_COR)
    fig = plt.figure(figsize=(7.5, 13.5))
    casos = ((0.0, 90.0, "Feixe na direção broadside (azimute 0°)", "#1f6fc0", "-"),
             (40.0, 90.0, "Feixe apontado a 40° de azimute", "#d42a2a", "--"))
    picos = []
    axp = fig.add_axes([0.14, 0.08, 0.58, 0.21], projection="polar")
    ph = np.linspace(-180.0, 180.0, 721)
    for k, (az, th, tit, cor, ls) in enumerate(casos):
        ant = antena(az, th)
        ax = fig.add_axes([0.04, 0.645 - 0.285 * k, 0.76, 0.285], projection="3d")
        G = painel(ax, ant, tit, cmap, norm)
        ax.patch.set_alpha(0.0)
        ax.set_zorder(0)
        picos.append(G.max())
        gh = ganho(ant, ph, np.full_like(ph, 90.0))
        axp.plot(np.radians(ph), np.maximum(gh, PISO), color=cor, ls=ls, lw=1.6, label=f"Feixe a {num(az)}°")
    print("pico (dBi):", [round(float(g), 2) for g in picos], "| peak_gain da classe:", round(ant.peak_gain, 2))
    axp.set_ylim(PISO, GMAX_COR)
    axp.set_zorder(1)
    axp.set_yticks(np.arange(PISO, GMAX_COR + 0.1, 8))
    axp.set_yticklabels([num(v) for v in np.arange(PISO, GMAX_COR + 0.1, 8)], fontsize=7)
    axp.set_rlabel_position(100)
    axp.set_thetagrids(range(0, 360, 30), [f"{t if t <= 180 else t - 360}°".replace("-", "−") for t in range(0, 360, 30)],
                       fontsize=7)
    axp.set_title("Corte no plano horizontal (dBi)\n(eixo x = 0°, eixo y = 90°)", fontsize=10, pad=14)
    axp.legend(loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8, frameon=False)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.82, 0.42, 0.022, 0.42])
    cb = fig.colorbar(sm, cax=cax, extend="min")
    cb.set_label("Ganho (dBi)")
    cb.set_ticks(np.arange(PISO, GMAX_COR + 0.1, 4))
    cb.ax.set_yticklabels([num(v) for v in np.arange(PISO, GMAX_COR + 0.1, 4)])
    fig.suptitle("Diagrama de radiação do AP Wi-Fi com beamforming (Grupo 3)\narranjo linear 1 × 4, "
                 "elementos isotrópicos, espaçamento 0,5 λ", fontsize=11, y=0.985)
    import textwrap
    nota = (f"Ganho de pico {num(picos[0], 1)} dBi; ganho médio na esfera normalizado para 0 dBi. Raio = ganho − ({num(PISO)} dBi); "
            f"ganhos abaixo de {num(PISO)} dBi aparecem na origem. O arranjo está ao longo do eixo y. "
            "Como os elementos são isotrópicos e não há plano de terra, o diagrama é simétrico em torno do eixo "
            "do arranjo: o lóbulo principal é um \"disco\" que cobre todas as elevações e se repete para trás do AP.")
    fig.text(0.03, 0.0, textwrap.fill(nota, 105), fontsize=8, color="0.3", va="bottom")
    os.makedirs(a.out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        out = os.path.join(a.out_dir, f"diagrama_antena_wifi_bf_1x4_3d.{ext}")
        fig.savefig(out, dpi=300 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        print("gravado:", out)


if __name__ == "__main__":
    main()
