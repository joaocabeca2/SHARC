"""
Plota em 3D uma realização (um snapshot) do cenário espacial de um caso
SHARC IMT x Wi-Fi: grid hexagonal, BSs, UEs, prédios, APs e STAs, com
linhas ligando os nós ativos (enlaces IMT BS-UE e Wi-Fi AP-STA).

O snapshot é reproduzido EXATAMENTE como na simulação: as sementes por
snapshot são geradas como em model.py (general.seed -> secondary_seeds) e o
snapshot roda o mesmo código de downlink/uplink. Nada é escrito nos
resultados: a gravação de resultados é desativada e nenhuma pasta de saída é
criada. Só as figuras são gravadas, em --out-dir.

Uso (a partir de sharc/):
    python plot_scenario_3d.py campaigns/WIFI_CoChannel/input/coch_DL_IMT_20_WIFI_50_NA_57.yaml
    python plot_scenario_3d.py <yaml> --snapshot 7 --show
    python plot_scenario_3d.py <yaml> --zoom-center 0 0 --zoom-size 400

Gera <caso>_snap<N>_overview.png/.pdf (cenário inteiro) e
<caso>_snap<N>_zoom.png/.pdf (recorte com escala vertical real).
"""
import argparse
import math
import os
import random
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SHARC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SHARC_DIR))

import sharc.simulation as simulation_module  # noqa: E402
from sharc.parameters.parameters import Parameters  # noqa: E402
from sharc.simulation_downlink import SimulationDownlink  # noqa: E402
from sharc.simulation_uplink import SimulationUplink  # noqa: E402

# Cores (IMT em azuis/preto, Wi-Fi em vermelhos, como nas figuras do relatório)
C_HEX = "0.55"
C_BUILDING = (0.6, 0.6, 0.6)
C_BS = "black"
C_UE = (0.0, 0.3, 1.0)
C_IMT_LINK = (0.0, 0.3, 1.0)
C_AP = (0.6, 0.0, 0.0)
C_STA = (1.0, 0.2, 0.2)
C_WIFI_LINK = (0.85, 0.1, 0.1)
C_INACTIVE = "0.75"


class _NoResults:
    """Substitui Results: não cria pastas nem grava arquivos."""

    def prepare_to_write(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        values = []
        setattr(self, name, values)
        return values


def run_snapshot(param_file: str, snapshot: int):
    """Reproduz o snapshot `snapshot` (1 = primeiro) do caso, sem gravar nada."""
    parameters = Parameters()
    parameters.set_file_name(param_file)
    parameters.read_params()
    if parameters.general.system != "WIFI":
        raise SystemExit("plot_scenario_3d: o caso precisa ter general.system = WIFI")
    if not 1 <= snapshot <= parameters.general.num_snapshots:
        raise SystemExit(f"plot_scenario_3d: snapshot deve estar entre 1 e "
                         f"{parameters.general.num_snapshots}")

    original_results = simulation_module.Results
    simulation_module.Results = _NoResults
    try:
        if parameters.general.imt_link == "DOWNLINK":
            sim = SimulationDownlink(parameters, param_file)
        else:
            sim = SimulationUplink(parameters, param_file)
        sim.initialize()
        sim.initialize_wifi()
    finally:
        simulation_module.Results = original_results

    # Sementes iguais às de model.py
    random.seed(parameters.general.seed)
    seeds = [random.randint(1, 2**32 - 1) for _ in range(snapshot)]

    sim.collect_results_wifi = lambda *args, **kwargs: None
    sim.snapshot(write_to_file=False, snapshot_number=snapshot, seed=seeds[-1])
    return sim, parameters


def hexagon_edges(topology):
    """Segmentos (x, y) dos setores hexagonais, como em TopologyMacrocell.plot."""
    r = topology.intersite_distance / 3
    polys = []
    for x, y, az in zip(topology.x, topology.y, topology.azimuth):
        pts = [(x, y)]
        angle = int(az - 60)
        for _ in range(6):
            px, py = pts[-1]
            pts.append((px + r * math.cos(math.radians(angle)),
                        py + r * math.sin(math.radians(angle))))
            angle += 60
        polys.append(np.array(pts))
    return polys


def building_faces(b):
    x0, x1, y0, y1, z1 = b.x_min, b.x_max, b.y_min, b.y_max, b.total_height
    v = [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    idx = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
           (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return [[v[i] for i in f] for f in idx]


def in_box(x, y, box):
    if box is None:
        return np.ones(np.shape(x), dtype=bool)
    xmin, xmax, ymin, ymax = box
    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)


def clip_segment(p0, p1, box):
    """Recorta o segmento 3D p0-p1 ao retângulo xy de box (Liang-Barsky).
    Devolve (q0, q1) ou None se o segmento estiver todo fora."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    if box is None:
        return p0, p1
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for p, q in ((-d[0], p0[0] - box[0]), (d[0], box[1] - p0[0]),
                 (-d[1], p0[1] - box[2]), (d[1], box[3] - p0[1])):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return None
    return p0 + t0 * d, p0 + t1 * d


def plot_segment(ax, p0, p1, box, **kwargs):
    seg = clip_segment(p0, p1, box)
    if seg is None:
        return False
    q0, q1 = seg
    ax.plot([q0[0], q1[0]], [q0[1], q1[1]], [q0[2], q1[2]], **kwargs)
    return True


def plot_scene(sim, parameters, title, box=None, z_scale=None, elev=28, azim=-55,
               show_inactive=True):
    """Desenha o cenário. box = (xmin, xmax, ymin, ymax) recorta a vista.
    z_scale: altura relativa do eixo z na caixa 3D (None = escala real)."""
    bs, ue = sim.bs, sim.ue
    ap, sta = sim.system.ap, sim.system.sta
    buildings = sim.system_topology.buildings

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection="3d")

    # Setores hexagonais no solo
    for poly in hexagon_edges(sim.topology):
        for a, b in zip(poly[:-1], poly[1:]):
            plot_segment(ax, (a[0], a[1], 0), (b[0], b[1], 0), box, color=C_HEX, lw=0.6)

    # Prédios
    faces = []
    for b in buildings:
        if box is None or in_box(np.array([b.x_center]), np.array([b.y_center]), box)[0]:
            faces.extend(building_faces(b))
    if faces:
        ax.add_collection3d(Poly3DCollection(
            faces, facecolor=C_BUILDING, edgecolor="0.35", linewidths=0.3, alpha=0.18))

    # Enlaces IMT (BS -> UE servidos)
    n_imt_links = 0
    for b_idx in np.where(bs.active)[0]:
        for u in sim.link.get(b_idx, []):
            if not ue.active[u]:
                continue
            n_imt_links += plot_segment(
                ax, (bs.x[b_idx], bs.y[b_idx], bs.z[b_idx]), (ue.x[u], ue.y[u], ue.z[u]),
                box, color=C_IMT_LINK, lw=0.7, alpha=0.6)

    # Enlaces Wi-Fi (AP <-> STA, ambos ativos)
    n_wifi_links = 0
    for a_idx, stas in sim.system.link.items():
        for s in stas:
            downlink = a_idx in sim.system.ap_tx and s in sim.system.sta_rx
            uplink = a_idx in sim.system.ap_rx and s in sim.system.sta_tx
            if not (downlink or uplink):
                continue
            n_wifi_links += plot_segment(
                ax, (ap.x[a_idx], ap.y[a_idx], ap.z[a_idx]), (sta.x[s], sta.y[s], sta.z[s]),
                box, color=C_WIFI_LINK, lw=1.2, alpha=0.9)

    # BSs: os 3 setores de um site ficam no mesmo ponto; um mastro e um
    # marcador por site, cheio se algum setor estiver ativo
    sites = {}
    for i in range(bs.num_stations):
        sites.setdefault((round(bs.x[i], 3), round(bs.y[i], 3)), []).append(i)
    site_xyz, site_act = [], []
    for (sx, sy), idx in sites.items():
        if not in_box(sx, sy, box):
            continue
        h = bs.z[idx[0]]
        ax.plot([sx, sx], [sy, sy], [0, h], color=C_BS, lw=1.2)
        site_xyz.append((sx, sy, h))
        site_act.append(bool(bs.active[idx].any()))
    site_xyz = np.array(site_xyz).reshape(-1, 3)
    site_act = np.array(site_act, dtype=bool)
    ax.scatter(*site_xyz[site_act].T, marker="^", s=80, color=C_BS, depthshade=False,
               label=f"Site com setor ativo ({bs.active.sum()}/{bs.num_stations} setores ativos)")
    if show_inactive and (~site_act).any():
        ax.scatter(*site_xyz[~site_act].T, marker="^", s=60, facecolor="white",
                   edgecolor=C_BS, depthshade=False, label="Site inativo")

    def nodes(st, color, marker, size, name):
        m = in_box(st.x, st.y, box)
        a = st.active & m
        ax.scatter(st.x[a], st.y[a], st.z[a], marker=marker, s=size, color=color,
                   edgecolor="black", linewidths=0.4, depthshade=False,
                   label=f"{name} ativa ({st.active.sum()}/{st.num_stations})")
        i = ~st.active & m
        if show_inactive and i.any():
            ax.scatter(st.x[i], st.y[i], st.z[i], marker=marker, s=size * 0.6,
                       color=C_INACTIVE, alpha=0.5, depthshade=False,
                       label=f"{name} inativa")

    nodes(ue, C_UE, "o", 14, "UE")
    nodes(ap, C_AP, "s", 30, "AP")
    nodes(sta, C_STA, "o", 16, "STA")

    # Entradas de legenda para enlaces e prédios
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color=C_IMT_LINK, lw=1.5),
                Line2D([], [], color=C_WIFI_LINK, lw=1.5),
                Patch(facecolor=C_BUILDING, edgecolor="0.35", alpha=0.5)]
    labels += [f"Enlace IMT (BS-UE): {n_imt_links}",
               f"Enlace Wi-Fi (AP-STA): {n_wifi_links}",
               f"Prédios ({len(buildings)} no cenário)"]

    # Limites e proporção
    if box is None:
        xs = np.concatenate([bs.x, ue.x, ap.x, sta.x])
        ys = np.concatenate([bs.y, ue.y, ap.y, sta.y])
        pad = 50
        box_lim = (xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad)
    else:
        box_lim = box
    zmax = max([b.total_height for b in buildings] + [float(bs.z.max()), 1.0]) * 1.1
    ax.set_xlim(box_lim[0], box_lim[1])
    ax.set_ylim(box_lim[2], box_lim[3])
    ax.set_zlim(0, zmax)
    dx, dy = box_lim[1] - box_lim[0], box_lim[3] - box_lim[2]
    dz = zmax if z_scale is None else z_scale * max(dx, dy)
    ax.set_box_aspect((dx, dy, dz))

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=11)
    ax.legend(handles, labels, loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plota uma realização 3D do cenário SHARC IMT x Wi-Fi")
    parser.add_argument("param_file", help="YAML do caso (pasta input/ ou cópia na pasta de saída)")
    parser.add_argument("--snapshot", type=int, default=1, help="snapshot a reproduzir (padrão 1)")
    parser.add_argument("--out-dir", default=os.path.join(SHARC_DIR, "scenario_figs"))
    parser.add_argument("--zoom-center", type=float, nargs=2, default=None, metavar=("X", "Y"),
                        help="centro do recorte (padrão: AP ativo mais próximo da origem)")
    parser.add_argument("--zoom-size", type=float, default=300.0, help="lado do recorte em m")
    parser.add_argument("--overview-z", type=float, default=0.12,
                        help="altura relativa do eixo z na vista geral (exagero vertical)")
    parser.add_argument("--no-inactive", action="store_true", help="omite nós inativos")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--show", action="store_true", help="abre janelas interativas (girar com o mouse)")
    args = parser.parse_args()

    param_file = os.path.abspath(args.param_file)
    if not args.show:
        matplotlib.use("Agg")
    os.chdir(SHARC_DIR)  # caminhos relativos do YAML (ex.: normalization_file)

    sim, parameters = run_snapshot(param_file, args.snapshot)
    name = os.path.splitext(os.path.basename(param_file))[0]
    link = "DL" if parameters.general.imt_link == "DOWNLINK" else "UL"
    channel = "adjacente" if parameters.general.enable_adjacent_channel else "co-canal"
    head = (f"{name} — snapshot {args.snapshot}\n"
            f"IMT {link}, {channel}, LF IMT={parameters.imt.bs.load_probability:g}, "
            f"LF Wi-Fi={parameters.wifi.ap.load_probability:g}, "
            f"{sim.system.ap.num_stations} APs")

    ap = sim.system.ap
    if args.zoom_center is None:
        cand = np.where(ap.active)[0] if ap.active.any() else np.arange(ap.num_stations)
        k = cand[np.argmin(np.hypot(ap.x[cand], ap.y[cand]))]
        cx, cy = float(ap.x[k]), float(ap.y[k])
    else:
        cx, cy = args.zoom_center
    half = args.zoom_size / 2
    box = (cx - half, cx + half, cy - half, cy + half)

    figs = {
        "overview": plot_scene(sim, parameters, head + " (escala vertical exagerada)",
                               z_scale=args.overview_z, show_inactive=not args.no_inactive),
        "zoom": plot_scene(sim, parameters,
                           head + f"\nrecorte {args.zoom_size:g} m x {args.zoom_size:g} m "
                                  f"em ({cx:.0f}, {cy:.0f})",
                           box=box, z_scale=None, elev=30, azim=-60,
                           show_inactive=not args.no_inactive),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    for key, fig in figs.items():
        for fmt in args.formats:
            out = os.path.join(args.out_dir, f"{name}_snap{args.snapshot}_{key}.{fmt}")
            fig.savefig(out, dpi=300 if fmt == "png" else None, bbox_inches="tight")
            print("gravado:", out)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
