"""
Mapas de potência recebida (IMT e Wi-Fi) em um snapshot de um caso SHARC
IMT x Wi-Fi com CCA por potência, para ilustrar o cenário no relatório.

O snapshot é reproduzido exatamente como na campanha (mesmas sementes de
model.py). Sobre ele, uma grade de receptores Wi-Fi hipotéticos (como a STA:
indoor, antena omni 0 dBi, mesma perda ôhmica) é posta no lugar das STAs só
durante o cálculo, e as próprias rotinas do simulador calculam:
  - potência do IMT no canal Wi-Fi: imt_to_wifi_linear (ERBs em DL, UEs em UL);
  - potência do Wi-Fi: calculate_intra_wifi_coupling_loss a partir dos
    transmissores Wi-Fi que venceram a contenção no snapshot.
As variáveis aleatórias de propagação (LOS, sombreamento, BEL) são sorteadas
--draws vezes com a geometria e os feixes fixos; o mapa mostra a mediana da
potência total em cada ponto. Potências em dBm na banda do Wi-Fi (80 MHz).

Escolha do snapshot: entre os --search primeiros, um andar com pelo menos
--min-nodes nós Wi-Fi dos enlaces candidatos (vencedores e bloqueados), um ou
mais enlaces ativos, num prédio a até --max-dist m de um site IMT com
setor ativo; preferência por andares com enlaces bloqueados (pelo IMT, em DL).

Figuras (em --out-dir):
  <caso>_snap<N>_mapa_area   : vista de cima, potência do IMT na altura do andar;
  <caso>_snap<N>_mapa_andar  : planta do andar: IMT, Wi-Fi e soma, com os nós;
  <caso>_snap<N>_mapa_corte  : corte vertical ERB -> prédio, por andar.

Uso (a partir de sharc/):
    python plot_coverage_map.py campaigns/WIFI_ITU_CCA/input/coch_DL_IMT_50_WIFI_90_NA_228.yaml
    python plot_coverage_map.py <yaml> --validate      # só a verificação da grade
"""
import argparse
import os
import random
import sys

import numpy as np
import matplotlib

SHARC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SHARC_DIR))

import sharc.simulation as simulation_module  # noqa: E402
from sharc.antenna.antenna_omni import AntennaOmni  # noqa: E402
from sharc.parameters.parameters import Parameters  # noqa: E402
from sharc.plot_scenario_3d import _NoResults, hexagon_edges  # noqa: E402
from sharc.simulation_downlink import SimulationDownlink  # noqa: E402
from sharc.simulation_uplink import SimulationUplink  # noqa: E402
from sharc.station_manager import StationManager  # noqa: E402
from sharc.support.enumerations import StationType  # noqa: E402
from sharc.system.system_wifi import SystemWifi  # noqa: E402

LIM_CRIT, LIM_ED, LIM_PD = -60.0, -56.0, -76.0      # dBm em 80 MHz (ED -62 e PD -82 dBm/20 MHz)
VMIN, VMAX = -110.0, -30.0
# building_id dos pontos da grade fora do prédio da planta. Não pode ser -1: é o
# valor padrão das estações IMT (sem building_id), e UEs indoor com -1 seriam
# tratadas como no mesmo prédio do ponto (P.1238, sem BEL).
FORA = -2
C_BS, C_UE, C_AP, C_STA = "black", (0.0, 0.3, 1.0), (0.6, 0.0, 0.0), (1.0, 0.2, 0.2)
CAUSA = {  # causa do bloqueio -> (rótulo, cor, marcador)
    "imt": ("bloqueado pelo IMT (ED)", (0.0, 0.3, 1.0), "X"),
    "andar": ("bloqueado por Wi-Fi do mesmo andar", (0.85, 0.45, 0.0), "X"),
    "outro_andar": ("bloqueado por Wi-Fi de outro andar", (0.55, 0.1, 0.55), "X"),
    "outro_predio": ("bloqueado por Wi-Fi de outro prédio", "0.35", "X"),
    "ed": ("bloqueado pela energia somada (ED)", "0.1", "X"),
}

# ----------------------------------------------------------------------------
# Captura da contenção do CCA (sem alterar o simulador)
# ----------------------------------------------------------------------------
CCA = {}
_orig_draw = SystemWifi.draw_csma_links
_orig_cont = SystemWifi.cca_power_contention


def _draw(self, rg):
    links = _orig_draw(self, rg)
    CCA["sys"], CCA["links"] = self, list(links)
    return links


def _tx_node(s, link):
    ap, sta, ap_tx = link
    return (s.ap, ap) if ap_tx else (s.sta, sta)


def _cont(rx_wifi_dbm, imt_dbm, bandwidth_mhz, pd, ed):
    transmits, blocked_imt = _orig_cont(rx_wifi_dbm, imt_dbm, bandwidth_mhz, pd, ed)
    s, links = CCA["sys"], CCA["links"]
    n = len(imt_dbm)
    to_20 = 10 * np.log10(20.0 / bandwidth_mhz)
    imt_lin = np.power(10, 0.1 * np.asarray(imt_dbm, dtype=float))
    rx = np.array(rx_wifi_dbm, dtype=float).reshape(n, n)
    causa = np.array(["imt" if b else "" for b in blocked_imt], dtype=object)
    blocked = blocked_imt.copy()
    tx = np.zeros(n, dtype=bool)
    wifi_lin = np.zeros(n)
    for i in range(n):            # mesma contenção de SystemWifi.cca_power_contention
        if blocked[i]:
            continue
        tx[i] = True
        row = rx[i].copy()
        row[i] = -np.inf
        wifi_lin += np.power(10, 0.1 * row)
        energy = 10 * np.log10(np.maximum(wifi_lin + imt_lin, 1e-300)) + to_20
        by_pd = row + to_20 > pd
        novo = ~blocked & ~tx & (by_pd | (energy > ed))
        mi, ii = _tx_node(s, links[i])
        for j in np.where(novo)[0]:
            if not by_pd[j]:
                causa[j] = "ed"
                continue
            mj, jj = _tx_node(s, links[j])
            if mi.building_id[ii] != mj.building_id[jj]:
                causa[j] = "outro_predio"
            elif mi.floor[ii] != mj.floor[jj]:
                causa[j] = "outro_andar"
            else:
                causa[j] = "andar"
        blocked |= novo
    assert np.array_equal(tx, transmits)
    CCA["transmits"], CCA["causa"] = transmits.copy(), causa
    CCA["imt"] = np.asarray(imt_dbm, dtype=float).copy()
    return transmits, blocked_imt


SystemWifi.draw_csma_links = _draw
SystemWifi.cca_power_contention = staticmethod(_cont)


# ----------------------------------------------------------------------------
# Simulação
# ----------------------------------------------------------------------------
class _Sink:
    """Resultados descartados: qualquer atributo é uma lista nova."""

    def __getattr__(self, name):
        lst = []
        setattr(self, name, lst)
        return lst


def build_sim(param_file):
    parameters = Parameters()
    parameters.set_file_name(param_file)
    parameters.read_params()
    if parameters.wifi.cca_mode != "POWER":
        raise SystemExit("plot_coverage_map: o caso precisa de wifi.cca_mode = POWER")
    original = simulation_module.Results
    simulation_module.Results = _NoResults
    try:
        cls = SimulationDownlink if parameters.general.imt_link == "DOWNLINK" else SimulationUplink
        sim = cls(parameters, param_file)
        sim.initialize()
        sim.initialize_wifi()
    finally:
        simulation_module.Results = original
    sim.collect_results_wifi = lambda *a, **k: None
    sim.results = _Sink()                          # o CCA anexa contagens aos resultados
    random.seed(parameters.general.seed)          # sementes iguais às de model.py
    seeds = [random.randint(1, 2**32 - 1) for _ in range(parameters.general.num_snapshots)]
    return sim, parameters, seeds


def run(sim, seeds, snap):
    CCA.clear()
    sim.snapshot(write_to_file=False, snapshot_number=snap, seed=seeds[snap - 1])
    return dict(CCA)


def bs_sites(sim):
    """{(x, y): [setores]} e altura; site ativo se algum setor ativo."""
    sites = {}
    for i in range(sim.bs.num_stations):
        sites.setdefault((round(sim.bs.x[i], 3), round(sim.bs.y[i], 3)), []).append(i)
    return sites


def candidate_floors(sim, cca, min_nodes, max_dist):
    """Andares com >= min_nodes nós Wi-Fi dos enlaces candidatos (vencedores e
    bloqueados pelo CCA), com pelo menos um enlace ativo, num prédio a até
    max_dist m de um site com setor ativo; melhor primeiro."""
    s = sim.system
    act = [(xy, idx) for xy, idx in bs_sites(sim).items() if sim.bs.active[idx].any()]
    if not act:
        return []
    site_xy = np.array([xy for xy, _ in act])
    nos, ativos, bloq, imt = {}, {}, {}, {}
    for link, t, c, pi in zip(cca["links"], cca["transmits"], cca["causa"], cca["imt"]):
        ap = link[0]
        key = (int(s.ap.building_id[ap]), int(s.ap.floor[ap]))   # a STA está no andar do seu AP
        nos[key] = nos.get(key, 0) + 2
        ativos[key] = ativos.get(key, 0) + 2 * bool(t)
        imt[key] = max(imt.get(key, -np.inf), pi)       # IMT no transmissor de cada candidato
        if c:
            bloq.setdefault(key, []).append(c)
    out = []
    for (b, f), n in nos.items():
        if n < min_nodes or not ativos[(b, f)]:
            continue
        bd = sim.system_topology.buildings[b]
        d = np.hypot(site_xy[:, 0] - bd.x_center, site_xy[:, 1] - bd.y_center)
        k = int(np.argmin(d))
        if d[k] > max_dist:
            continue
        c = bloq.get((b, f), [])
        # preferência: enlace bloqueado pelo IMT no andar, algum bloqueio, prédio
        # com 4+ andares (corte vertical mais informativo), maior IMT nos candidatos
        score = (c.count("imt") > 0, len(c) > 0, bd.floors >= 4, imt.get((b, f), -np.inf), n)
        out.append((score, b, f, tuple(site_xy[k]), float(d[k])))
    return sorted(out, reverse=True)


# ----------------------------------------------------------------------------
# Grade de receptores e potências
# ----------------------------------------------------------------------------
def make_grid(sim, x, y, z, building_id, floor, active=None, indoor=True):
    n = len(x)
    g = StationManager(n)
    g.station_type = StationType.WIFI_STA
    g.x, g.y = np.asarray(x, float), np.asarray(y, float)
    g.z = np.asarray(z, float) * np.ones(n)
    g.height = g.z.copy()
    g.indoor = np.asarray(indoor, dtype=bool) * np.ones(n, dtype=bool)
    g.building_id = np.asarray(building_id, int) * np.ones(n, dtype=int)
    g.floor = np.asarray(floor, int) * np.ones(n, dtype=int)
    g.azimuth, g.elevation = np.zeros(n), np.zeros(n)
    g.active = np.ones(n, dtype=bool) if active is None else np.array(active, dtype=bool)
    g.antenna = [AntennaOmni(0) for _ in range(n)]
    g.bandwidth = sim.param_system.bandwidth * np.ones(n)
    g.center_freq = sim.param_system.frequency * np.ones(n)
    g.noise_figure = sim.param_system.sta.noise_figure * np.ones(n)
    g.tx_power = np.full(n, -np.inf)
    for a in ("rx_power", "rx_interference", "ext_interference"):
        setattr(g, a, np.full(n, -500.0))
    return g


def _forget_path_loss(sim, station):
    sim.clear_wifi_coupling_cache(station)


def imt_lin(sim, grid):
    """Uma realização: potência do IMT [mW] em cada ponto (rotina do simulador)."""
    sta = sim.system.sta
    sim.system.sta = grid
    try:
        _forget_path_loss(sim, grid)
        _, lin = sim.imt_to_wifi_linear(np.array([], dtype=int), np.arange(grid.num_stations))
    finally:
        sim.system.sta = sta
    return lin


_GAINS = {}


def wifi_coupling_loss(sim, grid, mgr):
    """Uma realização de calculate_intra_wifi_coupling_loss(grid, mgr) [mgr x grid],
    com os ganhos de antena (determinísticos) calculados uma vez por grade; só a
    perda de propagação é sorteada de novo. Conferido contra a rotina do simulador
    em validate()."""
    key = (id(grid), id(mgr))
    if key not in _GAINS:
        _GAINS[key] = (sim.calculate_gains(mgr, grid),     # mgr -> grade  [mgr x grid]
                       sim.calculate_gains(grid, mgr))     # grade -> mgr  [grid x mgr]
    g_tx, g_rx = _GAINS[key]
    pl = sim.propagation_system.get_loss(sim.parameters, sim.parameters.wifi.frequency,
                                         grid, mgr, g_rx, g_tx)               # [grid x mgr]
    loss = (sim.parameters.wifi.ap.ohmic_loss + sim.parameters.wifi.sta.ohmic_loss
            + sim.parameters.wifi.sta.body_loss)
    return pl.T - g_tx - g_rx.T + loss


def wifi_lin(sim, grid):
    """Uma realização: potência dos transmissores Wi-Fi ativos [mW] em cada ponto."""
    s = sim.system
    tot = np.zeros(grid.num_stations)
    for mgr, tx in ((s.ap, s.ap_tx), (s.sta, s.sta_tx)):
        if len(tx) == 0:
            continue
        cl = wifi_coupling_loss(sim, grid, mgr)
        p = np.array([mgr.tx_power[i] for i in tx], dtype=float)           # tx_power é um dict
        tot += np.sum(np.power(10, 0.1 * (p[:, None] - cl[tx])), axis=0)
    return tot


CACHE = {"dir": None, "tag": ""}      # campos já calculados (main preenche: pasta e caso+snapshot)


def median_dbm(fn, sim, grid, draws):
    """Mediana em dBm de draws realizações de fn em cada ponto da grade. Com
    CACHE["dir"], o resultado fica gravado (chave: caso, snapshot, função,
    realizações e a própria grade), e uma nova figura só com mudança de estilo
    não recalcula nada."""
    path = None
    if CACHE["dir"]:
        import hashlib
        h = hashlib.sha1(f"{CACHE['tag']}|{fn.__name__}|{draws}".encode())
        for a in (grid.x, grid.y, grid.z, grid.building_id, grid.floor, grid.indoor):
            h.update(np.ascontiguousarray(a).tobytes())
        path = os.path.join(CACHE["dir"], h.hexdigest() + ".npy")
        if os.path.exists(path):
            return np.load(path)
    with np.errstate(divide="ignore"):
        z = np.median([10 * np.log10(fn(sim, grid)) for _ in range(draws)], axis=0)
    if path:
        np.save(path, z)
    return z


def validate(sim):
    """A grade (clone das STAs reais) dá exatamente o resultado das STAs reais."""
    s = sim.system
    sta = s.sta
    clone = make_grid(sim, sta.x, sta.y, sta.z, sta.building_id, sta.floor, active=sta.active)
    rng = sim.propagation_system.random_number_gen
    idx = np.where(sta.active)[0]
    if len(idx) == 0:
        return 0.0, 0.0
    _forget_path_loss(sim, sta)
    st = rng.get_state()
    _, ref = sim.imt_to_wifi_linear(np.array([], dtype=int), idx)
    rng.set_state(st)
    s.sta = clone
    try:
        _forget_path_loss(sim, clone)
        _, got = sim.imt_to_wifi_linear(np.array([], dtype=int), idx)
    finally:
        s.sta = sta
    d_imt = np.max(np.abs(10 * np.log10(ref[idx]) - 10 * np.log10(got[idx])))
    st = rng.get_state()
    ref = sim.calculate_intra_wifi_coupling_loss(sta, s.ap)
    rng.set_state(st)
    got = sim.calculate_intra_wifi_coupling_loss(clone, s.ap)
    m = np.ix_(np.where(s.ap.active)[0], idx)
    d_wifi = np.max(np.abs(ref[m] - got[m]))
    # versão com ganhos guardados (wifi_coupling_loss) x rotina do simulador
    rng.set_state(st)
    got2 = wifi_coupling_loss(sim, clone, s.ap)
    d_wifi = max(d_wifi, np.max(np.abs(ref[m] - got2[m])))
    _GAINS.clear()
    print(f"validação: |dif| máx IMT->STA {d_imt:.2e} dB ({len(idx)} STAs); "
          f"AP->STA {d_wifi:.2e} dB")
    return d_imt, d_wifi


# ----------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------
def smooth(Z, sigma=1.5, regions=None):
    """Suavização leve (só para traçar os contornos): a mediana por ponto tem
    ruído de amostragem, que deixaria as curvas serrilhadas.
    regions: rótulo por ponto (ex.: prédio ou fora); cada região é suavizada só
    com os próprios pontos, para não borrar o degrau da BEL nas paredes."""
    from scipy.ndimage import gaussian_filter
    if np.isnan(Z).any():
        return Z
    if regions is None:
        return gaussian_filter(Z, sigma, mode="nearest")
    out = np.empty_like(Z)
    for r in np.unique(regions):
        m = (regions == r).astype(float)
        num_ = gaussian_filter(Z * m, sigma, mode="nearest")
        den = gaussian_filter(m, sigma, mode="nearest")
        sel = regions == r
        out[sel] = num_[sel] / den[sel]
    return out


def contours(ax, X, Y, Z, levels, regions=None):
    import matplotlib.pyplot as plt  # noqa: F401
    sty = {LIM_CRIT: ("-", 1.6), LIM_ED: ("--", 1.4), LIM_PD: (":", 1.6)}
    Zs = smooth(Z, regions=regions)
    for lv in levels:
        if np.nanmax(Z) > lv > np.nanmin(Z):
            ax.contour(X, Y, Zs, levels=[lv], colors="black", linestyles=sty[lv][0],
                       linewidths=sty[lv][1])


def contour_handles(levels, ref=True):
    """ref=False: rótulo só com o nível (sem citar critério ou limiar do CCA)."""
    from matplotlib.lines import Line2D
    lab = {LIM_CRIT: "−60 dBm (critério de interferência; também ED −66 dBm/20 MHz)",
           LIM_ED: "−56 dBm (CCA, detecção de energia; −62 dBm/20 MHz)",
           LIM_PD: "−76 dBm (CCA, preâmbulo; −82 dBm/20 MHz)"}
    sty = {LIM_CRIT: "-", LIM_ED: "--", LIM_PD: ":"}
    return [Line2D([], [], color="black", ls=sty[lv], lw=1.5,
                   label=lab[lv] if ref else f"{num(lv)} dBm") for lv in levels]


def draw_buildings(ax, buildings, highlight=None, box=None):
    from matplotlib.patches import Rectangle
    for b in buildings:
        if box and (b.x_max < box[0] or b.x_min > box[1] or b.y_max < box[2] or b.y_min > box[3]):
            continue
        hl = b.build_id == highlight
        ax.add_patch(Rectangle((b.x_min, b.y_min), b.width, b.length, fill=False,
                               edgecolor=(0.85, 0.1, 0.1) if hl else "0.35", lw=2.0 if hl else 0.6))


def floor_z(bd, f, sim):
    return f * bd.floor_height + sim.param_system.sta.height


def fig_area(sim, info, draws, res, size, sensor=False):
    """sensor=False: receptor Wi-Fi indoor (com BEL) em todos os pontos.
    sensor=True: sensor de potência omni 0 dBi; dentro dos prédios (até a altura
    de cada um) é indoor, com BEL e o building_id do prédio, como um nó Wi-Fi
    ali; fora dos prédios é outdoor, sem BEL."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    b, f, site, dist = info["b"], info["f"], info["site"], info["dist"]
    bd = sim.system_topology.buildings[b]
    size = max(size, dist + 160)
    cx, cy = (bd.x_center + site[0]) / 2, (bd.y_center + site[1]) / 2
    xs = np.arange(cx - size / 2, cx + size / 2 + res / 2, res)
    ys = np.arange(cy - size / 2, cy + size / 2 + res / 2, res)
    X, Y = np.meshgrid(xs, ys)
    z = floor_z(bd, f, sim)
    bid = np.full(X.size, FORA)
    if sensor:
        for bb in sim.system_topology.buildings:
            if bb.floors > f:                     # o ponto está dentro do prédio nesta altura
                dentro = (X.ravel() >= bb.x_min) & (X.ravel() <= bb.x_max) &                     (Y.ravel() >= bb.y_min) & (Y.ravel() <= bb.y_max)
                bid[dentro] = bb.build_id
    grid = make_grid(sim, X.ravel(), Y.ravel(), z, bid, f, indoor=(bid != FORA) if sensor else True)
    Z = median_dbm(imt_lin, sim, grid, draws).reshape(X.shape)
    box = (xs[0], xs[-1], ys[0], ys[-1])

    fig, ax = plt.subplots(figsize=(9.2, 8))
    im = ax.pcolormesh(X, Y, Z, cmap="Blues", vmin=VMIN, vmax=VMAX, shading="auto")
    contours(ax, X, Y, Z, (LIM_CRIT,), regions=bid.reshape(X.shape) if sensor else None)
    for poly in hexagon_edges(sim.topology):
        ax.plot(poly[:, 0], poly[:, 1], color="0.55", lw=0.6)
    # mapa da área: todo ponto já tem BEL, então só o prédio da planta é desenhado;
    # no mapa tipo sensor os prédios marcam onde a BEL se aplica
    draw_buildings(ax, sim.system_topology.buildings if sensor else [bd], highlight=b, box=box)
    h = []
    for (sx, sy), idx in bs_sites(sim).items():
        if box[0] <= sx <= box[1] and box[2] <= sy <= box[3]:
            act = sim.bs.active[idx].any()
            ax.plot(sx, sy, "^", ms=11, mfc=C_BS if act else "white", mec=C_BS, mew=1.2)
    h.append(Line2D([], [], ls="", marker="^", ms=10, mfc=C_BS, mec=C_BS, label="Site IMT com setor ativo"))
    ue = sim.ue
    u = np.where(ue.active & (ue.x >= box[0]) & (ue.x <= box[1]) & (ue.y >= box[2]) & (ue.y <= box[3]))[0]
    ax.plot(ue.x[u], ue.y[u], "o", ms=5, mfc=C_UE, mec="black", mew=0.4)
    h.append(Line2D([], [], ls="", marker="o", ms=5, mfc=C_UE, mec="black", mew=0.4,
                    label="UE ativa" + (" (transmite)" if sim.parameters.general.imt_link == "UPLINK" else "")))
    s = sim.system
    for mgr, mk, col, nm in ((s.ap, "s", C_AP, "AP"), (s.sta, "o", C_STA, "STA")):
        a = np.where(mgr.active & (mgr.x >= box[0]) & (mgr.x <= box[1]) & (mgr.y >= box[2]) & (mgr.y <= box[3]))[0]
        ax.plot(mgr.x[a], mgr.y[a], mk, ms=4, mfc=col, mec="black", mew=0.3)
        h.append(Line2D([], [], ls="", marker=mk, ms=4, mfc=col, mec="black", mew=0.3,
                        label=f"{nm} ativo (qualquer andar)"))
    h.append(Line2D([], [], color=(0.85, 0.1, 0.1), lw=2.0, label=f"Prédio da planta ({bd.floors} andares)"))
    h += contour_handles((LIM_CRIT,), ref=False)
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    colorbar(fig, im, ax, "Potência do IMT no canal Wi-Fi (dBm em 80 MHz)", shrink=0.85, pad=0.02)
    ax.legend(handles=h, loc=info.get("leg_loc", "upper left"), fontsize=7.5, framealpha=0.92)
    if sensor:
        ax.set_title(f"{info['titulo']}\nPotência do IMT num sensor a {num(z, 1)} m de altura ({rot_andar(f)}): "
                     "dentro dos prédios com BEL, fora sem BEL", fontsize=10)
        txt = (f"Sensor de potência com antena omni 0 dBi. Dentro dos prédios com {f + 1} ou mais andares, é tratado "
               "como um nó Wi-Fi naquele prédio (indoor, com perda de entrada em edifício, BEL); fora deles, outdoor, "
               "sem BEL. Prédios no caminho não atenuam (só a probabilidade de visada do UMa). "
               f"Mediana de {draws} realizações de LOS, sombreamento e BEL, com posições e feixes do snapshot fixos.")
    else:
        ax.set_title(f"{info['titulo']}\nPotência do IMT recebida por um receptor Wi-Fi indoor a "
                     f"{num(z, 1)} m de altura ({rot_andar(f)})", fontsize=10)
        txt = nota(draws)
    fig.text(0.01, 0.005, txt, fontsize=7, color="0.3", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def num(x, d=None):
    """Número com vírgula decimal."""
    return (f"{x:g}" if d is None else f"{x:.{d}f}").replace(".", ",").replace("-", "−")


def colorbar(fig, im, ax, label, extend="min", **kw):
    cb = fig.colorbar(im, ax=ax, extend=extend, **kw)
    cb.set_ticks(np.arange(VMIN, VMAX + 1, 20))
    cb.set_label(label, fontsize=8)
    return cb


def rot_andar(f):
    return "térreo" if f == 0 else f"{f}º andar"


def nota(draws):
    return (f"Receptor hipotético como a STA (indoor, antena omni 0 dBi). Mediana de {draws} realizações de "
            "LOS, sombreamento e perda de entrada em edifício, com posições, feixes e transmissores do snapshot fixos.")


def fig_floor(sim, info, cca, draws, res):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    b, f = info["b"], info["f"]
    bd = sim.system_topology.buildings[b]
    xs = np.arange(bd.x_min + res / 2, bd.x_max, res)
    ys = np.arange(bd.y_min + res / 2, bd.y_max, res)
    X, Y = np.meshgrid(xs, ys)
    grid = make_grid(sim, X.ravel(), Y.ravel(), floor_z(bd, f, sim), b, f)
    zi = median_dbm(imt_lin, sim, grid, draws).reshape(X.shape)
    zw = median_dbm(wifi_lin, sim, grid, draws).reshape(X.shape)
    with np.errstate(divide="ignore"):
        zt = 10 * np.log10(10 ** (0.1 * zi) + 10 ** (0.1 * zw))   # soma das medianas, ver nota

    s = sim.system
    # três painéis empilhados (largura de uma página A4), posições em polegadas
    W, left, pw, cbw = 7.5, 0.85, 5.6, 0.16
    ph = pw * bd.length / bd.width
    if ph > 3.0:                                   # prédio "alto": limita a altura do painel
        ph, pw = 3.0, 3.0 * bd.width / bd.length
        left = (W - pw - 1.0) / 2                   # centraliza painel + barra de cores
    top, gap, xlab, leg = 0.95, 0.32, 0.5, 1.95
    H = top + 3 * (ph + gap) + xlab + leg
    fig = plt.figure(figsize=(W, H))
    axs = []
    for k in range(3):
        y0 = H - top - gap - k * (ph + gap) - ph
        axs.append(fig.add_axes((left / W, y0 / H, pw / W, ph / H), sharex=axs[0] if axs else None))
    paineis = ((zi, "Blues", "IMT", (LIM_CRIT,)),
               (zw, "Reds", "Wi-Fi (transmissores ativos)", (LIM_CRIT,)),
               (zt, "Purples", "IMT + Wi-Fi", (LIM_CRIT,)))
    on_floor = lambda m, i: m.building_id[i] == b and m.floor[i] == f      # noqa: E731
    for ax, (Z, cmap, nome, lv) in zip(axs, paineis):
        im = ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=VMIN, vmax=VMAX, shading="auto")
        contours(ax, X, Y, Z, lv)
        ax.add_patch(plt.Rectangle((bd.x_min, bd.y_min), bd.width, bd.length, fill=False, lw=2.0, ec=(0.85, 0.1, 0.1)))
        # enlaces que transmitem neste andar
        for link, t in zip(cca["links"], cca["transmits"]):
            ap, st, ap_tx = link
            if not on_floor(s.ap, ap):
                continue
            if t:
                ax.plot([s.ap.x[ap], s.sta.x[st]], [s.ap.y[ap], s.sta.y[st]], color=(0.85, 0.1, 0.1), lw=1.2)
                for mgr, i, mk, col in ((s.ap, ap, "s", C_AP), (s.sta, st, "o", C_STA)):
                    tx = (mgr is s.ap) == ap_tx
                    ax.plot(mgr.x[i], mgr.y[i], mk, ms=9 if tx else 7, mfc=col if tx else "white",
                            mec=col if not tx else "black", mew=1.6 if not tx else 0.6)
        for link, c in zip(cca["links"], cca["causa"]):
            ap, st, ap_tx = link
            if c and on_floor(s.ap, ap):
                ax.plot([s.ap.x[ap], s.sta.x[st]], [s.ap.y[ap], s.sta.y[st]], color="0.3", lw=1.0, ls="--")
                ax.plot(s.ap.x[ap], s.ap.y[ap], "s", ms=7, mfc="0.85", mec="0.3", mew=0.8)
                ax.plot(s.sta.x[st], s.sta.y[st], "o", ms=7, mfc="0.85", mec="0.3", mew=0.8)
                mgr, i = (s.ap, ap) if ap_tx else (s.sta, st)
                lab, col, mk = CAUSA[c]
                ax.plot(mgr.x[i], mgr.y[i], mk, ms=10, mfc=col, mec="white", mew=0.8)
        # transmissores nos andares vizinhos do mesmo prédio
        for mgr, tx in ((s.ap, s.ap_tx), (s.sta, s.sta_tx)):
            for i in tx:
                if mgr.building_id[i] == b and abs(int(mgr.floor[i]) - f) == 1:
                    ax.plot(mgr.x[i], mgr.y[i], "^" if mgr.floor[i] > f else "v", ms=8,
                            mfc="none", mec="0.15", mew=1.1)
        ax.set_aspect("equal")
        ax.set_xlim(bd.x_min, bd.x_max)
        ax.set_ylim(bd.y_min, bd.y_max)
        ax.set_ylabel("y (m)")
        ax.set_title(nome, fontsize=10)
        pos = ax.get_position()
        cax = fig.add_axes((pos.x1 + 0.15 / W, pos.y0, cbw / W, pos.height))
        colorbar(fig, im, None, "dBm em 80 MHz", extend="neither", cax=cax)
    for ax in axs[:-1]:
        ax.tick_params(labelbottom=False)
    axs[-1].set_xlabel("x (m)")
    h = [Line2D([], [], ls="", marker="s", ms=8, mfc=C_AP, mec="black", label="AP transmitindo"),
         Line2D([], [], ls="", marker="o", ms=8, mfc=C_STA, mec="black", label="STA transmitindo"),
         Line2D([], [], ls="", marker="s", ms=7, mfc="white", mec=C_AP, mew=1.6, label="AP recebendo"),
         Line2D([], [], ls="", marker="o", ms=7, mfc="white", mec=C_STA, mew=1.6, label="STA recebendo"),
         Line2D([], [], color=(0.85, 0.1, 0.1), lw=1.2, label="Enlace Wi-Fi ativo"),
         Line2D([], [], color="0.3", lw=1.0, ls="--", marker="s", ms=6, mfc="0.85", mec="0.3",
                label="Enlace bloqueado pelo CCA (não transmite)")]
    usados = {c for link, c in zip(cca["links"], cca["causa"]) if c and on_floor(s.ap, link[0])}
    h += [Line2D([], [], ls="", marker=CAUSA[c][2], ms=9, mfc=CAUSA[c][1], mec="white",
                 label="Transmissor " + CAUSA[c][0]) for c in CAUSA if c in usados]
    h += [Line2D([], [], ls="", marker="^", ms=8, mfc="none", mec="0.15", label="Transmissor no andar de cima"),
          Line2D([], [], ls="", marker="v", ms=8, mfc="none", mec="0.15", label="Transmissor no andar de baixo")]
    h += contour_handles((LIM_CRIT,), ref=False)
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, (leg - 0.05) / H), ncol=2, fontsize=7.5,
               framealpha=0.95)
    fig.suptitle(f"{info['titulo'].replace(' — IMT LF', chr(10) + 'IMT LF')}\nPlanta do {rot_andar(f)} do prédio ({bd.width:.0f} m × {bd.length:.0f} m, "
                 f"{bd.floors} andares), a {info['dist']:.0f} m do site IMT mais próximo com setor ativo",
                 fontsize=9.5, y=1 - 0.1 / H, va="top")
    fig.text(0.01, 0.005, nota(draws) + " Soma = soma das medianas de IMT e Wi-Fi. Contornos traçados sobre o mapa levemente suavizado. "
             "Os enlaces e bloqueios marcados são os da contenção no snapshot, que usa um único sorteio da propagação; "
             "por isso um transmissor bloqueado pelo IMT pode estar num ponto onde a mediana da potência do IMT é mais baixa.", fontsize=7, color="0.3", wrap=True)
    return fig


def fig_cut(sim, info, draws, res):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    b, f, site = info["b"], info["f"], info["site"]
    bd = sim.system_topology.buildings[b]
    v = np.array([bd.x_center - site[0], bd.y_center - site[1]])
    d0 = np.linalg.norm(v)
    u = v / d0
    # trecho do corte dentro do prédio (retângulo alinhado aos eixos)
    with np.errstate(divide="ignore"):
        tx = sorted(((bd.x_min - site[0]) / u[0], (bd.x_max - site[0]) / u[0])) if u[0] else (-np.inf, np.inf)
        ty = sorted(((bd.y_min - site[1]) / u[1], (bd.y_max - site[1]) / u[1])) if u[1] else (-np.inf, np.inf)
    t_in, t_out = max(tx[0], ty[0]), min(tx[1], ty[1])
    t = np.arange(res / 2, t_out + 25, res)
    nf = bd.floors
    T, F = np.meshgrid(t, np.arange(nf))
    px, py = site[0] + u[0] * T.ravel(), site[1] + u[1] * T.ravel()
    dentro = (T.ravel() >= t_in) & (T.ravel() <= t_out)
    zz = F.ravel() * bd.floor_height + sim.param_system.sta.height
    grid = make_grid(sim, px, py, zz, np.where(dentro, b, FORA), F.ravel())
    zi = median_dbm(imt_lin, sim, grid, draws).reshape(T.shape)
    zw = median_dbm(wifi_lin, sim, grid, draws).reshape(T.shape)
    zw[~dentro.reshape(T.shape)] = np.nan

    edges_t = np.concatenate([t - res / 2, [t[-1] + res / 2]])
    edges_z = np.arange(nf + 1) * bd.floor_height
    fig, axs = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
    bs_h = float(sim.bs.height[0]) if np.ndim(sim.bs.height) else float(sim.bs.height)
    for ax, Z, cmap, nome, lv in ((axs[0], zi, "Blues", "Potência do IMT", (LIM_CRIT,)),
                                  (axs[1], zw, "Reds", "Potência do Wi-Fi (transmissores ativos)",
                                   (LIM_CRIT, LIM_PD))):
        im = ax.pcolormesh(edges_t, edges_z, Z, cmap=cmap, vmin=VMIN, vmax=VMAX)
        Tc, Zc = np.meshgrid(t, (np.arange(nf) + 0.5) * bd.floor_height)
        contours(ax, Tc, Zc, Z, lv)
        ax.add_patch(plt.Rectangle((t_in, 0), t_out - t_in, nf * bd.floor_height, fill=False, lw=1.5))
        for k in range(1, nf):
            ax.plot([t_in, t_out], [k * bd.floor_height] * 2, color="0.25", lw=0.5)
        ax.add_patch(plt.Rectangle((t_in, f * bd.floor_height), t_out - t_in, bd.floor_height,
                                   fill=False, lw=1.8, ls="--", ec=(0.85, 0.1, 0.1)))
        ax.plot([0, 0], [0, bs_h], color=C_BS, lw=1.5)
        ax.plot(0, bs_h, "^", ms=11, color=C_BS)
        ax.set_ylim(0, max(bs_h, nf * bd.floor_height) + 3)
        ax.set_xlim(-5, edges_t[-1])
        ax.set_ylabel("altura (m)")
        ax.set_title(nome, fontsize=10)
        colorbar(fig, im, ax, "dBm em 80 MHz", pad=0.01)
    axs[1].set_xlabel("distância horizontal ao site IMT ao longo do corte (m)")
    h = [Line2D([], [], ls="", marker="^", ms=10, color=C_BS, label=f"Site IMT ({bs_h:.0f} m)"),
         Line2D([], [], color="black", lw=1.5, label="Prédio e lajes"),
         Line2D([], [], color=(0.85, 0.1, 0.1), lw=1.8, ls="--", label=f"Andar da planta ({rot_andar(f)})")]
    h += contour_handles((LIM_CRIT, LIM_PD))
    axs[0].legend(handles=h, loc="upper right", fontsize=7.5, framealpha=0.92)
    fig.suptitle(f"{info['titulo']}\nCorte vertical do site IMT ao prédio da planta; "
                 "cada faixa é um andar (receptor a 1,5 m do piso)", fontsize=10.5)
    fig.text(0.01, 0.005, nota(draws) + " Fora do prédio, o receptor hipotético também é indoor (com BEL).",
             fontsize=7, color="0.3", wrap=True)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("param_file")
    p.add_argument("--snapshot", type=int, help="usa este snapshot (sem busca)")
    p.add_argument("--search", type=int, default=100, help="snapshots examinados na busca")
    p.add_argument("--min-nodes", type=int, default=3)
    p.add_argument("--max-dist", type=float, default=200.0)
    p.add_argument("--draws", type=int, default=101, help="realizações para a mediana")
    p.add_argument("--area-size", type=float, default=300.0)
    p.add_argument("--out-dir", default=os.path.join(SHARC_DIR, "scenario_figs"))
    p.add_argument("--validate", action="store_true", help="só valida a grade e sai")
    p.add_argument("--figs", nargs="+", help="só estas figuras (ex.: mapa_andar mapa_area_sensor)")
    a = p.parse_args()
    matplotlib.use("Agg")
    param_file = os.path.abspath(a.param_file)
    case = os.path.splitext(os.path.basename(param_file))[0]
    import time
    t0 = time.time()
    sim, par, seeds = build_sim(param_file)

    best = None
    for n in range(1, (a.snapshot or a.search) + 1):
        cca = run(sim, seeds, n)
        if a.snapshot and n != a.snapshot:
            continue
        c = candidate_floors(sim, cca, a.min_nodes, a.max_dist)
        if c and (best is None or c[0][0] > best[0][0]):
            best = (c[0], n)
    if best is None:
        raise SystemExit("nenhum andar atende aos critérios; aumente --search ou --max-dist")
    (score, b, f, site, dist), n = best
    # O gerador da propagação continua de um snapshot para o outro: o snapshot n
    # só é reproduzido rodando 1..n em sequência a partir de uma simulação nova.
    sim, par, seeds = build_sim(param_file)
    for k in range(1, n + 1):
        cca = run(sim, seeds, k)
    s = sim.system
    print(f"{case}: snapshot {n}, prédio {b}, {rot_andar(f)}, site a {dist:.0f} m; "
          f"enlaces: {len(cca['links'])} candidatos, {int(cca['transmits'].sum())} transmitem")
    print(f"tempo: busca e reprodução do snapshot {time.time() - t0:.0f} s")
    d_imt, d_wifi = validate(sim)
    if max(d_imt, d_wifi) > 1e-6:
        raise SystemExit("validação falhou: a grade não reproduz as STAs")
    if a.validate:
        return

    link = "DL" if par.general.imt_link == "DOWNLINK" else "UL"
    canal = "cocanal" if par.general.enable_cochannel else "canal adjacente"
    CACHE["dir"] = os.path.join(a.out_dir, "cache")
    CACHE["tag"] = f"{param_file}|{n}"
    os.makedirs(CACHE["dir"], exist_ok=True)
    info = dict(b=b, f=f, site=site, dist=dist, leg_loc="lower right" if link == "DL" else "upper left",
                titulo=f"IMT em {link}, {canal}, CCA por potência (ED {num(par.wifi.cca_ed_threshold)} dBm/20 MHz) — IMT LF {num(par.imt.bs.load_probability)}, "
                       f"Wi-Fi LF {num(par.wifi.ap.load_probability)}, {s.ap.num_stations} APs — snapshot {n}")
    os.makedirs(a.out_dir, exist_ok=True)
    figs = (("mapa_andar", lambda: fig_floor(sim, info, cca, a.draws, 0.5)),
            ("mapa_corte", lambda: fig_cut(sim, info, a.draws, 1.0)),
            ("mapa_area", lambda: fig_area(sim, info, a.draws, 2.0, a.area_size)),
            ("mapa_area_sensor", lambda: fig_area(sim, info, a.draws, 2.0, a.area_size, sensor=True)))
    for nome, make in figs:
        if a.figs and nome not in a.figs:
            continue
        t1 = time.time()
        fig = make()
        print(f"tempo: {nome} {time.time() - t1:.0f} s")
        for ext in ("png", "pdf"):
            out = os.path.join(a.out_dir, f"{case}_snap{n}_{nome}.{ext}")
            fig.savefig(out, dpi=300 if ext == "png" else None, bbox_inches="tight", facecolor="white")
            print("gravado:", out)
        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__ == "__main__":
    main()
