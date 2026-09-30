#!/usr/bin/env python3
"""
Gera a estrutura de pastas de cada campanha de simulação:

    <CAMPANHA>/
    ├── input/     DL_LF20.yaml, DL_LF50.yaml, DL_LF90.yaml, UL_LF20.yaml, ...
    ├── output/    (resultados da simulação entram aqui)
    └── plot.py    (mostra throughput, interferência e INR)

Uso:
    python gerar_campanhas.py -b parameters_wifi.yaml                  # todas as campanhas de CAMPAIGNS
    python gerar_campanhas.py -b parameters_wifi.yaml -c MINHA_CAMP    # campanha nova, sem overrides
    python gerar_campanhas.py -b parameters_wifi.yaml -r /caminho/raiz

Para criar outra campanha, adicione uma entrada em CAMPAIGNS com os
parâmetros que mudam em relação ao YAML base.
"""
import argparse
import copy
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO (edite aqui)
# ---------------------------------------------------------------------------

LINKS = {"DL": "DOWNLINK", "UL": "UPLINK"}
OUTPUT_ROOT = "campaigns"            # output_dir = <OUTPUT_ROOT>/<CAMPANHA>/output/<DL|UL>_LF<xx>
LOAD_FACTORS = [20, 50, 90]          # em %  ->  bs_load_probability = LF/100

# nome da campanha -> {SEÇÃO: {chave: valor}}  (valores abaixo são PLACEHOLDERS)
CAMPAIGNS = {
    "WIFI_IMT_URBANO_LF_IMT": {
        "general":{
                "num_snapshots": 10000,
                "enable_cochannel": True,
                "enable_adjacent_channel": False,
             },
        "IMT": {
            "channel_model": "UMa",
            "intersite_distance": 500,
        },
    },
}

# ---------------------------------------------------------------------------
# plot.py (copiado para dentro de cada campanha)
# ---------------------------------------------------------------------------
PLOT_TEMPLATE = r'''#!/usr/bin/env python3
"""
Mostra os resultados da campanha: throughput, interferência e INR.

Uso (dentro da pasta da campanha):
    python plot.py             # CDF, abre as janelas e salva PNGs em output/plots
    python plot.py --ccdf      # CCDF (útil para INR)
    python plot.py --no-show   # só salva os PNGs

Espera encontrar em output/ uma pasta (ou arquivos) por simulação, com nome
começando em DL_LF20, UL_LF50, ... (sufixos de data são aceitos; se houver
mais de uma execução do mesmo caso, usa a mais recente).
Se os nomes dos arquivos de resultado forem diferentes, ajuste METRICS abaixo.
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"

# métrica -> (trechos do nome do arquivo, rótulo do eixo x)
METRICS = {
    "Throughput":   (["tput", "throughput"], "Throughput (Mbps)"),
    "Interferência": (["interf"],            "Interferência (dBm)"),
    "INR":          (["inr"],                "INR (dB)"),
}
EXTS = {".txt", ".csv", ".npz", ".npy"}
CASE_RE = re.compile(r"^(DL|UL)_LF(\d+)", re.IGNORECASE)


def load_array(path):
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as z:
            parts = [np.asarray(z[k], dtype=float).ravel() for k in z.files]
        data = np.concatenate(parts) if parts else np.array([])
    elif path.suffix == ".npy":
        data = np.asarray(np.load(path), dtype=float).ravel()
    else:
        try:
            data = np.loadtxt(path, comments="#", delimiter=None).ravel()
        except ValueError:
            data = np.genfromtxt(path, comments="#", delimiter=",", skip_header=1).ravel()
    data = data[np.isfinite(data)]
    return data


def find_cases():
    """{(link, lf): pasta_ou_arquivos} usando a execução mais recente de cada caso."""
    cases = {}
    if not OUT.exists():
        return cases
    for d in sorted(p for p in OUT.iterdir() if p.is_dir()):
        m = CASE_RE.match(d.name)
        if m:
            cases[(m.group(1).upper(), int(m.group(2)))] = d   # sorted -> último vence
    return cases


def files_for(case_dir, link, keys):
    files = [f for f in case_dir.rglob("*")
             if f.suffix.lower() in EXTS and any(k in f.name.lower() for k in keys)]
    # prefere arquivos do próprio enlace quando o nome traz DL/UL
    own = [f for f in files if link.lower() in f.name.lower()]
    return own or files


def curve(data, ccdf):
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, (1 - y if ccdf else y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ccdf", action="store_true", help="plota CCDF em vez de CDF")
    ap.add_argument("--no-show", action="store_true", help="não abre as janelas")
    args = ap.parse_args()

    cases = find_cases()
    if not cases:
        raise SystemExit(f"Nenhum resultado encontrado em {OUT} "
                         "(esperado: pastas DL_LF20, UL_LF50, ...)")

    links = sorted({l for l, _ in cases}, reverse=True)       # DL, UL
    fig, axes = plt.subplots(len(links), len(METRICS),
                             figsize=(5 * len(METRICS), 3.8 * len(links)),
                             squeeze=False)
    ylabel = "CCDF" if args.ccdf else "CDF"

    for r, link in enumerate(links):
        for c, (name, (keys, xlabel)) in enumerate(METRICS.items()):
            ax = axes[r][c]
            for (l, lf), d in sorted(cases.items(), key=lambda kv: kv[0][1]):
                if l != link:
                    continue
                arrays = []
                for f in files_for(d, link, keys):
                    try:
                        arrays.append(load_array(f))
                    except Exception as e:
                        print(f"aviso: não foi possível ler {f}: {e}")
                if not arrays:
                    continue
                x, y = curve(np.concatenate(arrays), args.ccdf)
                ax.plot(x, y, label=f"LF {lf}%")
            ax.set_title(f"{name} - {link}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend()
            else:
                ax.text(0.5, 0.5, "sem dados", ha="center", va="center",
                        transform=ax.transAxes)

    fig.suptitle(BASE.name)
    fig.tight_layout()
    plots = OUT / "plots"
    plots.mkdir(exist_ok=True)
    fig.savefig(plots / f"{BASE.name}_{ylabel.lower()}.png", dpi=150)
    print(f"figura salva em {plots}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
'''


def resolve_base(arg):
    """Procura o YAML base na pasta atual e, se não achar, na pasta do script."""
    p = Path(arg)
    candidates = [p] if p.is_absolute() else [Path.cwd() / p, Path(__file__).resolve().parent / p]
    for c in candidates:
        if c.is_file():
            return c
    here = Path(__file__).resolve().parent
    found = ", ".join(sorted(x.name for x in here.glob("*.y*ml"))) or "nenhum"
    raise SystemExit(
        f"YAML base '{arg}' não encontrado.\n"
        f"Procurei em: {', '.join(str(c.parent) for c in candidates)}\n"
        f"YAMLs na pasta do script: {found}\n"
        f"Use -b com o nome certo, ex.: -b {found.split(', ')[0]}"
    )


def build_yaml(base, campaign, overrides, link, lf):
    cfg = copy.deepcopy(base)
    for section, kv in overrides.items():
        cfg.setdefault(section, {}).update({k: str(v) for k, v in kv.items()})

    name = f"{link}_LF{lf}"
    gen = cfg.setdefault("GENERAL", {})
    gen["imt_link"] = LINKS[link]
    cfg.setdefault("IMT", {})["bs_load_probability"] = str(lf / 100)
    # output_dir sempre segue o nome da campanha (e o caso: DL_LF20, UL_LF50, ...)
    gen["output_dir"] = f"{OUTPUT_ROOT}/{campaign}/output/{name}"
    return name, cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-b", "--base", default="parameters.yaml", help="YAML base")
    ap.add_argument("-r", "--root", default=".", help="pasta onde as campanhas serão criadas")
    ap.add_argument("-c", "--campaign", nargs="+",
                    help="nome(s) da(s) campanha(s); padrão: todas de CAMPAIGNS")
    args = ap.parse_args()

    with open(resolve_base(args.base), encoding="utf-8") as f:
        base = yaml.safe_load(f)

    names = args.campaign or list(CAMPAIGNS)
    root = Path(args.root)

    for campaign in names:
        overrides = CAMPAIGNS.get(campaign, {})
        cdir = root / campaign
        (cdir / "input").mkdir(parents=True, exist_ok=True)
        (cdir / "output").mkdir(exist_ok=True)

        for link in LINKS:
            for lf in LOAD_FACTORS:
                name, cfg = build_yaml(base, campaign, overrides, link, lf)
                with open(cdir / "input" / f"{name}.yaml", "w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        (cdir / "plot.py").write_text(PLOT_TEMPLATE, encoding="utf-8")
        n = len(LINKS) * len(LOAD_FACTORS)
        print(f"{cdir}/  ->  {n} yamls em input/, output/ vazia, plot.py")


if __name__ == "__main__":
    main()