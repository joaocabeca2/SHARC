"""
Monitor de campanhas SHARC: percentual, tempo decorrido e ETA de cada caso,
e ETA da campanha inteira. Funciona só lendo as pastas de saída (não precisa
de nenhuma mudança no simulador nem no runner) e pode ser aberto e fechado a
qualquer momento, inclusive com a campanha já rodando.

Como o progresso é obtido:
  - concluído: o SHARC grava simulation_log_*.yaml só ao terminar o caso;
  - percentual: imt_{dl|ul}_tx_power.csv recebe, em média,
    (num. de BS) x ue.k x (carga do IMT) linhas por snapshot; as linhas são
    contadas de forma incremental (só os bytes novos a cada atualização).
    Os CSVs são gravados a cada 10 snapshots, então o valor anda em degraus.
  - ETA do caso: ritmo desde a criação da pasta de saída do caso;
  - ETA da campanha: simula a fila de processos (mesma ordem do runner),
    estimando a duração dos casos pendentes pelos já iniciados com o mesmo
    número de APs e carga do Wi-Fi.

Uso (a partir de sharc/):
    python campaign_monitor.py                         # janela, campanhas padrão
    python campaign_monitor.py WIFI_CoChannel           # só uma campanha
    python campaign_monitor.py --text                   # tabela no terminal
    python campaign_monitor.py --text --once            # imprime uma vez e sai
    python campaign_monitor.py --interval 10 --workers 22
"""
import argparse
import glob
import os
import re
import sys
import time
from datetime import datetime, timedelta

WORKFOLDER = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CAMPAIGNS = ["WIFI_CoChannel", "WIFI_Adjacent"]
STALL_SECONDS = 15 * 60   # sem gravar nada por 15 min -> "parado?"


# --------------------------------------------------------------------------
# Leitura dos parâmetros do caso (YAML de entrada)
# --------------------------------------------------------------------------
_KEY_RE = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:(.*)$")


def read_yaml_keys(path):
    """'secao.sub.chave' -> valor (texto), por indentação; ignora comentários."""
    values, stack = {}, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            m = _KEY_RE.match(line)
            if not m:
                continue
            indent = len(m.group(1))
            while stack and stack[-1][0] >= indent:
                stack.pop()
            stack.append((indent, m.group(2)))
            value = m.group(3).split("#")[0].strip()
            if value:
                values[".".join(k for _, k in stack)] = value
    return values


def wifi_case_cost(name):
    """Mesmo critério de ordenação do runner: num_aps x carga do Wi-Fi."""
    m = re.search(r"_WIFI_(\d+)_NA_(\d+)$", name)
    return int(m.group(2)) * int(m.group(1)) if m else 0


class Case:
    """Um caso (YAML de entrada) e o estado da sua pasta de saída."""

    def __init__(self, campaign, yaml_path):
        self.campaign = campaign
        self.name = os.path.splitext(os.path.basename(yaml_path))[0]
        keys = read_yaml_keys(yaml_path)
        self.snapshots = int(float(keys.get("general.num_snapshots", "0")))
        self.link = "ul" if keys.get("general.imt_link", "").upper() == "UPLINK" else "dl"
        self.output_dir = os.path.join(WORKFOLDER, keys.get("general.output_dir", ""))
        self.expected_rows = self._expected_tx_rows(keys)
        self.cost = wifi_case_cost(self.name)
        self.group = re.sub(r"^.*?_IMT_\d+_", "", self.name)  # ex.: WIFI_90_NA_228
        self._offset = 0
        self._rows = 0
        self._tx_file = None
        self.reset_state()

    def reset_state(self):
        self.status = "pendente"
        self.progress = 0.0
        self.start = None
        self.end = None
        self.last_write = None

    def _expected_tx_rows(self, keys):
        """Linhas esperadas em imt_*_tx_power.csv ao final do caso."""
        try:
            load = float(keys["imt.bs.load_probability"])
            k = int(float(keys.get("imt.ue.k", "3")))
            if keys.get("imt.topology.type", "").upper() != "MACROCELL":
                return None
            clusters = int(float(keys.get("imt.topology.macrocell.num_clusters", "1")))
            num_bs = 19 * 3 * clusters
            return self.snapshots * num_bs * k * load
        except (KeyError, ValueError):
            return None

    def _run_dir(self):
        """Pasta da execução mais recente do caso (output_*_NN)."""
        runs = [d for d in glob.glob(os.path.join(self.output_dir, "*")) if os.path.isdir(d)]
        return max(runs, key=os.path.getmtime) if runs else None

    def _count_new_rows(self, path):
        if path != self._tx_file:
            self._tx_file, self._offset, self._rows = path, 0, 0
        size = os.path.getsize(path)
        if size < self._offset:          # arquivo recriado
            self._offset, self._rows = 0, 0
        if size > self._offset:
            with open(path, "rb") as f:
                f.seek(self._offset)
                chunk = f.read()
            self._offset += len(chunk)
            self._rows += chunk.count(b"\n")
        return max(self._rows - 1, 0)    # sem o cabeçalho

    def update(self, now):
        self.reset_state()
        run = self._run_dir()
        if run is None:
            return
        st = os.stat(run)
        self.start = getattr(st, "st_birthtime", st.st_ctime)
        files = [os.path.join(run, f) for f in os.listdir(run)]
        self.last_write = max([os.path.getmtime(f) for f in files] + [self.start])

        logs = glob.glob(os.path.join(run, "simulation_*", "simulation_log_*.yaml"))
        if logs:
            self.status = "concluído"
            self.progress = 1.0
            self.end = os.path.getmtime(logs[0])
            return

        tx = os.path.join(run, f"imt_{self.link}_tx_power.csv")
        if self.expected_rows and os.path.exists(tx):
            self.progress = min(self._count_new_rows(tx) / self.expected_rows, 0.999)
        stalled = now - self.last_write > STALL_SECONDS
        self.status = "parado?" if stalled else "rodando"

    @property
    def elapsed(self):
        if self.start is None:
            return None
        return (self.end or time.time()) - self.start

    @property
    def estimated_total(self):
        """Duração total estimada (s), a partir do ritmo atual."""
        if self.status == "concluído":
            return self.elapsed
        if self.status in ("rodando", "parado?") and self.progress >= 0.01:
            return self.elapsed / self.progress
        return None

    @property
    def eta(self):
        total = self.estimated_total
        if total is None or self.status == "concluído":
            return None
        return max(total - self.elapsed, 0.0)


# --------------------------------------------------------------------------
# Campanha
# --------------------------------------------------------------------------
def load_cases(campaigns):
    cases = []
    for camp in campaigns:
        pattern = os.path.join(WORKFOLDER, "campaigns", camp, "input", "*.yaml")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"Aviso: nenhum YAML em {pattern}", file=sys.stderr)
        cases += [Case(camp, f) for f in files]
    # mesma ordem do runner (casos mais pesados primeiro, ordenação estável)
    cases.sort(key=lambda c: c.cost, reverse=True)
    return cases


def campaign_eta(cases, workers, now):
    """ETA da campanha: simula a fila de processos do runner."""
    durations = [c.estimated_total for c in cases if c.estimated_total]
    if not durations:
        return None
    by_group = {}
    for c in cases:
        if c.estimated_total:
            by_group.setdefault(c.group, []).append(c.estimated_total)
    mean_all = sum(durations) / len(durations)

    running = [c for c in cases if c.status in ("rodando", "parado?")]
    pending = [c for c in cases if c.status == "pendente"]
    free_at = [c.eta if c.eta is not None else mean_all for c in running]
    free_at += [0.0] * max(workers - len(free_at), 0)
    free_at.sort()
    finish = max(free_at) if free_at else 0.0
    for c in pending:
        est = by_group.get(c.group)
        dur = sum(est) / len(est) if est else mean_all
        t = free_at.pop(0) + dur
        free_at.append(t)
        free_at.sort()
        finish = max(finish, t)
    return finish


def summary(cases, workers, now):
    counts = {s: sum(c.status == s for c in cases)
              for s in ("concluído", "rodando", "parado?", "pendente")}
    overall = sum(c.progress for c in cases) / len(cases) if cases else 0.0
    eta = campaign_eta(cases, workers, now)
    return counts, overall, eta


def fmt_dur(seconds):
    if seconds is None:
        return "-"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def fmt_clock(seconds_from_now):
    if seconds_from_now is None:
        return "-"
    return (datetime.now() + timedelta(seconds=seconds_from_now)).strftime("%H:%M")


def bar(fraction, width=20):
    filled = int(round(fraction * width))
    return "#" * filled + "." * (width - filled)


ORDER = {"parado?": 0, "rodando": 1, "pendente": 2, "concluído": 3}


def sorted_rows(cases):
    return sorted(cases, key=lambda c: (ORDER[c.status], -c.progress, c.name))


def summary_line(counts, overall, eta):
    return (f"{counts['concluído']} concluídos | {counts['rodando']} rodando | "
            f"{counts['pendente']} pendentes"
            + (f" | {counts['parado?']} PARADOS?" if counts["parado?"] else "")
            + f" | total {overall * 100:5.1f}% | ETA {fmt_dur(eta)} (fim ~{fmt_clock(eta)})")


# --------------------------------------------------------------------------
# Modo texto
# --------------------------------------------------------------------------
def render_text(cases, workers):
    now = time.time()
    for c in cases:
        c.update(now)
    counts, overall, eta = summary(cases, workers, now)
    lines = [f"SHARC — {datetime.now():%H:%M:%S}  " + summary_line(counts, overall, eta), ""]
    lines.append(f"{'caso':<34} {'status':<10} {'progresso':<28} {'decorrido':>9} {'ETA':>9} {'fim':>6}")
    for c in sorted_rows(cases):
        end_clock = fmt_clock(c.eta) if c.eta is not None else (
            datetime.fromtimestamp(c.end).strftime("%H:%M") if c.end else "-")
        lines.append(f"{c.name:<34} {c.status:<10} {bar(c.progress)} {c.progress * 100:5.1f}% "
                     f"{fmt_dur(c.elapsed):>9} {fmt_dur(c.eta):>9} {end_clock:>6}")
    return "\n".join(lines)


def run_text(cases, workers, interval, once):
    if os.name == "nt":
        os.system("")  # habilita sequências ANSI no console do Windows
    while True:
        text = render_text(cases, workers)
        if once:
            print(text)
            return
        print("\x1b[2J\x1b[H" + text, flush=True)
        if all(c.status == "concluído" for c in cases):
            return
        time.sleep(interval)


# --------------------------------------------------------------------------
# Modo janela (tkinter)
# --------------------------------------------------------------------------
def run_gui(cases, workers, interval):
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("SHARC — progresso da campanha")
    root.geometry("1150x680")

    header = ttk.Label(root, font=("Segoe UI", 11, "bold"))
    header.pack(fill="x", padx=10, pady=(10, 4))
    overall_bar = ttk.Progressbar(root, maximum=100)
    overall_bar.pack(fill="x", padx=10, pady=(0, 8))

    columns = ("status", "progresso", "pct", "decorrido", "eta", "fim")
    tree = ttk.Treeview(root, columns=columns, show="tree headings")
    tree.heading("#0", text="Caso")
    tree.column("#0", width=290)
    for col, title, width, anchor in (
            ("status", "Status", 90, "center"), ("progresso", "Progresso", 190, "w"),
            ("pct", "%", 60, "e"), ("decorrido", "Decorrido", 90, "e"),
            ("eta", "ETA", 90, "e"), ("fim", "Fim previsto", 90, "center")):
        tree.heading(col, text=title)
        tree.column(col, width=width, anchor=anchor)
    tree.tag_configure("concluído", foreground="#1b7f1b")
    tree.tag_configure("rodando", foreground="#0b4fa8")
    tree.tag_configure("pendente", foreground="#777777")
    tree.tag_configure("parado?", foreground="#c00000")

    scroll = ttk.Scrollbar(root, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scroll.set)
    scroll.pack(side="right", fill="y")
    tree.pack(fill="both", expand=True, padx=(10, 0), pady=(0, 10))

    def refresh():
        now = time.time()
        for c in cases:
            c.update(now)
        counts, overall, eta = summary(cases, workers, now)
        header.config(text=summary_line(counts, overall, eta))
        overall_bar["value"] = overall * 100
        tree.delete(*tree.get_children())
        for c in sorted_rows(cases):
            end_clock = fmt_clock(c.eta) if c.eta is not None else (
                datetime.fromtimestamp(c.end).strftime("%H:%M") if c.end else "-")
            tree.insert("", "end", text=c.name, tags=(c.status,), values=(
                c.status, bar(c.progress, 24), f"{c.progress * 100:.1f}",
                fmt_dur(c.elapsed), fmt_dur(c.eta), end_clock))
        root.after(int(interval * 1000), refresh)

    refresh()
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Monitor de campanhas SHARC")
    parser.add_argument("campaigns", nargs="*", default=DEFAULT_CAMPAIGNS,
                        help="nomes das pastas em sharc/campaigns (padrão: %(default)s)")
    parser.add_argument("--text", action="store_true", help="tabela no terminal em vez de janela")
    parser.add_argument("--once", action="store_true", help="(com --text) imprime uma vez e sai")
    parser.add_argument("--interval", type=float, default=5.0, help="segundos entre atualizações")
    parser.add_argument("--workers", type=int, default=None,
                        help="processos em paralelo (padrão: como o runner, min(casos, CPUs))")
    args = parser.parse_args()

    cases = load_cases(args.campaigns)
    if not cases:
        sys.exit("Nenhum caso encontrado.")
    workers = args.workers or min(len(cases), os.cpu_count() or 1)

    if args.text:
        run_text(cases, workers, args.interval, args.once)
    else:
        run_gui(cases, workers, args.interval)


if __name__ == "__main__":
    main()
