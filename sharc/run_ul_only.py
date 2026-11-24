"""Roda so os casos UL (cocanal e adjacente) das campanhas Wi-Fi, mais pesados primeiro."""
import glob, os, sys
from concurrent.futures import ThreadPoolExecutor
from run_multiple_campaigns_mut_thread import run_command, wifi_case_cost

wf = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(wf, "campaigns", "WIFI_CoChannel", "input", "*_UL_*.yaml")) +
               glob.glob(os.path.join(wf, "campaigns", "WIFI_Adjacent", "input", "*_UL_*.yaml")),
               key=wifi_case_cost, reverse=True)
print(len(files), "casos", flush=True)
if not files:
    raise SystemExit("No campaign YAML files found; generate or provide the campaign inputs first")
with ThreadPoolExecutor(max_workers=min(len(files), os.cpu_count())) as ex:
    rc = list(ex.map(run_command, files, [os.path.join(wf, "main_cli.py")] * len(files)))
print(f"{sum(r == 0 for r in rc)}/{len(files)} casos OK", flush=True)
