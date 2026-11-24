"""Roda as campanhas com o modelo alinhado a ITU (06/11/2026), mais pesadas primeiro.

    WIFI_ITU_CoChannel  24 casos (Parte A: todos os casos, prob_DL 0,5, sem beamforming, sem BEL)
    WIFI_ITU_Adjacent   24 casos (Parte A)
    WIFI_ITU_BF_BEL      8 casos (Parte B: IMT 0,5 / Wi-Fi 0,9 / 228 APs, beamforming 4x1 no AP,
                                  limite de 24 dBm, BEL nas UEs indoor, prob_DL 0,5 e 0,75)
    WIFI_ITU_BEL         4 casos (opcionais: como a Parte B com prob_DL 0,5, mas SEM beamforming,
                                  para separar o efeito do beamforming do efeito da BEL)

YAML gerados por: python campaigns/wifi_campaign_generator.py --itu

Uso (a partir de sharc/):
    ..\\.venv\\Scripts\\python.exe run_itu_campaigns.py > campaigns\\itu_run.log 2> campaigns\\itu_run.err
e, em outro terminal, para o PC nao suspender:
    ..\\.venv\\Scripts\\python.exe keep_awake_until_done.py WIFI_ITU_CoChannel WIFI_ITU_Adjacent WIFI_ITU_BF_BEL WIFI_ITU_BEL --max-hours 14
"""
import glob
import os
from concurrent.futures import ThreadPoolExecutor

from run_multiple_campaigns_mut_thread import run_command, wifi_case_cost

CAMPAIGNS = ["WIFI_ITU_CoChannel", "WIFI_ITU_Adjacent", "WIFI_ITU_BF_BEL", "WIFI_ITU_BEL"]

wf = os.path.dirname(os.path.abspath(__file__))
files = sorted(
    [f for c in CAMPAIGNS for f in glob.glob(os.path.join(wf, "campaigns", c, "input", "*.yaml"))],
    key=wifi_case_cost, reverse=True)
print(len(files), "casos", flush=True)
if not files:
    raise SystemExit("No campaign YAML files found; generate or provide the campaign inputs first")
with ThreadPoolExecutor(max_workers=min(len(files), os.cpu_count())) as ex:
    rc = list(ex.map(run_command, files, [os.path.join(wf, "main_cli.py")] * len(files)))
print(f"{sum(r == 0 for r in rc)}/{len(files)} casos OK", flush=True)
