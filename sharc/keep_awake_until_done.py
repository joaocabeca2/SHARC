"""
Impede que o Windows entre em suspensão/hibernação por inatividade enquanto
uma campanha SHARC estiver rodando, e libera sozinho quando todos os casos
terminarem (usa a mesma detecção de concluído do campaign_monitor.py).

Usa SetThreadExecutionState (o mesmo mecanismo de players de vídeo): não
altera nenhuma configuração de energia e deixa de valer assim que este
processo termina. Não impede a suspensão ao FECHAR A TAMPA, nem por bateria
crítica: deixe o notebook aberto e na tomada.

Uso (a partir de sharc/):
    python keep_awake_until_done.py                    # campanhas padrão
    python keep_awake_until_done.py WIFI_CoChannel      # campanhas específicas
    python keep_awake_until_done.py --max-hours 12      # limite de segurança
"""
import argparse
import ctypes
import time
from datetime import datetime

from campaign_monitor import DEFAULT_CAMPAIGNS, load_cases

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def main():
    parser = argparse.ArgumentParser(description="Mantém o PC acordado até a campanha terminar")
    parser.add_argument("campaigns", nargs="*", default=DEFAULT_CAMPAIGNS)
    parser.add_argument("--interval", type=float, default=60.0, help="segundos entre verificações")
    parser.add_argument("--max-hours", type=float, default=24.0, help="libera depois deste tempo")
    args = parser.parse_args()

    cases = load_cases(args.campaigns)
    deadline = time.time() + args.max_hours * 3600
    kernel32 = ctypes.windll.kernel32

    print(f"{datetime.now():%H:%M:%S} mantendo o PC acordado: {len(cases)} casos em {args.campaigns}",
          flush=True)
    try:
        while time.time() < deadline:
            # Renova o pedido a cada verificação
            kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            now = time.time()
            for c in cases:
                c.update(now)
            done = sum(c.status == "concluído" for c in cases)
            print(f"{datetime.now():%H:%M:%S} {done}/{len(cases)} concluídos", flush=True)
            if done == len(cases):
                print("Campanha concluída.", flush=True)
                break
            time.sleep(args.interval)
        else:
            print(f"Limite de {args.max_hours} h atingido.", flush=True)
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)  # libera
        print(f"{datetime.now():%H:%M:%S} pedido liberado; o PC volta ao comportamento normal.",
              flush=True)


if __name__ == "__main__":
    main()
