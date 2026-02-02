import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Configuração dos Caminhos e Load Factors
# Ajuste 'base_dir' para onde as pastas de Load Factor (LF_10p, LF_30p...) estão
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))

# Dicionário: "Rótulo do Gráfico" -> "Nome da Pasta"
load_factors = {
    "10%": "LF10",
    "30%": "LF30",
    "50%": "LF50",
    "70%": "LF70",
    "100%": "LF100" # Altere para LF_90p se for o seu caso
}

# Nome do arquivo CSV que queremos ler em cada pasta
TARGET_FILE = "wifi_dl_inr.csv"

# 2. Configuração do Gráfico (CCDF)
plt.figure(figsize=(10, 6))
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.title("CCDF do INR no Wi-Fi para Diferentes Load Factors", fontsize=14)
plt.xlabel("Interference-to-Noise Ratio (INR) [dB]", fontsize=12)
plt.ylabel("Probabilidade Complementar (CCDF)", fontsize=12)

# Estilos de linha e cores para diferenciar as curvas
line_styles = ['-', '--', '-.', ':', '-']
colors = ['blue', 'green', 'orange', 'red', 'purple']

# 3. Loop de Leitura e Plotagem

for i, (lf_label, folder_name) in enumerate(load_factors.items()):
    csv_path = os.path.join(base_dir, folder_name, TARGET_FILE)
    
    if not os.path.exists(csv_path):
        print(f"[AVISO] Arquivo não encontrado: {csv_path}")
        continue
        
    print(f"Lendo dados de LF = {lf_label}...")
    
    # Lê o CSV usando pandas. A coluna chama-se 'samples'
    df = pd.read_csv(csv_path)
    inr_data = df['samples'].values
    
    # Opcional: Filtra valores absurdos (-500) se ainda existirem no CSV
    inr_data = inr_data[inr_data > -200]
    
    if len(inr_data) == 0:
        print(f"[AVISO] Sem dados válidos para LF = {lf_label}")
        continue

    # 4. Cálculo da CCDF (Complementary Cumulative Distribution Function)
    # CCDF mostra a probabilidade do INR ser MAIOR que X dB
    sorted_inr = np.sort(inr_data)
    # y_vals vai de 1.0 (100%) até próximo de 0.0
    y_vals = 1.0 - np.arange(1, len(sorted_inr) + 1) / len(sorted_inr)
    
    # Plotagem
    plt.plot(sorted_inr, y_vals, label=f"LF = {lf_label}", 
             linestyle=line_styles[i % len(line_styles)], color=colors[i], linewidth=2)

# 5. Finalização do Gráfico
# Escala Y Logarítmica é padrão para CCDF de interferência (destaca a cauda do gráfico)
plt.yscale('log')
plt.ylim(1e-4, 1.0) # Mostra até 0.01% de probabilidade
plt.xlim(-10, 50)   # Ajuste os limites do eixo X conforme os seus dados

plt.legend(fontsize=12, loc='upper right')
plt.tight_layout()

# Salva o gráfico na pasta atual
output_image = "comparativo_wifi_inr_ccdf.png"
plt.savefig(output_image, dpi=300)
print(f"Gráfico gerado com sucesso: {output_image}")
plt.show()