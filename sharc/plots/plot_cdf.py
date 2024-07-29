import os
import pandas as pd
import plotly.graph_objects as go

def plot_cdf(base_dir, file_prefix, label='CDF', valores_label=['0', '45', '90'], passo_xticks=5, xaxis_title='Value'):
    # Define o diretório base dinamicamente com base na entrada do usuário

    workfolder = os.path.dirname(os.path.abspath(__file__))
    #csv_folder = os.path.abspath(base_dir)
    csv_folder = os.path.abspath(os.path.join(workfolder, '..', "campaigns",base_dir,"output"))

    figs_folder = os.path.abspath(os.path.join(workfolder, '..', "campaigns",base_dir,"output","figs"))

    # Verifica se a pasta de saída das figuras existe, senão, cria-a
    if not os.path.exists(figs_folder):
        os.makedirs(figs_folder)
    
    # Lista todas as subpastas no diretório base
    subdirs = [os.path.join(csv_folder, d) for d in os.listdir(csv_folder) if os.path.isdir(os.path.join(csv_folder, d))]

    # Inicializa valores mínimos e máximos globais
    global_min = float('inf')
    global_max = float('-inf')

    # Primeiramente, calcula os valores mínimos e máximos globais
    for subdir in subdirs:
        all_files = [f for f in os.listdir(subdir) if f.endswith('.csv') and label in f and file_prefix in f]
        

        for file_name in all_files:
            file_path = os.path.join(subdir, file_name)
            if os.path.exists(file_path):
                try:
                    # Lê o arquivo .csv usando pandas
                    data = pd.read_csv(file_path)
                    
                    # Remove linhas que não contêm valores numéricos válidos
                    data = data.apply(pd.to_numeric, errors='coerce').dropna()
                    
                    if not data.empty:
                        global_min = min(global_min, data.iloc[:, 0].min())
                        global_max = max(global_max, data.iloc[:, 0].max())
                except Exception as e:
                    print(f"Error processing the file {file_name}: {e}")

    # Em seguida, plota os gráficos ajustando os eixos
    # Em seguida, plota os gráficos ajustando os eixos
    for subdir in subdirs:
        all_files = [f for f in os.listdir(subdir) if f.endswith('.csv') and label in f and file_prefix in f]

        for file_name in all_files:
            fig = go.Figure()
            file_path = os.path.join(subdir, file_name)
            if os.path.exists(file_path):
                try:
                    # Lê o arquivo .csv usando pandas
                    data = pd.read_csv(file_path)
                    
                    # Remove linhas que não contêm valores numéricos válidos
                    data = data.apply(pd.to_numeric, errors='coerce').dropna()
                    
                    # Verifica se há pontos de dados suficientes para plotar
                    if data.empty or data.shape[0] < 2:
                        print(f"The file {file_name} does not have enough data to plot.")
                        continue
                    
                    # Plota o CDF
                    fig.add_trace(go.Scatter(x=data.iloc[:, 0], y=data.iloc[:, 1], mode='lines', name=f'{file_name}'))
                except Exception as e:
                    print(f"Error processing the file {file_name}: {e}")
            
            # Configurações do gráfico
            fig.update_layout(
                title=f'CDF Plot for {file_name}',
                xaxis_title=xaxis_title,
                yaxis_title='CDF',
                yaxis=dict(tickmode='array', tickvals=[0, 0.25, 0.5, 0.75, 1]),
                xaxis=dict(tickmode='linear', tick0=int(global_min), dtick=passo_xticks),
                legend_title="Labels"
            )
            
            # Mostra a figura
            fig.show()
            
            # Salva a figura
            base_name_no_ext = os.path.splitext(file_name)[0]  # Remove a extensão .csv
            fig_file_path = os.path.join(figs_folder, f"CDF_Plot_{base_name_no_ext}.png")
            fig.write_image(fig_file_path)
            print(f"Figure saved: {fig_file_path}")

# Funções específicas para diferentes métricas
def plot_bs_antenna_gain_towards_the_ue(base_dir, label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf(base_dir, 'BS_antenna_gain_towards_the_UE', label, valores_label, passo_xticks, xaxis_title='Antenna Gain (dB)')

def plot_coupling_loss(base_dir, label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf(base_dir, 'coupling_loss', label, valores_label, passo_xticks, xaxis_title='Coupling Loss (dB)')

def plot_dl_sinr(base_dir, label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf(base_dir, 'DL_SINR', label, valores_label, passo_xticks, xaxis_title='DL SINR (dB)')

def plot_dl_snr(base_dir, label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf(base_dir, 'DL_SNR', label, valores_label, passo_xticks, xaxis_title='DL SNR (dB)')

def plot_dl_throughput(base_dir, label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf(base_dir, 'DL_throughput', label, valores_label, passo_xticks, xaxis_title='DL Throughput (Mbps)')

def plot_dl_transmit_power(base_dir, label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf(base_dir, 'DL_transmit_power', label, valores_label, passo_xticks, xaxis_title='DL Transmit Power (dBm)')

def plot_imt_station_antenna_gain_towards_system(base_dir, label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf(base_dir, 'IMT_station_antenna_gain_towards_system', label, valores_label, passo_xticks, xaxis_title='Antenna Gain (dB)')

def plot_path_loss(base_dir, label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf(base_dir, 'path_loss', label, valores_label, passo_xticks, xaxis_title='Path Loss (dB)')

def plot_ue_antenna_gain_towards_the_bs(base_dir, label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf(base_dir, 'UE_antenna_gain_towards_the_BS', label, valores_label, passo_xticks, xaxis_title='Antenna Gain (dB)')

def plot_imt_to_system_path_loss(base_dir, label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf(base_dir, 'IMT_to_system_path_loss', label, valores_label, passo_xticks, xaxis_title='Path Loss (dB)')

def plot_system_antenna_towards_imt_stations(base_dir, label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf(base_dir, 'system_antenna_gain_towards_IMT_stations', label, valores_label, passo_xticks, xaxis_title='Antenna Gain (dB)')

def plot_system_inr(base_dir, label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf(base_dir, 'system_INR', label, valores_label, passo_xticks, xaxis_title='INR (dB)')

def plot_system_interference_power_from_imt_dl(base_dir, label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf(base_dir, 'system_interference_power_from_IMT_DL', label, valores_label, passo_xticks, xaxis_title='Interference Power (dBm/MHz)')

def plot_system_pfd(base_dir, label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf(base_dir, 'system_PFD', label, valores_label, passo_xticks, xaxis_title='PFD (dBW/m²)')

def plot_inr_samples(base_dir, label='CDF', valores_label=['0', '45', '90'], passo_xticks=5):
    plot_cdf(base_dir, 'INR_samples', label, valores_label, passo_xticks, xaxis_title='INR Samples (dB)')

# Função principal para identificar labels e chamar as funções apropriadas
def main(base_dir):
    valores_label = ['0']
    plot_bs_antenna_gain_towards_the_ue(base_dir, valores_label=valores_label)
    plot_coupling_loss(base_dir, valores_label=valores_label)
    plot_dl_sinr(base_dir, valores_label=valores_label)
    plot_dl_snr(base_dir, valores_label=valores_label)
    plot_dl_throughput(base_dir, valores_label=valores_label)
    plot_dl_transmit_power(base_dir, valores_label=valores_label)
    plot_imt_station_antenna_gain_towards_system(base_dir, valores_label=valores_label)
    plot_path_loss(base_dir, valores_label=valores_label)
    plot_ue_antenna_gain_towards_the_bs(base_dir, valores_label=valores_label)
    plot_imt_to_system_path_loss(base_dir, valores_label=valores_label)
    plot_system_antenna_towards_imt_stations(base_dir, valores_label=valores_label)
    plot_system_inr(base_dir, valores_label=valores_label)
    plot_system_interference_power_from_imt_dl(base_dir, valores_label=valores_label)
    plot_system_pfd(base_dir, valores_label=valores_label)
    plot_inr_samples(base_dir, valores_label=valores_label)

if __name__ == "__main__":
    #import sys
    #if len(sys.argv) != 2:
    #    print("Usage: python main_cli.py <base_directory>")
    #else:
    #    main(sys.argv[1])
    name= "imt_hibs_ras_2600_MHz"
    main(name)

