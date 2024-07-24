import os
import pandas as pd
import sys

def salvar_em_csv(label, label2):
    # Obter o diretório atual do script
    workfolder = os.path.dirname(os.path.abspath(__file__))
    
    # Definir o caminho base dinamicamente com base no diretório atual
    name_campaigns = "imt_hibs_ras_2600_MHz"
    base_path = os.path.abspath(os.path.join(workfolder, '..', 'output'))
    subfolder_name ="output_"+name_campaigns + "_" + label
    input_folder = os.path.join(base_path, subfolder_name)
    
    # Verificar se o diretório existe
    if not os.path.exists(input_folder):
        print(f"O diretório {input_folder} não existe.")
        return

    # Criar a pasta output/RAS se não existir
    output_ras_folder = os.path.join(base_path, 'RAS_distance_csv')
    if not os.path.exists(output_ras_folder):
        os.makedirs(output_ras_folder)

    # Listar todos os arquivos .txt no diretório especificado
    all_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    print("Arquivos .txt encontrados:", all_files)

    for file in all_files:
        try:
            # Ler o arquivo .txt utilizando pandas, assumindo que há duas colunas
            file_path = os.path.join(input_folder, file)
            data = pd.read_csv(file_path, sep=r'\s+', header=None, usecols=[0, 1])
            
            # Criar o nome do arquivo .csv com o mesmo nome do arquivo .txt e adicionar label2
            nome_base = os.path.splitext(file)[0]
            nome_csv = f"RAS_{label2}_{nome_base}.csv"
            caminho_csv = os.path.join(output_ras_folder, nome_csv)
            
            # Salvar os dados em um arquivo .csv
            data.to_csv(caminho_csv, index=False)
            print(f"Arquivo salvo: {caminho_csv}")
        except Exception as e:
            print(f"Erro ao processar o arquivo {file}: {e}")

if __name__ == '__main__':
    dist = "500"
    label = dist+"km_2024-07-24_01"
    label2 = dist+"km"

    salvar_em_csv(label, label2)


