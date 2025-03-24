import os
from pathlib import Path

# Obtém o diretório base do projeto SHARC automaticamente
base_dir = Path(__file__).resolve().parent.parent.parent  # Caminha 3 níveis acima para alcançar SHARC

# Diretório de entrada e saída baseado no diretório do projeto
input_file = base_dir / "imt_fss_es_rel_dist_borda_lf20/scripts/base.yaml"
output_dir = base_dir / "imt_fss_es_rel_dist_borda_lf20/input/"
output_dir.mkdir(parents=True, exist_ok=True)

# Carregar arquivo de referência como texto
with open(input_file, "r") as f:
    reference_text = f.readlines()

# Gerar arquivos para cada ângulo de azimute
for dist_borda in [500, 1000, 2000]:
    for link_type in ["dl", "ul"]:
        dist = 1600
        x = dist+dist_borda
        modified_text = reference_text[:]
        
        # Alterar linha 7 para DL ou UL
        modified_text[17] = f"    imt_link: {'DOWNLINK' if link_type == 'dl' else 'UPLINK'}\n"

        # Alterar linha 23 e 26 sufixo do nome do arquivo
        modified_text[23] = f"    output_dir: campaigns/imt_fss_es_rel_dist_borda_lf20/output\n"
        modified_text[24] = f"    output_dir_prefix: output_imt_fss_es_{link_type}_dist_borda{x}\n"
        

        if link_type == "dl":
            modified_text[142] = f"      Hte: 18\n"
        else:
            modified_text[142] = f"      Hte: 1.5\n"

        modified_text[159] = f"        x: {x}\n"
        
        # Criar nome do arquivo de saída
        output_filename = f"parameters_imt_macro_fss_es_{link_type}_lf20_dist_borda{x}.yaml"
        output_path = output_dir / output_filename
        
        # Escrever o novo arquivo mantendo a estrutura original
        with open(output_path, "w") as out_file:
            out_file.writelines(modified_text)

print("Arquivos YAML gerados com sucesso!")