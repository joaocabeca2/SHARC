import math
import sys

import numpy as np
from sharc.antenna.antenna_omni import AntennaOmni
from sharc.parameters.wifi.parameters_wifi_system import ParametersWifiSystem
from sharc.station_manager import StationManager
from sharc.support.enumerations import StationType
from sharc.parameters.wifi.parameters_antenna_wifi import ParametersAntennaWifi
from sharc.mask.spectral_mask_wifi import SpectralMaskWifi
from sharc.support.sharc_utils import wrap2_180
class SystemWifi:
    """Implements a Wifi Network RLAN system"""

    def __init__(self, param, param_ant, random_number_gen, topology):
        self.parameters = param
        self.topology = topology
        self.parameters_antenna = param_ant

        # 1. Ajuste a contagem total de nós
        # Se você quer espalhar nós ao redor dos pontos da topologia, o total é (N_sites * K_nós_por_site)
        # Se 'k' não for mais relevante, defina um número fixo ou use param.sta.k como "densidade"
        self.nodes_per_site = self.parameters.sta.k 
        self.num_nodes = self.topology.num_base_stations

        # Inicializa o StationManager com a nova contagem total
        self.wifi = StationManager(self.num_nodes)
        self.wifi.station_type = StationType.WIFI # Certifique-se de ter criado este Enum

        # --- (Suas inicializações de array mantidas aqui, atualizadas para self.num_nodes) ---
        self.wifi.path_loss = np.empty([self.num_nodes, self.num_nodes])
        self.wifi.coupling_loss = np.empty([self.num_nodes, self.num_nodes])
        self.link = dict([(node, list()) for node in range(self.num_nodes)])
        
        # Inicialização básica de parâmetros de RF
        self.wifi.bandwidth = self.parameters.bandwidth * np.ones(self.num_nodes)
        self.wifi.center_freq = self.parameters.frequency * np.ones(self.num_nodes)
        self.wifi.thermal_noise = -500 * np.ones(self.num_nodes)
        self.wifi.noise_figure = self.parameters.sta.noise_figure * np.ones(self.num_nodes) # Exemplo
        
        # Antenas
        self.wifi.antenna = [AntennaOmni() for _ in range(self.num_nodes)]
        self.wifi.elevation = -param_ant.downtilt * np.ones(self.num_nodes)

        # --- LÓGICA DE POSICIONAMENTO (A parte que faltava) ---
        
        # Listas temporárias para acumular as coordenadas geradas
        node_x = []
        node_y = []
        node_z = []

        # Parâmetros de distribuição angular e radial
        azimuth_range = self.parameters.sta.azimuth_range
        
        # Gera ângulos e raios para TODOS os nós de uma vez (vetorizado)
        # Nota: random_number_gen.rand vs random_sample depende da sua versão do numpy, mantenha consistência
        angle = (azimuth_range[1] - azimuth_range[0]) * \
                random_number_gen.random_sample(self.num_nodes) + azimuth_range[0]

        # Distribuição de Distância (Ex: Uniforme no anel)
        r_min = self.parameters.minimum_separation_distance_ap_sta
        r_max = self.topology.cell_radius
        if self.parameters.sta.distribution_distance.upper() == "SQRT(UNIFORM)":
            radius = np.sqrt(
                random_number_gen.random_sample(self.num_nodes) * (r_max**2 - r_min**2) + r_min**2
            )
        else: # Fallback ou outra distribuição
            radius = r_min + random_number_gen.random_sample(self.num_nodes) * (r_max - r_min)

        # Loop para posicionar cada grupo de nós ao redor do seu respectivo "Site" da topologia
        for site_idx in range(self.topology.num_base_stations):
            # Índices dos nós que pertencem a este site/cluster
            idx_start = site_idx * self.nodes_per_site
            idx_end = idx_start + self.nodes_per_site
            indices = range(idx_start, idx_end)

            # 1. Coordenadas polares locais
            # Theta local relativo ao azimute do site (se houver rotação do site)
            theta = self.topology.azimuth[site_idx] + angle[indices]
            
            # 2. Converte para Cartesiano Local
            x_local = radius[indices] * np.cos(np.radians(theta))
            y_local = radius[indices] * np.sin(np.radians(theta))
            z_local = np.zeros_like(x_local) # Altura relativa inicial

            # 3. Transforma para Global (aplica rotação e translação do Site)
            # Usa o método da topologia para mover o ponto (x,y) para a posição do Site
            x_global, y_global, z_global = self.topology.transform_ue_xyz(
                site_idx, x_local, y_local, z_local
            )

            node_x.extend(x_global)
            node_y.extend(y_global)
            node_z.extend(z_global)

            # Define Azimute e Elevação finais do Nó
            # (Lógica original adaptada: Azimute do nó aponta 'para fora' ou aleatório?)
            # Aqui mantendo a lógica de "olhar para o centro" + 180 graus
            self.wifi.azimuth[indices] = (angle[indices] + self.topology.azimuth[site_idx] + 180) % 360
            
            # Cálculo de Elevação (Psi) baseada na distância e diferença de altura
            # Assumindo altura do nó = altura definida nos parâmetros + z_global
            # Se topology.z já inclui altura do terreno, cuidado para não somar duas vezes
            
            dist_2d = np.sqrt((self.topology.x[site_idx] - x_global)**2 + (self.topology.y[site_idx] - y_global)**2)
            
            # Exemplo: Elevação olhando para o horizonte ou para o site? 
            # Se for rede ad-hoc plana, elevation pode ser 0. 
            # Se mantiver a lógica original (olhando para o AP):
            psi = np.degrees(np.arctan((self.parameters.ap.height - self.parameters.sta.height) / dist_2d))
            self.wifi.elevation[indices] = -param_ant.downtilt + psi # Exemplo

        # Atribui as listas preenchidas ao StationManager
        self.wifi.x = np.array(node_x)
        self.wifi.y = np.array(node_y)
        
        # Altura: Topologia Z (terreno) + Altura do Mastro do Nó
        self.wifi.z = np.array(node_z) + self.parameters.sta.height 
        self.wifi.height = self.wifi.z

        # --- FIM DA LÓGICA DE POSICIONAMENTO ---

        # Configuração de Interferência e Potência (Inicialização)
        self.wifi.active = random_number_gen.rand(self.num_nodes) < self.parameters.ap.load_probability # Ou outra prob
        self.wifi.tx_power = self.parameters.sta.conducted_power * np.ones(self.num_nodes) # Potência de transmissão
        
        # Dicionários de Resultados (agora indexados de 0 a num_nodes)
        # Nota: Inicializar com array vazio ou valor default
        self.wifi.rx_power = np.full(self.num_nodes, -500.0) 
        self.wifi.sinr = np.full(self.num_nodes, -500.0)
        # Se precisar de histórico por RB ou Snapshot, a estrutura pode ser diferente (dict ou tensor)
        
        if self.parameters.spectral_mask == "WIFI-2020":
            self.wifi.spectral_mask = SpectralMaskWifi(
                self.parameters.frequency,
                self.parameters.bandwidth,
                StationType.WIFI,
                self.parameters.spurious_emissions,
            )
        self.wifi.spectral_mask.set_mask()

        if self.parameters.topology.type == 'HOTSPOT':
            self.wifi.intersite_dist = self.parameters.topology.hotspot.intersite_distance

    def run_csma_ca_scheduling(self, random_gen):
        # 1. Pré-calcular as matrizes de distância (Vetorizado e rápido)
        # Retornam matrizes NumPy [origem x destino]
        d_wifi_nodes = self.wifi.get_distance_to(self.wifi)
        
        # 2. Pegar os índices dos que 'querem' transmitir (Intent to transmit)
        wifi_candidates = np.where(self.wifi.active)[0]
        
        # Pool único de (Manager, Index)
        candidates = []
        candidates = list(wifi_candidates)
        
        # 3. Resetar o estado 'active' (agora ele representará 'vencedores do canal')
        self.wifi.active[:] = False

        # 4. Embaralhar para garantir justiça no sorteio (Simula Backoff)
        random_gen.shuffle(candidates)

        radius_km = self.parameters.max_dist_hotspot_ue

        # 5. Processo de Contenção (CSMA/CA)
        while candidates:
           # O primeiro da lista ganha o canal
            idx_tx = candidates.pop(0)
            self.wifi.active[idx_tx] = True 
            
            # Filtrar os demais candidatos:
            # Remove da lista de espera qualquer nó que esteja dentro do raio de detecção do vencedor
            remaining = []
            for idx_target in candidates:
                # Busca direta na matriz [N x N]
                dist = d_wifi_nodes[idx_tx, idx_target]
                
                # Se a distância for maior que o raio, o nó target NÃO ouve o transmissor
                # e portanto continua candidato a transmitir (reuso espacial)
                if dist >= radius_km:
                    remaining.append(idx_target)
            
            candidates = remaining
    
    def connect_wifi_sta_to_ap(self, parameters: ParametersWifiSystem, random_gen):
        """
        Link the Wi-Fi STA's to the serving AP. It is assumed that each group of K
        user equipments are distributed and pointed to a certain access point
        """
        # 2. Quem vai transmitir agora? (Vencedores do CSMA)
        active_nodes = np.where(self.wifi.active)[0]
        
        # 3. Pega a matriz de distâncias (já calculada/cacheada se possível)
        # Se get_distance_to for pesado, considere armazenar o resultado numa variável de classe
        d_matrix = self.wifi.get_distance_to(self.wifi)
        
        # 4. Define o alcance máximo (Ex: 300 metros ou parametrizado)
        # Tente pegar dos parametros, se não tiver, use um valor fixo seguro
        try:
            max_range = self.parameters.max_dist_communication
        except AttributeError:
            max_range = 0.3 # 300 metros (exemplo padrão Wi-Fi/DSRC)

        # 5. Loop para criar os pares
        for tx_node in active_nodes:
            # Encontra candidatos:
            # a) Distância <= max_range
            # b) Índice != tx_node (não pode falar consigo mesmo)
            # c) (Opcional) Rx não pode estar transmitindo (Half-duplex rígido) -> ignorado aqui para simplificar
            
            candidates_mask = (d_matrix[tx_node] <= max_range) & \
                              (np.arange(self.num_nodes) != tx_node)
            
            candidate_indices = np.where(candidates_mask)[0]
            
            if len(candidate_indices) > 0:
                # Escolhe UM vizinho aleatoriamente
                rx_node = random_gen.choice(candidate_indices)
                self.link[tx_node] = [rx_node]
            else:
                # Nó isolado (ninguém por perto), transmite para o "vazio"
                self.link[tx_node] = []

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from sharc.parameters.wifi.parameters_hotspot import ParametersHotspot
    from sharc.topology.topology_hotspot import TopologyHotspot

    wifi_ant_param = ParametersAntennaWifi()
    wifi_param = ParametersWifiSystem()
    t_param = ParametersHotspot()

    wifi_topology = TopologyHotspot(t_param, 321, 1)
    wifi_topology.calculate_coordinates()

    wifi_sys = SystemWifi(wifi_param, wifi_ant_param, np.random.RandomState(1234), wifi_topology)
    wifi = wifi_sys.wifi