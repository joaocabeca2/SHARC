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
        
        self.wifi.antenna = [AntennaOmni() for _ in range(self.num_nodes)]
        self.wifi.elevation = -param_ant.downtilt * np.ones(self.num_nodes)
        self.wifi.rx_power[:] = -500.0
        self.wifi.rx_interference[:] = -500.0
        self.wifi.total_interference[:] = -500.0
        self.wifi.sinr[:] = -500.0
        self.wifi.snr[:] = -500.0

        self.configure_node_parameters()
        node_x = []
        node_y = []
        node_z = []

        azimuth_range = self.parameters.sta.azimuth_range
        
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

        for site_idx in range(self.topology.num_base_stations):
            idx_start = site_idx * self.nodes_per_site
            idx_end = idx_start + self.nodes_per_site
            indices = range(idx_start, idx_end)

            theta = self.topology.azimuth[site_idx] + angle[indices]
            
            x_local = radius[indices] * np.cos(np.radians(theta))
            y_local = radius[indices] * np.sin(np.radians(theta))
            z_local = np.zeros_like(x_local) # Altura relativa inicial

            x_global, y_global, z_global = self.topology.transform_ue_xyz(
                site_idx, x_local, y_local, z_local
            )

            node_x.extend(x_global)
            node_y.extend(y_global)
            node_z.extend(z_global)

            self.wifi.azimuth[indices] = (angle[indices] + self.topology.azimuth[site_idx] + 180) % 360
            
            
            dist_2d = np.sqrt((self.topology.x[site_idx] - x_global)**2 + (self.topology.y[site_idx] - y_global)**2)
            
            psi = np.degrees(np.arctan((self.parameters.ap.height - self.parameters.sta.height) / dist_2d))
            self.wifi.elevation[indices] = -param_ant.downtilt + psi # Exemplo

        # Atribui as listas preenchidas ao StationManager
        self.wifi.x = np.array(node_x)
        self.wifi.y = np.array(node_y)
        
        # Altura: Topologia Z (terreno) + Altura do Mastro do Nó
        self.wifi.z = np.array(node_z) + self.parameters.sta.height 
        self.wifi.height = self.wifi.z

        self.wifi.active = random_number_gen.rand(self.num_nodes) < self.parameters.ap.load_probability # Ou outra prob
        
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
    
    def create_random_links(self, random_number_gen):
        
        # 2. Cria array de índices [0, 1, 2, ..., N]
        all_indices = np.arange(self.num_nodes)
        
        # 3. Embaralha usando a instância RandomState do SHARC
        # Isso altera 'all_indices' in-place mantendo a reprodutibilidade
        random_number_gen.shuffle(all_indices)
        
        # 4. Cria os pares
        limit = len(all_indices) - (len(all_indices) % 2)
        
        for i in range(0, limit, 2):
            node_a = all_indices[i]
            node_b = all_indices[i+1]
            
            # Link Bidirecional
            self.link[node_a] = [node_b]
            self.link[node_b] = [node_a]
            
    def configure_node_parameters(self):

        self.num_aps = self.num_nodes  // 2
        idx_aps = slice(0, self.num_aps)
        idx_stas = slice(self.num_aps, self.num_nodes)

        p_ap = self.parameters.ap  
        
        # Potência (dBm)
        self.wifi.tx_power[idx_aps] = p_ap.conducted_power
        
        self.wifi.height[idx_aps] = self.topology.z[:self.num_aps] + p_ap.height
        self.wifi.z[idx_aps] = self.wifi.height[idx_aps]

        # Ruído e Perdas
        self.wifi.noise_figure[idx_aps] = p_ap.noise_figure
       
        p_sta = self.parameters.sta 
        
        # Potência (dBm)
        self.wifi.tx_power[idx_stas] = p_sta.conducted_power
        
        ground_z_stas = self.wifi.z[idx_stas] # Assume que Z atual é o solo
        self.wifi.height[idx_stas] = ground_z_stas + p_sta.height
        self.wifi.z[idx_stas] = self.wifi.height[idx_stas]

        # Ruído e Perdas
        self.wifi.noise_figure[idx_stas] = p_sta.noise_figure

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