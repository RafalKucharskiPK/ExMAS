import visualising_functions as vis_eff
import os

os.chdir(os.path.dirname(os.getcwd()) + r"\\Topology\\")

date = "25-11-22"
name0 = "_full_n"
name1 = "_mini_n"
name2 = "_maxi_n"

config = vis_eff.config_initialisation('data/configs/topology_settings3.json', date)
config.sblts_exmas = "exmas"

# for var in ['profit', 'veh', 'utility', 'pass']:
vis_eff.mixed_datasets_kpi('profit', config, date, name0, name1, name2)

