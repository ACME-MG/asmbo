"""
 Title:         Main
 Description:   Adaptive surrogate model optimisation
 Author:        Janzen Choi

"""

# Libraries
import sys; sys.path += [".."]
import time
from asmbo.assessor import assess
from asmbo.processer import process
from asmbo.trainer import train
from asmbo.optimiser import optimise
from asmbo.simulator import simulate
from asmbo.plotter import plot_results
from asmbo.helper.general import safe_mkdir
from asmbo.helper.io import csv_to_dict

# Constants
MAX_SIM_TIME   = 20000
NUM_ITERATIONS = 100
STRAIN_FIELD   = "average_strain"
STRESS_FIELD   = "average_stress"
NUM_STRAINS    = 32
NUM_PROCESSORS = 190
CAL_GRAIN_IDS  = [59, 63, 86, 237, 303]
VAL_GRAIN_IDS  = [44, 53, 60, 78, 190]
PARAM_NAMES    = ["cp_lh_0", "cp_lh_1", "cp_tau_0", "cp_n", "cp_gamma_0"]
OPT_PARAMS     = [f"Param ({pn.replace('cp_','')})" for pn in PARAM_NAMES]
MESH_PATH      = f"data/mesh"
SAMPLE_PATH    = "data/sampled_data_0p1.csv"
EXP_PATH       = "data/617_s3_40um_exp.csv"
RESULTS_PATH   = "./results"

def main():
    """
    Main function
    """

    # Initialise
    get_prefix = lambda : f"{RESULTS_PATH}/" + time.strftime("%y%m%d%H%M%S", time.localtime(time.time()))
    train_dict = csv_to_dict(SAMPLE_PATH)
    safe_mkdir(RESULTS_PATH)
    params_dict_list = []

    # Define maximum strain
    max_strain = 0.1

    # Iterate
    for i in range(NUM_ITERATIONS):

        # Initialise
        progressor = Progresser(i+1)
        print("="*40)

        # 1) Train a surrogate model
        progressor.progress("Training")
        train_path = f"{get_prefix()}_i{i+1}_surrogate"
        safe_mkdir(train_path)
        train(train_dict, train_path, PARAM_NAMES, CAL_GRAIN_IDS, STRAIN_FIELD, STRESS_FIELD)

        # 2) Assesses the surrogate model on previously optimised parameters
        progressor.progress("Assessing")
        if params_dict_list == []:
            init_params = None
        else:
            init_params = assess(params_dict_list, train_path, EXP_PATH, max_strain, CAL_GRAIN_IDS)
            init_params = {key.replace("cp_","") : value for key, value in init_params.items()}

        # 3) Optimise surrogate model
        progressor.progress("Optimising")
        opt_path = f"{get_prefix()}_i{i+1}_optimise"
        safe_mkdir(opt_path)
        optimise(train_path, opt_path, EXP_PATH, max_strain, CAL_GRAIN_IDS, init_params)

        # 4) Run CPFEM with optimised parameters
        progressor.progress("Validating")
        sim_path = f"{get_prefix()}_i{i+1}_simulate"
        opt_dict = csv_to_dict(f"{opt_path}/params.csv")
        opt_params = [opt_dict[op][0] for op in OPT_PARAMS]
        safe_mkdir(sim_path)
        simulate(sim_path, MESH_PATH, EXP_PATH, PARAM_NAMES, opt_params, NUM_PROCESSORS, MAX_SIM_TIME)

        # 5) Plot CPFEM simulation results
        progressor.progress("Plotting")
        plot_results(sim_path, EXP_PATH, CAL_GRAIN_IDS, VAL_GRAIN_IDS, STRAIN_FIELD, STRESS_FIELD)

        # 6) Process simulation results
        progressor.progress("Processing")
        sim_dict = process(sim_path, PARAM_NAMES, STRAIN_FIELD, STRESS_FIELD, NUM_STRAINS, max_strain)
        sim_params = read_params(f"{sim_path}/params.txt")
        
        # 7) Add to training dictionary
        progressor.progress("Adding")
        combined_dict = {}
        for key in train_dict.keys():
            if key in PARAM_NAMES:
                combined_dict[key] = train_dict[key] + [sim_dict[key]]*NUM_STRAINS
            else:
                combined_dict[key] = train_dict[key] + sim_dict[key]
        train_dict = combined_dict
        params_dict_list.append(sim_params)

# Progress updater class
class Progresser:
    def __init__(self, iteration:int):
        self.iteration = iteration
        self.step = 1
    def progress(self, verb:str):
        message = f"===== {self.iteration}.{self.step}: {verb} ====="
        print("")
        print("="*len(message))
        print(message)
        print("="*len(message))
        print("")
        self.step += 1

def read_params(params_path:str) -> dict:
    """
    Reads parameters from a file

    Parameters:
    * `params_path`: The path to the parameters

    Returns a dictionary containing the parameter information
    """
    data_dict = {}
    with open(params_path, 'r') as file:
        for line in file:
            key, value = line.strip().split(": ")
            data_dict[key] = float(value)
    return data_dict

# Main function caller
if __name__ == "__main__":
    main()
