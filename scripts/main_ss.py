"""
 Title:         Main for sensitivity study with Voce hardening
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
from asmbo.helper.sampler import get_lhs

# Simulation constants
MAX_SIM_TIME   = 20000
NUM_ITERATIONS = 100
STRAIN_FIELD   = "average_strain"
STRESS_FIELD   = "average_stress"
NUM_STRAINS    = 32
NUM_PROCESSORS = 190

# Grain IDs
CAL_GRAIN_IDS = [51, 56, 72, 80, 126, 223, 237, 262]
VAL_GRAIN_IDS = [44, 60, 78, 86, 178, 190, 207, 244]

# Model information
PARAM_INFO = [
    {"name": "cp_tau_s", "bounds": (0, 2000)},
    {"name": "cp_b",     "bounds": (0, 20)},
    {"name": "cp_tau_0", "bounds": (0, 500)},
    {"name": "cp_n",     "bounds": (1, 20)},
]
PARAM_NAMES = [pi["name"] for pi in PARAM_INFO]
OPT_PARAMS  = [f"Param ({pn})" for pn in PARAM_NAMES]
OPT_MODEL   = "sm_617_s3_vh"
SIM_MODEL   = "deer/cpvh_ae"

# Paths
MESH_PATH    = f"data/mesh"
EXP_PATH     = "data/617_s3_40um_exp.csv"
RESULTS_PATH = "./results"

def main():
    """
    Main function
    """
    
    # Initialise
    get_prefix = lambda : f"{RESULTS_PATH}/" + time.strftime("%y%m%d%H%M%S", time.localtime(time.time()))
    safe_mkdir(RESULTS_PATH)
    params_dict_list = []

    # Define maximum strain
    exp_dict = csv_to_dict(EXP_PATH)
    max_strain = exp_dict["strain_intervals"][-1]

    # Sample parameter space
    num_params = int(sys.argv[1])
    param_info_dict = dict(zip([pi["name"] for pi in PARAM_INFO], [pi["bounds"] for pi in PARAM_INFO]))
    param_dict_list = get_lhs(param_info_dict, num_params)

    # Run model with sampled parameters
    for i, param_dict in enumerate(param_dict_list):
   
        # Initialise
        param_vals = [param_dict[pn] for pn in PARAM_NAMES]
        sim_path = f"{get_prefix()}_i1_initial_{i+1}"
        safe_mkdir(sim_path)
        
        # Simulate, plot, and process
        simulate(sim_path, MESH_PATH, EXP_PATH, PARAM_NAMES, param_vals, NUM_PROCESSORS, MAX_SIM_TIME, SIM_MODEL)
        plot_results(sim_path, EXP_PATH, CAL_GRAIN_IDS, VAL_GRAIN_IDS, STRAIN_FIELD, STRESS_FIELD)
        sim_dict = process(sim_path, PARAM_NAMES, STRAIN_FIELD, STRESS_FIELD, NUM_STRAINS, max_strain)

        # Update training dictionary
        if i == 0:
            train_dict = {}
            for key in sim_dict.keys():
                if key in PARAM_NAMES:
                    train_dict[key] = [sim_dict[key]]*NUM_STRAINS
                else:
                    train_dict[key] = sim_dict[key]
        else:
            train_dict = update_train_dict(train_dict, sim_dict)

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
            init_params = assess(params_dict_list, train_path, EXP_PATH, max_strain, CAL_GRAIN_IDS, PARAM_NAMES)

        # 3) Optimise surrogate model
        progressor.progress("Optimising")
        opt_path = f"{get_prefix()}_i{i+1}_optimise"
        safe_mkdir(opt_path)
        optimise(train_path, opt_path, EXP_PATH, max_strain, CAL_GRAIN_IDS, PARAM_INFO, OPT_MODEL, init_params)

        # 4) Run CPFEM with optimised parameters
        progressor.progress("Validating")
        sim_path = f"{get_prefix()}_i{i+1}_simulate"
        opt_dict = csv_to_dict(f"{opt_path}/params.csv")
        opt_params = [opt_dict[op][0] for op in OPT_PARAMS]
        safe_mkdir(sim_path)
        simulate(sim_path, MESH_PATH, EXP_PATH, PARAM_NAMES, opt_params, NUM_PROCESSORS, MAX_SIM_TIME, SIM_MODEL)

        # 5) Plot CPFEM simulation results
        progressor.progress("Plotting")
        plot_results(sim_path, EXP_PATH, CAL_GRAIN_IDS, VAL_GRAIN_IDS, STRAIN_FIELD, STRESS_FIELD)

        # 6) Process simulation results
        progressor.progress("Processing")
        sim_dict = process(sim_path, PARAM_NAMES, STRAIN_FIELD, STRESS_FIELD, NUM_STRAINS, max_strain)
        sim_params = read_params(f"{sim_path}/params.txt")
        
        # 7) Add to training dictionary
        progressor.progress("Adding")
        train_dict = update_train_dict(train_dict, sim_dict)
        params_dict_list.append(sim_params)

def update_train_dict(train_dict:dict, sim_dict:dict) -> dict:
    """
    Updates the training dictionary

    Parameters:
    * `train_dict`: The current training dictionary
    * `sim_dict`:   The dictionary to add

    Returns the combined dictionary
    """
    combined_dict = {}
    for key in train_dict.keys():
        if not key in sim_dict.keys():
            continue
        if key in PARAM_NAMES:
            combined_dict[key] = train_dict[key] + [sim_dict[key]]*NUM_STRAINS
        else:
            combined_dict[key] = train_dict[key] + sim_dict[key]
    return combined_dict

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
