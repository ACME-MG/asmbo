"""
 Title:         Main
 Description:   Adaptive surrogate model optimisation
 Author:        Janzen Choi

"""

# Libraries
import sys; sys.path += [".."]
import time
from asmbo.processer import process
from asmbo.trainer import train
from asmbo.optimiser import optimise
from asmbo.simulator import simulate
from asmbo.analyser import analyse
from asmbo.helper.general import combine_dict, safe_mkdir
from asmbo.helper.io import csv_to_dict

# Constants
NUM_ITERATIONS = 100
STRAIN_FIELD   = "average_strain"
STRESS_FIELD   = "average_stress"
NUM_STRAINS    = 32
NUM_PROCESSORS = 190
MAX_STRAIN     = 0.1
GRAIN_IDS      = [59, 63, 72, 237, 303]
PARAM_NAMES    = [f"cp_lh_{i}" for i in range(2)] + ["cp_tau_0", "cp_n", "cp_gamma_0"]
OPT_PARAMS     = [f"Param ({pn.replace('cp_','')})" for pn in PARAM_NAMES]
MESH_PATH      = f"data/mesh"
SAMPLE_PATH    = "data/sampled_data.csv"
EXP_PATH       = "data/617_s3_exp.csv"
RESULTS_PATH   = "./results"

def main():
    """
    Main function
    """

    # Initialise
    get_prefix = lambda : f"{RESULTS_PATH}/" + time.strftime("%y%m%d%H%M%S", time.localtime(time.time()))
    train_dict = csv_to_dict(SAMPLE_PATH)
    safe_mkdir(RESULTS_PATH)

    # Iterate
    for i in range(NUM_ITERATIONS):

        # Initialise
        progressor = Progresser(i+1)
        print("="*40)

        # 1) Train a surrogate model
        progressor.progress("Training")
        train_path = f"{get_prefix()}_i{i+1}_s1_sm"
        safe_mkdir(train_path)
        train(train_dict, train_path, PARAM_NAMES, GRAIN_IDS, STRAIN_FIELD, STRESS_FIELD)

        # 2) Optimise surrogate model
        progressor.progress("Optimising")
        opt_path = f"{get_prefix()}_i{i+1}_s2_opt"
        safe_mkdir(opt_path)
        optimise(train_path, opt_path, EXP_PATH, MAX_STRAIN, GRAIN_IDS)

        # 3) Run CPFEM with optimimsed parameters
        progressor.progress("Validating")
        sim_path = f"{get_prefix()}_i{i+1}_s3_sim"
        opt_dict = csv_to_dict(f"{opt_path}/params.csv")
        opt_params = [opt_dict[op][0] for op in OPT_PARAMS]
        safe_mkdir(sim_path)
        simulate(sim_path, MESH_PATH, EXP_PATH, PARAM_NAMES, opt_params, NUM_PROCESSORS)

        # 4) Analyse CPFEM simulation results
        progressor.progress("Analysing")
        analyse(sim_path, EXP_PATH, GRAIN_IDS, STRAIN_FIELD, STRESS_FIELD)

        # 5) Add to training dictionary
        progressor.progress("Adding")
        sim_dict = process(sim_path, PARAM_NAMES, STRAIN_FIELD, STRESS_FIELD, NUM_STRAINS, MAX_STRAIN)
        train_dict = combine_dict(train_dict, sim_dict)

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

# Main function caller
if __name__ == "__main__":
    main()
