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
GRAIN_IDS      = [207, 79, 164, 167, 309]
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
        progressor.start("Training")
        train_path = f"{get_prefix()}_i{i+1}_s1_sm"
        safe_mkdir(train_path)
        train(train_dict, train_path, PARAM_NAMES, GRAIN_IDS, STRAIN_FIELD, STRESS_FIELD)
        progressor.end()

        # 2) Optimise surrogate model
        progressor.start("Optimising")
        opt_path = f"{get_prefix()}_i{i+1}_s2_opt"
        safe_mkdir(opt_path)
        optimise(train_path, opt_path, EXP_PATH, MAX_STRAIN, GRAIN_IDS)
        progressor.end()

        # 3) Run CPFEM with optimimsed parameters
        progressor.start("Validating")
        sim_path = f"{get_prefix()}_i{i+1}_s3_sim"
        opt_dict = csv_to_dict(f"{opt_path}/params.csv")
        opt_params = [opt_dict[op][0] for op in OPT_PARAMS]
        safe_mkdir(sim_path)
        simulate(sim_path, MESH_PATH, EXP_PATH, PARAM_NAMES, opt_params, NUM_PROCESSORS)
        progressor.end()

        # 4) Analyse CPFEM simulation results
        progressor.start("Analysing")
        analyse(sim_path, EXP_PATH, GRAIN_IDS, STRAIN_FIELD, STRESS_FIELD)
        progressor.end()

        # 5) Add to training dictionary
        progressor.start("Adding")
        sim_dict = process(sim_path, PARAM_NAMES, STRAIN_FIELD, STRESS_FIELD, NUM_STRAINS, MAX_STRAIN)
        train_dict = combine_dict(train_dict, sim_dict)
        progressor.end()

# Progress updater class
class Progresser:
    def __init__(self, iteration:int, max_length:int=20):
        self.iteration = iteration
        self.max_length = max_length
        self.start_time = None
        self.step = 1
    def start(self, verb:str):
        message = f"{self.iteration}.{self.step}: {verb} ..."
        padding = self.max_length - len(message)
        print(f"  {message}{padding*'.'}" , end="")
        self.start_time = time.time()
    def end(self):
        duration = round(time.time()-self.start_time, 2)
        print(f" [Done] ({duration}s)")
        self.step += 1

# Main function caller
if __name__ == "__main__":
    main()