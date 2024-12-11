"""
 Title:         Summarise
 Description:   Summarises the results of the sampled simulations
 Author:        Janzen Choi

"""

# Libraries
import sys; sys.path += ["../.."]
import math, numpy as np, os, re
from moose_sim.helper.interpolator import Interpolator
from moose_sim.helper.general import transpose, round_sf
from moose_sim.helper.io import csv_to_dict, dict_to_csv
from moose_sim.analyse.plotter import Plotter, save_plot

# Simulation paths
RESULTS_DIR = "/mnt/c/Users/janzen/OneDrive - UNSW/PhD/results/moose_sim"
# SAMPLED_PATH = f"{RESULTS_DIR}/2024-11-28 (617_s3_40um_lh2_sm32_full)"
SAMPLED_PATH = f"{RESULTS_DIR}/2024-12-08 (617_s3_40um_lh2_sm48)"
SIM_PATHS = [f"{RESULTS_DIR}/{sim_dir}" for sim_dir in [
    # "2024-11-28 (617_s3_40um_lh2_i1)",
    # "2024-11-30 (617_s3_40um_lh2_i2)",
    # "2024-12-01 (617_s3_40um_lh2_i3)",
]]
SUMMARY_FILE = "617_s3_40um_lh2_sampled.csv"

# Constants
PARAMS = [f"cp_lh_{i}" for i in range(2)] + ["cp_tau_0", "cp_n", "cp_gamma_0"]
# STRAIN_FIELD = "average_grain_strain"
# STRESS_FIELD = "average_grain_stress"
STRAIN_FIELD = "average_strain"
STRESS_FIELD = "average_stress"
NUM_STRAINS = 32
MAX_STRAIN = 0.10

def main():
    """
    Main function
    """
    
    # Identify paths to simulations
    dir_path_list = [f"{SAMPLED_PATH}/{dir_path}" for dir_path in os.listdir(SAMPLED_PATH)
                    if os.path.exists(f"{SAMPLED_PATH}/{dir_path}/summary.csv")]
    print(f"Summarising {len(dir_path_list)}+{len(SIM_PATHS)} simulations ...")
    dir_path_list += SIM_PATHS

    # Read all summary files
    summary_path_list = [f"{dir_path}/summary.csv" for dir_path in dir_path_list]
    summary_dict_list = [csv_to_dict(summary_path) for summary_path in summary_path_list]
    param_dict_list = [get_param_dict(f"{dir_path}/params.txt") for dir_path in dir_path_list]

    # Process the dictionaries
    processed_dict_list = [process_data_dict(summary_dict) for summary_dict in summary_dict_list]
    key_list = list(param_dict_list[0].keys()) + list(processed_dict_list[0].keys())
    super_processed_dict = dict(zip(key_list, [[] for _ in range(len(key_list))]))

    # Initialise plotter
    plotter = Plotter(STRAIN_FIELD, STRESS_FIELD, "mm/mm", "MPa")
    plotter.prep_plot()

    # Iterate through the results
    for summary_dict, processed_dict, param_dict in zip(summary_dict_list, processed_dict_list, param_dict_list):
        
        # Plot unprocessed and processed data
        plotter.scat_plot(summary_dict, colour="silver")
        plotter.line_plot(processed_dict, colour="red")
        
        # Save to super dictionary 
        num_values = len(list(processed_dict.values())[0])
        for key in param_dict:
            super_processed_dict[key] += [round_sf(param_dict[key], 5)]*num_values
        for key in processed_dict:
            super_processed_dict[key] += round_sf(processed_dict[key], 5)

    # Save the plot and super summary dictionary
    save_plot("plot_ss.png")
    dict_to_csv(super_processed_dict, SUMMARY_FILE)

def get_param_dict(params_path:str) -> dict:
    """
    Gets the parameters from a file
    
    Parameters:
    * `params_path`: The path to the parameters file
    
    Returns a dictionary of the parameter values
    """
    with open(params_path, "r") as fh:
        line_list = [line.replace("\n", "").replace(":", "") for line in fh.readlines()]
        param_dict = {line.split(" ")[0]: float(line.split(" ")[1]) for line in line_list
                      if line.split(" ")[0] in PARAMS}
        return param_dict

def get_trajectories(data_dict:dict) -> dict:
    """
    Gets the reorientation trajectories

    Parameters:
    * `data_dict`: The data dictionary
    
    Return trajectories as a dictoinary of lists of euler angles
    """
    grain_ids = [int(key.replace("g","").replace("_phi_1","")) for key in data_dict.keys() if "_phi_1" in key]
    trajectories = {}
    for grain_id in grain_ids:
        trajectory = [data_dict[f"g{grain_id}_{phi}"] for phi in ["phi_1", "Phi", "phi_2"]]
        trajectory = transpose(trajectory)
        trajectories[grain_id] = trajectory
    return trajectories

def convert_grain_ids(data_dict:dict, grain_map_path:str) -> dict:
    """
    Converts the grain IDs of a dictionary
    
    Parameters:
    * `data_dict`:      The dictionary
    * `grain_map_path`: The path to the grain map
    
    Returns the dictionary with renamed keys
    """
    
    # Initialise conversion
    grain_map = csv_to_dict(grain_map_path)
    new_data_dict = {}
    mesh_to_ebsd = dict(zip(grain_map["mesh_id"], grain_map["ebsd_id"]))

    # Iterate through keys
    for key in data_dict:
        if bool(re.match(r'^g\d+.*$', key)):
            key_list = key.split("_")
            mesh_id = int(key_list[0].replace("g",""))
            new_key = f"g{int(mesh_to_ebsd[mesh_id])}_{'_'.join(key_list[1:])}"
            new_data_dict[new_key] = data_dict[key]
        else:
            new_data_dict[key] = data_dict[key]
    
    # Return
    return new_data_dict

def fix_angle(angle:float, l_bound:float=0.0, u_bound:float=2*math.pi) -> float:
    """
    Fixes an angle between two bounds
    
    Parameters:
    * `angle`: The angle (rads)

    Returns the fixed angle
    """
    if abs(angle-l_bound) < 1e-4 or abs(angle-u_bound) < 1e-4:
        return angle
    range = u_bound - l_bound
    if l_bound < angle and angle < u_bound:
        return angle
    elif angle < l_bound:
        return fix_angle(angle+range, l_bound, u_bound)
    else:
        return fix_angle(angle-range, l_bound, u_bound)

def process_data_dict(data_dict:dict, num_strains:int=NUM_STRAINS) -> dict:
    """
    Processes the data in the data dictionary;
    normalises strain difference
    
    Parameters:
    * `data_dict`:   The dictionary of data
    * `num_strains`: The number of strains to intervaluate

    Returns the processed dictionary
    """
    
    # Prepare old and new strain values
    strain_list = data_dict[STRAIN_FIELD]
    max_strain = max([s for s in data_dict[STRAIN_FIELD] if s < MAX_STRAIN])
    new_strain_list = list(np.linspace(0, max_strain, num_strains+1)[1:])
    
    # Prepare fields
    grain_ids = [int(key.replace("g","").replace("_phi_1","")) for key in data_dict.keys() if "_phi_1" in key]
    euler_fields = [f"g{grain_id}_{field}" for grain_id in grain_ids for field in ["phi_1", "Phi", "phi_2"]]
    field_list = [STRESS_FIELD] + euler_fields
    
    # Interpolate for each field and return
    processed_dict = {STRAIN_FIELD: new_strain_list}
    for field in field_list:
        field_itp = Interpolator(strain_list, data_dict[field], len(strain_list))
        new_list = field_itp.evaluate(new_strain_list)
        processed_dict[field] = new_list
    
    # Fix the domain of the euler-bunge angles
    for field in euler_fields:
        processed_dict[field] = [fix_angle(phi) for phi in processed_dict[field]]
    
    # Return processed dictionary
    return processed_dict

# Main function
if __name__ == "__main__":
    main()
