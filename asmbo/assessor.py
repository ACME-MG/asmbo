"""
 Title:         Assessor
 Description:   Assesses the CPFEM simulation results
 Author:        Janzen Choi

"""

# Libraries
import math, numpy as np, os
from asmbo.paths import MMS_PATH
import sys; sys.path += [MMS_PATH]
from asmbo.helper.analyse import get_stress, get_geodesics
from asmbo.helper.surrogate import Model
from asmbo.helper.io import csv_to_dict, dict_to_csv

# Constants
PARAM_NAMES = ["cp_lh_0", "cp_lh_1", "cp_tau_0", "cp_n", "cp_gamma_0"]

def assess(params_list:list, sm_path:str, exp_path:str, max_strain:float, grain_ids:list) -> dict:
    """
    Assesses the surrogate model using previously optimised simulation results

    Parameters:
    * `params_list`: List of parameter dictionaries
    * `sm_path`:     Path storing the surrogate model
    * `exp_path`:    Path to the experimental data
    * `max_strain`:  Maximum strain to consider
    * `grain_ids`:   List of grain IDs to conduct the training
    * `sim_keyword`: Keyword to identify directories containing simulation results

    Returns a dictionary of the best parameters
    """

    # Check if there are any parameters
    if params_list == []:
        return None

    # Initialise other information
    exp_dict = csv_to_dict(exp_path)
    eval_strains = list(np.linspace(0, max_strain, 50))
    fields = ["iteration"] + PARAM_NAMES + ["stress_error", "geodesic_error", "reduced_error"]
    error_dict = dict(zip(fields, [[] for _ in range(len(fields))]))

    # Calculate the errors for each set of parameters
    for params_dict in params_list:

        # Define model
        model = Model(
            sm_path    = f"{sm_path}/sm.pt",
            map_path   = f"{sm_path}/map.csv",
            exp_path   = exp_path,
            max_strain = max_strain,
        )

        # Evaluate the model's performance
        param_values = [params_dict[param_name] for param_name in PARAM_NAMES]
        res_dict = model.get_response(param_values)

        # Calculate stress error
        stress_error = get_stress(
            stress_list_1 = exp_dict["stress"],
            stress_list_2 = res_dict["stress"],
            strain_list_1 = exp_dict["strain"],
            strain_list_2 = res_dict["strain"],
            eval_strains  = eval_strains,
        )

        # Calculate orientation error
        geodesic_grid = get_geodesics(
            grain_ids     = grain_ids,
            data_dict_1   = exp_dict,
            data_dict_2   = res_dict,
            strain_list_1 = exp_dict["strain_intervals"],
            strain_list_2 = res_dict["strain"],
            eval_strains  = eval_strains
        )
        geodesic_error = np.average([np.sqrt(np.average([g**2 for g in gg])) for gg in geodesic_grid])

        # Add information
        error_dict["iteration"].append(len(error_dict["iteration"])+1)
        for param_name in PARAM_NAMES:
            error_dict[param_name].append(params_dict[param_name])
        error_dict["stress_error"].append(stress_error)
        error_dict["geodesic_error"].append(geodesic_error)
        error_dict["reduced_error"].append(stress_error+geodesic_error/math.pi)
        
    # Save error information
    dict_to_csv(error_dict, f"{sm_path}/errors.csv")

    # Return best parameters
    min_index = error_dict["reduced_error"].index(min(error_dict["reduced_error"]))
    return params_list[min_index]

def calculate_errors(sim_path:str, exp_path:str, grain_ids:list, max_strain:float,
                     strain_field:str, stress_field:str) -> tuple:
    """
    Calculates the errors of the simulation results
    
    Parameters:
    * `sim_path`:     Path that stores the simulation results
    * `exp_path`:     Path to the experimental data
    * `grain_ids`:    List of grain IDs to conduct the training
    * `max_strain`:   Maximum strain to consider
    * `strain_field`: Name of the field for the strain data
    * `stress_field`: Name of the field for the stress data

    Returns the stress error and orientation error
    """

    # Get all results
    sim_dict = csv_to_dict(f"{sim_path}/summary.csv")
    exp_dict = csv_to_dict(exp_path)

    # Determine strains to consider
    eval_strain_list = list(np.linspace(0, max_strain, 50))
    
    # Calculate errors
    stress_error = get_stress(exp_dict["stress"], sim_dict[strain_field], exp_dict["strain"], sim_dict[stress_field], eval_strain_list)
    geodesic_grid = get_geodesics(grain_ids, exp_dict, sim_dict, exp_dict["strain_intervals"], sim_dict["average_strain"], eval_strain_list)
    geodesic_error = np.average([np.sqrt(np.average([g**2 for g in gg])) for gg in geodesic_grid])

    # Return
    return stress_error, geodesic_error
