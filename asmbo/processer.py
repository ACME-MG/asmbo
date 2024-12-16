"""
 Title:         Process
 Description:   Processes the results of the sampled simulations
 Author:        Janzen Choi

"""

# Libraries
import math, numpy as np
from asmbo.helper.interpolator import Interpolator
from asmbo.helper.general import round_sf
from asmbo.helper.io import csv_to_dict

def process(sim_path:str, param_names:list, strain_field:str, stress_field:str,
            num_strains:int, max_strain:float) -> dict:
    """
    Summarises multiple simulations

    Parameters:
    * `sim_path`:     Path to the simulation summary
    * `param_names`:  List of parameter names
    * `strain_field`: Name of the field for the strain data
    * `stress_field`: Name of the field for the stress data
    * `num_strains`:  Number of strains to interpolate
    * `max_strain`:   Maximum strain to consider
    
    Returns the summary as a dictionary
    """

    # Read files
    sim_dict = csv_to_dict(f"{sim_path}/summary.csv")
    param_dict = get_param_dict(f"{sim_path}/params.txt", param_names)

    # Initialise strains
    strain_list = sim_dict[strain_field]
    max_strain = max([s for s in sim_dict[strain_field] if s <= max_strain])
    new_strain_list = list(np.linspace(0, max_strain, num_strains+1)[1:])
    
    # Prepare fields
    grain_ids = [int(key.replace("g","").replace("_phi_1","")) for key in sim_dict.keys() if "_phi_1" in key]
    euler_fields = [f"g{grain_id}_{field}" for grain_id in grain_ids for field in ["phi_1", "Phi", "phi_2"]]
    field_list = [stress_field] + euler_fields
    
    # Interpolate for each field and return
    processed_dict = {strain_field: new_strain_list}
    for field in field_list:
        field_itp = Interpolator(strain_list, sim_dict[field], len(strain_list))
        new_list = field_itp.evaluate(new_strain_list)
        processed_dict[field] = new_list
    
    # Fix the domain of the euler-bunge angles
    for field in euler_fields:
        processed_dict[field] = [fix_angle(phi) for phi in processed_dict[field]]
    
    # Combine, compress, and return
    combined_dict = {**param_dict, **processed_dict}
    for key in combined_dict:
        combined_dict[key] = round_sf(combined_dict[key], 5)
    return combined_dict

def get_param_dict(params_path:str, param_names:list) -> dict:
    """
    Gets the parameters from a file
    
    Parameters:
    * `params_path`: The path to the parameters file
    * `param_names`: List of parameter names
    
    Returns a dictionary of the parameter values
    """
    with open(params_path, "r") as fh:
        line_list = [line.replace("\n", "").replace(":", "") for line in fh.readlines()]
        param_dict = {line.split(" ")[0]: float(line.split(" ")[1]) for line in line_list
                      if line.split(" ")[0] in param_names}
        return param_dict

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
