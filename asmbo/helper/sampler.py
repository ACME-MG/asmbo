"""
 Title:         Sampler
 Description:   Auxiliary functions for running multiple simulations;
                Not linked to the interface
 Author:        Janzen Choi

"""

# Libraries
import pyDOE2 # type: ignore

def linear_scale(value:float, in_l_bound:float, in_u_bound:float, out_l_bound:float, out_u_bound:float) -> float:
    """
    Linearly scales a value

    Parameters:
    * `value`:       The value to be linearly scaled
    * `in_l_bound`:  The lower bound of the input values
    * `in_u_bound`:  The upper bound of the input values
    * `out_l_bound`: The lower bound of the linearly scaled values
    * `out_u_bound`: The upper bound of the linearly scaled values
    """
    in_range = in_u_bound - in_l_bound
    out_range = out_u_bound - out_l_bound
    scaled_value = (value-in_l_bound)*out_range/in_range + out_l_bound
    return scaled_value

def get_lhs(param_bounds:dict, num_points:int) -> list:
    """
    Generates points using latin hypercube sampling
    
    Parameters:
    * `param_bounds`: Dictionary of parameter bounds;
                      (i.e., `{"name": (l_bound, u_bound)}`)
    * `num_points`:   The number of points to sample
    
    Returns the list of dictionaries of parameter combinations
    """
    
    # Get unscaled LHS points
    num_params = len(param_bounds.keys())
    combinations = list(pyDOE2.lhs(num_params, samples=num_points))
    
    # Create linear scales for each parameter
    param_scales = {}
    for param in param_bounds.keys():
        param_scales[param] = lambda x : linear_scale(x, 0, 1, param_bounds[param][0], param_bounds[param][1])

    # Linearly scale the unscaled combinations
    scaled_dict_list = []
    for combination in combinations:
        scaled_dict = {}
        for i, param in enumerate(param_scales.keys()):
            scaled_param = param_scales[param](combination[i])
            scaled_dict[param] = scaled_param
        scaled_dict_list.append(scaled_dict)

    # Return scaled LHS points
    return scaled_dict_list

def get_ccd(param_bounds:dict, centre_points:int=4, alpha:str="r") -> list:
    """
    Generates points using central composite design

    Parameters:
    * `param_bounds`:  Dictionary of parameter bounds;
                       (i.e., `{"name": (l_bound, u_bound)}`)
    * `centre_points`: The number of centre points (usually 3-6)
    * `alpha`:         The effect on the variance ("r", "o")

    Returns the list of dictionaries of parameter combinations 
    """

    # Get unscaled CCD
    num_params = len(param_bounds.keys())
    combinations = pyDOE2.ccdesign(num_params, center=(centre_points,centre_points), alpha=alpha, face="ccc")

    # Create linear scales for each parameter
    param_scales = {}
    for param in param_bounds.keys():
        param_scales[param] = lambda x : linear_scale(x, -1, 1, param_bounds[param][0], param_bounds[param][1])

    # Linearly scale the unscaled combinations
    scaled_dict_list = []
    for combination in combinations:
        scaled_dict = {}
        for i, param in enumerate(param_scales.keys()):
            scaled_param = param_scales[param](combination[i])
            scaled_dict[param] = scaled_param
        scaled_dict_list.append(scaled_dict)

    # Return scaled CCD
    return scaled_dict_list

def get_domains(param_dict_list:list) -> dict:
    """
    Gets the domains of the sampled parameter combinations

    Parameters:
    * `param_dict_list`: List of dictionaries of parameter combinations
    
    Returns a dictionary of the parameter domains
    """
    domain_dict = {}
    for param_dict in param_dict_list:
        for param in param_dict.keys():
            if not param in domain_dict.keys():
                domain_dict[param] = []
            if not param_dict[param] in domain_dict[param]:
                domain_dict[param].append(param_dict[param])
    for param in domain_dict.keys():
        domain_dict[param].sort()
    return domain_dict
