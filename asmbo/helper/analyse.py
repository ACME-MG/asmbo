"""
 Title:         Analyse
 Description:   Analysis related functions for CPFEM simulations
 Author:        Janzen Choi

"""

# Libraries
import math, numpy as np
import matplotlib.pyplot as plt
from asmbo.helper.general import transpose
from asmbo.helper.interpolator import intervaluate
from asmbo.helper.orientation import euler_to_quat, get_geodesic

def get_stress(stress_list_1:list, stress_list_2:list, strain_list_1:list,
               strain_list_2:list, eval_strains:list) -> list:
    """
    Gets the stress error from two lists of stresses;
    normalisation conducted on first list of stresses

    Parameters:
    * `stress_list_1`: First list of stresses
    * `stress_list_2`: Second list of stresses
    * `strain_list_1`: First list of strain values
    * `strain_list_2`: Second list of strain values
    * `eval_strains`:  List of strains to evaluate
    
    Returns the normalised root mean square error for the stresses
    """
    max_strain = max(eval_strains)
    stress_list = [stress for stress, strain in zip(stress_list_1, strain_list_1) if strain <= max_strain]
    eval_stress_list_1 = intervaluate(strain_list_1, stress_list_1, eval_strains)
    eval_stress_list_2 = intervaluate(strain_list_2, stress_list_2, eval_strains)
    mse = np.average([math.pow(es_1-es_2, 2) for es_1, es_2 in zip(eval_stress_list_1, eval_stress_list_2)])
    nrmse = math.sqrt(mse)/np.average(stress_list)
    return nrmse

def get_geodesics(grain_ids:list, data_dict_1:dict, data_dict_2:dict,
                  strain_list_1:list, strain_list_2:list, eval_strains:list) -> list:
    """
    Gets the geodesic distances from two sets of data
    
    Parameters:
    * `grain_ids`:     List of grain IDs to conduct evaluation
    * `data_dict_1`:   First set of data
    * `data_dict_2`:   Second set of data
    * `strain_list_1`: First list of strain values
    * `strain_list_2`: Second list of strain values
    * `eval_strains`:  List of strains to evaluate
    
    Returns list of lists of geodesic distances
    """

    # Initialise
    geodesic_grid = []

    # Iterate through grain IDs
    for grain_id in grain_ids:

        # Get list of euler angles at specific strains
        quick_ie = lambda data_dict, strain_list : intervaluate_eulers(*[data_dict[f"g{grain_id}_{phi}"]
                   for phi in ["phi_1", "Phi", "phi_2"]], strain_list, eval_strains)
        euler_list_1 = quick_ie(data_dict_1, strain_list_1)
        euler_list_2 = quick_ie(data_dict_2, strain_list_2)
        
        # Calculate geodesic distances of orientations at the same strains
        geodesic_list = []
        for euler_1, euler_2 in zip(euler_list_1, euler_list_2):
            quat_1 = euler_to_quat(euler_1)
            quat_2 = euler_to_quat(euler_2)
            geodesic = get_geodesic(quat_1, quat_2)
            geodesic_list.append(geodesic)
        geodesic_grid.append(geodesic_list)

    # Return list of lists of geodesic distances
    return geodesic_grid

def intervaluate_eulers(phi_1_list:list, Phi_list:list, phi_2_list:list,
                       strain_list:list, eval_strains:list) -> list:
    """
    Interpolates the euler-bunge (rads) components and evaluates
    the components and certain strain values

    Parameters:
    * `phi_1_list`:   List of phi_1 components
    * `Phi_list`:     List of Phi components
    * `phi_2_list`:   List of phi_2 components
    * `strain_list`:  List of strain values to interpolate
    * `eval_strains`: List of strain values to evaluate
    
    Returns the evaluated orientations as a list of euler-bunge values (in lists)
    """
    phi_1_list = intervaluate(strain_list, phi_1_list, eval_strains)
    Phi_list = intervaluate(strain_list, Phi_list, eval_strains)
    phi_2_list = intervaluate(strain_list, phi_2_list, eval_strains)
    euler_list = transpose([phi_1_list, Phi_list, phi_2_list])
    return euler_list

def plot_boxplots(y_list_list:list, colours:list) -> None:
    """
    Plots several boxplots together

    Parameters:
    * `y_list_list`: List of data lists
    * `colours`:     List of colours
    """

    # Plot boxplots
    boxplots = plt.boxplot(y_list_list, showfliers=False, patch_artist=True, vert=True, widths=0.7)
    
    # Add scattered data and format each boxplot
    for i, (y_list, colour) in enumerate(zip(y_list_list, colours)):

        # Format boxplot face
        patch = boxplots["boxes"][i]
        patch.set_facecolor(colour)
        patch.set_alpha(0.4)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.5)

        # Format median line
        median = boxplots["medians"][i]
        median.set(color="black", linewidth=1)

        # Add scattered data
        x_list = np.random.normal(i+1, 0.04, size=len(y_list))
        plt.scatter(x_list, y_list, s=4**2, color=colour, edgecolors="black")
