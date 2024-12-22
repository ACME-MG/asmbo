"""
 Title:         Plotter
 Description:   Plots the CPFEM simulation results
 Author:        Janzen Choi

"""

# Libraries
from asmbo.helper.io import csv_to_dict
from asmbo.helper.general import transpose
from asmbo.helper.pole_figure import get_lattice, IPF
from asmbo.helper.plotter import define_legend, save_plot, Plotter

def plot_results(sim_path:str, exp_path:str, cal_grain_ids:list,
                 val_grain_ids:list, strain_field:str, stress_field:str):
    """
    Plots the simulation results
    
    Parameters:
    * `sim_path`:      Path that stores the simulation results
    * `exp_path`:      Path to the experimental data
    * `cal_grain_ids`: List of grain IDs to conduct the calibration
    * `val_grain_ids`: List of grain IDs to conduct the validation
    * `strain_field`:  Name of the field for the strain data
    * `stress_field`:  Name of the field for the stress data
    """

    # Get all results
    res_dict = csv_to_dict(f"{sim_path}/summary.csv")
    exp_dict = csv_to_dict(exp_path)

    # Plot reorientation trajectories
    plot_trajectories(exp_dict, res_dict, cal_grain_ids, "green", "Calibration", f"{sim_path}/plot_opt_cal_rt.png")
    plot_trajectories(exp_dict, res_dict, val_grain_ids, "red",   "Validation",  f"{sim_path}/plot_opt_val_rt.png")

    # Plot stress-strain curve
    res_dict["strain"] = res_dict[strain_field]
    res_dict["stress"] = res_dict[stress_field]
    plotter = Plotter("strain", "stress", "mm/mm", "MPa")
    plotter.prep_plot()
    plotter.scat_plot(exp_dict, "silver", "Experimental")
    plotter.line_plot(res_dict, "green", "Calibration")
    plotter.set_legend()
    save_plot(f"{sim_path}/plot_opt_ss.png")

def plot_trajectories(exp_dict:dict, sim_dict:dict, grain_ids:list, sim_colour:str,
                      sim_label:str, path:str) -> None:
    """
    Plots the experimental and simulated reorientation trajectories

    Parameters:
    * `exp_dict`:   Dictionary of experimental data
    * `sim_dict`:   Dictionary of simulated data
    * `grain_ids`:  List of grain IDs
    * `sim_colour`: Colour to plot the simulated trajectories
    * `sim_label`:  Label for the simulated data
    * `path`:       Path to save the plot
    """

    # Initialise IPF
    ipf = IPF(get_lattice("fcc"))
    direction = [1,0,0]
    get_trajectories = lambda dict : [transpose([dict[f"g{grain_id}_{phi}"] for phi in ["phi_1", "Phi", "phi_2"]]) for grain_id in grain_ids]

    # Plot experimental reorientation trajectories
    exp_trajectories = get_trajectories(exp_dict)
    ipf.plot_ipf_trajectory(exp_trajectories, direction, "plot", {"color": "silver", "linewidth": 2})
    ipf.plot_ipf_trajectory(exp_trajectories, direction, "arrow", {"color": "silver", "head_width": 0.01, "head_length": 0.015})
    ipf.plot_ipf_trajectory([[et[0]] for et in exp_trajectories], direction, "scatter", {"color": "silver", "s": 8**2})
    for exp_trajectory, grain_id in zip(exp_trajectories, grain_ids):
        ipf.plot_ipf_trajectory([[exp_trajectory[0]]], direction, "text", {"color": "black", "fontsize": 8, "s": grain_id})

    # Plot calibration reorientation trajectories
    sim_trajectories = get_trajectories(sim_dict)
    ipf.plot_ipf_trajectory(sim_trajectories, direction, "plot", {"color": sim_colour, "linewidth": 1, "zorder": 3})
    ipf.plot_ipf_trajectory(sim_trajectories, direction, "arrow", {"color": sim_colour, "head_width": 0.0075, "head_length": 0.0075*1.5, "zorder": 3})
    ipf.plot_ipf_trajectory([[st[0]] for st in sim_trajectories], direction, "scatter", {"color": sim_colour, "s": 6**2, "zorder": 3})

    # Save IPF
    define_legend(["silver", sim_colour], ["Experimental", sim_label], ["scatter", "line"])
    save_plot(path)
