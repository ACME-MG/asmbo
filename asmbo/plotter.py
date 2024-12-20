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

def plot_results(sim_path:str, exp_path:str, grain_ids:list, strain_field:str, stress_field:str):
    """
    Plots the simulation results
    
    Parameters:
    * `sim_path`:     Path that stores the simulation results
    * `exp_path`:     Path to the experimental data
    * `grain_ids`:    List of grain IDs to conduct the training
    * `strain_field`: Name of the field for the strain data
    * `stress_field`: Name of the field for the stress data
    """

    # Get all results
    res_dict = csv_to_dict(f"{sim_path}/summary.csv")
    exp_dict = csv_to_dict(exp_path)
    get_trajectories = lambda dict, grain_ids : [transpose([dict[f"g{grain_id}_{phi}"] for phi in ["phi_1", "Phi", "phi_2"]]) for grain_id in grain_ids]

    # Initialise IPF
    ipf = IPF(get_lattice("fcc"))
    direction = [1,0,0]

    # Plot experimental reorientation trajectories
    exp_trajectories = get_trajectories(exp_dict, grain_ids)
    ipf.plot_ipf_trajectory(exp_trajectories, direction, "plot", {"color": "silver", "linewidth": 2})
    ipf.plot_ipf_trajectory(exp_trajectories, direction, "arrow", {"color": "silver", "head_width": 0.01, "head_length": 0.015})
    ipf.plot_ipf_trajectory([[et[0]] for et in exp_trajectories], direction, "scatter", {"color": "silver", "s": 8**2})
    for exp_trajectory, grain_id in zip(exp_trajectories, grain_ids):
        ipf.plot_ipf_trajectory([[exp_trajectory[0]]], direction, "text", {"color": "black", "fontsize": 8, "s": grain_id})

    # Plot calibration reorientation trajectories
    cal_trajectories = get_trajectories(res_dict, grain_ids)
    ipf.plot_ipf_trajectory(cal_trajectories, direction, "plot", {"color": "green", "linewidth": 1, "zorder": 3})
    ipf.plot_ipf_trajectory(cal_trajectories, direction, "arrow", {"color": "green", "head_width": 0.0075, "head_length": 0.0075*1.5, "zorder": 3})
    ipf.plot_ipf_trajectory([[ct[0]] for ct in cal_trajectories], direction, "scatter", {"color": "green", "s": 6**2, "zorder": 3})

    # Save IPF
    define_legend(["silver", "green", "red"], ["Experimental", "Calibration", "Validation"], ["scatter", "line", "line"])
    save_plot(f"{sim_path}/plot_opt_rt.png")

    # Plot stress-strain curve
    res_dict["strain"] = res_dict[strain_field]
    res_dict["stress"] = res_dict[stress_field]
    plotter = Plotter("strain", "stress", "mm/mm", "MPa")
    plotter.prep_plot()
    plotter.scat_plot(exp_dict, "silver", "Experimental")
    plotter.line_plot(res_dict, "green", "Calibration")
    plotter.set_legend()
    save_plot(f"{sim_path}/plot_opt_ss.png")
