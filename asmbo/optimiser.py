"""
 Title:         Optimiser
 Description:   Optimises a surrogate model
 Author:        Janzen Choi

"""

# Libraries
from asmbo.paths import OPT_PATH, MMS_PATH
import sys; sys.path += [OPT_PATH]
from opt_all.interface import Interface
from matplotlib.pyplot import figure
from asmbo.helper.general import transpose
from asmbo.helper.plotter import define_legend, save_plot
from asmbo.helper.pole_figure import get_lattice, IPF

def optimise(train_path:str, opt_path:str, exp_path:str, max_strain:float, grain_ids:list):
    """
    Trains a surrogate model
    
    Parameters:
    * `train_path`:   Path that stores the training results
    * `opt_path`:     Path to store the optimisation results
    * `exp_path`:     Path to the experimental data
    * `max_strain`:   Maximum strain to consider
    * `grain_ids`:    List of grain IDs to conduct the training
    """

    # Initialise interface
    itf = Interface(input_path=".", output_here=True, verbose=False)
    itf.__output_path__ = opt_path
    itf.__get_output__ = lambda x : f"{itf.__output_path__}/{x}"

    # Define mmodel
    itf.define_model(
        model_name = f"sm_617_s3_lh2",
        mms_path   = MMS_PATH,
        sm_path    = f"{train_path}/sm.pt",
        map_path   = f"{train_path}/map.csv",
        exp_path   = exp_path
    )

    # Bind parameters
    itf.bind_param("lh_0",    0, 400)
    itf.bind_param("lh_1",    0, 400)
    itf.bind_param("tau_0",   0, 200)
    itf.bind_param("n",       1, 16)
    itf.bind_param("gamma_0", 0, 1e-4)
    
    # Read data
    itf.read_data(exp_path, thin_data=False)
    itf.remove_data("strain", max_strain)
    
    # Add errors
    itf.add_error("area", labels=["strain", "stress"], group="curve", max_value=0.1, weight=3.1415*len(grain_ids))
    for i in grain_ids:
        itf.add_error(
            error_name  = "geodesic",
            labels      = ["strain_intervals"] + [f"g{i}_{phi}" for phi in ["phi_1", "Phi", "phi_2"]],
            eval_x_list = [0.02, 0.04, 0.06, 0.08, 0.1],
            group       = f"g{i}",
        )

    def plot_ipf(exp_dict:dict, sim_dict:dict, output_path:str) -> None:
        """
        Plots an IPF plot

        Parameters:
        * `exp_dict`:    Dictionary containing the experimental data
        * `sim_dict`:    Dictionary containing the simulated data
        * `output_path`: Path to output the plot
        """
        
        # Initialise
        figure()
        ipf = IPF(get_lattice("fcc"))
        direction = [1,0,0]
        max_strain = 0.1
        get_trajectories = lambda dict, grain_ids, strain_field : [
            transpose([[g for g, s in zip(dict[f"g{grain_id}_{phi}"], dict[strain_field]) if s <= max_strain] for phi in ["phi_1", "Phi", "phi_2"]])
        for grain_id in grain_ids]

        # Plot experimental reorientation trajectories
        exp_trajectories = get_trajectories(exp_dict, grain_ids, "strain_intervals")
        ipf.plot_ipf_trajectory(exp_trajectories, direction, "plot", {"color": "silver", "linewidth": 2})
        ipf.plot_ipf_trajectory(exp_trajectories, direction, "arrow", {"color": "silver", "head_width": 0.01, "head_length": 0.015})
        ipf.plot_ipf_trajectory([[et[0]] for et in exp_trajectories], direction, "scatter", {"color": "silver", "s": 8**2})
        for exp_trajectory, grain_id in zip(exp_trajectories, grain_ids):
            ipf.plot_ipf_trajectory([[exp_trajectory[0]]], direction, "text", {"color": "black", "fontsize": 8, "s": grain_id})

        # Plot simulation reorientation trajectories
        sim_trajectories = get_trajectories(sim_dict, grain_ids, "strain")
        ipf.plot_ipf_trajectory(sim_trajectories, direction, "plot", {"color": "green", "linewidth": 1, "zorder": 3})
        ipf.plot_ipf_trajectory(sim_trajectories, direction, "arrow", {"color": "green", "head_width": 0.0075, "head_length": 0.0075*1.5, "zorder": 3})
        ipf.plot_ipf_trajectory([[st[0]] for st in sim_trajectories], direction, "scatter", {"color": "green", "s": 6**2, "zorder": 3})
        
        # Save plot
        define_legend(["silver", "green"], ["Experimental", "Simulation"], ["scatter", "line"])
        save_plot(f"{output_path}/plot_rt.png")

    # Record plots
    itf.record_plot("strain", "stress")
    itf.set_function(plot_ipf)

    # Optimise and record
    itf.start_recorder(interval=1000)
    # itf.optimise("moga", num_gens=1000, population=100, offspring=100, crossover=0.8, mutation=0.01)
    itf.optimise("moga", num_gens=2, population=100, offspring=100, crossover=0.8, mutation=0.01)
