"""
 Title:         Simulator
 Description:   Simulates the CPFEM simulation based on the optimised surrogate model
 Author:        Janzen Choi

"""

# Libraries
from asmbo.paths import SIM_PATH
import sys; sys.path += [SIM_PATH]
from moose_sim.interface import Interface
from asmbo.helper.io import csv_to_dict
import math

def simulate(sim_path:str, mesh_path:str, exp_path:str, param_names:list,
             opt_params:list, num_processors:list):
    """
    Trains a surrogate model
    
    Parameters:
    * `sim_path`:       Path to store simulation results
    * `mesh_path`:      Path to the mesh
    * `exp_path`:       Path to the experimental data
    * `param_names`:    List of parameter names
    * `opt_params`:     List of optimised parameters
    * `num_processors`: Number of processors to use
    """
    
    # Initialise interface
    itf = Interface(input_path=mesh_path, output_here=True, verbose=False)
    itf.__output_path__ = sim_path
    itf.__get_output__ = lambda x : f"{itf.__output_path__}/{x}"

    # Define mesh
    itf.define_mesh("mesh.e", "element_stats.csv", degrees=False, active=False)
    dimensions = itf.get_dimensions()

    # Defines the material parameters
    itf.define_material(
        material_path   = "deer/cplh_ae",
        material_params = dict(zip(param_names, opt_params)),
        c_11            = 250000,
        c_12            = 151000,
        c_44            = 123000,
    )
    
    # Defines the simulation parameters
    exp_dict = csv_to_dict(exp_path)
    eng_strain = math.exp(exp_dict["strain_intervals"][-1])-1
    itf.define_simulation(
        simulation_path = "deer/1to1_ui_cp",
        end_time        = exp_dict["time_intervals"][-1],
        end_strain      = eng_strain*dimensions["x"]
    )

    # Runs the model and saves results
    itf.export_params()
    itf.simulate("~/moose/deer/deer-opt", num_processors, 100000)

    # Conduct post processing
    itf.compress_csv(sf=5, exclude=["x", "y", "z"])
    itf.post_process(grain_map_path=f"{mesh_path}/grain_map.csv")
    itf.remove_files(["mesh.e", "element_stats.csv", "results", "simulation_out_cp"])