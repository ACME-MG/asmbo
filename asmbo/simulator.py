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
             opt_params:list, num_processors:list, max_time:float, mat_model:str,
             sim_model:str):
    """
    Trains a surrogate model
    
    Parameters:
    * `sim_path`:       Path to store simulation results
    * `mesh_path`:      Path to the mesh
    * `exp_path`:       Path to the experimental data
    * `param_names`:    List of parameter names
    * `opt_params`:     List of optimised parameters
    * `num_processors`: Number of processors to use
    * `max_float`:      Maximum time to run the simulation before terminating
    * `mat_model`:      Name of the material model
    * `sim_model`:      Name of the simulation model
    """
    
    # Initialise interface
    itf = Interface(input_path=mesh_path, output_here=True, verbose=True)
    itf.__output_path__ = sim_path
    itf.__get_output__ = lambda x : f"{itf.__output_path__}/{x}"

    # Define mesh
    itf.define_mesh("mesh.e", "element_stats.csv", degrees=False, active=False)
    dimensions = itf.get_dimensions()

    # Defines the material parameters
    param_dict = dict(zip(param_names, opt_params))
    param_dict["cp_gamma_0"] = 3.25e-5
    itf.define_material(
        material_path   = mat_model,
        material_params = param_dict,
        c_11            = 250000,
        c_12            = 151000,
        c_44            = 123000,
    )
    
    # Defines the simulation parameters
    exp_dict = csv_to_dict(exp_path)
    eng_strain = math.exp(exp_dict["strain_intervals"][-1])-1
    itf.define_simulation(
        simulation_path = sim_model,
        end_time        = exp_dict["time_intervals"][-1],
        end_strain      = eng_strain*dimensions["x"]
    )

    # Runs the model and saves results
    itf.export_params()
    itf.simulate("~/moose/deer/deer-opt", num_processors, max_time)

    # Conduct post processing
    itf.compress_csv(sf=5, exclude=["x", "y", "z"])
    itf.post_process(grain_map_path=f"{mesh_path}/grain_map.csv")
    itf.remove_files(["mesh.e", "element_stats.csv", "results", "simulation_out_cp"])
