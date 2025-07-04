"""
 Title:         Trainer
 Description:   Trains a surrogate model based on the results of the sampled simulations
 Author:        Janzen Choi

"""

# Libraries
from asmbo.paths import MMS_PATH
import sys; sys.path += [MMS_PATH]
from mms.interface import Interface
from asmbo.helper.io import dict_to_csv

def train(train_dict:dict, train_path:str, param_names:list, grain_ids:list,
          strain_field:str, stress_field:str, num_threads:int):
    """
    Trains a surrogate model
    
    Parameters:
    * `train_dict`:   Dictionary containing training data
    * `train_path`:   Path to store the training results
    * `param_names`:  List of parameter names
    * `grain_ids`:    List of grain IDs to conduct the training
    * `strain_field`: Name of the field for the strain data
    * `stress_field`: Name of the field for the stress data
    * `num_threads`:  The number of threads to use
    """

    # Initialise interface
    itf = Interface(input_path=".", output_here=True, verbose=True)
    itf.__output_path__ = train_path
    itf.__get_output__ = lambda x : f"{itf.__output_path__}/{x}"

    # Read data
    sampled_data_path = f"{train_path}/sampled_data.csv"
    dict_to_csv(train_dict, sampled_data_path)
    itf.read_data(sampled_data_path)

    # Identify and scale inputs
    input_list = param_names + [strain_field]
    for input in input_list:
        itf.add_input(input, ["log", "linear"])
    
    # Identify and scale output
    output_list = [stress_field] + [f"g{grain_id}_{field}" for grain_id in grain_ids for field in ["phi_1", "Phi", "phi_2"]]
    for output in output_list:
        itf.add_output(output, ["log", "linear"])

    # Train surrogate model
    itf.set_num_threads(num_threads)
    itf.define_surrogate("kfold_2", "cpu", num_splits=5, epochs=1000, batch_size=32, verbose=True)
    itf.add_training_data()
    itf.train()
    itf.plot_loss_history()

    # Save surrogate model and mapping
    itf.save("sm")
    itf.export_maps("map")

    # Validate the trained model
    itf.get_validation_data()
    itf.print_validation(use_log=True, print_table=False)
    itf.plot_validation(
        headers   = [stress_field],
        label     = "Stress (MPa)",
        use_log   = False,
        plot_file = "stress"
    )
    # itf.plot_validation(
    #     headers   = [f"g{grain_id}_{phi}" for grain_id in grain_ids for phi in ["phi_1", "Phi", "phi_2"]],
    #     label     = "Orientation (rads)",
    #     use_log   = False,
    #     plot_file = "all_phi"
    # )
