"""
 Title:         Model Information
 Description:   Contains constants and parameters for each model
 Author:        Janzen Choi

"""

# VH information
VH_PARAM_INFO = [
    {"name": "cp_tau_s", "bounds": (0, 2000)},
    {"name": "cp_b",     "bounds": (0, 10)},
    {"name": "cp_tau_0", "bounds": (0, 500)},
    {"name": "cp_n",     "bounds": (1, 20)},
]
VH_OPT_MODEL   = "sm_617_s3_vh"
VH_MAT_MODEL   = "deer/cpvh_ae"

# LH2 information
LH2_PARAM_INFO = [
    {"name": "cp_lh_0",  "bounds": (0, 1000)},
    {"name": "cp_lh_1",  "bounds": (0, 1000)},
    {"name": "cp_tau_0", "bounds": (0, 500)},
    {"name": "cp_n",     "bounds": (1, 20)},
]
LH2_OPT_MODEL   = "sm_617_s3_lh2"
LH2_MAT_MODEL   = "deer/cplh_ae"

# LH6 information
LH6_PARAM_INFO = [
    {"name": "cp_lh_0",  "bounds": (0, 1000)},
    {"name": "cp_lh_1",  "bounds": (0, 1000)},
    {"name": "cp_lh_2",  "bounds": (0, 1000)},
    {"name": "cp_lh_3",  "bounds": (0, 1000)},
    {"name": "cp_lh_4",  "bounds": (0, 1000)},
    {"name": "cp_lh_5",  "bounds": (0, 1000)},
    {"name": "cp_tau_0", "bounds": (0, 500)},
    {"name": "cp_n",     "bounds": (1, 20)},
]
LH6_OPT_MODEL   = "sm_617_s3_lh6"
LH6_MAT_MODEL   = "deer/cplh6_ae"

def get_model_info(model_name:str) -> tuple:
    """
    Gets simulation information about the model
    
    Parameters:
    * `model_name`: Name of the model

    Returns the parameters information, optimisation model,
    and simulation model, as a tuple
    """
    if model_name == "vh":
        return VH_PARAM_INFO, VH_OPT_MODEL, VH_MAT_MODEL
    elif model_name == "lh2":
        return LH2_PARAM_INFO, LH2_OPT_MODEL, LH2_MAT_MODEL
    elif model_name == "lh6":
        return LH6_PARAM_INFO, LH6_OPT_MODEL, LH6_MAT_MODEL
