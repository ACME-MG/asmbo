"""
 Title:         Interpolator
 Description:   Interpolates data
 Author:        Janzen Choi

"""

# Libraries
import numpy as np
from scipy.interpolate import splev, splrep, splder
from asmbo.helper.general import get_thinned_list

# The Interpolator Class
class Interpolator:

    def __init__(self, x_list:list, y_list:list, resolution:int=50, smooth:bool=False):
        """
        Class for interpolating two lists of values

        Parameters:
        * `x_list`:     List of x values
        * `y_list`:     List of y values
        * `resolution`: The resolution used for the interpolation
        * `smooth`:     Whether to smooth the interpolation
        """
        x_list, indices = np.unique(np.array(x_list), return_index=True)
        y_list = np.array(y_list)[indices]
        if len(x_list) > resolution:
            x_list = get_thinned_list(list(x_list), resolution)
            y_list = get_thinned_list(list(y_list), resolution)
        else:
            resolution = len(x_list)
        smooth_amount = resolution if smooth else 0
        self.spl = splrep(x_list, y_list, s=smooth_amount)
    
    def differentiate(self) -> None:
        """
        Differentiate the interpolator
        """
        self.spl = splder(self.spl)

    def evaluate(self, x_list:list) -> list:
        """
        Run the interpolator for specific values

        Parameters
        * `x_list`: The list of x values

        Returns the evaluated values
        """
        return list(splev(x_list, self.spl))

def intervaluate(x_list:list, y_list:list, x_value:float) -> float:
    """
    Quickly interpolates using two lists of values and evaluates
    the interpolator at a single point
    
    Parameters:
    * `x_list`:  List of x values
    * `y_list`:  List of y values
    * `x_value`: Value to evaluate at
    
    Returns the evaluated y value
    """
    interpolator = Interpolator(x_list, y_list, len(x_list))
    y_value = interpolator.evaluate([x_value])[0]
    return y_value
