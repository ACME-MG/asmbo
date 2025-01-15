"""
 Title:         Plotter
 Description:   For plotting data
 Author:        Janzen Choi
 
"""

# Libraries
import matplotlib.pyplot as plt
import matplotlib.colors as mcolours

# Constants
EXP_COLOUR   = "darkgray"
CAL_COLOUR   = "green"
VAL_COLOUR   = "red"
ALL_COLOURS  = list(mcolours.TABLEAU_COLORS) + list(mcolours.BASE_COLORS) + list(mcolours.CSS4_COLORS)

# Plotter class
class Plotter:

    def __init__(self, x_label:str="x", y_label:str="y", x_units:str="", y_units:str=""):
        """
        Class for plotting data

        Parameters:
        * `path`:    The path to save the plot
        * `x_label`: The label for the x axis
        * `y_label`: The label for the y axis
        * `x_units`: The units for the x axis
        * `y_units`: The units for the y axis
        """

        # Initialise input parameters
        self.x_label = x_label
        self.y_label = y_label
        self.x_units = x_units
        self.y_units = y_units

        # Initialise internal variables
        self.name_list   = []
        self.colour_list = []
        self.type_list   = []

    def prep_plot(self, title:str="", size:int=12) -> None:
        """
        Prepares the plot
        
        Parameters:
        * `title`: The title of the plot
        * `size`:  The size of the font
        """

        # Set figure size and title
        plt.figure(figsize=(5,5), dpi=200)
        plt.title(title, fontsize=size+3, fontweight="bold", y=1.05)
        plt.gca().set_position([0.17, 0.12, 0.75, 0.75])
        plt.gca().grid(which="major", axis="both", color="SlateGray", linewidth=1, linestyle=":", alpha=0.5)

        # Set x and y labels
        x_unit_str = f" ({self.x_units})" if self.x_units != "" else ""
        y_unit_str = f" ({self.y_units})" if self.y_units != "" else ""
        plt.xlabel(f"{self.x_label.replace('_', ' ').capitalize()}{x_unit_str}", fontsize=size)
        plt.ylabel(f"{self.y_label.replace('_', ' ').capitalize()}{y_unit_str}", fontsize=size)
    
    def set_limits(self, x_limits:tuple=None, y_limits:tuple=None) -> None:
        """
        Sets the limits of the x and y scales

        Parameters:
        * `x_limits`: The upper and lower bounds of the plot for the x scale
        * `y_limits`: The upper and lower bounds bound of the plot for the y scale
        """
        if x_limits != None:
            plt.xlim(*x_limits)
        if y_limits != None:
            plt.ylim(*y_limits)

    def set_log_scale(self, x_log:bool=False, y_log:bool=False) -> None:
        """
        Changes the scale of the plot
        
        Parameters:
        * `x_log`: Whether to log the x scale
        * `y_log`: Whether to log the y scale
        """
        if x_log:
            plt.xscale("log")
        if y_log:
            plt.yscale("log")
    
    def scat_plot(self, data_dict:dict, colour:str=CAL_COLOUR, name:str=None) -> None:
        """
        Plots the experimental data using a scatter plot

        Parameters:
        * `data_dict`: The dictionary to store the data
        * `colour`:    The colour to plot the data
        * `name`:      The name of the curve
        """

        # Plot the data
        x_list = data_dict[self.x_label]
        if self.x_label == "time":
            x_list = [x/3600 for x in x_list]
        plt.scatter(x_list, data_dict[self.y_label], s=5**2,
                    marker="o", color=colour, linewidth=1, zorder=1)
        
        # Save name if defined
        if name != None:
            self.name_list.append(name)
            self.colour_list.append(colour)
            self.type_list.append("scatter")

    def line_plot(self, data_dict:dict, colour=CAL_COLOUR, name:str=None) -> None:
        """
        Plots the experimental data using a line plot

        Parameters:
        * `data_dict`: The dictionary to store the data
        * `colour`:    The colour to plot the data
        * `name`:      The name of the curve
        """

        # Plot the data
        x_list = data_dict[self.x_label]
        if self.x_label == "time":
            x_list = [x/3600 for x in x_list]
        plt.plot(x_list, data_dict[self.y_label], colour, zorder=1)

        # Save name if defined
        if name != None:
            self.name_list.append(name)
            self.colour_list.append(colour)
            self.type_list.append("line")

    def set_legend(self) -> None:
        """
        Defines the plot legend
        """
        for i in range(len(self.name_list)):
            if self.type_list[i] == "scatter":
                plt.scatter([], [], color=self.colour_list[i], label=self.name_list[i], s=7**2)
            elif self.type_list[i] == "line":
                plt.plot([], [], color=self.colour_list[i], label=self.name_list[i], linewidth=2)
        plt.legend(framealpha=1, edgecolor="black", fancybox=True, facecolor="white")

def define_legend(colour_list:list, label_list:list, type_list:list, **kwargs) -> None:
    """
    Manually defines the plot legend
    
    Parameters:
    * `colour_list`: The colours in the legend
    * `label_list`:  The keys to add to the legend
    * `type_list`:   The type of the icons in the legend
    """
    for i in range(len(colour_list)):
        if type_list[i] == "scatter":
            plt.scatter([], [], color=colour_list[i], label=label_list[i], s=7**2)
        elif type_list[i] == "line":
            plt.plot([], [], color=colour_list[i], label=label_list[i], linewidth=2)
    plt.legend(framealpha=1, edgecolor="black", fancybox=True, facecolor="white", **kwargs)

def save_plot(file_path:str) -> None:
    """
    Saves the plot and clears the figure

    Parameters:
    * `file_path`: The path to save the plot
    """
    plt.savefig(file_path)
    plt.cla()
    plt.clf()
    plt.close()
