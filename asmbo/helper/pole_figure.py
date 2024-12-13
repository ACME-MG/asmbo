"""
 Title:         Pole figure
 Description:   Contains code to plot orientations on PF and IPF plots;
 References:    https://github.com/Argonne-National-Laboratory/neml/blob/dev/neml/cp/polefigures.py 
 Author:        Janzen Choi

"""

# Libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.collections import PathCollection
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from neml.math import rotations, tensors
from neml.cp import crystallography
from opt_all.helper.general import flatten, round_sf

# Pole figure class
class PF:
    
    def __init__(self, lattice, sample_symmetry:list=[2,2,2], x_direction=[1.0,0,0], y_direction=[0,1.0,0]):
        """
        Constructor for PF class
        
        Parameters:
        * `lattice`:         The lattice object
        * `sample_symmetry`: Sample direction of the projection ([1] is no symmetry)
        * `x_direction`:     X crystallographic direction of the projection
        * `y_direction`:     Y crystallographic direction of the projection
        """
        self.lattice = lattice
        sample_symmetry_str = "".join([str(ss) for ss in sample_symmetry])
        self.sample_symmetry = crystallography.symmetry_rotations(sample_symmetry_str)
        self.x_direction = [float(x) for x in x_direction]
        self.y_direction = [float(y) for y in y_direction]

    def __init__(self, lattice, sample_symmetry:list=[2,2,2], x_direction=[1.0,0,0], y_direction=[0,1.0,0]):
        """
        Constructor for PF class
        
        Parameters:
        * `lattice`:         The lattice object
        * `sample_symmetry`: Sample direction of the projection
        * `x_direction`:     X crystallographic direction of the projection
        * `y_direction`:     Y crystallographic direction of the projection
        """
        self.lattice = lattice
        sample_symmetry_str = "".join([str(ss) for ss in sample_symmetry])
        self.sample_symmetry = crystallography.symmetry_rotations(sample_symmetry_str)
        self.x_direction = [float(x) for x in x_direction]
        self.y_direction = [float(y) for y in y_direction]

    def get_equivalent_poles(self, plane:list) -> list:
        """
        Gets the equivalent poles

        Parameters:
        * `plane`: Plane of the projection (i.e., crystal direction)

        Returns the list of equivalent poles
        """
        poles = self.lattice.miller2cart_direction(plane)
        eq_poles = self.lattice.equivalent_vectors(poles)
        eq_poles = [pole.normalize() for pole in eq_poles]
        eq_poles = [op.apply(p) for p in eq_poles for op in self.sample_symmetry]
        return eq_poles

    def initialise_polar_grid(self) -> plt.Axes:
        """
        Initialises the polar grid;
        returns the axis
        """
        _, axis = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})
        axis.grid(False)
        axis.get_yaxis().set_visible(False)
        plt.ylim([0, 1.0])
        plt.xticks([0, np.pi/2], ["  RD", "TD"], fontsize=15)
        return axis

    def get_polar_points(self, euler:list, eq_poles:list) -> list:
        """
        Converts the euler-bunge angles into polar points
        
        Parameters:
        * `euler`:    Orientation in euler-bunge angles (rads)
        * `eq_poles`: List of equivalent poles

        Returns the list of polar points
        """
        standard_rotation = rotations.Orientation(tensors.Vector(self.x_direction), tensors.Vector(self.y_direction))
        orientation = rotations.CrystalOrientation(euler[0], euler[1], euler[2], angle_type="radians", convention="bunge")
        points = [standard_rotation.apply(orientation.inverse().apply(pp)) for pp in eq_poles]
        points = [point for point in points if point[2] >= 0.0] # rid of points in the lower hemisphere
        cart_points = np.array([project_stereographic(v) for v in points])
        polar_points = np.array([cart2pol(cp) for cp in cart_points])
        return polar_points

    def plot_pf(self, euler_list:list, plane:list, colour_list:list=None, size_list:list=None) -> None:
        """
        Plots a standard pole figure using a stereographic projection;
        only works for cubic structures

        Parameters:
        * `euler_list`:  The list of orientations in euler-bunge form (rads)
        * `plane`:       Plane of the projection (i.e., crystal direction)
        * `colour_list`: List of values to define the colouring scheme
        * `size_list`:   List of values to define the sizing scheme
        
        Plots the PF of the orientations
        """
        
        # Define the colouring and sizing scheme
        rgb_colours = get_colours(euler_list, colour_list)
        norm_size_list = get_sizes(euler_list, size_list)

        # Get the standard rotationn and equivalent poles, and creates the grid
        eq_poles = self.get_equivalent_poles(plane)
        axis = self.initialise_polar_grid()

        # Iterate through the orientations
        for i, euler in enumerate(euler_list):
            polar_points = self.get_polar_points(euler, eq_poles)
            size = norm_size_list[i] if size_list != None else 3
            colour = rgb_colours[i] if colour_list != None else None
            plot_points(axis, polar_points, size, colour)

    def plot_pf_density(self, euler_list:list, plane:list) -> None:
        """
        Plots a standard pole figure with contoured colours based on point density
        
        Parameters:
        * `euler_list`:  The list of orientations in euler-bunge form (rads)
        * `plane`:       Plane of the projection (i.e., crystal direction)
        """
        eq_poles = self.get_equivalent_poles(plane)
        self.initialise_polar_grid()
        radius_list, theta_list = [], [] 
        for euler in euler_list:
            polar_points = self.get_polar_points(euler, eq_poles)
            radius_list += list(polar_points[:,0])
            theta_list += list(polar_points[:,1])
        sns.kdeplot(x=radius_list, y=theta_list, cmap="viridis", levels=5, thresh=0)

# Inverse pole figure class
class IPF:

    def __init__(self, lattice, sample_symmetry:list=[2,2,2], x_direction=[1,0,0],
                 y_direction=[0,1,0], colour_limits:tuple=None, size_limits:tuple=None):
        """
        Constructor for IPF class

        Parameters:
        * `lattice`:         The lattice object
        * `sample_symmetry`: Sample direction of the projection ([1] is no symmetry)
        * `x_direction`:     X crystallographic direction of the projection
        * `y_direction`:     Y crystallographic direction of the projection
        * `colour_limits`:   A tuple of the minimum and maximum stress
        """
        self.lattice = lattice
        sample_symmetry_str = "".join([str(ss) for ss in sample_symmetry])
        self.sample_symmetry = crystallography.symmetry_rotations(sample_symmetry_str)
        self.vectors = (np.array([0,0,1.0]), np.array([1.0,0,1]), np.array([1.0,1,1])) # force float
        self.norm_vectors = [vector / np.linalg.norm(np.array(vector)) for vector in self.vectors]
        self.x_direction = [int(x) for x in x_direction]
        self.y_direction = [int(y) for y in y_direction]
        self.colour_limits = colour_limits
        self.size_limits = size_limits

    def project_ipf(self, quaternion:np.array, direction:list) -> None:
        """
        Projects a single sample direction onto a crystal

        Parameters:
        * `quaternion`: Orientation in quaternion form
        * `direction`:  The sample direction

        Returns the projected points
        """

        # Normalise lattice directions
        norm_x  = self.lattice.miller2cart_direction(self.x_direction).normalize()
        norm_y  = self.lattice.miller2cart_direction(self.y_direction).normalize()
        if not np.isclose(norm_x.dot(norm_y), 0.0):
            raise ValueError("Lattice directions are not orthogonal!")
        norm_z = norm_x.cross(norm_y)
        trans  = rotations.Orientation(np.vstack((norm_x.data, norm_y.data, norm_z.data)))
        norm_d = tensors.Vector(np.array(direction)).normalize()

        # Populate the points
        points_list = []
        for rotation in self.sample_symmetry:
            sample_points = rotation.apply(norm_d)
            crystal_points = quaternion.apply(sample_points)
            points = [trans.apply(op.apply(crystal_points)).data for op in self.lattice.symmetry.ops]
            points_list += points

        # Format the points in the upper hemisphere and return
        points_list = np.array(points_list)
        points_list = points_list[points_list[:,2] > 0]
        return points_list

    def reduce_points_triangle(self, points:tuple) -> list:
        """
        Reduce points to a standard stereographic triangle

        Parameters:
        * `points`: The projected points
        
        Returns the reduced points
        """
        norm_0 = np.cross(self.norm_vectors[0], self.norm_vectors[1])
        norm_1 = np.cross(self.norm_vectors[1], self.norm_vectors[2])
        norm_2 = np.cross(self.norm_vectors[2], self.norm_vectors[0])
        reduced_points = [p for p in points if np.dot(p, norm_0) >= 0 and np.dot(p, norm_1) >= 0 and np.dot(p, norm_2) >= 0]
        return reduced_points

    def get_outline(self) -> list:
        """
        Returns a list of tuples for outlines of an IPF plot
        """
        all_points = []
        for i,j in ((0,1), (2,0), (1,2)):
            fs = np.linspace(0, 1, 5)
            points = [project_stereographic((f*self.vectors[i]+(1-f)*self.vectors[j]) / 
                      np.linalg.norm(f*self.vectors[i]+(1-f)*self.vectors[j])) for f in fs]
            all_points += points
        all_points = [tuple(point) for point in all_points]
        return all_points

    def plot_outline(self) -> None:
        """
        Plots the outline
        """
        all_points = self.get_outline()
        x_list = [point[0] for point in all_points]
        y_list = [point[1] for point in all_points]
        plt.plot(x_list, y_list, color="black")

    def initialise_ipf(self) -> tuple:
        """
        Initialises the format and border of the IPF plot

        Returns the axis and patch
        """

        # Create the plot
        axis = plt.subplot(111)
        axis.axis("off")
        plt.text(0.10, 0.09, "[1 0 0]", transform=plt.gcf().transFigure)
        plt.text(0.86, 0.09, "[1 1 0]", transform=plt.gcf().transFigure)
        plt.text(0.74, 0.88, "[1 1 1]", transform=plt.gcf().transFigure)

        # Plot the outline and set clip
        all_points = self.get_outline()
        path_codes = [Path.MOVETO] + [Path.LINETO]*(len(all_points)-2) + [Path.CLOSEPOLY]
        path = Path(all_points, path_codes)
        patch = PathPatch(path, facecolor="white", edgecolor="black")
        axis.add_patch(patch)

        # Returns the axis and patch
        return axis, patch

    def get_points(self, euler:list, direction:list) -> list:
        """
        Converts an euler orientation into stereo points

        Parameters:
        * `euler`:     The orientation in euler-bunge form (rads)
        * `direction`: The sample direction

        Returns a list of stereo points
        """
        orientation = rotations.CrystalOrientation(euler[0], euler[1], euler[2], angle_type="radians", convention="bunge")
        projected_points = self.project_ipf(orientation, direction)
        projected_points = np.vstack(tuple(projected_points))
        reduced_points = self.reduce_points_triangle(projected_points)
        stereo_points  = np.array([project_stereographic(point) for point in reduced_points])
        return stereo_points

    def plot_ipf(self, euler_list:list, direction:list, colour_list:list=None, size_list:list=None) -> None:
        """
        Plot an inverse pole figure given a collection of discrete points;
        only works for cubic structures

        Parameters:
        * `euler_list`:  The list of orientations in euler-bunge form (rads)
        * `direction`:   The sample direction
        * `colour_list`: List of values to define the colouring scheme
        * `size_list`:   List of values to define the sizing scheme

        Plots the IPF of the orientations
        """

        # Initialise
        rgb_colours = get_colours(euler_list, colour_list, self.colour_limits)
        norm_size_list = get_sizes(euler_list, size_list, self.size_limits)
        axis, patch = self.initialise_ipf()

        # Iterate and plot the orientations
        for i, euler in enumerate(euler_list):
            
            # Initialise the points
            points = self.get_points(euler, direction)
            size = norm_size_list[i] if size_list != None else 8 # 5
            colour = rgb_colours[i] if colour_list != None else None
            
            # Plot and clip
            pc = plot_points(axis, points, size, colour)
            pc.set_clip_path(patch)
        
        # Add final outline
        self.plot_outline()

    def plot_ipf_trajectory(self, trajectories:list, direction:list, function:str="scatter", settings:dict={}) -> None:
        """
        Plot an inverse pole figure to display the reorientation trajectories

        Parameters:
        * `trajectories`: The list of reorientation trajectories
        * `direction`:    The sample direction
        * `function`      Function to plot points
        * `settings`:     The plotting settings
        
        Plots the IPF of the reorientation trajectories
        """

        # Initialise the IPF
        axis, patch = self.initialise_ipf()
        
        # Add zorder in settings if not defined
        if not "zorder" in settings.keys():
            settings["zorder"] = 2
        
        # Iterate through trajectories and plot each
        for trajectory in trajectories:

            # Get points
            points = np.array(flatten([self.get_points(euler, direction) for euler in trajectory]))
            if len(points) == 0:
                continue
            
            # Plot the points
            if function == "arrow": # experimental
                pc = axis.arrow(points[-3,0], points[-3,1], points[-1,0]-points[-3,0], points[-1,1]-points[-3,1], **settings)
            elif function == "text":
                pc = axis.text(points[0,0], points[0,1], **settings)
            elif function == "plot":
                pc = plt.plot(points[:,0], points[:,1], **settings)[0]
            elif function == "scatter":
                pc = plt.scatter(points[:,0], points[:,1], **settings)

            # Clip the points
            pc.set_clip_path(patch)
            
        # Add final outline
        self.plot_outline()

def get_lattice(structure:str="fcc"):
    """
    Gets the lattice object

    Parameters:
    * `structure`: The crystal structure

    Returns the lattice object
    """
    lattice = crystallography.CubicLattice(1.0)
    if structure == "fcc":
        lattice.add_slip_system([1,1,0], [1,1,1])
    elif structure == "bcc":
        lattice.add_slip_system([1,1,1], [1,1,0])
        lattice.add_slip_system([1,1,1], [1,2,3])
        lattice.add_slip_system([1,1,1], [1,1,2])
    else:
        raise ValueError(f"Crystal structure '{structure}' unsupported!")
    return lattice

def get_trajectories(euler_history:list, index_list:list=None) -> list:
    """
    Converts a history of euler angles into a list of trajectories

    Parameters:
    * `euler_history`: The history of orientations in euler-bunge form (rads)
    * `index_list`:    The list of indexes to include; if undefined, includes
                       all the trajectories

    Returns the list of trajectories
    """
    trajectories = []
    for i in range(len(euler_history[0])):
        if index_list != None and not i in index_list:
            continue
        trajectory = [euler_state[i] for euler_state in euler_history]
        trajectories.append(trajectory)
    return trajectories

def get_colours(orientations:list, values:list, colour_limits:tuple=None) -> list:
    """
    Checks the colour list and returns a list of RGB colours

    Parameters:
    * `orientations`: The list of orientations
    * `values`:       The list of values
    * `norm_limits`:  A tuple of predefined minimum and maximum values

    Returns the list of colours
    """

    # Checks the values
    if values == None:
        return None
    if len(values) != len(orientations):
        raise ValueError("The 'colour_list' does not have the same number of values as the quaternions!")
    
    # Normalise values
    norm_values = np.array(values)
    if colour_limits == None:
        min_value = min(norm_values)
        max_value = max(norm_values)
    else:
        min_value = colour_limits[0]
        max_value = colour_limits[1]
    norm_values = (norm_values-min_value)/(max_value-min_value)

    # Define colour map and return
    colours = cm.jet(norm_values)
    return np.array(colours)

def get_colour_map(values:list, orientation:str="horizontal") -> None:
    """
    Plots a colour map

    Parameters:
    * `values`:      The list of values
    * `orientation`: The orientation of the colour bar;
                     ("horizontal", "vertical")
    """

    # Initialise figure
    if orientation == "horizontal":
        figure = plt.figure(figsize=(5, 1.5))
        axis = figure.add_axes([0.15, 0.15, 0.7, 0.3]) # x_lower, y_lower, x_size, y_size
    else:
        figure = plt.figure(figsize=(1.5, 5))
        axis = figure.add_axes([0.15, 0.15, 0.3, 0.7]) # x_lower, y_lower, x_size, y_size

    # Get normalised values
    min_value = min(values)
    max_value = max(values)
    norm = Normalize(vmin=min_value, vmax=max_value)

    # Create colour bar
    sm = plt.cm.ScalarMappable(cmap="jet", norm=norm)
    sm.set_array([])
    colour_bar = ColorbarBase(axis, cmap=cm.jet, norm=norm, orientation=orientation)

    # Define the ticks
    ticks = [round_sf(tick, 5) for tick in np.linspace(min_value, max_value, 5)]
    colour_bar.set_ticks(ticks)
    colour_bar.set_ticklabels([str(tick) for tick in ticks])

def get_sizes(orientations:list, values:list, size_limits:tuple=None) -> list:
    """
    Checks the colour list and returns a list of RGB colours

    Parameters:
    * `orientations`: The list of orientations
    * `values`:       The list of value

    Returns the list of colours
    """

    # Checks values
    if values == None:
        return None
    if len(values) != len(orientations):
        raise ValueError("The 'size_list' does not have the same number of values as the quaternions!")
    
    # Normalise sizes and return
    min_norm = 1.0
    max_norm = 32.0
    if size_limits == None:
        min_value = min(values)
        max_value = max(values)
    else:
        min_value = size_limits[0]
        max_value = size_limits[1]
    norm_size_list = [min_norm+((value-min_value)/(max_value-min_value))*(max_norm-min_norm) for value in values]
    return norm_size_list

def project_stereographic(vector:np.array) -> np.array:
    """
    Stereographic projection of the given vector into a numpy array

    Parameters:
    * `vector`: Unprojected vector
    
    Returns the projected vector
    """
    return np.array([vector[0]/(1.0+vector[2]), vector[1]/(1.0+vector[2])])

def cart2pol(cart_point:np.array):
    """
    Convert a cartesian point into polar coordinates

    Parameters:
    * `cart_point`: Cartesian point

    Returns the polar coordinates
    """
    return np.array([np.arctan2(cart_point[1], cart_point[0]), np.linalg.norm(cart_point)])

def plot_points(axis:plt.Axes, points:list, size:float, colour:np.ndarray) -> PathCollection:
    """
    Plots the points on a plot

    Parameters:
    * `axis`:      The axis to plot the points on
    * `points`:    The points to be plotted
    * `size`:      The size of the points
    * `colour`:    The colour of the points; None if not defined
    * `is_family`: Whether the grain is part of the family; None if not defined

    Returns the scatter object
    """
    if not isinstance(colour, np.ndarray):
        scatter = axis.scatter(points[:,0], points[:,1], c="black", s=size**2)
    else:
        scatter = axis.scatter(points[:,0], points[:,1], color=colour, edgecolor="black", linewidth=0.25, s=size**2)
    return scatter
