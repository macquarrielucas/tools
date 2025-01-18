import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
#from matplotlib.colors import ListedColormap
from scipy import integrate, optimize
#The Model Class
class speedGraph:
    """
    A class to simulate and visualize the dynamics of a system of ODEs.

    This class integrates a system of ordinary differential equations (ODEs)
    and provides tools to visualize the trajectory, phase plane, and velocities
    of the system.

    Attributes:
        model (callable): The ODE system to integrate. Must accept a state vector and time.
        initValues (list[float]): Initial conditions for the system.
        var_labels (list[str]): Names of the state variables.
        vars_to_plot (list[str]): Variables to plot on the phase plane.
        tInit (float): Initial time for integration. Default is 0.
        tFinal (float): Final time for integration. Default is 1000.
        tSpace (int): Number of time steps for integration. Default is 100000.
        data (dict): Stores the simulation data for each variable.
        eq_points (ndarray): Calculated equilibrium points of the system.
    """
    def __init__(self, model, initValues, var_labels, vars_to_plot, with_eqs =False, tInit=0, tFinal=300, tSpace=1000):
        """
        Initializes the speedGraph class.

        Parameters:
            model (callable): The ODE system to integrate.
            initValues (list[float]): Initial conditions for the system.
            var_labels (list[str]): Names of the state variables.
            vars_to_plot (list[str]): Variables to plot on the phase plane.
            with_eqs (Boolean): Whether or not to calculate the equilibria on init
            tInit (float, optional): Initial time for integration. Default is 0.
            tFinal (float, optional): Final time for integration. Default is 1000.
            tSpace (int, optional): Number of time steps for integration. Default is 100000.

        Example:
            >>> def model(t, X):
            ...     x, y = X
            ...     return [-y, x]
            >>> graph = speedGraph(model, [1, 0], ['x', 'y'], ['x', 'y'])
        """
        self.EQTOLERANCE = 5
        self.model = model
        self.initValues = initValues
        self.tInit = tInit
        self.tFinal = tFinal
        self.tSpace = tSpace
        self.var_labels = var_labels
        self.vars_to_plot = vars_to_plot

        # Initialize time and data
        self.X0 = np.array(initValues)
        self.t = np.linspace(tInit, tFinal, tSpace)
        self.data = {name: np.zeros(self.t.size) for name in var_labels}  # Store all variables

        # Compute the data from odeint
        X = integrate.odeint(model, self.X0, self.t)
        for i, name in enumerate(var_labels):
            self.data[name] = X[:, i]  # Store each variable

        # Extract variables for plotting
        self.x = self.data[self.vars_to_plot[0]]
        self.y = self.data[self.vars_to_plot[1]]

        # Calculate velocities for all variables
        self.vel_components = np.zeros((self.t.size, len(var_labels)))
        for ts in range(self.t.size):
            self.vel_components[ts] = self.model(X[ts])  # Velocity for all variables at time step `ts`

        # Calculate velocity magnitude
        self.vel = np.sqrt(np.sum(self.vel_components**2, axis=1))

        if with_eqs:
           self.eq_points =  self.calcEquillibriumPoints()

    def calcEquillibriumPoints(self, decimals=2):
        """
        Find the equilibrium points of the system for an arbitrary number of variables.
        """
        # Generate a grid of initial points for optimization
        ranges = [
            np.linspace(
                self.data[var].min() - self.EQTOLERANCE,
                self.data[var].max() + self.EQTOLERANCE,
                10
            )
            for var in self.var_labels
        ]

        # Create a meshgrid for all variable combinations
        grid = np.array(np.meshgrid(*ranges)).T.reshape(-1, len(self.var_labels))

        # Find equilibrium points using optimization
        eq_points = []
        for initial_point in grid:
            root = optimize.root(self.model, initial_point).x
            eq_points.append(root)

        # Round the results and remove duplicates
        return np.unique(np.around(eq_points, decimals=decimals), axis=0)

    #Set up the graphs
    def plotGraphs(self, with_eqs = False):
        """
        Plots the trajectory, phase plane, and time evolution of the system.

        This method creates subplots for the phase plane and the time evolution
        of the selected variables.
        Parameters:
        with_eqs (Boolean): Whether or not to calculate and plot the equilibria based on a root finding algorithm. Terribly optimized ATM. Use with caution.

        Example:
            >>> graph.plotGraphs()
        """
        pa = 'Uninfected Cells'
        pb = 'Infected Cells'
        pc = 'Phase plane'
        figs, axs = plt.subplot_mosaic([
            [[[pa],[pb]],pc]
        ],
        width_ratios=[1,1])
        self.figs=figs
        self.axs=axs
        #Calculate data for colour map
        #The code is mostly from here https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/multicolored_line.html
        #This is the location of the points on the line
        points = np.array([self.x, self.y]).T.reshape(-1, 1, 2)
        #Turn the points into line segments
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        #Create the norm
        norm = plt.Normalize(self.vel.min(), self.vel.max())
        #Join the segments into a line Collection
        lc = LineCollection(segments, cmap='jet', norm=norm)
        # Set the values used for colormapping
        lc.set_array(self.vel)
        lc.set_linewidth(2)
        #Graph to the phaseplane
        line = axs[pc].add_collection(lc)
        figs.colorbar(line, ax=axs[pc])
        #Plotting the phase plane
        #Initial Condition
        axs[pc].plot(self.x[0],self.y[0],'ro')
        #Fixed points
        if with_eqs:        # Calculate equilibrium points
            self.eq_points = self.calcEquillibriumPoints()
            for fixed in self.eq_points:
                axs[pc].plot(fixed[0],fixed[1],'rP')
        #Labels
        axs[pc].set_xlabel(self.vars_to_plot[0])
        axs[pc].set_ylabel(self.vars_to_plot[1])
        #Plotting functions of time
        axs[pa].plot(self.t,self.x, 'b-', label=self.vars_to_plot[0])
        axs[pa].set_ylabel(self.vars_to_plot[0])
        axs[pb].plot(self.t,self.y, 'r-', label=self.vars_to_plot[1])
        axs[pb].set_xlabel('time')
        axs[pb].set_ylabel(self.vars_to_plot[1])
        #Place the legend
        for ax in [axs[pa],axs[pb]]:
            ax.grid()
            ax.legend(loc='best')
        plt.tight_layout()
        plt.show()