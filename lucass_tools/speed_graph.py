import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy import integrate, optimize


#The Model Class
class SpeedGraph:
    """A class to simulate and visualize the dynamics of a system of ODEs.

    This class integrates a system of ordinary differential equations (ODEs)
    and provides tools to visualize the trajectory, phase plane, and velocities
    of the system.

    Attributes:
        model (callable): The ODE system to integrate. Must accept a state vector and time.
        init_values (list[float]): Initial conditions for the system.
        var_labels (list[str]): Names of the state variables.
        vars_to_plot (list[str]): Variables to plot on the phase plane.
        t_init (float): Initial time for integration. Default is 0.
        t_final (float): Final time for integration. Default is 1000.
        t_space (int): Number of time steps for integration. Default is 100000.
        data (dict): Stores the simulation data for each variable.
        eq_points (ndarray): Calculated equilibrium points of the system.

    """

    def __init__(self,  # noqa: PLR0913
                 model: callable,
                 init_values: list[float],
                 var_labels: list[str],
                 vars_to_plot: list[str],
                 t_init: float=0,
                 t_final: float=300,
                 t_space: int=1000,
                 with_eqs: bool =False) -> None:  # noqa: FBT001, FBT002
        """Initialize the speedGraph class.

            model : callable
                The ODE system to integrate.
            init_values : list[float]
                Initial conditions for the system.
            var_labels : list[str]
                Names of the state variables.
            vars_to_plot : list[str]
                Variables to plot on the phase plane.
            t_init : float, optional
                Initial time for integration. Default is 0.
            t_final : float, optional
                Final time for integration. Default is 300.
            t_space : int, optional
                Number of time steps for integration. Default is 1000.
            with_eqs : bool, optional
                Whether or not to calculate the equilibria on init. Default is False.

        Example:
            >>> def model(t, X):
            ...     x, y = X
            ...     return [-y, x]
            >>> graph = speedGraph(model, [1, 0], ['x', 'y'], ['x', 'y'])

        """
        self.EQTOLERANCE = 5
        self.model = model
        self.init_values = init_values
        self.t_init = t_init
        self.t_final = t_final
        self.t_space = t_space
        self.var_labels = var_labels
        self.vars_to_plot = vars_to_plot

        # Initialize time and data
        self.X0 = np.array(init_values)
        self.t = np.linspace(t_init, t_final, t_space)
        #store the data in a dictionary
        self.data = {name: np.zeros(self.t.size) for name in var_labels}
        # Compute the data from odeint
        X = integrate.odeint(model, self.X0, self.t)  # noqa: N806
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
           self.eq_points =  self.estimate_equilibria()

    def estimate_equilibria(self, decimals: int =2) -> list[float]:
        """Find the equilibrium points of the system for an arbitrary number of variables.

        This method generates a grid of initial points for optimization, creates a meshgrid 
        for all variable combinations, and finds equilibrium points using optimization. 
        The results are then rounded to the specified number of decimal places and duplicates 
        are removed.

        Args:
            decimals (int): The number of decimal places to round the equilibrium points. 
                            Default is 2.

        Returns:
            list[float]: A list of unique equilibrium points rounded to the specified number 
                            of decimal places.

        """
        # Generate a grid of initial points for optimization
        ranges = [
            np.linspace(
                self.data[var].min() - self.EQTOLERANCE,
                self.data[var].max() + self.EQTOLERANCE,
                10,
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
    def plot_graphs(self, **kwargs: dict) -> plt:
        """Plot the trajectory, phase plane, and time evolution of the system.

        kwargs : dict, optional
            Additional keyword arguments to control the plotting behavior.
            - with_eqs (bool): Whether or not to calculate and plot the equilibria
              based on a root finding algorithm. Terribly optimized ATM. Use with caution.

        Returns
        -------
        plt : matplotlib.pyplot
            The matplotlib.pyplot object containing the plots.

        """
        with_eqs = kwargs.get("with_eqs", False)
        #These actually dont need to be named.
        pa = "Uninfected Cells"
        pb = "Infected Cells"
        pc = "Phase plane"
        figs, axs = plt.subplot_mosaic([
            [[[pa],[pb]],pc],
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
        lc = LineCollection(segments, cmap="jet", norm=norm)
        # Set the values used for colormapping
        lc.set_array(self.vel)
        lc.set_linewidth(2)
        #Graph to the phaseplane
        line = axs[pc].add_collection(lc)
        figs.colorbar(line, ax=axs[pc])
        #Plotting the phase plane
        #Initial Condition
        axs[pc].plot(self.x[0],self.y[0],"ro")
        #Fixed points
        if with_eqs:        # Calculate equilibrium points
            self.eq_points = self.estimate_equilibria()
            for fixed in self.eq_points:
                axs[pc].plot(fixed[0],fixed[1],"rP")
        #Labels
        axs[pc].set_xlabel(self.vars_to_plot[0])
        axs[pc].set_ylabel(self.vars_to_plot[1])
        #Plotting functions of time
        axs[pa].plot(self.t,self.x, "b-", label=self.vars_to_plot[0])
        axs[pa].set_ylabel(self.vars_to_plot[0])
        axs[pb].plot(self.t,self.y, "r-", label=self.vars_to_plot[1])
        axs[pb].set_xlabel("time")
        axs[pb].set_ylabel(self.vars_to_plot[1])
        plt.tight_layout()
        plt.show()
        return plt
