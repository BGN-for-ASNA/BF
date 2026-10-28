import pandas as pd
import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt
import arviz as az
import numpyro
from BayesForge.Distributions.np_dists import UnifiedDist as dist
dist = dist()

import jax
from numpyro.distributions import constraints as _constraints


class _LogGamma(numpyro.distributions.Distribution):
    r"""Distribution of :math:`\log X` where :math:`X \sim \mathrm{Gamma}(c, r)`.

    Sampling the piecewise baseline hazard directly as a Gamma with a tiny
    concentration (the usual vague ``Gamma(0.01, 0.01)``) is unusable under
    NumPyro's NUTS: the unconstrained value ``u`` drifts far negative,
    ``exp(u)`` underflows to exactly 0, and ``Gamma.log_prob(0)`` is ``-inf``,
    so every proposal is flagged divergent and the chains never leave their
    initialisation (r_hat ~ 1e15, ESS ~ 2). Evaluating the same density in log
    space never forms that 0, so the geometry is well behaved while the implied
    prior on ``exp(value)`` is unchanged.
    """

    support = _constraints.real
    arg_constraints = {}

    def __init__(self, concentration, rate=1.0, validate_args=None):
        self.concentration = concentration
        self.rate = rate
        super().__init__(batch_shape=(), event_shape=(), validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        x = numpyro.distributions.Gamma(self.concentration, self.rate).sample(key, sample_shape)
        return jnp.log(x)

    def log_prob(self, value):
        # log-density of log X, i.e. Gamma log-density + the log |dx/du| = u
        # Jacobian, collapsed analytically so exp(value) is the only exp taken.
        return (
            self.concentration * value
            - self.rate * jnp.exp(value)
            + self.concentration * jnp.log(self.rate)
            - jax.scipy.special.gammaln(self.concentration)
        )


class survival_old:
    """    The survival class is designed to handle survival analysis data and perform various operations related to time-to-event data. It provides methods for extracting basic information from the dataset, plotting censoring status, converting continuous time data into discrete intervals, calculating cumulative hazards and survival probabilities, and visualizing the results. This class serves as a high-level interface for managing survival data, allowing users to easily analyze and interpret time-to-event outcomes in a structured manner.
    """
    def __init__(self, parent):
        self.parent = parent 
        self.n_patients = None
        self.patients = None
        self.time = None
        self.event = None
        self.cov = None
        self.interval_length = None
        self.interval_bounds = None
        self.n_intervals = None
        self.death = None
        self.exposure = None
        self.base_hazard = None
        self.met_hazard = None  
        self.data_on_model = {}
        self.cov = None
        self.df = None
        self.dist = dist()
    
    @property
    def df(self):
        return self.parent.df  # always get from BayesForge

    @df.setter
    def df(self, value):
        self.parent.df = value  # always set in BF

    def get_basic_info(self, event='event', time='time', cov=None):
        ''' Get basic information about the dataset

            Parameters
            ----------
                event : str, optional
                    Name of the column containing the event status, by default 'event'
                time : str, optional
                    Name of the column containing the time, by default 'time'
                cov : str, optional
                    Name of the column containing the covariate, by default None

            Returns
            -------
                None

            Notes
            -----
                The function returns the following attributes:
                    - n_patients : int
                        Number of patients in the dataset
                    - patients : np.array
                        Array of patient indices
                    - time : np.array
                        Array of time points
                    - event : np.array
                        Array of event status
                    - data_on_model : dict
                        Dictionary containing the data of the model present ib the dataset                        
                    - cov : str
                        Name of the covariate
                    - df : pd.DataFrame
                        DataFrame containing the dataset

        ''' 

        # Number of patients in the dataset
        self.n_patients = self.df.shape[0]
        self.patients = np.arange(self.n_patients)  # Array of patient indices
        self.time = self.df.loc[:, time].values
        self.event = self.df.loc[:, event].values

        if self.data_on_model is None:
            self.data_on_model = {}
            
        if type(cov) is str:
            self.cov = cov # covariate
            tmp = jnp.reshape(self.df[cov].values, (1, len(self.df[cov].values)))
            self.data_on_model[cov] = tmp

        elif type(cov) is list:
            self.cov = cov
            a = 0
            for item in cov:
                if a == 0:
                    self.data_on_model['cov'] = jnp.array(self.df[item].values)
                    a += 1
                else:
                    self.data_on_model['cov'] = jnp.stack([self.data_on_model['cov'] , jnp.array(self.df[item].values)])
       
    def plot_censoring(self, event='event', time='time', cov='metastasized', xlabel='Time', ylabel='Subject'):
        """
        Plots the censoring status of subjects in a time-to-event dataset.

        Parameters:
        -----------
        df : pandas.DataFrame
            A DataFrame containing the time-to-event data.
        event : str, optional
            The name of the column in `df` indicating the event status (1 = event occurred, 0 = censored).
            Default is 'event'.
        time : str, optional
            The name of the column in `df` representing the time variable. Default is 'time'.
        cov : str, optional
            The name of the column in `df` representing a covariate, such as metastasized status.
            Default is 'metastasized'.
        xlabel : str, optional
            Label for the x-axis. Default is 'Time'.
        ylabel : str, optional
            Label for the y-axis. Default is 'Subject'.

        Returns:
        --------
        None
            This function generates a plot showing censored and uncensored subjects along with a specified covariate.


        """
        self.get_basic_info(event, time, cov)

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot censored subjects (event = 0) as red horizontal lines
        ax.hlines(
            self.patients[ self.event == 0], 0,  self.df[ self.event == 0].loc[:, 'time'], color="C3", label="Censored"
        )

        # Plot uncensored subjects (event = 1) as gray horizontal lines
        ax.hlines(
             self.patients[ self.event == 1], 0,  self.df[ self.event == 1].loc[:, 'time'], color="C7", label="Uncensored"
        )

        # Add scatter points for subjects with the specified covariate (e.g., metastasized = 1)
        ax.scatter(
            self.df[self.df.loc[:,cov] == 1].loc[:, time],
            self.patients[self.df.loc[:,cov] == 1],
            color="k",
            zorder=10,
            label=cov,
        )

        # Set plot limits and labels
        ax.set_xlim(left=0)
        ax.set_xlabel(xlabel)
        ax.set_yticks([])
        ax.set_ylabel(ylabel)

        # Set y-axis limits to provide padding around subjects
        ax.set_ylim(-0.25, self.n_patients + 0.25)

        # Add legend to the plot
        ax.legend(loc="center right")

    def surv_object(self, time='time', event='event', cov=None, interval_length=3):
        """
        Converts continuous time and event data into discrete time intervals for survival analysis.

        Parameters:
        -----------
        time : str, optional
            The name of the column in `df` representing the continuous time variable (default is 'time').
        event : str, optional
            The name of the column in `df` representing the event indicator (default is 'event').
        interval_length : int, optional
            The length of each discrete time interval (default is 3).

        Returns:
        --------
        interval_bounds : numpy.ndarray
            Array of boundaries for discrete time intervals.
        n_intervals : int
            The total number of discrete intervals.
        intervals : numpy.ndarray
            Array of interval indices.
        death : numpy.ndarray
            A binary matrix (n_patients x n_intervals) indicating whether each subject experienced the event
            in each interval (1 if the event occurred, 0 otherwise).
        exposure : numpy.ndarray
            A matrix (n_patients x n_intervals) indicating the time each subject was exposed in each interval.

        Notes:
        ------
        - The function assumes that subjects who experienced the event did so at the end of the time period.
        - Exposure is capped by the interval bounds, and the last interval reflects the remaining time to the event or censoring.

        """
        print("Survival analysis is still in development. Use it with caution and report any issues.")
        self.get_basic_info(time = time, event = event, cov = cov)
        self.interval_length = interval_length
        
        # Define interval bounds and calculate the number of intervals
        interval_bounds = np.arange(0, self.time.max() + interval_length + 1, interval_length)
        n_intervals = interval_bounds.size - 1
        intervals = np.arange(n_intervals)
        self.n_intervals = n_intervals

        # Determine the last interval each patient belongs to
        last_period = np.floor((self.time - 0.01) / self.interval_length).astype(int)
        self.last_period = last_period

        # Create a binary death matrix (n_patients x n_intervals)
        death = np.zeros((self.n_patients, self.n_intervals))
        death[self.patients, last_period] = self.event

        # Calculate exposure times for each interval
        # Strict `>`: with `>=`, a subject whose time falls exactly on an interval
        # bound is credited a full extra interval past its own last_period.
        exposure = np.greater.outer(self.time, interval_bounds[:-1]).astype(float) * interval_length
        exposure[self.patients, last_period] = self.time - interval_bounds[last_period]

        self.interval_bounds = interval_bounds # Array of boundaries for discrete time intervals.
        self.intervals = intervals # Array of interval indices.
        self.death = death # Binary matrix (n_patients x n_intervals) indicating whether each subject experienced the event in each interval (1 if the event occurred, 0 otherwise).
        self.exposure = exposure # Matrix (n_patients x n_intervals) indicating the time each subject was exposed in each interval.

        # data for the model
        if self.data_on_model is None:
            self.data_on_model = {}
        self.data_on_model['intervals'] = jnp.array(intervals)
        self.data_on_model['death'] = jnp.array(death)
        self.data_on_model['exposure']= jnp.array(exposure)

        if type(cov) is str:
            tmp = jnp.reshape(self.df[cov].values, (1, len(self.df[cov].values)))
            self.data_on_model[cov] = tmp
        elif type(cov) is list:
            for item in cov:
                self.data_on_model[item] = jnp.array(self.df[item].values)
        self.parent.data_on_model = self.data_on_model  # update the parent data_on_model

    def cum_hazard(self, hazard):
        """
        Calculates the cumulative hazard from a given hazard rate.

        Parameters:
        -----------
        hazard : numpy.ndarray or jax.numpy.ndarray
            A 1D or 2D array representing the hazard rate at each time step.

        Returns:
        --------
        numpy.ndarray or jax.numpy.ndarray
            The cumulative hazard calculated by summing the hazard over time steps.

        Notes:
        ------
        - The cumulative hazard is computed as the cumulative sum of the hazard values, 
          scaled by the `interval_length` factor.

        """
        return (self.interval_length * hazard).cumsum(axis=-1)

    def survival(self, hazard):
        """
        Calculates the survival probability from a given hazard rate.

        Parameters:
        -----------
        hazard : numpy.ndarray or jax.numpy.ndarray
            A 1D or 2D array representing the hazard rate at each time step.

        Returns:
        --------
        numpy.ndarray or jax.numpy.ndarray
            The survival probability, computed as the exponential of the negative cumulative hazard.

        Notes:
        ------
        - Survival is calculated as `exp(-cumulative hazard)` where the cumulative hazard 
          is calculated using the `cum_hazard` function.
        """
        return np.exp(-self.cum_hazard(hazard))

    def hazards(self, m, lambda0 = 'lambda0', beta = 'beta'):
        """
        Calculates two hazard values: the base hazard and the covariates hazard, based on posterior samples of 
        the parameters `lambda0` and `beta`.

        Parameters:
        -----------
        m : object
            An object that contains posterior samples in the `posteriors` attribute. This attribute should include 
            the parameters `lambda0` and `beta` for calculating the hazards.

        lambda0 : str, optional, default='lambda0'
            The key for the base hazard parameter in the `posteriors` dictionary.

        beta : str, optional, default='beta'
            The key for the covariate effect parameter in the `posteriors` dictionary.

        Returns:
        --------
        tuple of numpy.ndarray or jax.numpy.ndarray
            - base_hazard2 : The base hazard calculated from `lambda0` parameter.
            - met_hazard2 : The metastasis hazard calculated as the product of `lambda0` and `exp(beta)`.

        Notes:
        ------
        - The base hazard is derived from the posterior samples of `lambda0`, while the covariate hazard 
          is computed by multiplying `lambda0` with the exponential of `beta`.
        - `np.expand_dims` is used to ensure the correct dimensions when computing the product.

        """
        base_hazard = m.posteriors["lambda0"]
        array_expanded = jnp.expand_dims(np.exp(m.posteriors["beta"]), axis=-1)
        met_hazard =m.posteriors["lambda0"] * array_expanded

        self.base_hazard = base_hazard
        self.met_hazard = met_hazard
        return base_hazard, met_hazard 

    def plot_surv(self, lambda0 = 'lambda0', beta = 'beta',
                  xlab='Time', ylab='Survival', covlab = 'treated', title = "Bayesian survival model"):

        base_hazard = self.parent.posteriors[lambda0]        
        met_hazard =self.parent.posteriors[lambda0] * self.parent.posteriors[beta]

        fig, (hazard_ax, surv_ax) = plt.subplots(ncols=2, sharex=True, sharey=False, figsize=(16, 6))   

        az.plot_hdi(
            self.interval_bounds[:-1],
            self.cum_hazard(base_hazard),
            ax=hazard_ax,
            smooth=False,
            color="C0",
            fill_kwargs={"label": "Had not metastasized"},
        )
        
        az.plot_hdi(
            self.interval_bounds[:-1],
            self.cum_hazard(met_hazard),
            ax=hazard_ax,
            smooth=False,
            color="C1",
            fill_kwargs={"label": "Metastasized"},
        )   

        hazard_ax.plot(self.interval_bounds[:-1], jnp.mean(self.cum_hazard(base_hazard), axis = 0), color="darkblue")
        hazard_ax.plot(self.interval_bounds[:-1], jnp.mean(self.cum_hazard(met_hazard), axis = 0), color="maroon")   

        hazard_ax.set_xlim(0, self.time.max())
        hazard_ax.set_xlabel(xlab)
        hazard_ax.set_ylabel(r"Cumulative hazard $\Lambda(t)$")
        hazard_ax.legend(loc=2) 

        az.plot_hdi(self.interval_bounds[:-1], self.survival(base_hazard), ax=surv_ax, smooth=False, color="C0")
        az.plot_hdi(self.interval_bounds[:-1], self.survival(met_hazard), ax=surv_ax, smooth=False, color="C1")  

        surv_ax.plot(self.interval_bounds[:-1], jnp.mean(self.survival(base_hazard), axis = 0), color="darkblue")
        surv_ax.plot(self.interval_bounds[:-1], jnp.mean(self.survival(met_hazard), axis = 0), color="maroon")   

        surv_ax.set_xlim(0, self.time.max())
        surv_ax.set_xlabel(ylab)
        surv_ax.set_ylabel("Survival function $S(t)$")  

        fig.suptitle(title);

    def mu(self, cov,  exposure, lambda0, beta):
        lambda_ = numpyro.deterministic('lambda_', jnp.outer(jnp.exp(beta * cov), lambda0)) 
        mu = numpyro.deterministic('mu', exposure * lambda_)
        return lambda_, mu

    def hazard_rate(self, cov, beta, lambda0):
        lambda_ = numpyro.deterministic('lambda_', jnp.outer(jnp.exp(beta @ cov), lambda0)) 
        return lambda_
    
    def model(self,intervals, death, metastasized, exposure):
        # Parameters priors distributions-------------------------
        ## Base hazard distribution
        lambda0 = self.dist.gamma(0.01, 0.01, shape= intervals.shape, name = 'lambda0')
        ## Covariate effect distribution
        beta = self.dist.normal(0, 1000, shape = (1,),  name='beta')
        ## Likelihood
        ### Compute hazard rate based on covariate effect
        lambda_ = self.hazard_rate(cov = metastasized, beta = beta, lambda0 = lambda0)
        ### Compute exposure rates
        mu = exposure * lambda_
    
        # Likelihood calculation
        self.dist.poisson(mu + jnp.finfo(mu.dtype).tiny, obs = death)

import jax
import jax.numpy as jnp
import numpy as np

# estimation will change because alphabetic names of prior generating different seed to get same resutls as PyMC, gange names of priors to 'lambda0', 'beta'
class survival():
    def __init__(self,parent):
        self.n_patients = None
        self.patients = None
        self.time = None
        self.event = None
        self.cov = None
        self.interval_length = None
        self.interval_bounds = None
        self.n_intervals = None
        self.death = None
        self.exposure = None
        self.base_hazard = None
        self.met_hazard = None  
        self.data_on_model = {}
        self.cov_names = [] # Time-invariant covariates names
        self.cov = np.array([])  # Time-invariant covariates
        self.cov_v_names = []  # Time-varying covariates names
        self.cov_v = None  # Time-varying covariates 
        self.df = None
        self.oberved = None
        self.parent = parent
        self.parent.model_name = 'survival'
        # Prior hyperparameters, overridable before fitting.
        # Baseline_rate ~ Gamma(concentration, rate) per interval. Sampled in
        # log space on the numpyro backend, see _LogGamma. The posterior is
        # sensitive to this choice: with more intervals than events, anything
        # much less concentrated at 0 than (0.01, 0.01) inflates the empty
        # late intervals and can flip the sign of the covariate effect.
        self.baseline_rate_prior = (0.01, 0.01)
        # Hazard_rate_<covariate> ~ Normal(0, scale)
        self.hazard_rate_prior_scale = 10.0
        #self.surv_object(time, event, interval_length)


    def import_time_even(self, time, event, interval_length=1):
        """
        Convert data into of survival information into a surv object.
        Converts continuous time and event data into discrete time intervals for survival analysis if interval_length > 1.

        Parameters:
        -----------
        time : 1D jnp.ndarray of shape (N,) representing the continuous time variable.
        event : 1D Binary jnp.ndarray of shape (N,) representing the event indicator.
        interval_length : int, optional
            The length of each discrete time interval (default is 3).

        Returns:
        --------
        - n_patients : int
            Number of patients in the dataset
        - patients : np.array
            Array of patient indices
        - time : np.array
            Array of time points
        - event : np.array
            Array of event status
        - interval_bounds : jnp.ndarray
            Array of boundaries for discrete time intervals.
        - n_intervals : int
            The total number of discrete intervals.
        - intervals : jnp.ndarray
            Array of interval indices.
        - death : jnp.ndarray
            A binary matrix (n_patients x n_intervals) indicating whether each subject experienced the event
            in each interval (1 if the event occurred, 0 otherwise).
        - exposure : jnp.ndarray
            A matrix (n_patients x n_intervals) indicating the time each subject was exposed in each interval.

        Notes:
        ------
        - The function assumes that subjects who experienced the event did so at the end of the time period.
        - Exposure is capped by the interval bounds, and the last interval reflects the remaining time to the event or censoring.

        """
        self.time = time
        self.event = event
        self.n_patients = self.time.shape[0]
        self.patients = jnp.arange(self.n_patients)  # Array of patient indices
        self.interval_length = interval_length
        
        # Define interval bounds and calculate the number of intervals
        interval_bounds = np.arange(0, self.time.max() + interval_length + 0.01, interval_length)
        n_intervals = interval_bounds.size - 1
        intervals = np.arange(n_intervals)
        self.n_intervals = n_intervals

        # Determine the last interval each patient belongs to
        last_period = np.floor((self.time - 1e-7) / self.interval_length).astype(int)
        last_period = np.clip(last_period, 0, n_intervals - 1)
        self.last_period = last_period

        # Create a binary death matrix (n_patients x n_intervals)
        death = np.zeros((self.n_patients, self.n_intervals))
        death[self.patients, last_period] = self.event

        # Calculate exposure times for each interval
        # Strict `>`: with `>=`, a subject whose time falls exactly on an interval
        # bound is credited a full extra interval past its own last_period.
        exposure = np.greater.outer(self.time, interval_bounds[:-1]).astype(float) * interval_length
        exposure[self.patients, last_period] = self.time - interval_bounds[last_period]

        self.interval_bounds = interval_bounds # Array of boundaries for discrete time intervals.
        self.intervals = intervals # Array of interval indices.
        self.death = death # Binary matrix (n_patients x n_intervals) indicating whether each subject experienced the event in each interval (1 if the event occurred, 0 otherwise).
        self.exposure = exposure # Matrix (n_patients x n_intervals) indicating the time each subject was exposed in each interval.
        
        print("------------------------------------------------------------------------------")
        print(f'Survival concern {self.n_patients} individuals in {self.n_intervals} intervals.')
        print(f'{self.death.sum()} individuals experienced the event.')

        self.observed = jnp.ones((self.n_patients, self.n_intervals))
        self.build_data()
        self.parent.data_on_model = self.data_on_model

    def import_covF(self, cov, names):
        """
        Import fixed covariates (patient-level).
        """
        cov = np.asarray(cov)  # make sure it's array
    
        # validate
        if cov.ndim == 1:
            if len(cov) != self.n_patients:
                raise ValueError(f'Length mismatch: covariate ({len(cov)}) vs patients ({self.n_patients})')
            if len(names) != 1:
                raise ValueError(f'Names list must have length 1 for 1D covariate')
        else:
            if cov.shape[0] != self.n_patients:
                raise ValueError(f'Length mismatch: covariate ({cov.shape[0]}) vs patients ({self.n_patients})')
            if cov.shape[1] != len(names):
                raise ValueError(f'Number of names ({len(names)}) does not match covariates ({cov.shape[1]})')
    
        # add names
        self.cov_names += names
    
        # stack into existing covariates
        if self.cov.size == 0:
            self.cov = cov.reshape(-1, 1) if cov.ndim == 1 else cov
        else:
            self.cov = np.column_stack([self.cov, cov])
        print("------------------------------------------------------------------------------")
        print(f'Covariates imported: {names}')
        print(f'Surv object now has {self.cov.shape[1]} covariates: {self.cov_names}')
        self.build_data()
        self.n_cov = len(self.cov_names)
        self.parent.data_on_model = self.data_on_model

    def import_covV(self, cov, names):
        """
        Import time-varying covariates.
    
        Parameters
        ----------
        cov : np.ndarray
            Array of shape (N, T) for one covariate, 
            or (N, T, V) for multiple covariates.
        names : list of str
            Names of the covariates.
        """
        cov = np.asarray(cov)
    
        # Handle single covariate case (N,T) → (N,T,1)
        if cov.ndim == 2:
            if cov.shape[0] != self.n_patients or cov.shape[1] != self.n_intervals:
                raise ValueError(
                    f"Expected shape ({self.n_patients}, {self.n_intervals}), got {cov.shape}"
                )
            cov = cov[..., np.newaxis]   # (N,T,1)
            if len(names) != 1:
                raise ValueError("Must provide one name for a single covariate.")
        
        # Handle multiple covariates (N,T,V)
        elif cov.ndim == 3:
            if cov.shape[0] != self.n_patients or cov.shape[1] != self.n_intervals:
                raise ValueError(
                    f"Expected shape ({self.n_patients}, {self.n_intervals}, V), got {cov.shape}"
                )
            if cov.shape[2] != len(names):
                raise ValueError(
                    f"Number of names ({len(names)}) must match V ({cov.shape[2]})"
                )
        else:
            raise ValueError("cov must be 2D (N,T) or 3D (N,T,V).")
    
        # Initialize or stack
        print(self.cov_v)
        if self.cov_v is None:
            self.cov_v = cov
        else:
            self.cov_v = np.concatenate([self.cov_v, cov], axis=-1)
    
        # Store names
        if not hasattr(self, "cov_v_names"):
            self.cov_v_names = []
        self.cov_v_names.extend(names)
        self.build_data()
        self.parent.data_on_model = self.data_on_model
        print("------------------------------------------------------------------------------")
        print(f"Imported covariates {names}.")
        print(f"Surv object now has {self.cov_v.shape[2]} time-varying covariates: {self.cov_v_names}")

    def build_surv_covariates(self):
        """
        Build a full covariate array for survival modeling.
    
        - Fixed covariates (N,) are expanded to (N,T)
        - Stacked with existing time-varying covariates (N, T, V)
        - Updates covariate names with fixed covariates first
        """
        if len(self.cov_v_names) == 0:
            self.cov_all_names = self.cov_names
            self.cov_all = self.cov
            return None
            
        # 1️ Expand fixed covariates to (N, T)
        cov_fixed_expanded = None
        if self.cov is not None and self.cov.size > 0:
            if self.cov.ndim == 1:
                # single covariate → shape (N,T)
                cov_fixed_expanded = np.repeat(self.cov[:, np.newaxis], self.n_intervals, axis=1)
            elif self.cov.ndim == 2:
                # multiple fixed covariates → shape (N,T,F)
                cov_fixed_expanded = np.repeat(self.cov[:, np.newaxis, :], self.n_intervals, axis=1)
                # axes to (N,T,F)
            else:
                raise ValueError(f"Unexpected fixed covariate shape: {self.cov.shape}")
    
        # 2️ Time-varying covariates
        cov_v = getattr(self, "cov_v", None)
    
        # 3️ Combine fixed and time-varying covariates
        if cov_fixed_expanded is not None and cov_v is not None:
            # If cov_fixed_expanded has shape (N,T) → add last axis for single variable
            if cov_fixed_expanded.ndim == 2:
                cov_fixed_expanded = cov_fixed_expanded[..., np.newaxis]  # (N,T,1)
            self.cov_all = np.concatenate([cov_fixed_expanded, cov_v], axis=-1)
        elif cov_fixed_expanded is not None:
            self.cov_all = cov_fixed_expanded
        elif cov_v is not None:
            self.cov_all = cov_v
        else:
            raise ValueError("No covariates available to build.")
    
        # 4️ Update covariate names: fixed first, then time-varying
        fixed_names = self.cov_names if self.cov is not None else []
        v_names = getattr(self, "cov_v_names", []) if cov_v is not None else []
        self.cov_all_names = fixed_names + v_names
    
        print("------------------------------------------------------------------------------")
        print(f"Built all covariates array with shape {self.cov_all.shape}")
        print(f"Covariate names order: {self.cov_all_names}")

        self.cov_all = jnp.array(self.cov_all)

    def observed(self, observed):
        """
        Add observed data to the survival object.
        Args:
            observed (np.ndarray): Array of observed data with shape (N, T). This array represent if individuals have been observed at each time point and thus should be 1 if observed and 0 if censored.
        Raises:
            ValueError: If observed data shape does not match the survival object. 
        """
        if observed.shape[0] != self.n_patients:
            raise ValueError(f"Expected observed data shape ({self.n_patients}), got {observed.shape}")
        if observed.shape[1] != self.n_intervals:
            raise ValueError(f"Expected observed data shape ({self.n_intervals}), got {observed.shape}")
        self.observed = observed
    
    def cum_hazard(self, hazard):
        """
        Calculates the cumulative hazard from a given hazard rate.

        Parameters:
        -----------
        hazard : numpy.ndarray or jax.numpy.ndarray
            A 1D or 2D array representing the hazard rate at each time step.

        Returns:
        --------
        numpy.ndarray or jax.numpy.ndarray
            The cumulative hazard calculated by summing the hazard over time steps.

        Notes:
        ------
        - The cumulative hazard is computed as the cumulative sum of the hazard values, 
          scaled by the `interval_length` factor.

        """
        return (self.interval_length * hazard).cumsum(axis=-1)
    
    
    def survival(self, hazard):
        """
        Calculates the survival probability from a given hazard rate.

        Parameters:
        -----------
        hazard : numpy.ndarray or jax.numpy.ndarray
            A 1D or 2D array representing the hazard rate at each time step.

        Returns:
        --------
        numpy.ndarray or jax.numpy.ndarray
            The survival probability, computed as the exponential of the negative cumulative hazard.

        Notes:
        ------
        - Survival is calculated as `exp(-cumulative hazard)` where the cumulative hazard 
          is calculated using the `cum_hazard` function.
        """
        return np.exp(-self.cum_hazard(hazard))

    @staticmethod
    @jax.jit
    def calculate_hazard_rate_multi_cov(beta, cov, lambda0):
        return jnp.outer(jnp.exp(cov @ beta), lambda0)
    
    @staticmethod
    @jax.jit
    def calculate_hazard_rate_uni_cov(beta, cov, lambda0):
        cov_1d = cov.squeeze() if cov.ndim > 1 else cov
        return jnp.exp(beta * cov_1d)[:, None] * lambda0[None, :]
    
    @staticmethod
    @jax.jit
    def calculate_hazard_rate_time_varying_cov(beta, cov, lambda0):
        tmp = cov @ beta
        return jnp.exp(tmp[:,:,0]) * lambda0
    
    def build_data(self):
        self.build_surv_covariates()
        self.data_on_model['cov'] = self.cov_all
        self.data_on_model['death'] = self.death
        self.data_on_model['exposure'] = self.exposure
        #self.data_on_model['T'] = self.n_intervals
        #self.data_on_model['N_cov'] = self.cov_all.shape[1]

    def priors(self, sample = False):
        conc, rate = self.baseline_rate_prior
        scale = self.hazard_rate_prior_scale

        ## Base hazard distribution
        if not sample and getattr(self.parent, 'backend', None) == 'numpyro':
            log_lambda0 = numpyro.sample(
                'log_Baseline_rate',
                _LogGamma(conc, rate).expand((self.n_intervals,)).to_event(1),
            )
            lambda0 = numpyro.deterministic('Baseline_rate', jnp.exp(log_lambda0))
        else:
            lambda0 = self.parent.dist.gamma(conc, rate, shape= (self.n_intervals,), name = 'Baseline_rate', sample = sample)

        if self.cov_all.ndim == 2:
            ## Covariate effect distribution
            if self.cov_all.shape[1] == 1:
                beta = self.parent.dist.normal(0, scale, shape = (1,),  name=f'Hazard_rate_{self.cov_all_names[0]}', sample = sample)
            else:
                beta = []
                for i in range(self.cov_all.shape[1]):
                    beta.append(self.parent.dist.normal(0, scale, shape = (1,),  name=f'Hazard_rate_{self.cov_all_names[i]}', sample = sample))
                beta = jnp.array(beta)
        elif self.cov_all.ndim == 3:
            ## Covariate effect distribution
            beta = []
            for i in range(self.cov_all.shape[2]):
                beta.append(self.parent.dist.normal(0, scale, shape = (1,),  name=f'Hazard_rate_{self.cov_all_names[i]}', sample = sample))
            beta = jnp.array(beta)

        return lambda0, beta

    def model_univariate(self, death, cov,exposure, censoring = None):
        # Parameters priors distributions-------------------------
        lambda0, beta = self.priors()

        ## Likelihood
        ### Compute hazard rate based on covariate effect
        lambda_ =  self.calculate_hazard_rate_uni_cov(beta, cov, lambda0)

        ### Compute exposure rates
        mu =  exposure * lambda_

        # Likelihood calculation
        self.parent.dist.poisson(rate = mu + jnp.finfo(mu.dtype).tiny, obs = death)

    def model_multivariate(self, death, cov,exposure, censoring = None):
        # Parameters priors distributions-------------------------
        lambda0, beta = self.priors()

        ## Likelihood
        ### Compute hazard rate based on covariate effect
        lambda_ =  self.calculate_hazard_rate_multi_cov(beta, cov, lambda0)

        ### Compute exposure rates
        mu =  exposure * lambda_

        # Likelihood calculation
        self.parent.dist.poisson(mu + jnp.finfo(mu.dtype).tiny, obs = death)

    def model_time_varying(self, death, cov,exposure, censoring = None):
        # Parameters priors distributions-------------------------
        lambda0, beta = self.priors()

        ## Likelihood
        ### Compute hazard rate based on covariate effect
        lambda_ =  self.calculate_hazard_rate_time_varying_cov(beta, cov, lambda0)
        ### Compute exposure rates
        mu =  exposure * lambda_

        # Likelihood calculation
        self.parent.dist.poisson(mu + jnp.finfo(mu.dtype).tiny, obs = death)
    
    def model(self, death, cov,exposure, censoring = None):
        print("⚠️This function is still in development. Use it with caution. ⚠️")

        if self.cov_all.ndim == 2:
            if self.cov_all.shape[1] == 1: 
                return self.model_univariate(death, cov, exposure, censoring)
            else:
                return self.model_multivariate(death, cov, exposure, censoring)
        elif self.cov_all.ndim == 3:
                return self.model_time_varying(death, cov, exposure, censoring)

    def plot_censoring(self, cov=None, event='event', time='time', xlabel='Time', ylabel='Subject'):
        """
        Plots the censoring status of subjects in a time-to-event dataset.

        Uses the arrays populated by ``import_time_even`` (``self.time``,
        ``self.event``, ``self.patients``) and, when ``cov`` names a fixed
        covariate imported via ``import_covF``, overlays the subjects for
        which that covariate equals 1.

        Parameters:
        -----------
        cov : str, optional
            Name of a fixed covariate (as passed to ``import_covF``) to
            overlay as scatter points. If ``None`` or not found, no overlay
            is drawn.
        event, time : str, optional
            Kept for backwards compatibility; unused.
        xlabel : str, optional
            Label for the x-axis. Default is 'Time'.
        ylabel : str, optional
            Label for the y-axis. Default is 'Subject'.

        Returns:
        --------
        (fig, ax)
            The Matplotlib figure and axis showing censored and uncensored
            subjects along with the specified covariate overlay.


        """
        if self.time is None or self.event is None:
            raise RuntimeError(
                "No survival data imported yet. Call import_time_even(...) first."
            )

        t = np.asarray(self.time)
        ev = np.asarray(self.event)
        patients = np.asarray(self.patients)
        censored = ev == 0
        uncensored = ev == 1

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot censored subjects (event = 0) as red horizontal lines
        ax.hlines(patients[censored], 0, t[censored], color="C3", label="Censored")

        # Plot uncensored subjects (event = 1) as gray horizontal lines
        ax.hlines(patients[uncensored], 0, t[uncensored], color="C7", label="Uncensored")

        # Add scatter points for subjects with the specified covariate == 1
        if cov is not None and cov in self.cov_names:
            cov_vec = np.asarray(self.cov)[:, self.cov_names.index(cov)]
            mask = cov_vec == 1
            ax.scatter(t[mask], patients[mask], color="k", zorder=10, label=cov)

        # Set plot limits and labels
        ax.set_xlim(left=0)
        ax.set_xlabel(xlabel)
        ax.set_yticks([])
        ax.set_ylabel(ylabel)

        # Set y-axis limits to provide padding around subjects
        ax.set_ylim(-0.25, self.n_patients + 0.25)

        # Add legend to the plot
        ax.legend(loc="center right")
        return fig, ax

    def _posterior_draws(self, lambda0, beta, n_draws):
        """Posteriors sliced to ``(S, K)`` baseline rates and ``(S, n)``
        linear predictors ``cov[i] @ beta[s]`` (zeros with no covariate)."""
        K = int(self.n_intervals)
        n = int(self.n_patients)
        posteriors = self.parent.posteriors

        L0 = np.asarray(posteriors[lambda0], dtype=float).reshape(-1, K)

        cov = np.asarray(self.cov, dtype=float)
        if cov.size == 0:
            lp = np.zeros((L0.shape[0], n))
        else:
            cov = cov.reshape(n, -1)
            if beta in posteriors:
                B = np.asarray(posteriors[beta], dtype=float).reshape(-1, cov.shape[1])
            elif all(f"Hazard_rate_{name}" in posteriors for name in self.cov_names):
                cols = [np.asarray(posteriors[f"Hazard_rate_{name}"], dtype=float).reshape(-1, 1)
                        for name in self.cov_names]
                B = np.column_stack(cols)
            else:
                B = np.asarray(posteriors[beta], dtype=float).reshape(-1, cov.shape[1])
            lp = (cov @ B.T).T  # (S, n)

        if n_draws is not None:
            L0 = L0[:int(n_draws)]
            lp = lp[:int(n_draws)]
        return L0, lp

    def sample_times(self, lambda0 = 'Baseline_rate', beta = 'beta',
                     n_draws = None, seed = 0):
        """Posterior predictive event times, one row per posterior draw.

        Returns a ``(n_draws, n_patients)`` numpy array of replicated event
        times. For each posterior draw the piecewise-exponential rate is
        ``rate[s, i, k] = lambda0[s, k] * exp(cov[i] @ beta[s])`` and the event
        time is the exponential draw inverted through its cumulative hazard.

        The interval indicators in ``yrep`` are not durations, so this is what
        a bayesplot-style KM overlay needs: ``m.diag.ppc_km_overlay(y=s.time,
        yrep=s.sample_times(), status_y=s.event)``. Like bayesplot's predicted
        curves these times carry no censoring: a draw beyond the last interval
        bound is extrapolated with the last interval's rate, and a zero rate
        gives ``np.inf``. For one panel per covariate see ``ppc_km_by_cov``.
        """
        K = int(self.n_intervals)
        bounds = np.asarray(self.interval_bounds, dtype=float)
        widths = np.diff(bounds)  # (K,)

        L0, lp = self._posterior_draws(lambda0, beta, n_draws)
        S, n = L0.shape[0], lp.shape[1]

        rate = np.exp(lp)[:, :, None] * L0[:, None, :]  # (S, n, K)

        # Cumulative hazard at the interval bounds: H[s, i, 0] = 0.
        H = np.zeros((S, n, K + 1))
        H[:, :, 1:] = np.cumsum(rate * widths, axis=2)

        rng = np.random.default_rng(seed)
        E = rng.exponential(size=(S, n))

        # Interval holding each draw: first k with H[k + 1] >= E. Draws past the
        # last bound keep the last interval's rate, which continues the hazard
        # linearly beyond bounds[K].
        reached = H[:, :, 1:] >= E[:, :, None]
        k = np.argmax(reached, axis=2)
        k = np.where(reached.any(axis=2), k, K - 1)

        Hk = np.take_along_axis(H[:, :, :-1], k[:, :, None], axis=2)[:, :, 0]
        rk = np.take_along_axis(rate, k[:, :, None], axis=2)[:, :, 0]

        with np.errstate(divide='ignore', invalid='ignore'):
            times = bounds[k] + (E - Hk) / rk
        # A zero rate never reaches E, so such a subject never fails.
        times = np.where(rk > 0, times, np.inf)
        return np.asarray(times, dtype=float)

    @staticmethod
    def _quantile_groups(values, n_groups):
        """Labels "Q1".."Qn" from quantile bins of ``values`` (ties may leave
        a bin empty; only nonempty bins get a panel)."""
        edges = np.quantile(values, np.linspace(0.0, 1.0, n_groups + 1)[1:-1])
        codes = np.searchsorted(edges, values, side="right")
        return np.array([f"Q{i + 1}" for i in codes])

    def _cov_groups(self, name, n_groups):
        """Grouping vector for covariate ``name``: its values when it has at
        most ``n_groups`` distinct ones, quantile-bin labels otherwise."""
        col = np.asarray(self.cov, dtype=float)[:, self.cov_names.index(name)]
        if np.unique(col).size <= n_groups:
            return col
        return self._quantile_groups(col, n_groups)

    def ppc_km_by_cov(self, covs = None, n_groups = 3, n_draws = None, seed = 0,
                      n = 50, extrapolation_factor = 1.2, risk = True,
                      lambda0 = 'Baseline_rate', beta = 'beta', facet = False):
        """Grouped KM-overlay PPCs of the replicated times, one figure per
        covariate.

        Builds on ``sample_times()`` and ``m.diag.ppc_km_overlay(group=...)``:
        each covariate groups the patients -- at most ``n_groups`` distinct
        values are grouped as-is, anything more is binned into ``n_groups``
        quantile bins (Q1..Qn) -- so a multivariate model gets one
        bayesplot-style grouped KM check per covariate instead of a single
        pooled curve.

        By default (``facet=False``), each figure overlays the survival curves
        of all covariate levels on the same plot with distinct colors per level.
        Setting ``facet=True`` separates the levels into subplots (one panel
        per level).

        With ``risk=True`` the dict also holds a "risk" figure grouping the
        patients by quantiles of the posterior-mean linear predictor: one
        panel per quantile of predicted hazard, which checks all covariates
        at once (Q1 = lowest predicted hazard).

        Args:
            covs: Covariate name, list of names, or None for all of
                ``self.cov_names``.
            n_groups: Max number of distinct values grouped as-is, and
                number of quantile bins otherwise.
            n_draws, seed: Passed to ``sample_times``.
            n, extrapolation_factor: Passed to ``ppc_km_overlay``.
            risk: Also build the linear-predictor risk-group figure.
            lambda0, beta: Posterior names, as in ``sample_times``.
            facet: When False (the default), overlays groups on a single
                plot with distinct colors. When True, separates them into
                subplots (one panel per level).
        Returns:
            dict {label: plotly Figure} keyed by covariate name, plus
            "risk" when ``risk=True``.
        """
        if covs is None:
            covs = list(self.cov_names)
        elif isinstance(covs, str):
            covs = [covs]
        for c in covs:
            if c not in self.cov_names:
                raise ValueError(
                    f"Unknown covariate {c!r}. Available: {self.cov_names}")
        if not covs and not risk:
            raise ValueError("No covariate to plot and risk=False.")

        yrep = self.sample_times(lambda0=lambda0, beta=beta,
                                 n_draws=n_draws, seed=seed)
        overlay = self.parent.diag.ppc_km_overlay
        figs = {}
        for c in covs:
            figs[c] = overlay(y=self.time, yrep=yrep, status_y=self.event,
                              n=n, extrapolation_factor=extrapolation_factor,
                              group=self._cov_groups(c, n_groups),
                              group_name=c, name="time", facet=facet)
        if risk:
            _, lp = self._posterior_draws(lambda0, beta, n_draws)
            figs["risk"] = overlay(
                y=self.time, yrep=yrep, status_y=self.event,
                n=n, extrapolation_factor=extrapolation_factor,
                group=self._quantile_groups(lp.mean(axis=0), n_groups),
                group_name="risk", name="time", facet=facet)
        return figs

    def plot_surv(self, lambda0 = 'Baseline_rate', beta = 'beta',
                  xlab='Time', ylab='Survival', covlab = 'treated', title = "Bayesian survival model",
                  hdi_prob = 0.94):

        base_hazard = self.parent.posteriors[lambda0]
        met_hazard = self.parent.posteriors[lambda0] * self.parent.posteriors[beta]

        # Drop trailing intervals that carry no data (zero total exposure).
        # Their Baseline_rate stays at the Gamma(0.01, 0.01) prior, whose heavy
        # tail otherwise blows up the cumulative-hazard mean and HDI band at the
        # right edge of the plot.
        exposure_by_interval = np.asarray(self.exposure).sum(axis=0)
        nz = np.nonzero(exposure_by_interval > 0)[0]
        n_valid = int(nz.max()) + 1 if nz.size else self.n_intervals

        x = self.interval_bounds[:-1][:n_valid]

        def _band(ax, draws, color, label=None):
            """Equal-tailed credible band over the draw axis (axis 0).

            Replaces az.plot_hdi, which was removed in arviz 1.0 / arviz-stats.
            """
            draws = np.asarray(draws)[..., :n_valid]
            q = 100 * (1 - hdi_prob) / 2
            lo, hi = np.percentile(draws, [q, 100 - q], axis=0)
            ax.fill_between(x, lo, hi, color=color, alpha=0.3, label=label)

        fig, (hazard_ax, surv_ax) = plt.subplots(ncols=2, sharex=True, sharey=False, figsize=(16, 6))

        _band(hazard_ax, self.cum_hazard(base_hazard), "C0", "Had not metastasized")
        _band(hazard_ax, self.cum_hazard(met_hazard), "C1", "Metastasized")

        hazard_ax.plot(x, jnp.mean(self.cum_hazard(base_hazard), axis = 0)[:n_valid], color="darkblue")
        hazard_ax.plot(x, jnp.mean(self.cum_hazard(met_hazard), axis = 0)[:n_valid], color="maroon")

        hazard_ax.set_xlim(0, self.time.max())
        hazard_ax.set_xlabel(xlab)
        hazard_ax.set_ylabel(r"Cumulative hazard $\Lambda(t)$")
        hazard_ax.legend(loc=2)

        _band(surv_ax, self.survival(base_hazard), "C0")
        _band(surv_ax, self.survival(met_hazard), "C1")

        surv_ax.plot(x, jnp.mean(self.survival(base_hazard), axis = 0)[:n_valid], color="darkblue")
        surv_ax.plot(x, jnp.mean(self.survival(met_hazard), axis = 0)[:n_valid], color="maroon")

        surv_ax.set_xlim(0, self.time.max())
        surv_ax.set_xlabel(ylab)
        surv_ax.set_ylabel("Survival function $S(t)$")

        fig.suptitle(title)
        return fig, (hazard_ax, surv_ax)
