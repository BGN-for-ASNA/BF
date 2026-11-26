"""
Network-Based Diffusion Analysis (NBDA) models.

Backend note: this module uses numpyro directly —
``numpyro.handlers.mask`` and ``numpyro.distributions.Binomial`` in
``NBDA.model`` — instead of the BayesForge backend proxy used by the rest
of the library (``dist`` / ``bf.dist``). NBDA therefore runs on the numpyro
backend only and cannot be switched to another backend.
"""
import warnings

import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap
from BayesForge.Distributions.np_dists import UnifiedDist as _UnifiedDist
dist = _UnifiedDist()
from BayesForge.Utils.link import link
import numpyro
try:
    from IPython.display import Markdown, display
except ImportError:
    Markdown = str
    display = print

class NBDA:
    """
    The NBDA class is a comprehensive framework for conducting Network-Based Diffusion Analysis. It is designed to build, specify, and prepare complex statistical models that analyze how a behavior, trait, or piece of information spreads through a social network over time. The class systematically handles the data preparation and model specification process.

    The core functionality of the class is to disentangle the two primary mechanisms of     diffusion: asocial learning (spontaneous adoption) and social learning (adoption through    influence from network neighbors). It allows for these learning rates to be influenced by a    rich set of covariates.

    Notes:
        - ``self.status`` is stored internally as a 3-dimensional array of
          shape ``(n, t, 1)`` (the last axis is the single supported process),
          even though the constructor accepts ``(n, t)`` or ``(n, t, 1)``
          input.
    """
    def __init__(self,network=None,status=None,names_network=None,names_status=None): 
        """
        Initialize an NBDA object with network and status arrays.

        Args:
            network (Array): A 2-dimensional array of shape (n, n), a 3-dimensional array of shape (n, n, t)
                             or a 4-dimensional array of shape (n, n, t, num_networks) with num_networks == 1. 
                             If 2D, it is repeated across time. If 3D or 4D, it should be created by the user.
            status (Array): A 2-dimensional array of shape (n, t) representing the status of nodes over time.
                            A 3-dimensional array of shape (n, t, 1) is squeezed to (n, t); several
                            processes (n, t, P>1) are not supported yet.

        Returns:
            None
        """
        
        self.cov = False
        self.random_factor = False
        self.time_varying_network = False
        self.multi_network = False
        self.multiprocess = False

        if network is None or status is None:
            # RELEASE_AUDIT m4: an NBDA built without network/status is a
            # placeholder: every attribute exists (set to None) and the
            # methods that need the data raise ValueError (see
            # _require_data) instead of AttributeError.
            self.nbdaModel = False
            self.status = None
            self.network = None
            self.names = None
            self.n = None
            self.t = None
            self.N_process = None
            self.intercept = None
            self.intercept_asocial = None
            self.covNF_i = None
            self.covNF_j = None
            self.covNV_i = None
            self.covNV_j = None
            self.covDF = None
            self.covDV = None
            self.observed = None
            self.T_social = None
            self.T_asocial = None
            self.objects = None
            self.locations = None
            self.covNF_get_j = None
            self.covNV_get_j = None
        else:
            # Names of the networks and status
            self.names = dict(
                intercept = [],
                status = [],
                covNF = [],
                covNV = [],
                covDF = [],
                covDV = [],
                network = [],
                observed = [],
            )
            self.nbdaModel = True

            # Covariates locations
            self.locations = dict(
                covNF_i = [],
                covNF_j = [],
                covNV_i = [],                
                covNV_j = [],
            )

            # Get j covariate
            self.covNF_get_j = []
            self.covNV_get_j = []

            # Status 
            if len(status.shape) == 3: # (n, t, P): only a single process is supported (RELEASE_AUDIT M7)
                if status.shape[2] == 1:
                    status = status[:, :, 0] # squeeze to (n, t) and continue as a single process
                else:
                    raise NotImplementedError("NBDA: status with several processes (n, t, P>1) is not supported yet; fit each process separately")
            elif len(status.shape) > 3:
                raise ValueError(f"NBDA: status must be a 2 (n, t) or 3 (n, t, 1) dimensional array, got shape {status.shape}")
            self.N_process = 1

            self.status=status
            self.n=status.shape[0]
            self.t=status.shape[1]

            # Validate the data before any conversion (RELEASE_AUDIT M13c)
            if len(network.shape) >= 2:
                if network.shape[0] != network.shape[1]:
                    raise ValueError(f"NBDA: network must be square in its first two dims, got shape {network.shape}")
                if network.shape[0] != self.n:
                    raise ValueError(f"NBDA: network has {network.shape[0]} individuals but status has {self.n} (network shape {network.shape}, status shape {status.shape})")
            if len(network.shape) in (3, 4) and network.shape[2] != self.t:
                raise ValueError(f"NBDA: network has {network.shape[2]} time steps but status has {self.t} (network shape {network.shape}, status shape {status.shape})")
            status_arr = np.asarray(status)
            if np.isnan(status_arr).any():
                raise ValueError("NBDA: status contains NaN, only 0 and 1 are allowed")
            bad = np.unique(status_arr[(status_arr != 0) & (status_arr != 1)])
            if bad.size:
                raise ValueError(f"NBDA: status must contain only 0 and 1, got {bad.tolist()}")

            self.intercept_asocial = jnp.ones((self.n,self.t,self.N_process)) # Intercept for asocial learning

            ## Status at t-1
            self.convert_status(status) # stores self.status as (n, t, 1) (RELEASE_AUDIT m5/m6)
            if names_status is None:
                self.names['status'].append('status')
            else:
                self.names['status'].append(names_status)

            # Network
            self.network=None
            if len(network.shape)==2: # 2D arrays representing a single network (n, n)
                self.convert_network(network)
            elif (len(network.shape)==3): # 3D arrays representing networks through time (n, n, t)
                self.network=network[..., None] # add the network axis -> (n, n, t, 1) so model()'s network[:, :, t, 0] works (RELEASE_AUDIT M9)
                self.time_varying_network = True
            elif (len(network.shape)==4):
                if network.shape[3] > 1: # RELEASE_AUDIT M8: model() only fits network[:, :, t, 0]
                    raise NotImplementedError("NBDA: several networks (n, n, t, K>1) are not supported by model() yet; the extra networks would be ignored")
                self.network=network # 4D arrays representing multiple networks through time (n, n, t, num_networks), to represent multiple networks that do not vary through time, just repeat the network   rough time axis
                self.time_varying_network = True
            else:
                raise ValueError("Network must be a 2 (n, n), 3 (n, n, t) or 4 dimensional array (n, n, t, num_networks)")
            if names_network is None:
                self.names['network'].append('network')
            else:
                self.names['network'].append(names_network)

            # Intercetps for the network
            self.intercept = jnp.ones(self.network.shape)

            # fixed nodal covariates
            self.covNF_i=None
            self.covNF_j=None

            # Time-varying nodal covariates
            self.covNV_i=None
            self.covNV_j=None

            # fixed dyadic covariates
            self.covDF=None

            # Time-varying dyadic covariates
            self.covDV=None

            # individual observed
            self.observed=None

            # Conatenated covariates
            self.T_social=None
            self.T_asocial=None

            self.objects = None

    def _require_data(self, method):
        """
        RELEASE_AUDIT m4: raise a clear ValueError instead of an
        AttributeError when the object was built without network/status.
        """
        if self.network is None or self.status is None:
            raise ValueError(
                "NBDA requires network and status: build the object with "
                f"NBDA(network, status) before calling {method}"
            )

    @staticmethod
    @jax.jit
    def scale_data(x):
        std = x.std()
        return (x - x.mean()) / jnp.where(std > 0, std, 1.0)

    @staticmethod
    def scale_along_time(covNV):
        # Standardize each series along the last (time) axis only: per (variable,
        # node) for covNV (var, n, t) and per (i, j) for covDV (n, n, t).
        x = jnp.asarray(covNV, dtype=float)
        m = x.mean(-1, keepdims=True)
        s = x.std(-1, keepdims=True)
        return (x - m) / jnp.where(s > 0, s, 1.0)

    @staticmethod
    def _check_no_nan(data, method):
        if np.isnan(np.asarray(data, dtype=float)).any():
            raise ValueError(f"{method}: input contains NaN")

    def convert_network(self, network, names = None): # To be used only if there is a single network

        """
        Convert a single network array to a 4D array.

        Args:
            network (Array): A 2-dimensional array of shape (n, n).

        Returns:
            network: A 4-dimensional array of shape (n, n, t, 1) after repeating the network across time.
        """

        self.network=jnp.repeat(network[jnp.newaxis, :, :,jnp.newaxis], self.t, axis=0).transpose((1,2,0,3))

        if names is not None:
            if len(names)!=self.network.shape[3]:
                raise ValueError("The length of names must be equal to the number of networks.")
            self.names['network']=[name for name in names]
        return self.network

    def convert_status(self, status):#A 2-dimension arrray of (n,t)
        """
        Store the status internally as a 3-dimensional array of shape (n, t, 1).

        Args:
            status (Array): A 2-dimensional array of shape (n, t).

        Returns:
            self.status (Array): The same values as status, shaped (n, t, 1)
            (last axis = the single supported process).

        RELEASE_AUDIT m6/m7: the lagged (n, n, t) status_i/status_j arrays
        were removed — no code read them (O(n^2 t) memory for nothing). The
        (n, t, 1) view is now built with a reshape instead of the old Python
        loop of full copies; values are identical.
        """
        self.status = jnp.asarray(status)[:, :, None]
        return self.status

    def covNF_dims(self,df, n, t, num_variables):
        """
        Convert fixed nodal covariates into 4D arrays.

        Args:
            df (DataFrame or Array): A 2-dimensional array of shape (n, num_variables).
            n (int): Number of nodes.
            t (int): Number of time points.
            num_variables (int): Number of covariates.

        Returns:
            result_array_i (Array): A 4-dimensional array of shape (n, n, t, num_variables) for i.
            result_array_j (Array): A 4-dimensional array of shape (n, n, t, num_variables) for j.
        """
        # Create arrays using broadcasting
        df = jnp.asarray(df)[:, :num_variables]
        # axis 0 = learner i, axis 1 = demonstrator j
        result_array_i = jnp.broadcast_to(df[:, None, None, :], (n, n, t, num_variables))  # (n, n, t, num_variables)
        result_array_j = jnp.broadcast_to(df[None, :, None, :], (n, n, t, num_variables))  # (n, n, t, num_variables)

        return result_array_i, result_array_j

    def convert_covNF(self, df, n, t, num_variables, scale = True):
        """
        Convert fixed nodal covariates into 4D arrays.

        Args:
            df (DataFrame or Array): A 2-dimensional array of shape (n, num_variables).
            n (int): Number of nodes.
            t (int): Number of time points.
            num_variables (int): Number of covariates.

        Returns:
            tuple: A tuple of two 4-dimensional arrays (result_array_i, result_array_j).
        """
        if isinstance(df, pd.DataFrame):
            if scale:
                df = df.apply(lambda x: self.scale_data(x.values), axis=0)
                df = jnp.array(df)
            else:
                df = jnp.array(df)
        else:
            if len(df.shape)>2:
                raise ValueError("covariates must be a data frame or a 2-dimensional array")
            else:
                if scale:
                    df = vmap(lambda x: self.scale_data(x), in_axes=1, out_axes=1)(df)

        return  self.covNF_dims(df, n, t, num_variables)

    def import_covNF(self, df, names=None, where = 'both', get_cov_j = True, scale = True):
        """
        Import fixed nodal covariates.

        Args:
            df (DataFrame or Array): A 2-dimensional array of shape (n, num_variables).

        Returns:
            tuple: A tuple of two 4-dimensional arrays (covNF_i, covNF_j).
        """
        self._require_data('import_covNF')
        self._check_no_nan(df, 'import_covNF')
        if where not in ('both', 'social', 'asocial'):
            raise ValueError(f"where must be 'both', 'social' or 'asocial', got {where!r}")
        if not isinstance(df, pd.DataFrame):
            df = np.asarray(df, dtype=float)
            if df.ndim == 1:
                df = df.reshape(-1, 1)
        if df.shape[0] != self.n:
            raise ValueError(f"import_covNF: input has {df.shape[0]} rows, expected n = {self.n}")
        n_new = df.shape[1]

        covNF_i, covNF_j = self.convert_covNF(df, self.n, self.t, n_new, scale = scale)

        # Update self.covNF_i
        if where == 'asocial':
            get_cov_j = False
        if self.covNF_i is None:
            self.covNF_i = covNF_i
        else:
            self.covNF_i = jnp.concatenate((self.covNF_i, covNF_i), axis = -1)

        # Build cov from demonstrator
        if get_cov_j:
            if self.covNF_j is None:
                self.covNF_j = covNF_j
            else:
                self.covNF_j = jnp.concatenate((self.covNF_j, covNF_j), axis = -1)

        if names is None:
            self.names['covNF'].extend([f"covNF{len(self.names['covNF']) + i}" for i in range(n_new)])
        else:
            if len(names) != n_new:
                raise ValueError(f"import_covNF: names must have {n_new} entries, got {len(names)}")
            self.names['covNF'].extend(names)

        # One entry per new column, counted on the imported block: self.covNF_j
        # stays None when where='asocial' (M1).
        self.covNF_get_j.extend([get_cov_j]*n_new)
        self.locations['covNF_i'].extend([where]*n_new)
        self.locations['covNF_j'].extend([where]*n_new)

    def convert_covNV(self, covV):
        """
        Convert time-varying nodal covariates into 4D arrays.

        Args:
            covV (Array): A 3-dimensional array of shape (num_variables, n, t).

        Returns:
            tuple: A tuple of two 4-dimensional arrays (result_array_i, result_array_j).
        """
        c = jnp.transpose(jnp.asarray(covV), (1, 2, 0))  # (n, t, num_variables)
        shape = (self.n, self.n, c.shape[1], c.shape[2])
        # axis 0 = learner i, axis 1 = demonstrator j
        result_array_i = jnp.broadcast_to(c[:, None, :, :], shape)
        result_array_j = jnp.broadcast_to(c[None, :, :, :], shape)
        
        return result_array_i, result_array_j

    def import_covNV(self, covV, names = None, scale = True, where = 'both', get_cov_j = True): #covV need to be a 3 dimensional array of shape (num_var, n, t)i.e. A list of matrices of time-varying covariates
        self._require_data('import_covNV')
        self._check_no_nan(covV, 'import_covNV')
        if where not in ('both', 'social', 'asocial'):
            raise ValueError(f"where must be 'both', 'social' or 'asocial', got {where!r}")
        if scale:
            covV = self.scale_along_time(covV)
        else: 
            # RELEASE_AUDIT m9: warnings.warn (filterable) instead of print;
            # the two misspellings in the old message are fixed.
            warnings.warn(
                "Not scaling covariates along time can result in correlation "
                "between regression coefficients of temporal covariates and "
                "the intercepts (i.e. social rate). An alternative solution "
                "would be to add time varying coefficients, but it does "
                "affect computational time drastically.",
                UserWarning,
                stacklevel=2,
            )

        covNV_i, covNV_j = self.convert_covNV(covV)
        n_new = covNV_i.shape[3]

        if where == 'asocial':
            get_cov_j = False
        if self.covNV_i is None:
            self.covNV_i = covNV_i
        else:
            self.covNV_i = jnp.concatenate((self.covNV_i, covNV_i), axis = -1)

        # Build cov from demonstrator
        if get_cov_j:
            if self.covNV_j is None:
                self.covNV_j = covNV_j
            else:
                self.covNV_j = jnp.concatenate((self.covNV_j, covNV_j), axis = -1)

        if names is None:
            self.names['covNV'].extend([f"covNV{len(self.names['covNV']) + i}" for i in range(n_new)])
        else:
            if len(names) != n_new:
                raise ValueError(f"import_covNV: names must have {n_new} entries, got {len(names)}")
            self.names['covNV'].extend(names)

        # One entry per new column, counted on the imported block: self.covNV_j
        # stays None when where='asocial' (M1).
        self.covNV_get_j.extend([get_cov_j]*n_new)
        self.locations['covNV_i'].extend([where]*n_new)
        self.locations['covNV_j'].extend([where]*n_new)

    def import_covDF(self, covDF, names = None,  scale = True):
        """
        Import fixed dyadic covariates.

        Args:
            covDF (Array): A 2-dimensional array of shape (n, n) or a 3-dimensional array of shape (n, n, num_dyads).

        Returns:
            Array: A 4-dimensional array of shape (n, n, t, num_variables) if input is 3D, otherwise shape (n, n, t, 1).
        """
        self._require_data('import_covDF')
        self._check_no_nan(covDF, 'import_covDF')
        if len(covDF.shape)==2:
            tmp = self.scale_data(covDF) if scale else jnp.asarray(covDF)
            covDF = jnp.repeat(tmp[jnp.newaxis, :, :,jnp.newaxis], self.t, axis=0).transpose((1,2,0,3))

        else:
            # RELEASE_AUDIT m7: scale each (n, n) matrix, then broadcast the
            # time axis instead of materialising one full (n, n, t) copy per
            # variable (identical values, no per-variable time copies).
            mats = jnp.asarray(covDF)
            if scale:
                mats = vmap(lambda x: self.scale_data(x), in_axes=2, out_axes=2)(mats)
            covDF = jnp.broadcast_to(mats[:, :, None, :], (self.n, self.n, self.t, mats.shape[2]))

        n_new = covDF.shape[3]
        if self.covDF is None:
            self.covDF = covDF
        else:
            self.covDF = jnp.concatenate((self.covDF, covDF), axis = -1)


        if names is None:
            self.names['covDF'].extend([f"covDF{len(self.names['covDF']) + i}" for i in range(n_new)])
        else:
            if len(names) != n_new:
                raise ValueError(f"import_covDF: names must have {n_new} entries, got {len(names)}")
            self.names['covDF'].extend(names)

    def import_covDV(self, covDV, names = None, scale = True, where = 'both'):
        """
        Import time-varying dyadic covariates.

        Args:
            covDV (Array): A 3-dimensional array of shape (n, n, t) or a
                4-dimensional array of shape (num_variables, n, n, t): the
                code iterates covDV[i, :, :] over shape[0], so in 4-D the
                first axis is the variable axis (the
                docstring now matches the code's axis order; the code is
                unchanged).

        Returns:
            Array: A 4-dimensional array of shape (n, n, t, num_variables).

        Memory: the 4-D path builds one full (n, n, t)
        copy per variable in a Python loop, i.e. an
        O(num_variables * n^2 * t) materialisation.
        """
        
        self._require_data('import_covDV')
        self._check_no_nan(covDV, 'import_covDV')
        if where not in ('both', 'social', 'asocial'):
            raise ValueError(f"where must be 'both', 'social' or 'asocial', got {where!r}")
        
        if len(covDV.shape)==3:# A list of matrices of a single time-varying covariate
            if scale:
                covDV =  self.scale_along_time(covDV)
            else:
                covDV = jnp.asarray(covDV)
            covDV = covDV[:, :, :,jnp.newaxis]

        elif len(covDV.shape)==4:# A ist of list of matrices of a single time-varying covariate
            tmp = jnp.array([covDV[i,:, :,None]*jnp.ones((self.n, self.n, self.t)) for i in range(covDV.shape[0])])  # (num, n, n, t)
            if scale:
                tmp = self.scale_along_time(tmp)  # same scaling on both paths: each (num, i, j) series over time
            covDV = tmp.transpose((1,2,3,0))

        n_new = covDV.shape[3]
        if self.covDV is None:
            self.covDV = covDV
        else:
            self.covDV = jnp.concatenate((self.covDV, covDV), axis = -1)

        if names is None:
            self.names['covDV'].extend([f"covDV{len(self.names['covDV']) + i}" for i in range(n_new)])
        else:
            if len(names) != n_new:
                raise ValueError(f"import_covDV: names must have {n_new} entries, got {len(names)}")
            self.names['covDV'].extend(names)

    def stack_cov(self):
        """
        Stack all covariates into a dictionary.

        Returns:
            dict: A dictionary containing all covariates.
        """
        self._require_data('stack_cov')
        tmp=dict(
            intercept_social = self.intercept,
            intercept_asocial = self.intercept_asocial,
            status = self.status,
            covNF_i = self.covNF_i,
            covNF_j = self.covNF_j,
            covNV_i = self.covNV_i,
            covNV_j = self.covNV_j,
            covDF = self.covDF,
            covDV = self.covDV,
            network = self.network,
            observed = self.observed,
        )
        return {k: v for k, v in tmp.items() if v is not None}

    def get_cov(self):
        """
        Get all covariates in the NBDA object.

        Returns:
            tuple: A tuple of two arrays (D_social, D_asocial).
        """
        self._require_data('get_cov')
        objects = self.stack_cov()
        self.objects = objects
        D_social = [objects['intercept_social']]
        D_asocial = [objects['intercept_asocial']]
        D_social_names = ['social_learning']
        D_asocial_names = ['asocial_learning']

        for k in objects.keys():
            if k not in ['intercept_social', 'intercept_asocial', 'status', 'network']:
                # RELEASE_AUDIT m11: the old identity test on the loop key
                # always passed (keys are strings); None values are already
                # filtered out by stack_cov.
                if k in ['covNF_i', 'covNV_i']:
                    for i in range(objects[k].shape[3]):
                        loc = self.locations[k][i]
                        if loc == 'both':
                            D_social.append(jnp.expand_dims(objects[k][:,:,:,i],-1))
                            D_asocial.append(jnp.expand_dims(objects[k][:,0,:,i],-1))
                            if 'covNF' in k:
                                D_social_names.append(k + '_' + self.names['covNF'][i])
                                D_asocial_names.append(k + '_' + self.names['covNF'][i])
                            else:
                                D_social_names.append(k + '_' + self.names['covNV'][i])
                                D_asocial_names.append(k + '_' + self.names['covNV'][i])                                    

                        elif loc == 'social':
                            D_social.append(jnp.expand_dims(objects[k][:,:,:,i],-1))
                            if 'covNF' in k:
                                D_social_names.append(k + '_' + self.names['covNF'][i])
                            else:
                                D_social_names.append(k + '_' + self.names['covNV'][i])    

                        elif loc == 'asocial':
                            D_asocial.append(jnp.expand_dims(objects[k][:,0,:,i],-1))
                            if 'covNF' in k:
                                D_asocial_names.append(k + '_' + self.names['covNF'][i])
                            else:
                                D_asocial_names.append(k + '_' + self.names['covNV'][i])

                        else:
                            raise ValueError(f"Unknown location {loc!r} for column {i} of {k}: must be 'both', 'social' or 'asocial'")
                else:
                    D_social.append(objects[k])   
                    if 'covNF' in k:
                        filtered = [s for b, s in zip(self.covNF_get_j, self.names['covNF']) if b]
                        for i in range(len(filtered)):
                            D_social_names.append(k + '_' + filtered[i])
                    elif 'covNV' in k:
                        filtered = [s for b, s in zip(self.covNV_get_j, self.names['covNV']) if b]
                        for i in range(len(filtered)):
                            D_social_names.append(k + '_' + filtered[i])      
                    elif 'covDF' in k:
                        for i  in range(objects[k].shape[3]):
                            D_social_names.append(k + '_' + self.names['covDF'][i])  
                    elif 'covDV' in k:
                        for i  in range(objects[k].shape[3]):
                            D_social_names.append(k + '_' + self.names['covDV'][i])  

        self.T_social = jnp.concatenate(D_social, axis=-1)
        self.T_asocial = jnp.concatenate(D_asocial, axis=-1)
        self.T_social_names = D_social_names
        self.T_asocial_names = D_asocial_names
        #return dict(D_social=self.D_social, D_asocial=self.D_asocial, status=self.status, network=self.network)
    
    @staticmethod
    @jax.jit
    def ces(K, L, alpha):
        """
        Linear mixture of the two attention-weighted network orientations:
        Q = alpha * K + (1 - alpha) * L.

        RELEASE_AUDIT m13: this is a linear mixture (perfect substitutes,
        constant elasticity = infinity), not a CES aggregator with a
        curvature parameter rho as in the cited supplement (aax9070). The
        name is kept for backward compatibility.

        RELEASE_AUDIT m8: jitted as a staticmethod so the compilation cache
        is tied to the function, not to each NBDA instance (the old
        static_argnums=(0,) made every object recompile and retained memory
        for no benefit; the values are unchanged).
        """
        Q = (alpha * K + (1 - alpha) * L)
        return Q

    def priors(self, CES = True):
        T_social_shape = self.T_social.shape[3]
        T_asocial_shape = self.T_asocial.shape[2]
        if CES:
            ces_alpha =  dist.uniform(0.0001, 0.9999, name = 'ces_alpha')
        if T_social_shape > 1: # Covariates have been provided for social formula
            # Priors for social effect covariates
            alpha_soc = dist.normal(0, 2, shape = (1,), sample=False, name='social_learning')
            betas_soc = []
            for a in range(T_social_shape-1):
                beta_soc = dist.normal(0, 2, shape = (1,), sample=False, name=f"Beta_social_{self.T_social_names[a+1]}")
                betas_soc.append(beta_soc)
            betas_soc  = jnp.concatenate(betas_soc, -1)
            soc = jnp.concatenate((alpha_soc, betas_soc), axis = -1)
        else:
            soc = dist.normal(0, 2, shape = (1,), sample=False, name='social_learning')
        if T_asocial_shape > 1: # Covariates have been provided for asocial formula
            # Priors for asocial effect covariates
            alpha_asoc = dist.normal(0, 2,  shape = (1,), sample=False, name='asocial_learning')

            betas_asoc = []
            for a in range(T_asocial_shape-1):
                beta_asoc = dist.normal(0, 2, shape = (1,), sample=False, name=f"Beta_asocial_{self.T_asocial_names[a+1]}")
                betas_asoc.append(beta_asoc)
            betas_asoc  = jnp.concatenate(betas_asoc, -1)
            asoc = jnp.concatenate((alpha_asoc, betas_asoc), axis = -1)
        else:
            asoc = dist.normal(0, 2, shape = (1,), sample=False, name='asocial_learning')

        if CES :
            return soc, asoc, ces_alpha
        else:
            return soc, asoc
    
    # Not wired: model() never routes to priors_multinetwork (RELEASE_AUDIT M8); kept for a future multi-network model.
    def priors_multinetwork(self, N_networks, CES = True):
        T_social_shape = self.T_social.shape[3]
        T_asocial_shape = self.T_asocial.shape[2]
        if CES:
            ces_alpha =  dist.uniform(0.0001, 0.9999, name = 'ces_alpha')
        if T_social_shape > 1: # Covariates have been provided for social formula
            # Priors for social effect covariates
            alpha_soc = dist.normal(0, 2, shape = (N_networks, 1), sample=False, name='social_learning')
            betas_soc = []
            for a in range(T_social_shape-1):
                beta_soc = dist.normal(0, 2, shape = (N_networks,1), sample=False, name=f"Beta_social_{self.T_social_names[a+1]}")
                betas_soc.append(beta_soc)
            betas_soc  = jnp.concatenate(betas_soc, -1)
            soc = jnp.concatenate((alpha_soc, betas_soc), axis = -1)
        else:
            soc = dist.normal(0, 2, shape = (1,), sample=False, name='social_learning')
        if T_asocial_shape > 1: # Covariates have been provided for asocial formula
            # Priors for asocial effect covariates
            alpha_asoc = dist.normal(0, 2,  shape = (1,), sample=False, name='asocial_learning')

            betas_asoc = []
            for a in range(T_asocial_shape-1):
                beta_asoc = dist.normal(0, 2, shape = (1,), sample=False, name=f"Beta_asocial_{self.T_asocial_names[a+1]}")
                betas_asoc.append(beta_asoc)
            betas_asoc  = jnp.concatenate(betas_asoc, -1)
            asoc = jnp.concatenate((alpha_asoc, betas_asoc), axis = -1)
        else:
            asoc = dist.normal(0, 2, shape = (1,), sample=False, name='asocial_learning')

        if CES :
            return soc, asoc, ces_alpha
        else:
            return soc, asoc
    
    def model(self, CES = True):
        """
        Network-Based Diffusion Analysis (NBDA) probabilistic model.

        Parameters:
        - CES: If True, combines the social and asocial transmission rates
          through the linear mixture Q = alpha * K + (1 - alpha) * L of the
          two attention-weighted networks (perfect substitutes; not a CES
          with curvature parameter rho, see NBDA.ces).

        Returns:
        - Probabilistic model sampling diffusion events conditional on network structure and status.

        The likelihood conditions on the initial state status[:, 0] (t = 0 is not an event).

        NaN sentinel: NaN in the likelihood matrix ``lk``
        is the mask for "already informed" entries (t = 0 and status[:, t-1]
        == 1). A NaN produced accidentally by theta or the social weights for
        an individual that is still naive would therefore be masked out of
        the likelihood silently. The check cannot raise here: this code is
        traced by numpyro/JAX at fit time, so a concrete-input check would
        have to touch the traced likelihood.
        """
        self._require_data('model')
        if self.T_social is None: # RELEASE_AUDIT M13b: build the design matrices on demand
            self.get_cov()
        status = self.status
        network = self.network
        D_asocial = self.T_asocial
        D_social = self.T_social

        N = status.shape[0] # Number of individuals
        T = status.shape[1] # Number of time steps

        if CES:
            soc, asoc, ces_alpha = self.priors(CES)
            lk = self.compute_probs_ces(D_asocial, asoc,
                            D_social, soc,
                            ces_alpha,
                            network,status,N,T)
        else:
            soc, asoc = self.priors(CES)
            lk = self.compute_probs(D_asocial, asoc,
                D_social, soc,
                network,status,N,T)

        mask = ~jnp.isnan(lk)
        with numpyro.handlers.mask(mask=mask): 
            # masked (already informed) entries are NaN: give them a valid dummy
            # probability so the distribution validates; the mask zeroes them.
            numpyro.sample("y", numpyro.distributions.Binomial(probs=jnp.where(mask, lk, 0.5)), obs=status[:,:,0])
 
    def compute_probs(self,D_asocial, asoc,
                      D_social, soc,
                      network,status,N,T):
        lk = jnp.zeros((N,T))
        # t = 0 conditions on the initial state: NaN masks it out of the likelihood (RELEASE_AUDIT M10)
        lk = lk.at[:,0].set(jnp.nan)

        for t in range(1,T):
            ## Social learning-----------------------
            R_social = jnp.tensordot(D_social[:,:,t,:], soc, axes=(-1, 0))
            phi = link.inv_logit(R_social)
            attention_weigthed_network = phi*network[:,:,t,0]
            social_influence_weight = jnp.clip(link.inv_logit_scale(jnp.tensordot(attention_weigthed_network[:,:], status[:,t-1], axes=(-1, 0))), 0.0, 1.0)

            ## Asocial learning -----------------------
            R_asocial = jnp.tensordot(D_asocial[:,t,:], asoc, axes=(-1, 0))
            theta = link.inv_logit(R_asocial)

            # Informed update at t!= 0-----------------------
            lk = lk.at[:,t].set(jnp.where(status[:, t-1][:,0] == 1, jnp.nan, theta + (1-theta)*social_influence_weight[:,0]))

        return lk

    # Not wired: model() never routes to compute_probs_multinetwork (RELEASE_AUDIT M8); kept for a future multi-network model.
    def compute_probs_multinetwork(self,D_asocial, asoc,
                                   D_social, soc,
                                   network,status,N,T):
        N_networks = network.shape[3]
        lk = jnp.zeros((N,T))
        # t = 0 conditions on the initial state: NaN masks it out of the likelihood (RELEASE_AUDIT M10)
        lk = lk.at[:,0].set(jnp.nan)

        for t in range(1,T):

            ## Social learning-----------------------
            R_social = jnp.tensordot(D_social[:,:,t,:], soc, axes=(-1, 0))
            phi = link.inv_logit(R_social)
            attention_weigthed_network = phi*network[:,:,t,0]
            social_influence_weight = jnp.clip(link.inv_logit_scale(jnp.tensordot(attention_weigthed_network[:,:], status[:,t-1], axes=(-1, 0))), 0.0, 1.0)

            ## Asocial learning -----------------------
            R_asocial = jnp.tensordot(D_asocial[:,t,:], asoc, axes=(-1, 0))
            theta = link.inv_logit(R_asocial)

            # Informed update at t!= 0-----------------------
            lk = lk.at[:,t].set(jnp.where(status[:, t-1][:,0] == 1, jnp.nan, theta + (1-theta)*social_influence_weight[:,0]))

        return lk    
    
    def compute_probs_ces(self,D_asocial, asoc,
                      D_social, soc,
                      ces_alpha,
                      network,status,N,T):
        lk = jnp.zeros((N,T))
        # t = 0 conditions on the initial state: NaN masks it out of the likelihood (RELEASE_AUDIT M10)
        lk = lk.at[:,0].set(jnp.nan)

        for t in range(1,T):
            ## Social learning-----------------------
            R_social = jnp.tensordot(D_social[:,:,t,:], soc, axes=(-1, 0))
            phi = link.inv_logit(R_social)
            attention_weigthed_network = self.ces(phi*network[:,:,t,0], phi*network[:,:,t,0].T, ces_alpha)
            social_influence_weight = jnp.clip(link.inv_logit_scale(jnp.tensordot(attention_weigthed_network[:,:], status[:,t-1], axes=(-1, 0))), 0.0, 1.0)

            ## Asocial learning -----------------------
            R_asocial = jnp.tensordot(D_asocial[:,t,:], asoc, axes=(-1, 0))
            theta = link.inv_logit(R_asocial)

            # Informed update at t!= 0-----------------------
            lk = lk.at[:,t].set(jnp.where(status[:, t-1][:,0] == 1, jnp.nan, theta + (1-theta)*social_influence_weight[:,0]))
        return lk
    
    def print_model(self):
        self._require_data('print_model')
        # RELEASE_AUDIT m10: the priors in priors() are dist.normal(0, 2, ...)
        # i.e. scale = 2, so the LaTeX shows Normal(0,2) (was Normal(0,4) for
        # the alphas and Normal(0,1) for the betas).
        r="""$$
        \\text{Informed} = \\text{Binomial}(\\text{LK}) \\newline
        \\text{LK} = \\left[ \\theta + (1-\\theta)S \\right] (1- z_i) \\newline
        \\theta = \\alpha_a \\newline
        S = \\alpha_s \\left( \\sum_{j = 1}^{N} A_{ij} z_{j} \\right) \\newline 
        \\alpha_a\\sim Normal(0,2) \\newline
        \\alpha_s \\sim Normal(0,2) \\newline
        """

        if len(self.names) > 2:
            fa="""""" 
            fs=""""""
            count = 0
            for k in self.names.keys():
                if k in ['covNF', 'covNV', 'covDF','covDV']:
                    tmp=self.names[k]
                    if k in ['covNF', 'covNV']:
                        # locations are stored per column (M3): check each column
                        locs = self.locations[k + '_i']
                        for i in range(len(tmp)):
                            loc = locs[i] if i < len(locs) else 'both'
                            if loc in ('both', 'asocial'):
                                fa=fa+f"\\beta_{{a{{{count}}}}} {tmp[i]} + "
                                count += 1
                            if loc in ('both', 'social'):
                                fs=fs+f"\\beta_{{s{{{count}}}}} {tmp[i]} + "
                                count += 1
                    else:   # Dyadic variables can't be in asocial
                        for i in range(len(tmp)):
                            fs=fs+f"\\beta_{{s{{{count}}}}} {tmp[i]} + "
                            count += 1

                    count += 1

            # RELEASE_AUDIT m10: the old first pattern ("\sum A_{ij}" without
            # the "_{j = 1}^{N}" limits) never matched the model string above,
            # so the replacement was silently a no-op. The pattern now matches
            # and each replace only runs when it has covariates to insert.
            if fs:
                r = r.replace(r"\alpha_s \left( \sum_{j = 1}^{N} A_{ij} z_{j} \right)",
                """\\left(\\alpha_s + X\\right) \\sum_{j = 1}^{N} A_{ij} z_{j}  \\newline  X = """ + fs.rstrip(' +') + """  \\newline""")

            if fa:
                r = r.replace(r"\alpha_a ", 
                """\\alpha_a + Z  \\newline  Z = """ + fa.rstrip(' +') + """ \\newline""")

            r = r+"""\\beta_{(s)} \\sim Normal(0,2) \\newline"""
        r=r+"""$$"""
        display(Markdown(r))

    # We can add individual observation information in the same forme as  an input time varying cov
    # We can add multiple behaviors acquisition in the form of a (n,n,t,num_behaviors)
    # Random variable to add
    # (RELEASE_AUDIT m6: the lagged status_i/status_j arrays this note referred
    # to were removed; nothing read them. Re-add them only together with the
    # net*status_j / cov*(1-status_i) filtering they were meant for.)
