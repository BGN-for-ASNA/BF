from tensorflow_probability.substrates.jax.distributions import JointDistributionCoroutine
from tensorflow_probability.substrates import jax as tfp
tfb = tfp.bijectors
import jax.numpy as jnp
import inspect
import re
import ast
import textwrap
import warnings


class model_handler():
    """    The model_handler class is designed to facilitate the management and manipulation of Bayesian models within the BF framework. It provides methods to extract model arguments, variables, and distributions from a given model function. The class also includes functionality to initialize model parameters and bijectors based on the model's structure and the nature of its variables. This class serves as a bridge between raw model definitions and their practical application in BayesForge, ensuring that models are correctly configured for fitting and analysis.
    """
    def __init__(self):
        self.data_on_model = None # Come from data manip data_to_model function
        self.args = None
        self.vars = None
        self.model = None
        self.model_to_send = None
        self.model_info= None
        self.init_params2 = None
        self.bijectors = None
        # The attribute must exist before run() sets it, so direct
        # init_Model()/attribute users never hit an AttributeError.
        self.obs_names = None

    def get_model_args(self, model):
        # Get the signature of the function
        signature = inspect.signature(model)

        # Extract argument names
        self.args = [param.name for param in signature.parameters.values()]
        return self.args 

    def get_model_var(self, model):
        arguments = self.get_model_args(model)
        # The old list-comprehension line was dead (overwritten here).
        var_model = {key: self.data_on_model[key] for key in arguments if key in self.data_on_model}
        self.vars= var_model # data for the model converted as jnp arrays
    

    def yield_sites(self, model):
        """
        Parse the source of `model` and return every yielded distribution call
        in source order (sorted by the (lineno, col_offset) of the `yield`).

        Returns:
            list of dicts, one per yielded call, with keys:
            "target": assigned variable name (single-name `ast.Assign` parent), else None,
            "func": name of the called function/attribute, e.g. 'normal',
            "obs": ast.unparse of the `obs=` keyword value, else None,
            "name": literal string value of the `name=` keyword, else None.
        """
        source_code = textwrap.dedent(inspect.getsource(model))
        tree = ast.parse(source_code)

        # Map each Yield node to the variable it is assigned to (single ast.Name target).
        targets = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Yield):
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    targets[id(node.value)] = node.targets[0].id

        yields = [node for node in ast.walk(tree)
                  if isinstance(node, ast.Yield) and isinstance(node.value, ast.Call)]
        yields.sort(key=lambda node: (node.lineno, node.col_offset))

        sites = []
        for node in yields:
            call = node.value
            if isinstance(call.func, ast.Attribute):
                func = call.func.attr
            elif isinstance(call.func, ast.Name):
                func = call.func.id
            else:
                func = ast.unparse(call.func)
            obs = None
            name = None
            for keyword in call.keywords:
                if keyword.arg == 'obs':
                    obs = ast.unparse(keyword.value)
                elif keyword.arg == 'name':
                    if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                        name = keyword.value.value
            sites.append({
                "target": targets.get(id(node)),
                "func": func,
                "obs": obs,
                "name": name,
            })
        return sites

    def check_obs_last(self, sites):
        """
        Raise ValueError if any latent (unobserved) yield comes after an observed one:
        the TFP joint distribution requires the observed sites to be the last yields.
        """
        obs_seen = False
        misplaced = []
        for i, site in enumerate(sites):
            if site["obs"] is not None:
                obs_seen = True
            elif obs_seen:
                misplaced.append(site["target"] or site["name"] or f"param_{i}")
        if misplaced:
            raise ValueError(
                "TFP backend: observed sites (obs=) must be the last yields of "
                f"the model; move {misplaced} after the parameters."
            )

    def get_model_distributions(self, model):
        # M6: AST-based site extraction; no substring test on the source lines,
        # so a parameter whose name merely contains "obs" is kept.
        latents = [s for s in self.yield_sites(model) if s["obs"] is None]
        variables = {}
        for i, site in enumerate(latents):
            key = site["target"] or site["name"] or f"param_{i}"
            variables[key] = {'distribution': site["func"]}
        self.model_info = variables

    def find_obs_in_model(self, model):
        """
        Extract observed argument names from `obs` in calls in `model`,
        in source order of the yields.
        """
        return [s["obs"] for s in self.yield_sites(model) if s["obs"] is not None]

    # You need to pass the name of the observed variable to this function
    def initialise(self, infos, init_params, obs_names=None):
        if obs_names is None:
            obs_names = []
        if isinstance(obs_names, str):
            obs_names = [obs_names]

        # Minor 4: fail with a clear message instead of IndexError when the
        # parsed model_info does not match the number of initial values.
        n_latent = sum(1 for key in infos if key not in obs_names)
        if len(init_params) != n_latent:
            raise ValueError(
                f"initialise: model_info has {len(infos)} site(s) of which "
                f"{len(obs_names)} are observed, so {n_latent} initial value(s) "
                f"are expected, but init_params has {len(init_params)}.")

        init_params2 = []
        bijectors = []
        i = 0 # This now correctly tracks the index for init_params

        for key in infos.keys():
            # --- NEW: Check if this is one of the observed variables ---
            if key in obs_names:
                continue  # Skip to the next key in the loop
            # --- END NEW ---

            dist_name = infos[key]['distribution'].lower()
            
            # Now it's safe to access init_params[i] because we've skipped
            # the observed variable, and `i` corresponds to the unobserved params.
            param_shape = init_params[i].shape

            # --- Correlation Matrix ---
            if 'lkj' in dist_name or 'correlation' in dist_name:
                init_params2.append(jnp.eye(param_shape[0]))
                bijectors.append(tfb.CorrelationCholesky())

            # --- Positive-Only Parameters (scale, rates) ---
            elif 'exponential' in dist_name or 'half' in dist_name or 'gamma' in dist_name or 'chi2' in dist_name:
                init_params2.append(jnp.ones_like(init_params[i]))
                bijectors.append(tfb.Exp())

            # --- Default: Unconstrained Parameters ---
            else:
                # Minor 3: a library warning, not a stdout print.
                warnings.warn(
                    f"No specific bijector found for '{key}' (dist: {dist_name}). "
                    "Using Identity bijector (unconstrained support assumed).",
                    UserWarning, stacklevel=2)
                init_params2.append(jnp.zeros_like(init_params[i])) # Start at 0 for unconstrained
                bijectors.append(tfb.Identity())

            # IMPORTANT: Only increment `i` for unobserved variables
            i += 1
            
        return init_params2, bijectors