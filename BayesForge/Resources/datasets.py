"""
Dataset loader utilities for BF package.

This module provides sklearn-style dataset loaders for the CSV datasets
shipped in the BayesForge/Resources directory. By default (``frame=True``)
each loader method returns a pandas DataFrame; ``frame=False`` returns a
Bunch object with the data, feature names and dataset description, and
``only_path=True`` returns the path of the underlying file as a ``str``.

Example:
    >>> from BayesForge.Resources.datasets import load
    >>> howell = load().howell1()
    >>> print(howell.shape)
    >>> print(list(howell.columns))
"""

import warnings
import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------------------------------------
# m-7 (RELEASE_AUDIT): packaged data hygiene.
# pyproject.toml ships the *.csv / *.rda / *.json of this directory (package-data
# "BayesForge.Resources") so that a reader of the documentation can run every
# example from an installed package, without downloading this folder. Clean-up
# decided after the audit:
#   - NBDA_data.json (24.5 MB, no loader) is excluded from the build in
#     pyproject.toml ([tool.setuptools.exclude-package-data]); the file stays in
#     the repository.
#   - "KosterLeckie copy.rda" (byte-identical duplicate of KosterLeckie.rda) was
#     removed.
#   - The four phylogenetic models (simple, Poisson, repeated measures, meta-
#     analysis) are fitted on the same 200-taxa tree, so they share one kinship
#     Cholesky factor: phylo_L_simple.csv. phylo_L_slopes.csv has its own content.
#   - Howell2.csv (no loader yet) stays packaged: it is a shipped dataset and
#     removing it would break readers who open it by path.
# --------------------------------------------------------------------------------

# Kinship Cholesky factor shared by the four phylogenetic models (same tree).
_PHYLO_L_SHARED = "phylo_L_simple.csv"

class Bunch(dict):
    """
    Container object exposing keys as attributes.

    Bunch objects are sometimes used as an output for functions and methods.
    They extend dictionaries by enabling values to be accessed by key,
    `bunch["value_key"]`, or by an attribute, `bunch.value_key`.

    Examples:
        >>> b = Bunch(a=1, b=2)
        >>> b['b']
        2
        >>> b.b
        2
        >>> b.a = 3
        >>> b['a']
        3
        >>> b.c = 6
        >>> b['c']
        6
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.__dict__ = self


def _bunch(df, filename, description, data=None):
    """
    Build the common Bunch schema used by every ``frame=False`` loader.

    Keys: data, feature_names, DESCR, frame, filename.
    Warns when the frame holds non-numeric columns: ``data`` then keeps
    dtype object (no silent encoding is applied).
    """
    if data is None:
        data = df.values
    if df.values.dtype == object:
        non_numeric = [str(col) for col, dtype in df.dtypes.items()
                       if not pd.api.types.is_numeric_dtype(dtype)]
        warnings.warn(
            f"{filename}: columns {non_numeric} are not numeric, so Bunch.data "
            "has dtype object; use frame=True or encode them "
            "(e.g. pandas.factorize) before modelling.",
            UserWarning,
            stacklevel=3,
        )
    return Bunch(
        data=data,
        feature_names=df.columns.tolist(),
        DESCR=description,
        frame=df,
        filename=filename,
    )


class load:
    """
    Dataset loader class for BF datasets.

    This class provides methods to load various datasets stored in CSV format.
    By default (``frame=True``) every dataset method returns a plain pandas
    DataFrame of the file. Pass ``frame=False`` to get a Bunch object with the
    following attributes:
    - data: numpy array of features (dtype object when some columns are text;
      frame=False warns in that case, use frame=True or encode the columns)
    - feature_names: list of feature column names
    - DESCR: description of the dataset
    - frame: pandas DataFrame (full data including all columns)

    Pass ``only_path=True`` to any dataset method to get the path of the
    underlying CSV file as a ``str`` (the majority type, used consistently by
    every loader); the file is not read in that case.

    Naming: the class is named ``load`` (lowercase) and the method names mix
    snake_case (``howell1``, ``ucbadmit``, ``phylo_simple``) with PascalCase
    acronyms (``WaffleDivorce``, ``NBDA_events``, ``NBDA_network``). These
    public names are kept unchanged for backward compatibility; use the
    documented spelling when resolving them by name (getattr, generic scripts).

    The class has two roles:
    - dataset loader: instantiate it with ``load()`` and call a dataset method;
    - saved-session loader: ``load("bf_model.pkl")`` (handled by ``__new__``)
      and ``load()("bf_model.pkl")`` (handled by ``__call__``) both delegate
      to ``bf.load(path)`` and return the saved ``bf`` object.

    Security: loading uses cloudpickle, which runs arbitrary code from the
    file. Only load files you created or trust.
    """

    def __new__(cls, path=None):
        """Return ``bf.load(path)`` when a path is given, else a loader instance."""
        if path is not None:
            from BayesForge.Main.main import bf
            return bf.load(path=path)
        return super().__new__(cls)

    def __init__(self, path=None):
        """Initialize the dataset loader with the data directory path.

        ``path`` is accepted and ignored here: a non-None path is handled by
        ``__new__``, which never builds a loader instance in that case.
        """
        self.data_dir = Path(__file__).parent

    def __call__(self, path):
        """Load a saved ``bf`` session object from disk (alias for ``bf.load(path)``).

        ``path`` is required: name the file to load, e.g. ``m.load('bf_model.pkl')``.

        Security: loading uses cloudpickle, which runs arbitrary code from the
        file. Only load files you created or trust.
        """
        if path is None:
            raise TypeError("load()(path): pass the file to load, e.g. m.load('bf_model.pkl')")
        from BayesForge.Main.main import bf
        return bf.load(path=path)

    def _load_csv(self, filename, description="", frame=True, only_path=False, index_col=None, sep=","):
        """
        Load a CSV file and return a DataFrame (default) or a Bunch object.

        Parameters:
        -----------
        filename : str
            Name of the CSV file to load
        description : str
            Description of the dataset
        frame : bool
            If True, return the data as a pandas DataFrame. Else, return a dictionary with the data and feature names
        only_path : bool
            If True, return only the file path as a str (the file is not read)
        index_col : int, str, sequence of str or None
            Column(s) to use as the row index, passed to pandas.read_csv
            (the resulting index name is dropped)
        sep : str
            Explicit field separator for this file ("," or ";"), selected per
            CSV instead of pandas' automatic delimiter sniffing (python engine)

        Returns:
        --------
        df : pandas.DataFrame (frame=True, default), bunch : Bunch
            (frame=False), or path : str (only_path=True)
        """
        filepath = self.data_dir / filename
        if only_path:
            # return as string
            return str(filepath)
        # Explicit separator per file (m-10): the C engine is faster than the
        # python engine and cannot mis-sniff an ambiguous first line, unlike
        # automatic delimiter detection with the python engine.
        df = pd.read_csv(filepath, sep=sep, index_col=index_col)
        if index_col is not None:
            df.index.name = None

        if frame:
            return df
        else:
            return _bunch(df, filename, description)

    def howell1(self, frame=True, only_path=False):
        """
        Load the Howell1 dataset.

        Demographic data including height, weight, age, and sex.

        Returns:
        --------
        bunch : Bunch
            Dataset with 544 samples and 4 features
            Features: height, weight, age, male
        """
        description = """
        Howell1 Dataset
        ===============

        Demographic data from the Dobe area !Kung San people.

        Features:
        - height: Height in cm
        - weight: Weight in kg
        - age: Age in years
        - male: Sex (1=male, 0=female)

        Samples: 544
        """
        return self._load_csv("Howell1.csv", description, frame=frame, only_path=only_path, sep=";")

    def milk(self, frame=True, only_path=False):
        """
        Load the milk dataset.

        Primate milk composition data.

        Missing values: ``neocortex.perc`` is NA in 12 of 29 rows; ``clade``
        and ``species`` are text columns (``frame=False`` then yields
        dtype-object data, and modelling ``neocortex.perc`` needs dropna or
        imputation first).

        Returns:
        --------
        bunch : Bunch
            Dataset with milk composition data for various primate species
            Features: clade, species, kcal.per.g, perc.fat, perc.protein,
                     perc.lactose, mass, neocortex.perc
        """
        description = """
        Primate Milk Composition Dataset
        =================================

        Milk composition data for various primate species.

        Features:
        - clade: Taxonomic clade
        - species: Species name
        - kcal.per.g: Kilocalories per gram of milk
        - perc.fat: Percent fat
        - perc.protein: Percent protein
        - perc.lactose: Percent lactose
        - mass: Body mass (kg)
        - neocortex.perc: Percent of brain that is neocortex

        Missing values: neocortex.perc is NA in 12 of 29 rows; clade and
        species are text columns. Drop or impute before modelling.
        """
        return self._load_csv("milk.csv", description, frame=frame, only_path=only_path, sep=";")

    def kline(self, frame=True, only_path=False):
        """
        Load the Kline dataset.

        Cultural evolution data from Oceanic societies.

        Returns:
        --------
        bunch : Bunch
            Dataset with cultural and technological data
        """
        description = """
        Kline Dataset
        =============

        Data on tool complexity and population size in Oceanic societies.

        Used for studying cultural evolution and the relationship between
        population size and technological complexity.
        """
        return self._load_csv("Kline.csv", description, frame=frame, only_path=only_path, sep=";")

    def kline2(self, frame=True, only_path=False):
        """
        Load the Kline2 dataset.

        Extended version of the Kline cultural evolution dataset.

        Returns:
        --------
        bunch : Bunch
            Dataset with cultural and technological data
        """
        description = """
        Kline2 Dataset
        ==============

        Extended data on tool complexity and population size in Oceanic societies.

        Used for studying cultural evolution and the relationship between
        population size and technological complexity.
        """
        return self._load_csv("Kline2.csv", description, frame=frame, only_path=only_path, sep=";")

    def ucbadmit(self, frame=True, only_path=False):
        """
        Load the UC Berkeley admissions dataset.

        Famous dataset illustrating Simpson's paradox.

        Returns:
        --------
        bunch : Bunch
            Dataset with UC Berkeley graduate admissions data
        """
        description = """
        UC Berkeley Admissions Dataset
        ===============================

        Graduate school admissions data from UC Berkeley, 1973.

        This dataset is famous for illustrating Simpson's paradox:
        when examining admission rates by department, there was no
        evidence of discrimination, but aggregated data suggested bias.

        Features include department, gender, and admission outcomes.
        """
        return self._load_csv("UCBadmit.csv", description, frame=frame, only_path=only_path, index_col="case", sep=";")

    def chimpanzees(self, frame=True, only_path=False):
        """
        Load the chimpanzees dataset.

        Chimpanzee behavioral experiment data.

        Missing values: the ``recipient`` column is NA in 252 of 504 rows
        (the other half holds the recipient index); drop or encode it before
        modelling.

        Returns:
        --------
        bunch : Bunch
            Dataset with chimpanzee behavioral data
        """
        description = """
        Chimpanzees Dataset
        ===================

        Data from experiments on chimpanzee behavior and social learning.

        Contains behavioral observations and experimental outcomes
        from chimpanzee studies.

        Missing values: the recipient column is NA in 252 of 504 rows.
        """
        return self._load_csv("chimpanzees.csv", description, frame=frame, only_path=only_path, sep=";")

    def mastectomy(self, frame=True, only_path=False):
        """
        Load the mastectomy dataset.

        Survival data for mastectomy patients.

        Returns:
        --------
        bunch : Bunch
            Dataset with patient survival data
        """
        description = """
        Mastectomy Dataset
        ==================

        Survival data for breast cancer patients who underwent mastectomy.

        Contains survival times and related clinical information.
        """
        return self._load_csv("mastectomy.csv", description, frame=frame, only_path=only_path, sep=",")

    def reedfrogs(self, frame=True, only_path=False):
        """
        Load the reedfrogs dataset.

        Reed frog predation and survival experiment data.

        Returns:
        --------
        bunch : Bunch
            Dataset with experimental data on reed frog survival
        """
        description = """
        Reed Frogs Dataset
        ==================

        Experimental data on reed frog tadpole survival under various conditions.

        Contains data from experiments studying predation and density effects
        on tadpole survival rates.
        """
        return self._load_csv("reedfrogs.csv", description, frame=frame, only_path=only_path, sep=";")

    def tulips(self, frame=True, only_path=False):
        """
        Load the tulips dataset.

        Tulip growth experiment data.

        Returns:
        --------
        bunch : Bunch
            Dataset with experimental data on tulip growth
        """
        description = """
        Tulips Dataset
        ==============

        Experimental data on tulip growth under different conditions.

        Contains measurements of tulip growth with different levels of
        water and shade treatments.
        """
        return self._load_csv("tulips.csv", description, frame=frame, only_path=only_path, sep=";")

    def islands_dist_matrix(self, frame=True, only_path=False):
        """
        Load the islands distance matrix dataset.

        Distance matrix between islands.

        Returns:
        --------
        bunch : Bunch
            Dataset with pairwise distances between islands
        """
        description = """
        Islands Distance Matrix
        =======================

        Matrix of distances between various islands.

        Used for spatial analysis and modeling of geographic relationships.
        """
        filepath = self.data_dir / "islandsDistMatrix.csv"
        if only_path:
            return str(filepath)
        df = pd.read_csv(filepath, sep=",", index_col=0)
        if frame:
            return df

        bunch = _bunch(df, "islandsDistMatrix.csv", description,
                       data=np.asarray(df.values, dtype=float))
        bunch["row_names"] = df.index.tolist()
        return bunch

    def sim_gamma_poisson(self, frame=True, only_path=False):
        """
        Load the simulated Gamma-Poisson dataset.

        Simulated data with Gamma-Poisson distribution.

        Returns:
        --------
        bunch : Bunch
            Simulated dataset for testing Gamma-Poisson models
        """
        description = """
        Simulated Gamma-Poisson Dataset
        ================================

        Simulated data generated from a Gamma-Poisson distribution.

        Used for testing and validating Gamma-Poisson statistical models.
        """
        return self._load_csv("Sim dat Gamma poisson.csv", description, frame=frame, only_path=only_path, sep=",")

    def sim_multinomial(self, frame=True, only_path=False):
        """
        Load the simulated multinomial dataset.

        Simulated multinomial data.

        Returns:
        --------
        bunch : Bunch
            Simulated dataset for testing multinomial models
        """
        description = """
        Simulated Multinomial Dataset
        ==============================

        Simulated data from a multinomial distribution.

        Used for testing and validating multinomial statistical models.
        """
        return self._load_csv("Sim data multinomial.csv", description, frame=frame, only_path=only_path, sep=",")

    def sim_multivariate_normal(self, frame=True, only_path=False):
        """
        Load the simulated multivariate normal dataset.

        Simulated multivariate normal data.

        Returns:
        --------
        bunch : Bunch
            Simulated dataset for testing multivariate normal models
        """
        description = """
        Simulated Multivariate Normal Dataset
        ======================================

        Simulated data from a multivariate normal distribution.

        Used for testing and validating multivariate normal statistical models.
        """
        filepath = self.data_dir / "Sim data multivariatenormal.csv"
        if only_path:
            return str(filepath)

        # Load the CSV using pandas to easily handle row and column names
        df = pd.read_csv(filepath, sep=",", index_col=0)

        if frame:
            return df

        data = df.values
        feature_names = df.columns.tolist()
        row_names = df.index.tolist()

        return Bunch(
            data=data,
            feature_names=feature_names,
            row_names=row_names,  # Added row_names
            DESCR=description,
            frame=df,  # frame is now the loaded DataFrame with index
            filename="Sim data multivariatenormal.csv"
        )

    def WaffleDivorce(self, frame=True, only_path=False):
        """
        Load the WaffleDivorce dataset.

        Returns:
        --------
        bunch : Bunch
            Observational dataset with 50 rows (one per US state) and 13 columns
        """
        description = """
        Waffle Divorce Dataset
        =======================

        Observational data, one row per US state (50 rows): divorce and
        marriage rates, waffle house counts and demographic covariates
        measured across states.

        Columns: Location, Loc, Population, MedianAgeMarriage, Marriage,
        Marriage SE, Divorce, Divorce SE, WaffleHouses, South, Slaves1860,
        Population1860, PropSlaves1860.
        """
        return self._load_csv("WaffleDivorce.csv", description, frame=frame, only_path=only_path, sep=";")

    def elephants(self, frame=True, only_path=False):
        """
        Load the elephants dataset.

        Returns:
        --------
        bunch : Bunch
            Dataset with 41 records and 2 columns: AGE, MATINGS
        """
        description = """
        Elephants Dataset
        =================

        Mating records for 41 elephants: 2 integer columns.

        Columns:
        - AGE: age in years
        - MATINGS: number of matings recorded
        """
        return self._load_csv("elephants.csv", description, frame=frame, only_path=only_path, sep=";")

    def iris(self, frame=True, only_path=False):
        """
        Load the iris dataset.

        The four morphological features are standardised (zero mean, unit
        variance). Columns: ``sepal_length``, ``sepal_width``,
        ``petal_length``, ``petal_width`` (scaled), ``target`` (int 0-2),
        ``species`` (name). The file carries no index column.

        Returns:
        --------
        bunch : Bunch
            Dataset with iris data
        """
        description = """
        Iris Dataset
        ============

        150 samples, 3 species, 4 standardised morphological features
        (sepal length, sepal width, petal length, petal width).
        """
        return self._load_csv("iris.csv", description, frame=frame, only_path=only_path, sep=",")

    def NBDA_events(self, frame=True, only_path=False):
        """Load NBDA event data (STbayes::event_data format)."""
        description = """
        NBDA Event Data
        ===============

        Simulated acquisition times for 50 individuals across 1 trial.
        Columns: id, time, t_end, trial
        Source: STbayes example data (Chimento 2025).
        """
        return self._load_csv("NBDA_event_data.csv", description, frame=frame, only_path=only_path, sep=",")

    def NBDA_network(self, frame=True, only_path=False):
        """Load NBDA network edge list (STbayes::edge_list format)."""
        description = """
        NBDA Network Edge List
        ======================

        Association strengths for 100 dyads across 1 trial.
        Columns: focal, other, trial, assoc
        Source: STbayes example data (Chimento 2025).
        """
        return self._load_csv("NBDA_edge_list.csv", description, frame=frame, only_path=only_path, sep=",")


    # Phylogenetic Datasets -------------------------------------------------------------------

    def phylo_simple(self, frame=True, only_path=False):
        """Load simulated data for the Simple Phylogenetic Model (Gaussian)."""
        description = "Simulated dataset with phenotype 'y', covariate 'x', and species identity 'phylo'."
        return self._load_csv("phylo_data_simple.csv", description, frame=frame, only_path=only_path, sep=",")

    def phylo_L_simple(self, frame=True, only_path=False):
        """Load the Cholesky factor of the kinship matrix for the Simple Phylogenetic Model."""
        filepath = self.data_dir / "phylo_L_simple.csv"
        if only_path: return str(filepath)
        df = pd.read_csv(filepath, sep=",")
        if frame:
            return df
        bunch = _bunch(df, "phylo_L_simple.csv",
                       "Load the Cholesky factor of the kinship matrix for the Simple Phylogenetic Model.",
                       data=np.asarray(df.values, dtype=float))
        bunch["columns"] = df.columns.tolist()
        return bunch

    def phylo_poisson(self, frame=True, only_path=False):
        """Load simulated data for the Phylogenetic Poisson Model."""
        description = "Simulated count dataset with phenotype 'y', covariate 'x', and species 'phylo'."
        return self._load_csv("phylo_data_poisson.csv", description, frame=frame, only_path=only_path, sep=",")

    def phylo_L_poisson(self, frame=True, only_path=False):
        """Load the Cholesky factor of the kinship matrix for the Poisson model.

        The four phylogenetic models share the same tree, so this loader reads
        the single shared file ``phylo_L_simple.csv``.
        """
        filepath = self.data_dir / _PHYLO_L_SHARED
        if only_path: return str(filepath)
        df = pd.read_csv(filepath, sep=",")
        if frame:
            return df
        bunch = _bunch(df, _PHYLO_L_SHARED,
                       "Cholesky factor of the kinship matrix for the Poisson model "
                       "(shared file: the four phylogenetic models use the same tree).",
                       data=np.asarray(df.values, dtype=float))
        bunch["columns"] = df.columns.tolist()
        return bunch

    def phylo_repeated(self, frame=True, only_path=False):
        """Load simulated data for the Repeated Measurements Phylogenetic Model."""
        description = "Dataset with multiple observations per species for variance partitioning."
        return self._load_csv("phylo_data_repeated.csv", description, frame=frame, only_path=only_path, sep=",")

    def phylo_L_repeated(self, frame=True, only_path=False):
        """Load the Cholesky factor of the kinship matrix for the Repeated measurements model.

        The four phylogenetic models share the same tree, so this loader reads
        the single shared file ``phylo_L_simple.csv``.
        """
        filepath = self.data_dir / _PHYLO_L_SHARED
        if only_path: return str(filepath)
        df = pd.read_csv(filepath, sep=",")
        if frame:
            return df
        bunch = _bunch(df, _PHYLO_L_SHARED,
                       "Cholesky factor of the kinship matrix for the Repeated measurements "
                       "model (shared file: the four phylogenetic models use the same tree).",
                       data=np.asarray(df.values, dtype=float))
        bunch["columns"] = df.columns.tolist()
        return bunch

    def phylo_meta(self, frame=True, only_path=False):
        """Load data for a Phylogenetic Meta-analysis."""
        description = "Dataset with Fisher's Z effect sizes ('y'), standard errors ('se'), and species."
        return self._load_csv("phylo_data_meta.csv", description, frame=frame, only_path=only_path, sep=",")

    def phylo_L_meta(self, frame=True, only_path=False):
        """Load the Cholesky factor of the kinship matrix for Meta-analysis.

        The four phylogenetic models share the same tree, so this loader reads
        the single shared file ``phylo_L_simple.csv``.
        """
        filepath = self.data_dir / _PHYLO_L_SHARED
        if only_path: return str(filepath)
        df = pd.read_csv(filepath, sep=",")
        if frame:
            return df
        bunch = _bunch(df, _PHYLO_L_SHARED,
                       "Cholesky factor of the kinship matrix for Meta-analysis "
                       "(shared file: the four phylogenetic models use the same tree).",
                       data=np.asarray(df.values, dtype=float))
        bunch["columns"] = df.columns.tolist()
        return bunch

    def phylo_slopes(self, frame=True, only_path=False):
        """Load data for the Phylogenetic Varying Slopes Model."""
        description = "Dataset for modeling evolutionary shifts in trait correlations."
        return self._load_csv("phylo_data_slopes.csv", description, frame=frame, only_path=only_path, sep=",")

    def phylo_L_slopes(self, frame=True, only_path=False):
        """Load the Cholesky factor of the kinship matrix for the Varying Slopes model."""
        filepath = self.data_dir / "phylo_L_slopes.csv"
        if only_path: return str(filepath)
        df = pd.read_csv(filepath, sep=",")
        if frame:
            return df
        bunch = _bunch(df, "phylo_L_slopes.csv",
                       "Load the Cholesky factor of the kinship matrix for the Varying Slopes model.",
                       data=np.asarray(df.values, dtype=float))
        bunch["columns"] = df.columns.tolist()
        return bunch

    # Panama EWA -----------------------------------------------------------------------

    def sim_nested_effects(self, frame=True, only_path=False):
        """Load simulated nested varying-effects dataset.

        Simulated data with observations nested in groups, groups nested in regions.
        Generated with known true parameters for model validation.

        DGP parameters
        --------------
        N_regions=5, N_groups_per_region=4, N_obs_per_group=20 (400 total obs)
        mu_a_region=5.0,  mu_b_region=-1.0
        sigma_a_region=1.0, sigma_b_region=0.5, rho_region=-0.5
        sigma_a_group=0.5,  sigma_b_group=0.2,  rho_group=0.3
        sigma_obs=0.5, seed=42

        Columns
        -------
        y         : float — outcome
        x         : float — continuous predictor (standardised)
        group_id  : int   — group index 0..19
        region_id : int   — region index 0..4
        """
        return self._load_csv("sim_nested_effects.csv",
                              description="Simulated nested varying-effects dataset.",
                              frame=frame, only_path=only_path, sep=",")

    def panama_ewa(self, frame=True, only_path=False):
        """Load the Panama EWA social learning dataset (Barrett et al.).

        Capuchin monkey foraging data with 7 techniques across 14 days.
        Columns include technique choice, payoffs, social cues (payoff bias,
        kin, prestige, cohort, age-of-acquisition), bout index, individual id,
        and centered age.

        Missing values: the ``RCI`` column is NA in 792 of 1441 rows.
        """
        description = ("Panama capuchin EWA social learning dataset (Barrett et al.). "
                       "Missing values: the RCI column is NA in 792 of 1441 rows.")
        return self._load_csv("panama_ewa.csv", description, frame=frame, only_path=only_path, sep=",")



