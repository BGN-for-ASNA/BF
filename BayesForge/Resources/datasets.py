"""
Dataset loader utilities for BF package.

This module provides sklearn-style dataset loaders for datasets stored in 
the BF/resources/data directory. Each loader returns a Bunch object containing
the data, feature names, and dataset description.

Example:
    >>> from BayesForge.resources.data import datasets
    >>> howell = datasets.load_howell1()
    >>> print(howell.data.shape)
    >>> print(howell.feature_names)
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import jax.numpy as jnp

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


class load:
    """
    Dataset loader class for BF datasets.
    
    This class provides methods to load various datasets stored in CSV format.
    Each method returns a Bunch object with the following attributes:
    - data: numpy array of features
    - feature_names: list of feature column names
    - DESCR: description of the dataset
    - frame: pandas DataFrame (optional, full data including all columns)
    """
    
    def __init__(self):
        """Initialize the dataset loader with the data directory path."""
        self.data_dir = Path(__file__).parent

    def __call__(self, path=None):
        """Load a saved ``bf`` session object from disk (alias for ``bf.load(path)``)."""
        from BayesForge.Main.main import bf
        return bf.load(path=path)
    
    def _load_csv(self, filename, description="", frame=True, only_path=False):
        """
        Load a CSV file and return a Bunch object.
        
        Parameters:
        -----------
        filename : str
            Name of the CSV file to load
        description : str
            Description of the dataset
        frame : bool
            If True, return the data as a pandas DataFrame. Else, return a dictionary with the data and feature names
        only_path : bool
            If True, return only the file path
            
        Returns:
        --------
        bunch : Bunch
            Dataset as a Bunch object
        """
        filepath = self.data_dir / filename
        if only_path:
            # return as string
            return str(filepath)
        # Load the data using pandas with auto-delimiter detection
        # sep=None tells pandas to auto-detect the delimiter
        # engine='python' is required when using sep=None
        df = pd.read_csv(filepath, sep=None, engine='python')
        
        if frame:
            return df
        else:
            # Get feature names
            feature_names = df.columns.tolist()
            
            # Convert to numpy array
            data = df.values
        
            return Bunch(
                data=data,
                feature_names=feature_names,
                DESCR=description,
                frame=df,
                filename=filename
            )

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
        return self._load_csv("Howell1.csv", description, frame=frame, only_path=only_path)
    
    def milk(self, frame=True, only_path=False):
        """
        Load the milk dataset.
        
        Primate milk composition data.
        
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
        """
        return self._load_csv("milk.csv", description, frame=frame, only_path=only_path)
    
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
        return self._load_csv("Kline.csv", description, frame=frame, only_path=only_path)
    
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
        return self._load_csv("Kline2.csv", description, frame=frame, only_path=only_path)
    
    def trolley(self, frame=True, only_path=False):
        """
        Load the Trolley dataset.
        
        Trolley problem moral judgment experiment data.
        
        Returns:
        --------
        bunch : Bunch
            Dataset with moral judgment responses
        """
        description = """
        Trolley Problem Dataset
        =======================
        
        Data from experiments on moral judgment using trolley problem scenarios.
        
        Contains responses to various moral dilemmas involving the trolley problem,
        a thought experiment in ethics.
        
        Samples: Large dataset with multiple experimental conditions
        """
        return self._load_csv("Trolley.csv", description, frame=frame, only_path=only_path)
    
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
        return self._load_csv("UCBadmit.csv", description, frame=frame, only_path=only_path)
    
    def chimpanzees(self, frame=True, only_path=False):
        """
        Load the chimpanzees dataset.
        
        Chimpanzee behavioral experiment data.
        
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
        """
        return self._load_csv("chimpanzees.csv", description, frame=frame, only_path=only_path)
    
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
        return self._load_csv("mastectomy.csv", description, frame=frame, only_path=only_path)
    
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
        return self._load_csv("reedfrogs.csv", description, frame=frame, only_path=only_path)
    
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
        return self._load_csv("tulips.csv", description, frame=frame, only_path=only_path)
    
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
        df = pd.read_csv(str(self.data_dir / "islandsDistMatrix.csv"), index_col=0)
        if only_path:
            return str(self.data_dir / "islandsDistMatrix.csv")
        if frame:
            return df
        
        return Bunch(
            data=jnp.array(df.values),
            feature_names=df.columns.tolist(),
            row_names=df.index.tolist(),
            DESCR=description,            
            filename="islandsDistMatrix.csv"
        )
    
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
        return self._load_csv("Sim dat Gamma poisson.csv", description, frame=frame, only_path=only_path)
    
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
        return self._load_csv("Sim data multinomial.csv", description, frame=frame, only_path=only_path)
    
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
        self.data_dir = Path(__file__).parent
        filepath = self.data_dir / "Sim data multivariatenormal.csv"
        if only_path:
            return filepath
        
        # Load the CSV using pandas to easily handle row and column names
        df = pd.read_csv(filepath, index_col=0)
        
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
            Dataset with waffle divorce data
        """
        description = """
        Waffle Divorce Dataset
        =======================
        
        Data from experiments on waffle divorce.
        
        Contains data from experiments studying the effect of waffles on divorce rates.
        """
        if only_path:
            return self.data_dir / "WaffleDivorce.csv"
        if frame:
            return pd.read_csv(self.data_dir / "WaffleDivorce.csv", index_col=0)
        
        return self._load_csv("WaffleDivorce.csv", description, frame=frame, only_path=only_path)

    def elephants(self, frame=True, only_path=False):
        """
        Load the elephants dataset.
        
        Returns:
        --------
        bunch : Bunch
            Dataset with elephant data
        """
        description = """
        Elephants Dataset
        =================
        
        Data from experiments on elephant aggression.
        
        Contains data from experiments studying the effect of elephant aggression on survival.
        """
        if only_path:
            return self.data_dir / "elephants.csv"
        if frame:
            return pd.read_csv(self.data_dir / "elephants.csv", index_col=0)
        
        return self._load_csv("elephants.csv", description, frame=frame, only_path=only_path)

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
        return self._load_csv("iris.csv", description, frame=frame, only_path=only_path)

    def NBDA_events(self, frame=True, only_path=False):
        """Load NBDA event data (STbayes::event_data format)."""
        description = """
        NBDA Event Data
        ===============

        Simulated acquisition times for 50 individuals across 1 trial.
        Columns: id, time, t_end, trial
        Source: STbayes example data (Chimento 2025).
        """
        return self._load_csv("NBDA_event_data.csv", description, frame=frame, only_path=only_path)

    def NBDA_network(self, frame=True, only_path=False):
        """Load NBDA network edge list (STbayes::edge_list format)."""
        description = """
        NBDA Network Edge List
        ======================

        Association strengths for 100 dyads across 1 trial.
        Columns: focal, other, trial, assoc
        Source: STbayes example data (Chimento 2025).
        """
        return self._load_csv("NBDA_edge_list.csv", description, frame=frame, only_path=only_path)


    # Phylogenetic Datasets -------------------------------------------------------------------

    def phylo_simple(self, frame=True, only_path=False):
        """Load simulated data for the Simple Phylogenetic Model (Gaussian)."""
        description = "Simulated dataset with phenotype 'y', covariate 'x', and species identity 'phylo'."
        return self._load_csv("phylo_data_simple.csv", description, frame=frame, only_path=only_path)

    def phylo_L_simple(self, frame=True, only_path=False):
        """Load the Cholesky factor of the kinship matrix for the Simple Phylogenetic Model."""
        filepath = self.data_dir / "phylo_L_simple.csv"
        if only_path: return str(filepath)
        df = pd.read_csv(filepath)
        return df if frame else Bunch(data=jnp.array(df.values), columns=df.columns.tolist())

    def phylo_poisson(self, frame=True, only_path=False):
        """Load simulated data for the Phylogenetic Poisson Model."""
        description = "Simulated count dataset with phenotype 'y', covariate 'x', and species 'phylo'."
        return self._load_csv("phylo_data_poisson.csv", description, frame=frame, only_path=only_path)

    def phylo_L_poisson(self, frame=True, only_path=False):
        """Load the Cholesky factor of the kinship matrix for the Poisson model."""
        filepath = self.data_dir / "phylo_L_poisson.csv"
        if only_path: return str(filepath)
        df = pd.read_csv(filepath)
        return df if frame else Bunch(data=jnp.array(df.values), columns=df.columns.tolist())

    def phylo_repeated(self, frame=True, only_path=False):
        """Load simulated data for the Repeated Measurements Phylogenetic Model."""
        description = "Dataset with multiple observations per species for variance partitioning."
        return self._load_csv("phylo_data_repeated.csv", description, frame=frame, only_path=only_path)

    def phylo_L_repeated(self, frame=True, only_path=False):
        """Load the Cholesky factor of the kinship matrix for the Repeated measurements model."""
        filepath = self.data_dir / "phylo_L_repeated.csv"
        if only_path: return str(filepath)
        df = pd.read_csv(filepath)
        return df if frame else Bunch(data=jnp.array(df.values), columns=df.columns.tolist())

    def phylo_meta(self, frame=True, only_path=False):
        """Load data for a Phylogenetic Meta-analysis."""
        description = "Dataset with Fisher's Z effect sizes ('y'), standard errors ('se'), and species."
        return self._load_csv("phylo_data_meta.csv", description, frame=frame, only_path=only_path)

    def phylo_L_meta(self, frame=True, only_path=False):
        """Load the Cholesky factor of the kinship matrix for Meta-analysis."""
        filepath = self.data_dir / "phylo_L_meta.csv"
        if only_path: return str(filepath)
        df = pd.read_csv(filepath)
        return df if frame else Bunch(data=jnp.array(df.values), columns=df.columns.tolist())

    def phylo_slopes(self, frame=True, only_path=False):
        """Load data for the Phylogenetic Varying Slopes Model."""
        description = "Dataset for modeling evolutionary shifts in trait correlations."
        return self._load_csv("phylo_data_slopes.csv", description, frame=frame, only_path=only_path)

    def phylo_L_slopes(self, frame=True, only_path=False):
        """Load the Cholesky factor of the kinship matrix for the Varying Slopes model."""
        filepath = self.data_dir / "phylo_L_slopes.csv"
        if only_path: return str(filepath)
        df = pd.read_csv(filepath)
        return df if frame else Bunch(data=jnp.array(df.values), columns=df.columns.tolist())

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
                              frame=frame, only_path=only_path)

    def panama_ewa(self, frame=True, only_path=False):
        """Load the Panama EWA social learning dataset (Barrett et al.).

        Capuchin monkey foraging data with 7 techniques across 14 days.
        Columns include technique choice, payoffs, social cues (payoff bias,
        kin, prestige, cohort, age-of-acquisition), bout index, individual id,
        and centered age.
        """
        description = "Panama capuchin EWA social learning dataset (Barrett et al.)."
        return self._load_csv("panama_ewa.csv", description, frame=frame, only_path=only_path)



