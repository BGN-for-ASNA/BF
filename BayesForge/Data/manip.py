import os
import pandas as pd
import jax.numpy as jnp
import numpy as np
import inspect
import jax

class manip():
    """
    The manip class serves as a comprehensive data preprocessing and manipulation tool, specifically designed to prepare data for use with JAX-based models. It encapsulates common data transformation tasks, starting from loading data with pandas, applying various encodings and scaling, and finally converting the data into JAX-compatible arrays.
    """
    def __init__(self):
        """## Initialize the manip class with data modification tracking and dtype mapping"""
        self.data_modification = {}
        self.pandas_to_jax_dtype_map = {
            'int64': jnp.int64,
            'int32': jnp.int32,
            'int16': jnp.int32,
            'float64': jnp.float64,
            'float32': jnp.float32,
            'float16': jnp.float16,
        }
        self.pandas_to_jax_dtype_map_force32 = {
            'int64': jnp.int32,
            'int32': jnp.int32,
            'int16': jnp.int32,
            'float64': jnp.float32,
            'float32': jnp.float32,
            'float16': jnp.float16,
        }

    # Import data----------------------------
    def data(self, path, **kwargs):
        """## Load tabular data.

        ### Args:
            - *path* (str, path-like or data frame): Path to a CSV file, or an
              in-memory table: a pandas DataFrame, or anything pandas.DataFrame()
              accepts (e.g. a dict of columns). R data.frames passed through
              reticulate arrive as pandas DataFrames, so R users can pass their
              data directly without writing a file.
            - * **kwargs*: Additional arguments for pd.read_csv (CSV paths only)

        ### Returns:
            pd.DataFrame: Loaded dataframe
        """
        if isinstance(path, (str, os.PathLike)):
            self.data_original_path = path
            self.data_args = kwargs
            self.df = pd.read_csv(path, **kwargs)
        else:
            if kwargs:
                raise TypeError(
                    f"read_csv arguments {sorted(kwargs)} only apply when a file path is given."
                )
            self.data_original_path = None
            self.data_args = {}
            # Copy so later manipulations (scale, OHE, ...) never modify the caller's table.
            self.df = pd.DataFrame(path).copy()
        self.data_modification = {}
        return self.df

    def OHE(self, cols = 'all'):
        """## Perform one-hot encoding on specified columns
        
        ### Args:
            - *cols* (str or list): Columns to encode. Use 'all' for all object-type columns
            
        ### Returns:
            - *pd.DataFrame*: DataFrame with encoded columns
        """
        if cols == 'all':
            colCat = list(self.df.select_dtypes(['object']).columns)    
            OHE = pd.get_dummies(self.df, columns=colCat, dtype=int)
        else:
            # a single name, or any sequence of names (list, tuple, Julia Vector)
            cols = [cols] if isinstance(cols, str) else [str(c) for c in cols]
            OHE = pd.get_dummies(self.df, columns=cols, dtype=int)

        OHE.columns = OHE.columns.str.replace('.', '_')
        OHE.columns = OHE.columns.str.replace(' ', '_')


        self.df = pd.concat([self.df , OHE], axis=1)
        self.data_modification['OHE'] = cols
        return OHE

    def index(self, cols = 'all'):
        """## Create index encoding for categorical columns
        
        ### Args:
            - *cols* (str or list): Columns to encode. Use 'all' for all text
              (object) and categorical columns, e.g. R factors.

        Codes are 0-based and follow the category order: the level order of a
        pandas Categorical or R factor, alphabetical for text columns.
        ``m.index_map[col]`` maps each code to its level.

        ### Returns:
            - *pd.DataFrame*: DataFrame with encoded columns
        """
        self.index_map = {}
        if cols == 'all':
            # 'category' too: an R factor arrives as a pandas Categorical, whose
            # level order (set in R) is kept by .astype("category") below.
            colCat = list(self.df.select_dtypes(['object', 'category']).columns)
            for a in range(len(colCat)):                
                self.df["index_"+ colCat[a]] =  self.df.loc[:,colCat[a]].astype("category").cat.codes
                self.df["index_"+ colCat[a]] = self.df["index_"+ colCat[a]].astype(np.int64)
                self.index_map[colCat[a]] = dict(enumerate(self.df[colCat[a]].astype("category").cat.categories ) )
        else:
            # a single name, or any sequence of names (list, tuple, Julia Vector)
            cols = [cols] if isinstance(cols, str) else [str(c) for c in cols]
            for a in range(len(cols)):
                self.df["index_"+ cols[a]] =  self.df.loc[:,cols[a]].astype("category").cat.codes
                self.df["index_"+ cols[a]] = self.df["index_"+ cols[a]].astype(np.int64)

                self.index_map[cols[a]] = dict(enumerate(self.df[cols[a]].astype("category").cat.categories ) )

        self.df.columns = self.df.columns.str.replace('.', '_')
        self.df.columns = self.df.columns.str.replace(' ', '_')

        self.data_modification['index'] = cols # store info of indexed columns
        
        return self.df
    
    @jax.jit
    def scale_var(self, x):
        """## JAX-jitted function to scale/standardize a single variable"""
        return (x - x.mean()) / x.std()

    def scale(self, data = 'all'):
        """## Standardize specified columns.
        
        ### Args:
            - *data* (str or list): Columns to standardize. Use 'all' for all columns
            
        ### Returns:
            - *pd.DataFrame*: Standardized dataframe
        """        
        if type(data) == str:
            return self.scale_var(data)
        else:
            if data == 'all':
                for col in self.df.columns:     
                    self.df[col] = self.df[col].astype(float)         
                    self.df.loc[:, col] = (self.df.loc[:,col] - self.df.loc[:,col].mean())/self.df.loc[:,col].sd()

            else:
                for a in range(len(data)):
                    self.df[data[a]] = self.df[data[a]].astype('float64') 
                    self.df.loc[:, data[a]] = (self.df.loc[:, data[a]] - self.df.loc[:, data[a]].mean()) / self.df.loc[:, data[a]].std()


            self.data_modification['scale'] = data # store info of scaled columns

        return self.df

    def z_score(self,X, axis=0):
        """
        Performs Z-score scaling (standardization) on a JAX array.
        (X - mean) / std_dev

        Args:
            X (jnp.ndarray): The input JAX array (e.g., shape (n_samples, n_features)).
            axis (int): The axis along which to calculate mean and std. 
                        axis=0 scales features (columns).

        Returns:
            jnp.ndarray: The scaled array.
        """
        # Calculate mean and std, preserving dimensions for correct broadcasting
        mean_arr = jnp.mean(X, axis=axis, keepdims=True)
        std_arr = jnp.std(X, axis=axis, keepdims=True)

        # Handle division by zero (for constant features) by replacing 0 with 1
        # This prevents NaN, as (X - mean) will be 0 when std_arr is 0.
        std_arr = jnp.where(std_arr == 0, 1.0, std_arr)

        X_scaled = (X - mean_arr) / std_arr
        return X_scaled
    
    def to_float(self, cols = 'all', type = 'float32'):
        """## Convert specified columns to float type
        
        ### Args:
            - *cols* (str or list): Columns to convert. Use 'all' for all columns
            - *type* (str): Float type to convert to (default: float32)
            
        ### Returns:
            - *pd.DataFrame*: Converted dataframe
        """        
        if cols == 'all':
            for col in self.df.columns:                
                self.df.loc[:, col] = self.df.iloc[:,col].astype(type)

        else:
            for a in range(len(cols)):
                self.df.loc[:, cols[a]] = self.df.loc[:,cols[a]].astype(type)


        self.data_modification['float'] = cols # store info of scaled columns
        
        return self.df

    def to_int(self, cols = 'all', type = 'int32'):
        """## Convert specified columns to integer type
        
        ### Args:
            - *cols* (str or list): Columns to convert. Use 'all' for all columns
            - *type* (str): Integer type to convert to (default: int32)
            
        ### Returns:
            - *pd.DataFrame*: Converted dataframe
        """
        if cols == 'all':
            for col in self.df.columns:                
                self.df.iloc[:, cols] = self.df.iloc[:,col].astype(type)

        else:
            for a in range(len(cols)):
                self.df.loc[:, cols[a]] = self.df.iloc[:,cols[a]].astype(type)


        self.data_modification['int'] = cols # store info of scaled columns

    def pd_to_jax(self, model, bit = None):
        """## Convert pandas dataframe to JAX compatible format for a model.
        
        ### Args:
            - *model*: JAX model to prepare data for
            - *bit* (str): Bit precision for numbers (default: 32)
            
        ### Returns:
            - *dict*: JAX formatted dictionary
        """        
        params = inspect.signature(model).parameters
        args_without_defaults = []
        args_with_defaults = {}
        for param_name, param in params.items():
            if param.default == inspect.Parameter.empty:
                args_without_defaults.append(param_name)
            else:
                args_with_defaults[param_name] = (param.default, type(param.default).__name__)

        test = all(elem in self.df.columns for elem in args_without_defaults)
        result = dict()
        if test:
            for arg in args_without_defaults:
                varType = str(self.df[arg].dtype)
                if bit is None:                    
                    result[arg] = jnp.array(self.df[arg], dtype = self.pandas_to_jax_dtype_map.get(varType))
                else: 
                    result[arg] = jnp.array(self.df[arg], dtype = self.pandas_to_jax_dtype_map_force32.get(varType))

        else:
            return "Error, no data found"

        for k in args_with_defaults.keys():
            val, type_name = args_with_defaults[k]
            if type_name in ('int', 'float', 'bool'):
                # Scalar primitives stay as Python scalars — keeps them static for
                # JAX shape arguments (e.g. shape=(N_groups,) requires a Python int).
                result[k] = val
            elif bit is not None:
                result[k] = jnp.array(val, dtype=self.pandas_to_jax_dtype_map_force32.get(type_name + bit))
            else:
                result[k] = jnp.array(val, dtype=self.pandas_to_jax_dtype_map.get(type_name))

        return result     

    def data_to_model(self, cols):
        """## Prepare data for model input in JAX format
        
        ### Args:
            - *cols* (list): List of columns to include in model data
            
        ### Returns:
            - *dict*: JAX formatted dictionary
        """       
        jax_dict = {}
        for col in cols:
            jax_dict[col] = jnp.array(self.df.loc[:,col].values)
        self.data_modification['data_on_model'] = cols # store info of data used in the model
        self.data_on_model = jax_dict
