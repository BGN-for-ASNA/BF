import os
import warnings
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
        # m1: the category decode map is a user artefact — created once here and
        # updated per column by index(), never reset by later index() calls.
        self.index_map = {}
        # {original: normalized} column names, filled by data().
        self.column_aliases = {}
        self.pandas_to_jax_dtype_map = {
            'int64': jnp.int64,
            'int32': jnp.int32,
            'int16': jnp.int32,
            'int8': jnp.int32,
            'uint8': jnp.int32,
            'uint16': jnp.int32,
            'uint32': jnp.int64,
            'bool': jnp.bool_,
            'float64': jnp.float64,
            'float32': jnp.float32,
            'float16': jnp.float16,
        }
        self.pandas_to_jax_dtype_map_force32 = {
            'int64': jnp.int32,
            'int32': jnp.int32,
            'int16': jnp.int32,
            'int8': jnp.int32,
            'uint8': jnp.int32,
            'uint16': jnp.int32,
            'uint32': jnp.int32,
            'bool': jnp.bool_,
            'float64': jnp.float32,
            'float32': jnp.float32,
            'float16': jnp.float16,
        }

    def _resolve_columns(self, cols):
        """## Resolve a sequence of column labels without stringifying them (m12).

        Each label is used as given when it is a column of ``m.df``; only when
        it is not found as-is does it fall back to the unique column whose
        ``str()`` equals ``str(label)`` (so ``m.index([0])`` works on an
        integer-labelled column and ``m.index(['0'])`` still finds it).
        A label that ``data()`` normalized (``'kcal.per.g'`` loaded as
        ``'kcal_per_g'``) is accepted under its original spelling too.
        Unknown labels are returned unchanged so pandas raises the usual KeyError.
        """
        aliases = getattr(self, "column_aliases", {}) or {}
        resolved = []
        for c in cols:
            if c in self.df.columns:
                resolved.append(c)
                continue
            if isinstance(c, str) and aliases.get(c) in self.df.columns:
                resolved.append(aliases[c])
                continue
            match = [col for col in self.df.columns if str(col) == str(c)]
            if len(match) > 1:
                raise KeyError(f"column label {c!r} matches several columns {match!r} when compared as text")
            resolved.append(match[0] if match else c)
        return resolved

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
            - *pd.DataFrame*: Loaded dataframe (`m.df`).
            - **Updates on `m`**: sets `m.df` (copy of the loaded table), `m.data_original_path`, `m.data_args`, `m.column_aliases`, and resets `m.data_modification = {}`.

        ### Column names:
            '.' and spaces in column names are replaced by '_' at load
            ('kcal.per.g' becomes 'kcal_per_g'), so every column can be the
            name of a model argument. `m.column_aliases` maps each original
            name to its new one, and the methods of `m` (scale, index, OHE,
            to_float, to_int, data_to_model) also accept the original names.
            A message lists the renamed columns. Two names that would
            collide after the replacement raise ValueError.
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
            # RELEASE_AUDIT X13: a scikit-learn style Bunch (m.load.<name>(frame=False))
            # holds 2-D arrays that pd.DataFrame() rejects with an obscure error;
            # use its "frame" table, or say how to get one.
            if hasattr(path, "keys") and "data" in path and not isinstance(path, pd.DataFrame):
                frame = path.get("frame") if hasattr(path, "get") else None
                if frame is not None:
                    path = frame
                elif getattr(path.get("data"), "ndim", 1) > 1:
                    raise ValueError(
                        "data(): this looks like a dataset bunch with a 2-D 'data' array "
                        "and no 'frame' table; load it with frame=True (the default), "
                        "or pass a pandas DataFrame.")
            # Copy so later manipulations (scale, OHE, ...) never modify the caller's table.
            self.df = pd.DataFrame(path).copy()
        self._normalize_column_names()
        # pandas >= 3 gives text categories the "str" dtype, which reticulate
        # cannot convert back to an R factor (m$data(df) then fails). Store the
        # categories as object: same values, same level order.
        for col in self.df.select_dtypes(['category']).columns:
            cats = self.df[col].cat.categories
            if cats.dtype != object and pd.api.types.is_string_dtype(cats.dtype):
                self.df[col] = self.df[col].cat.set_categories(cats.astype(object))
        self.data_modification = {}
        return self.df

    def _normalize_column_names(self):
        """Replace '.' and ' ' by '_' in the text column names of ``m.df``.

        A model argument cannot be called 'kcal.per.g', so the columns get
        names usable as arguments once, at load, instead of as a side effect
        of a later call. ``m.column_aliases`` keeps {original: new} so the
        original names stay accepted (see ``_resolve_columns``).
        """
        renamed = {c: c.replace('.', '_').replace(' ', '_')
                   for c in self.df.columns if isinstance(c, str)}
        renamed = {old: new for old, new in renamed.items() if new != old}
        if renamed:
            final = [renamed.get(c, c) for c in self.df.columns]
            clashes = sorted({str(n) for n in final if final.count(n) > 1})
            if clashes:
                raise ValueError(
                    f"data(): column names {clashes} collide once '.' and ' ' are "
                    "replaced by '_' (needed to use them as model arguments); "
                    "rename the columns first.")
            self.df = self.df.rename(columns=renamed)
            pairs = ", ".join(f"{old!r} -> {new!r}" for old, new in renamed.items())
            print(f"data(): renamed columns so they can be model arguments: {pairs}. "
                  "The original names are still accepted by the methods of m.")
        self.column_aliases = renamed

    def OHE(self, cols = 'all', missing = 'zero'):
        """## Perform one-hot encoding on specified columns
        
        ### Args:
            - *cols* (str or list): Columns to encode. Use 'all' for all object
              and categorical columns, e.g. R factors
            - *missing* (str): 'zero' (default) -> rows with a missing category get all-zero dummies; 'category' -> an extra `<col>_nan` indicator column (1 where the value is missing) is added, only for columns that actually contain NaN (mirrors index(missing='category')).
            
        ### Returns:
            - *pd.DataFrame*: DataFrame containing the newly created one-hot encoded columns.
            - **Updates on `m`**: appends the one-hot encoded columns in-place to `m.df` (`pd.concat([m.df, OHE], axis=1)`) and records `m.data_modification['OHE']` as the resolved list of encoded columns (with 'all', the columns selected from `m.df`, not the string 'all').

        NaN / missing categories: `pd.get_dummies` is called
        with its default `dummy_na=False`, so a row whose category is missing
        gets an **all-zero dummy row** for that variable (no `_<na>` column is
        created). The original column is **kept** in `m.df` next to the new
        dummy columns. This is the default (missing='zero'); pass
        missing='category' to get an explicit indicator column.
        """
        if missing not in ('zero', 'category'):
            raise ValueError("OHE(): missing must be 'zero' or 'category', got " + repr(missing))
        if cols == 'all':
            # 'category' too: an R factor arrives as a pandas Categorical and
            # must be encoded like a text column (RELEASE_AUDIT M4).
            encoded = list(self.df.select_dtypes(['object', 'category']).columns)
            OHE = pd.get_dummies(self.df[encoded], columns=encoded, dtype=int)
        else:
            # a single name, or any sequence of names (list, tuple, Julia Vector);
            # m12: labels are used as given, never stringified.
            encoded = self._resolve_columns([cols] if isinstance(cols, str) else cols)
            OHE = pd.get_dummies(self.df[encoded], columns=encoded, dtype=int)

        if missing == 'category':
            for c in encoded:
                if self.df[c].isna().any():
                    name = f"{c}_nan"
                    if name in OHE.columns or name in self.df.columns:
                        raise ValueError(f"OHE(): column {name!r} already exists; rename it before using missing='category'.")
                    OHE[name] = self.df[c].isna().astype(int)

        # m4: regex=False pins literal matching on every pandas version
        # (pandas < 2.0 defaulted to regex=True, where '.' matched any character).
        OHE.columns = OHE.columns.str.replace('.', '_', regex=False)
        OHE.columns = OHE.columns.str.replace(' ', '_', regex=False)


        self.df = pd.concat([self.df , OHE], axis=1)
        self.data_modification['OHE'] = encoded # m2: resolved columns, not 'all'
        return OHE

    def _index_column(self, c, missing):
        """Encode one column for index(); see index() for the missing-value rule."""
        cat = self.df[c].astype("category")
        codes = cat.cat.codes.to_numpy().astype(np.int64)
        levels = list(cat.cat.categories)
        n_missing = int((codes == -1).sum())
        if n_missing:
            if missing == 'raise':
                raise ValueError(f"index(): column {c!r} has {n_missing} missing value(s); "
                                 "pandas would code them -1, which JAX reads as the last group. "
                                 "Drop or impute them first, or pass missing='category' "
                                 "to give them their own group.")
            if "missing" in levels:
                raise ValueError(f"index(): column {c!r} already has a level 'missing'; "
                                 "rename it before using missing='category'.")
            codes[codes == -1] = len(levels)
            levels.append("missing")
        self.df[f"index_{c}"] = codes
        self.index_map[c] = dict(enumerate(levels))

    def index(self, cols = 'all', missing = 'raise'):
        """## Create index encoding for categorical columns
        
        ### Args:
            - *cols* (str or list): Columns to encode. Use 'all' for all text
              (object) and categorical columns, e.g. R factors.
            - *missing* (str): What to do with missing values (NaN/None).
              ``'raise'`` (default) raises ``ValueError``: pandas would code them
              ``-1``, which JAX indexing reads as the last group.
              ``'category'`` gives them their own group: code ``K`` (after the
              ``K`` observed levels) and ``m.index_map[col][K] == 'missing'``.
              Raises ``ValueError`` if the column already has a level ``'missing'``.

        Codes are 0-based and follow the category order: the level order of a
        pandas Categorical or R factor, alphabetical for text columns.
        ``m.index_map[col]`` maps each code to its level.

        ### Returns:
            - *pd.DataFrame*: Updated dataframe `m.df` with added `index_<col>` columns.
            - **Updates on `m`**: adds 0-based integer columns `index_<col>` (`int64`) to `m.df`, replaces `.` and spaces with `_` in the names of the added `index_<col>` columns only (the other columns of `m.df` keep their original names), updates `m.index_map[col]` (`{code: category_label}`) for the encoded columns while keeping the entries of earlier `index()` calls, and records `m.data_modification['index']` as the resolved list of encoded columns (with 'all', the columns selected from `m.df`, not the string 'all').
        """
        if not hasattr(self, "index_map"):
            # m1: the map is created in __init__ and only updated per column;
            # this guard covers instances built before that fix.
            self.index_map = {}
        if missing not in ('raise', 'category'):
            raise ValueError(f"index(): missing must be 'raise' or 'category', got {missing!r}")
        if cols == 'all':
            # 'category' too: an R factor arrives as a pandas Categorical, whose
            # level order (set in R) is kept by .astype("category") below.
            encoded = list(self.df.select_dtypes(['object', 'category']).columns)
        else:
            # a single name, or any sequence of names (list, tuple, Julia Vector);
            # m12: labels are used as given, never stringified, so integer or
            # tuple column labels keep working (m.index([0]) on column 0).
            encoded = self._resolve_columns([cols] if isinstance(cols, str) else cols)
        for c in encoded:
            self._index_column(c, missing)

        # M5: sanitize only the names of the index columns just added. Renaming
        # every column of m.df here used to break later calls made with the
        # original names (index() renamed 'my.col' to 'my_col', then
        # m.scale(['my.col']) raised KeyError).
        for c in encoded:
            old = f"index_{c}"
            new = old.replace('.', '_').replace(' ', '_')
            if new != old:
                self.df = self.df.rename(columns={old: new})

        self.data_modification['index'] = encoded # m2: resolved columns, not 'all'
        
        return self.df
    
    def scale(self, data = 'all'):
        """## Standardize specified columns.
        
        ### Args:
            - *data* (str or list): 'all' (default) = every numeric column except the index_* codes; a single name or a list of names.
            
        ### Returns:
            - *pd.DataFrame*: Standardized dataframe `m.df`.
            - **Updates on `m`**: standardizes the specified columns in-place in `m.df` (zero mean, unit **sample** standard deviation, `ddof=1` — the module-wide convention shared by `z_score`) and records `m.data_modification['scale'] = data`. Raises `ValueError` when a column has zero or undefined standard deviation (constant column, single row, or NaN values): standardizing it would silently produce NaN.
        """        
        if isinstance(data, str) and data == 'all':
            data = [c for c in self.df.select_dtypes('number').columns
                    if not str(c).startswith('index_')]
        elif isinstance(data, str):
            data = [data]
        data = self._resolve_columns(data)
        # Check every column before writing anything, so a constant column
        # cannot leave the other columns half-scaled (RELEASE_AUDIT M1).
        # ddof=1 (pandas Series.std default) is the convention the reference
        # documents for scale; z_score matches it (RELEASE_AUDIT M2).
        stats = {}
        bad = []
        for col in data:
            x = self.df[col].astype('float64')
            std = x.std(ddof=1)
            if std == 0 or not np.isfinite(std):
                bad.append(col)
            else:
                stats[col] = (x.mean(), std)
        if bad:
            raise ValueError(f"scale(): column(s) {bad!r} have zero or undefined standard deviation "
                             "(constant column, fewer than 2 rows, or NaN values); standardizing them "
                             "would produce NaN. Drop them or add variation first.")
        for col in data:
            mean, std = stats[col]
            self.df[col] = (self.df[col].astype('float64') - mean) / std
        self.data_modification['scale'] = data # store info of scaled columns
        return self.df

    def z_score(self,X, axis=0):
        """
        Performs Z-score scaling (standardization) on a JAX array.
        (X - mean) / std_dev, with the sample standard deviation (ddof=1),
        the same convention as scale.

        Args:
            X (jnp.ndarray): The input JAX array (e.g., shape (n_samples, n_features)).
            axis (int): The axis along which to calculate mean and std. 
                        axis=0 scales features (columns).

        Returns:
            jnp.ndarray: The scaled array.

        NaN values: a NaN used to poison the mean and std
        along the whole axis, turning every row NaN without a word. When the
        input contains NaN, a `UserWarning` names the affected columns (or
        rows, for axis=1) and the statistics are computed over the non-NaN
        entries (`nanmean`/`nanstd`); the NaN entries themselves stay NaN.
        """
        X = jnp.asarray(X)
        nan_mask = jnp.isnan(X) if jnp.issubdtype(X.dtype, jnp.inexact) else None
        if nan_mask is not None and bool(nan_mask.any()):
            # m7: nanmean/nanstd per slice so non-NaN rows are still scaled;
            # the ddof=1 correction uses the non-NaN count of each slice.
            affected = jnp.nonzero(jnp.atleast_1d(nan_mask.any(axis=axis)))[0].tolist()
            label = "column(s)" if axis == 0 else "row(s)"
            warnings.warn(f"z_score(): {label} {affected} contain NaN; mean and std are "
                          "computed over the non-NaN entries (nanmean/nanstd) and the "
                          "NaN entries stay NaN.", UserWarning)
            mean_arr = jnp.nanmean(X, axis=axis, keepdims=True)
            std_arr = jnp.nanstd(X, axis=axis, keepdims=True)
            n = jnp.sum(~nan_mask, axis=axis, keepdims=True)
            std_arr = std_arr * jnp.where(n > 1, jnp.sqrt(n / jnp.maximum(n - 1, 1)), 1.0)
        else:
            # Calculate mean and std, preserving dimensions for correct broadcasting
            mean_arr = jnp.mean(X, axis=axis, keepdims=True)
            std_arr = jnp.std(X, axis=axis, keepdims=True)
            n = X.shape[axis]
            if n > 1:
                # sample standard deviation (ddof=1), matching pandas Series.std in scale
                std_arr = std_arr * jnp.sqrt(n / (n - 1))

        # Handle division by zero (for constant features) by replacing 0 with 1
        # This prevents NaN, as (X - mean) will be 0 when std_arr is 0.
        std_arr = jnp.where(std_arr == 0, 1.0, std_arr)

        X_scaled = (X - mean_arr) / std_arr
        return X_scaled
    
    # m9: the `type` keyword shadows the builtin inside these two methods on
    # purpose — it is part of the public keyword-argument contract of the `bf`
    # class (m.to_float(cols, type='float64')) and must not be renamed.
    def to_float(self, cols = 'all', type = 'float32'):
        """## Convert specified columns to float type
        
        ### Args:
            - *cols* (str or list): Columns to convert. Use 'all' for all columns
            - *type* (str): Float type to convert to (default: float32)
            
        ### Returns:
            - *pd.DataFrame*: Converted dataframe `m.df`.
            - **Updates on `m`**: casts the specified columns of `m.df` in-place to `type` and records `m.data_modification['float'] = cols`.
        """        
        if isinstance(cols, str):
            cols = list(self.df.columns) if cols == 'all' else [cols]
        cols = self._resolve_columns(cols)
        for col in cols:
            self.df[col] = self.df[col].astype(type)

        self.data_modification['float'] = cols # store info of scaled columns
        
        return self.df

    def to_int(self, cols = 'all', type = 'int32'):  # m9: see note on to_float
        """## Convert specified columns to integer type
        
        ### Args:
            - *cols* (str or list): Columns to convert. Use 'all' for all columns
            - *type* (str): Integer type to convert to (default: int32)
            
        ### Returns:
            - *pd.DataFrame*: converted m.df.
            - **Updates on `m`**: casts the specified columns of `m.df` in-place to `type` and records `m.data_modification['int'] = cols`.
        """
        if isinstance(cols, str):
            cols = list(self.df.columns) if cols == 'all' else [cols]
        cols = self._resolve_columns(cols)
        for col in cols:
            self.df[col] = self.df[col].astype(type)

        self.data_modification['int'] = cols # store info of scaled columns

        return self.df

    def pd_to_jax(self, model, bit = None):
        """## Convert pandas dataframe to JAX compatible format for a model.
        
        ### Args:
            - *model*: JAX model to prepare data for
            - *bit* (str or int, optional): Bit precision to force: '32'/32 forces
              32-bit dtypes, '64'/64 requests 64-bit dtypes (produced only when
              jax_enable_x64 is on, otherwise downcast to 32-bit). Default None
              follows `jax_enable_x64`. Any other value raises
              `ValueError`.
            
        ### Returns:
            - *dict*: Dictionary mapping model argument names to JAX arrays (extracted from `m.df`) or static default values. Called automatically by `m.fit()` and `m.svi()` to populate `m.data_on_model` when `m.data_on_model is None`. Raises `KeyError` when a model argument without a default has no column in `m.df`, and `TypeError` when a required column has a dtype absent from the conversion map (e.g. `category`, `datetime64[ns]`, `complex`, `object`): cast it first, e.g. with `m.to_float(['col'])`.

        64-bit JAX dtypes (`float64`, `int64`) are produced only when `jax_enable_x64` is on; with x64 off (the default) such columns are converted to `float32`/`int32` directly, without a silent 32-bit truncation warning.
        """        
        params = inspect.signature(model).parameters
        args_without_defaults = []
        args_with_defaults = {}
        for param_name, param in params.items():
            if param.default is inspect.Parameter.empty:
                args_without_defaults.append(param_name)
            else:
                args_with_defaults[param_name] = (param.default, type(param.default).__name__)

        missing = [a for a in args_without_defaults if a not in self.df.columns]
        if missing:
            raise KeyError(f"Model arguments {missing} have no default and no column in m.df "
                           f"(columns: {list(self.df.columns)}); pass them through fit(data=...) or add them to the data.")
        # m5: accept '32'/'64'/32/64 robustly instead of concatenating the
        # value into a dtype name ('float64' + '64' used to give 'float6464'
        # -> .get -> None -> silent inference). None follows jax_enable_x64.
        if bit is None:
            dtype_map = self.pandas_to_jax_dtype_map
        elif bit in (32, '32'):
            dtype_map = self.pandas_to_jax_dtype_map_force32
        elif bit in (64, '64'):
            dtype_map = self.pandas_to_jax_dtype_map
        else:
            raise ValueError(f"pd_to_jax: bit must be None, 32, '32', 64 or '64' (got {bit!r}).")
        if not jax.config.jax_enable_x64:
            # With x64 off, JAX truncates any explicitly requested 64-bit dtype
            # to 32 bits with a UserWarning; request the 32-bit dtype directly
            # instead (RELEASE_AUDIT M3).
            to32 = {jnp.float64: jnp.float32, jnp.int64: jnp.int32}
            dtype_map = {k: to32.get(v, v) for k, v in dtype_map.items()}
        result = dict()
        for arg in args_without_defaults:
            varType = str(self.df[arg].dtype)
            jax_dtype = dtype_map.get(varType)
            if jax_dtype is None:
                # An unmapped dtype must not fall through to dtype=None inference;
                # name the column and the dtype (RELEASE_AUDIT M6).
                raise TypeError(f"pd_to_jax: column {arg!r} has dtype {varType} which cannot be converted; "
                                f"cast it first (e.g. m.df[{arg!r}].astype(float)).")
            result[arg] = jnp.array(self.df[arg], dtype = jax_dtype)

        for k in args_with_defaults.keys():
            val, type_name = args_with_defaults[k]
            if val is None or isinstance(val, (int, float, bool, str)):
                # Scalars, None and strings stay as Python values: static for
                # JAX shape arguments and never forced into a tensor.
                result[k] = val
            elif isinstance(val, (np.ndarray, list, tuple)) or hasattr(val, '__jax_array__') or isinstance(val, jax.Array):
                arr = np.asarray(val)
                key = str(arr.dtype)
                result[k] = jnp.asarray(arr, dtype=dtype_map.get(key))
            else:
                result[k] = val

        return result     

    def data_to_model(self, cols):
        """## Prepare data for model input in JAX format
        
        ### Args:
            - *cols* (list): List of columns to include in model data
            
        ### Returns:
            - *None* (populates `m.data_on_model` in-place).
            - **Updates on `m`**: sets `m.data_on_model` to the dictionary `{col: jnp.array(m.df[col])}` and records `m.data_modification['data_on_model'] = cols`.

        Raises `ValueError` naming the column when a requested
        name is **duplicated** in `m.df` (e.g. after an `OHE` that re-created the
        original name).
        """       
        cols = self._resolve_columns([cols] if isinstance(cols, str) else cols)
        dupes = set(self.df.columns[self.df.columns.duplicated()])
        jax_dict = {}
        for col in cols:
            if col in dupes:
                raise ValueError(f"data_to_model: column name {col!r} is duplicated in m.df "
                                 f"(e.g. after OHE); m.df.loc[:, {col!r}] would return a DataFrame and "
                                 "silently build a 2-D array. Drop or rename one of the duplicates first.")
            values = self.df.loc[:, col]
            if str(values.dtype) == 'int64':
                # m6: request the int dtype matching the x64 setting explicitly;
                # warn only when int32 would actually truncate the values.
                if jax.config.jax_enable_x64:
                    jax_dict[col] = jnp.array(values.values, dtype=jnp.int64)
                else:
                    arr = values.values
                    if arr.size and (int(arr.min()) < np.iinfo(np.int32).min
                                     or int(arr.max()) > np.iinfo(np.int32).max):
                        warnings.warn(f"data_to_model: column {col!r} holds int64 values outside the "
                                      "int32 range; with jax_enable_x64 off JAX stores them as int32 "
                                      "and they will be truncated.", UserWarning)
                    jax_dict[col] = jnp.array(arr, dtype=jnp.int32)
            else:
                jax_dict[col] = jnp.array(values.values)
        self.data_modification['data_on_model'] = cols # store info of data used in the model
        self.data_on_model = jax_dict
