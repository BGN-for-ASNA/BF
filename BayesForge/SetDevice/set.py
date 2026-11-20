import os
import re
import warnings

# Set once setup_device has configured jax_enable_x64 (see the x64 warning below).
_X64_CONFIGURED = False


def set_deallocate():
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"


def _n_cpus():
    """CPU budget: sched_getaffinity (respects taskset/cgroups), else cpu_count.

    Same helper as BayesForge/Parallel/simulate.py (copied, not imported, to
    keep SetDevice dependency-free): os.cpu_count() reports the host's
    physical cores even under SLURM/cpuset limits, which oversubscribes XLA.
    """
    if hasattr(os, "sched_getaffinity"):
        try:
            n = len(os.sched_getaffinity(0))
        except OSError:
            n = 0
        if n >= 1:
            return n
    return os.cpu_count() or 1


def _validate_cores(cores):
    """Validate a ``cores`` argument: a positive integer (bool rejected).

    Returns the value as int, or None when cores is None. Shared by
    setup_device and setup_multi_device (RELEASE_AUDIT m8) so both raise the
    same ValueError message.
    """
    if cores is None:
        return None
    if isinstance(cores, bool):
        raise ValueError(f"cores must be a positive integer, got {cores!r}")
    if isinstance(cores, int) or (isinstance(cores, float) and cores.is_integer()):
        cores = int(cores)
    else:
        raise ValueError(f"cores must be a positive integer, got {cores!r}")
    if cores < 1:
        raise ValueError(f"cores must be a positive integer, got {cores!r}")
    return cores


def _validate_gpu_index(gpu_index):
    """Validate a ``gpu_index`` argument (RELEASE_AUDIT m1).

    Accepts an int >= 0 (bool rejected) or a comma-separated string of such
    ints, e.g. ``0``, ``"0"`` or ``"0,1"``. Returns the string to put in
    ``CUDA_VISIBLE_DEVICES``.
    """
    msg = ("gpu_index must be a non-negative integer or a comma-separated "
           f"string of such integers, got {gpu_index!r}")
    if isinstance(gpu_index, bool):
        raise ValueError(msg)
    if isinstance(gpu_index, int):
        if gpu_index < 0:
            raise ValueError(msg)
        return str(gpu_index)
    if isinstance(gpu_index, str):
        values = []
        for part in gpu_index.split(","):
            try:
                value = int(part.strip())
            except ValueError:
                raise ValueError(msg) from None
            if value < 0:
                raise ValueError(msg)
            values.append(value)
        return ",".join(str(v) for v in values)
    raise ValueError(msg)


def _backend_initialized():
    """True once JAX has created its backends (device settings are then frozen)."""
    try:
        from jax._src import xla_bridge
        return bool(xla_bridge.backends_are_initialized())
    except Exception:
        return False


def _warn_frozen_settings(platform, cores, gpu_index, deallocate):
    """Warn about requested device settings a live JAX backend will ignore."""
    import jax
    ignored = []
    try:
        current_platform = jax.default_backend()
    except Exception:
        current_platform = None
    if current_platform is not None and platform is not None and platform != current_platform:
        ignored.append(f"platform={platform!r} (backend is {current_platform!r})")
    if (platform in (None, 'cpu')) and cores is not None:
        try:
            n_cpu = len(jax.devices('cpu'))
        except Exception:
            n_cpu = None
        if n_cpu is not None and n_cpu != cores:
            ignored.append(f"cores={cores} ({n_cpu} CPU device(s) already created)")
    if gpu_index is not None:
        ignored.append(f"gpu_index={gpu_index} (CUDA_VISIBLE_DEVICES is read at backend start)")
    if deallocate:
        ignored.append("deallocate=True (XLA allocator flags are read at backend start)")
    if ignored:
        warnings.warn(
            "JAX is already initialised in this process, so these setup_device "
            "settings have no effect: " + "; ".join(ignored) + ". Set them in the first "
            "bf()/setup_device call of a fresh process (float_precision still applies).",
            RuntimeWarning, stacklevel=3)
    return ignored


def setup_device(platform='cpu', cores=None, gpu_index=None, deallocate=False,
                 print_devices_found=True, float_precision=64):
    """## Configures JAX for distributed computation.

    This function sets up the JAX computing environment by specifying the hardware
    platform and managing CPU core allocation. It also selects the XLA allocator
    and configures XLA flags appropriately.

    ### Args:

    - *platform:* str, optional
        The hardware platform to use for computation. Options include:
        - 'cpu': Use CPU(s) for computation
        - 'gpu': Use GPU(s) for computation
        - 'tpu': Use TPU(s) for computation
        Defaults to 'cpu'.

    - *cores:* int, optional
        Number of CPU cores to allocate for computation. If None, all available CPU
        cores will be used, or the value of the ``BF_MAX_CORES`` environment
        variable when it is set (``run_simulations`` sets it in each worker so a
        bare ``bf()`` inside a simulation takes only its own share of the
        machine). An explicit ``cores`` above ``BF_MAX_CORES`` is capped with a
        warning. Only applicable when platform is 'cpu'.

    - *gpu_index:* int or str, optional
        The index of the GPU to use (e.g., 0 or 1), or a comma-separated
        string of indices for several GPUs (e.g., '0,1'). Must be a
        non-negative integer or such a string. Only applicable when
        platform is 'gpu'.

    - *deallocate:* bool, optional
        Selects the XLA "platform" allocator (no preallocation) for this
        process via the ``XLA_PYTHON_CLIENT_*`` flags. It does not free
        existing devices, and once the backend has started the choice
        cannot be undone in the same process. Defaults to False.

    - *float_precision:* int or str, optional
        Floating-point precision for JAX computations. Accepts ``64`` or ``32``
        (also ``'float64'`` / ``'float32'`` for compatibility).
        - ``64``: Enable 64-bit precision (``jax_enable_x64=True``).
        - ``32``: Use default 32-bit precision (``jax_enable_x64=False``).
        Defaults to ``64``.
        Note: x64 is process-global and still applies to a live backend, so
        changing it between ``bf()`` instances mixes float32 and float64
        arrays (arrays created earlier keep their dtype). Such a switch emits
        a ``RuntimeWarning``.

    ### Notes
    Device settings (platform, cores, gpu_index, deallocate) take effect only
    before the JAX backend is initialised (the first computation or device
    query); later calls warn and change nothing (float_precision is the
    exception: it still applies). It configures the
    XLA_FLAGS environment variable to specify the number of CPU cores to use. The
    XLA_FORCE_HOST_PLATFORM_DEVICE_COUNT flag is particularly important for properly
    distributing computation across multiple CPUs.

    setup_device mutates process-global state (``os.environ``, ``jax.config``)
    and is not thread-safe: call it once, before starting threads.

     ### Examples

    Basic usage:
    >>> setup_device(platform='cpu')

    Specifying CPU cores and 32-bit precision:
    >>> setup_device(platform='cpu', cores=4, float_precision='float32')

     ### Returns
     *None*
    """
    if float_precision in (32, '32', 'float32'):
        float_precision = 'float32'
    elif float_precision in (64, '64', 'float64'):
        float_precision = 'float64'
    else:
        raise ValueError(f"float_precision must be 32 or 64, got {float_precision!r}")

    # M1: validate platform before any environment or JAX config change.
    if platform is not None:
        if not isinstance(platform, str):
            raise ValueError(f"platform must be 'cpu', 'gpu' or 'tpu', got {platform!r}")
        if platform.lower() not in ('cpu', 'gpu', 'tpu'):
            raise ValueError(f"platform must be 'cpu', 'gpu' or 'tpu', got {platform!r}")
        platform = platform.lower()

    # M3: validate cores before any XLA_FLAGS write (bool is an int subclass).
    cores = _validate_cores(cores)

    # m1: validate gpu_index before any environment change (also on 'cpu', so
    # a typo is caught instead of silently ignored).
    if gpu_index is not None:
        gpu_index = _validate_gpu_index(gpu_index)

    # BF_MAX_CORES is set by run_simulations in each worker process so that a
    # bare bf() inside a simulation claims only that worker's share of the
    # machine instead of every core on it.
    raw = os.environ.get("BF_MAX_CORES")
    max_cores = None
    if raw is not None and raw.strip():
        try:
            max_cores = int(raw)
        except ValueError:
            raise ValueError(
                f"environment variable BF_MAX_CORES must be a positive integer, got {raw!r}"
            ) from None
        if max_cores < 1:
            raise ValueError(
                f"environment variable BF_MAX_CORES must be a positive integer, got {raw!r}"
            )

    if platform == 'gpu' and gpu_index is not None:
        # Still set before import jax (the backend reads it at start-up); the
        # success print is deferred until the platform probe below confirms a
        # GPU backend exists (M1: no misleading message before the crash).
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    if cores is None:
        cores = max_cores if max_cores is not None else _n_cpus()
    elif max_cores is not None and cores > max_cores:
        warnings.warn(
            f"cores={cores} exceeds BF_MAX_CORES={max_cores} (this process is one "
            f"of several parallel simulation workers); capping to {max_cores}.",
            stacklevel=2,
        )
        cores = max_cores

    backend_live = _backend_initialized()
    if backend_live:
        _warn_frozen_settings(platform, cores, gpu_index, deallocate)

    if deallocate:
        set_deallocate()

    # Set the XLA flags before importing jax
    xla_flags = os.getenv("XLA_FLAGS", "")
    # m4: warn before silently replacing a --xla_force_host_platform_device_count
    # the user (or a scheduler) already put in XLA_FLAGS.
    prev = re.search(r"--xla_force_host_platform_device_count=(\S+)", xla_flags)
    if prev:
        try:
            unchanged = int(prev.group(1)) == cores
        except ValueError:
            unchanged = False
        if not unchanged:
            warnings.warn(
                f"XLA_FLAGS already sets --xla_force_host_platform_device_count="
                f"{prev.group(1)}; setup_device replaces it with {cores}.",
                stacklevel=2,
            )
        xla_flags = re.sub(r"--xla_force_host_platform_device_count=\S+", "", xla_flags)
    os.environ["XLA_FLAGS"] = " ".join(["--xla_force_host_platform_device_count={}".format(cores)] + xla_flags.split())

    # Now import jax
    import jax as jax

    # M1: probe the accelerator before writing it into the config, so a failed
    # setup never leaves jax_platform_name set to an unavailable backend.
    if platform in ('gpu', 'tpu'):
        try:
            same_backend_live = backend_live and jax.default_backend() == platform
        except Exception:
            same_backend_live = False
        if not same_backend_live:
            try:
                jax.devices(platform)
            except RuntimeError as e:
                raise RuntimeError(
                    f"platform={platform!r} requested but no {platform.upper()} backend "
                    f"is available ({e}); rerun with platform='cpu'."
                ) from e

    # m1: with a live backend the write above has no effect; the frozen-settings
    # warning already covers it, so do not announce a success that will not happen.
    if platform == 'gpu' and gpu_index is not None and not backend_live:
        print(f"Setting CUDA_VISIBLE_DEVICES to: {gpu_index}")

    # Explicitly update the configuration after import
    jax.config.update("jax_platform_name", platform)
    # jax_enable_x64 is process-global: a second bf()/setup_device with another
    # precision silently switches every live instance. Warn when that happens
    # (only after a first explicit configuration, so the initial call is quiet).
    global _X64_CONFIGURED
    want_x64 = float_precision == 'float64'
    if _X64_CONFIGURED and bool(jax.config.jax_enable_x64) != want_x64:
        warnings.warn(
            f"float_precision: switching the process-global jax_enable_x64 to "
            f"{want_x64} ({'64' if want_x64 else '32'} bits); every bf instance "
            f"of this process, including earlier ones, now runs at this precision.",
            RuntimeWarning, stacklevel=2)
    jax.config.update("jax_enable_x64", want_x64)
    _X64_CONFIGURED = True

    if print_devices_found:
        # m7: count the requested platform (the probe above confirmed it is
        # available), not the default backend, which may be a different one.
        print(f"jax.local_device_count({platform}) = {jax.local_device_count(platform)}")


def setup_multi_device(platform='cpu', cores=None):
    """Set up a JAX device mesh across all available devices for the given platform.

    Works on **both CPU and GPU**:

    * **CPU** — JAX exposes as many virtual CPU devices as cores were
      requested via the ``cores`` argument of :func:`setup_device` (backed
      by ``--xla_force_host_platform_device_count``).  Sharding distributes
      array slices across these virtual devices, which execute in parallel
      on separate OS threads / XLA partitions.

    * **GPU** — each physical GPU becomes one device; sharding puts
      different observation slices in separate GPU memory spaces for true
      hardware parallelism.

    In both cases the API and the model code are identical — only the
    underlying hardware changes.

    Parameters
    ----------
    platform : str
        ``'cpu'``, ``'gpu'``, or ``'tpu'``.  Must match the platform
        passed to :func:`setup_device`.

    cores : int, optional
        Cap the number of devices used.  Validated exactly like the
        ``cores`` argument of :func:`setup_device` (positive integer,
        bool rejected); ``None`` keeps every available device.

    Returns
    -------
    mesh : jax.sharding.Mesh or None
        1-D logical device mesh (axis ``'device'``), or ``None`` when only
        one device is available (no sharding is possible).
    data_sharding : jax.sharding.NamedSharding or None
        Shard arrays along their leading (observation) axis.
        Apply with ``m.shard(array)``.
    rep_sharding : jax.sharding.NamedSharding or None
        Replicate an array identically on every device.
        Apply with ``m.replicate(array)``.

    Notes
    -----
    **Chain parallelism vs data parallelism — both work on CPU and GPU**

    * *Chain parallelism* — ``num_chains=N`` with ``chain_method='parallel'``
      (the BayesForge default) dispatches each MCMC chain to a separate
      device via ``jax.pmap``.  No model changes required.

    * *Data parallelism* — shard observation arrays with ``m.shard(arr)``
      before passing them to the model.  XLA computes each device's slice
      of the log-likelihood concurrently and aggregates automatically.

    Both can be active simultaneously.

    Examples
    --------
    CPU — split across 8 cores:

    >>> m = bf(platform='cpu', cores=8)
    >>> Y_sharded = m.shard(jnp.array(Y))
    >>> m.fit(model, obs=dict(Y=Y_sharded), num_chains=m.n_devices)

    GPU — split across 4 GPUs:

    >>> m = bf(platform='gpu')
    >>> Y_sharded = m.shard(jnp.array(Y))
    >>> m.fit(model, obs=dict(Y=Y_sharded), num_chains=m.n_devices)
    """
    # m8: same validation as setup_device, before touching JAX, so a
    # non-numeric cores raises the same ValueError message.
    cores = _validate_cores(cores)

    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    from jax.experimental import mesh_utils

    devices = jax.devices(platform)
    if cores is not None and len(devices) > cores:
        devices = devices[:cores]
    n = len(devices)
    if n < 2:
        return None, None, None   # single device — sharding unavailable

    device_mesh = mesh_utils.create_device_mesh((n,), devices=devices)
    mesh = Mesh(device_mesh, axis_names=('device',))

    # Shard along the leading (observation) axis across devices
    data_sharding = NamedSharding(mesh, PartitionSpec('device'))
    # Replicate the same array on every device
    rep_sharding  = NamedSharding(mesh, PartitionSpec())

    return mesh, data_sharding, rep_sharding


# Keep the old name as an alias so any external code that calls it still works
setup_multi_gpu = setup_multi_device
