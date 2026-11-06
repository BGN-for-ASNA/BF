import os
import re


def set_deallocate():
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
    
def setup_device(platform='cpu', cores=None, gpu_index=None, deallocate=False,
                 print_devices_found=True, float_precision=64):
    """## Configures JAX for distributed computation.

    This function sets up the JAX computing environment by specifying the hardware
    platform and managing CPU core allocation. It also handles deallocation of existing
    devices and configures XLA flags appropriately.

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

    - *gpu_index:* int, optional
        The index of the GPU to use (e.g., 0 or 1). Only applicable when platform is 'gpu'.

    - *deallocate:* bool, optional
        Whether to deallocate any existing devices before setting up new configuration.
        Defaults to False.

    - *float_precision:* int or str, optional
        Floating-point precision for JAX computations. Accepts ``64`` or ``32``
        (also ``'float64'`` / ``'float32'`` for compatibility).
        - ``64``: Enable 64-bit precision (``jax_enable_x64=True``).
        - ``32``: Use default 32-bit precision (``jax_enable_x64=False``).
        Defaults to ``64``.

    ### Notes
    This function must be called before any JAX imports or usage. It configures the
    XLA_FLAGS environment variable to specify the number of CPU cores to use. The
    XLA_FORCE_HOST_PLATFORM_DEVICE_COUNT flag is particularly important for properly
    distributing computation across multiple CPUs.

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

    if platform == 'gpu' and gpu_index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        print(f"Setting CUDA_VISIBLE_DEVICES to: {gpu_index}")

    # BF_MAX_CORES is set by run_simulations in each worker process so that a
    # bare bf() inside a simulation claims only that worker's share of the
    # machine instead of every core on it.
    max_cores = os.environ.get("BF_MAX_CORES")
    max_cores = int(max_cores) if max_cores else None

    if cores is None:
        cores = max_cores if max_cores is not None else os.cpu_count()
    elif max_cores is not None and cores > max_cores:
        import warnings
        warnings.warn(
            f"cores={cores} exceeds BF_MAX_CORES={max_cores} (this process is one "
            f"of several parallel simulation workers); capping to {max_cores}.",
            stacklevel=2,
        )
        cores = max_cores

    if deallocate:
        set_deallocate()

    # Set the XLA flags before importing jax
    xla_flags = os.getenv("XLA_FLAGS", "")
    xla_flags = re.sub(r"--xla_force_host_platform_device_count=\S+", "", xla_flags).split()
    os.environ["XLA_FLAGS"] = " ".join(["--xla_force_host_platform_device_count={}".format(cores)] + xla_flags)

    # Now import jax
    import jax as jax

    # Explicitly update the configuration after import
    jax.config.update("jax_platform_name", platform)
    jax.config.update("jax_enable_x64", float_precision == 'float64')

    if print_devices_found:
        print('jax.local_device_count', jax.local_device_count(backend=None))


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
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    from jax.experimental import mesh_utils

    devices = jax.devices(platform)
    if cores is not None and int(cores) > 0 and len(devices) > int(cores):
        devices = devices[:int(cores)]
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
