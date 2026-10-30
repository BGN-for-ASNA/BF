# Same model as repro.py through PyBayesForge (see Test/CROSS_LANGUAGE.md).
using PyBayesForge, PythonCall
m = importBF(platform="cpu")
m.data_on_model = pybuiltins.dict(y = jnp.array([0.1, 0.5, -0.3, 1.2, 0.8]))
@BF function model(y)
    alpha = m.dist.normal(0, 1, name = "alpha")
    s = m.dist.exponential(1, name = "s")
    m.dist.normal(alpha, s, obs = y, name = "y")
end
m.fit(model, num_samples = 200, num_warmup = 200, num_chains = 1, seed = 0, progress_bar = false)
println(m.summary(round_to=8).loc[pylist(["alpha", "s"]), "mean"].tolist())
