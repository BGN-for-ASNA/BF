library(reticulate)
library(BayesForge)
library(bisonR)
library(cmdstanr)

RESULTS_DIR <- "results"
stan_out_dir <- tempfile("stan_output_")  # per-run dir: shared stan_output/ gave empty-chain read errors
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(stan_out_dir, showWarnings = FALSE, recursive = TRUE)
options(cmdstanr_output_dir = stan_out_dir)

source("R_scripts/Modified simulate_bison_model.R")
assignInNamespace("simulate_bison_model", simulate_bison_model, ns = "bisonR")
source("R_scripts/BF_model_count.R")

m <- importBF("cpu")
jnp <- import("jax.numpy")

source("R_scripts/util.R")

# ---- main ----
NUM_INDIVIDUALS <- 20L
ITER_WARMUP <- 1000L
ITER_SAMPLES <- 1000L
NUM_CHAINS <- 8L

test_name <- "count_full_undirected_no_zi"
out_dir <- file.path(RESULTS_DIR, test_name)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cat("\n=======================================================\n")
cat("Running:", test_name, "\n")
cat("  Stan + BF: warmup=", ITER_WARMUP, ", samples=", ITER_SAMPLES, ", chains=", NUM_CHAINS, "\n")
cat("=======================================================\n")

set.seed(42)
sim <- simulate_bison_model("count",
  aggregated = TRUE,
  location_effect = TRUE, age_diff_effect = TRUE,
  num_nodes = NUM_INDIVIDUALS, num_locations = 5, max_obs = 10
)
df <- sim$df_sim

formula <- as.formula(
  "(event | duration) ~ dyad(node_1_id, node_2_id) + age_diff + (1 | node_1_id) + (1 | node_2_id)"
)

# --- Stan fit via bisonR ---
cat("  Fitting Stan model...\n")
fit_bison <- tryCatch(
  bison_model(formula,
    data = df, model_type = "count",
    directed = FALSE, partial_pooling = TRUE, zero_inflated = FALSE,
    iter_sampling = 10, iter_warmup = 10, refresh = 0, mc_cores = NUM_CHAINS
  ),
  error = function(e) {
    cat("  Stan failed:", conditionMessage(e), "\n")
    NULL
  }
)
if (is.null(fit_bison)) stop("Stan failed")

stan_data_raw <- fit_bison$model_data
BF_data <- build_stan_data_bison(stan_data_raw, "count", TRUE, FALSE, directed = FALSE)
print_priors(BF_data)

cat("  Re-running Stan model directly (adapt_delta=0.999)...\n")
t_stan_start <- proc.time()
clean_stan_fit <- sample_with_retry(fit_bison$stan_model,
  data = stan_data_raw, refresh = 0,
  chains = NUM_CHAINS, parallel_chains = NUM_CHAINS,
  iter_warmup = ITER_WARMUP, iter_sampling = ITER_SAMPLES,
  adapt_delta = 0.999, step_size = 0.05, max_treedepth = 20
)
t_stan_elapsed <- (proc.time() - t_stan_start)[["elapsed"]]

stan_params <- c(
  "edge_weight", "edge_sigma", "beta_fixed", "beta_random",
  "random_group_mu", "random_group_sigma"
)
stan_draws <- get_stan_draws(clean_stan_fit, stan_params)
stan_draws <- center_undirected_random_effects(stan_draws, BF_data)

# --- BF fit ---
cat("  Fitting BF model...\n")
BF_draws <- list()
t_bi_start <- proc.time()
tryCatch(
  {
    m$data_on_model <- list(data = make_jax_data(BF_data, "count"))
    m$fit(BF_model_count,
      num_warmup = ITER_WARMUP, num_samples = ITER_SAMPLES,
      num_chains = NUM_CHAINS, target_accept_prob = 0.999
    )
    raw_BF <- get_bi_draws(m$posteriors)
    BF_draws <- normalize_bi_names(
      center_undirected_random_effects(apply_non_centered_transform(raw_BF, BF_data), BF_data),
      stan_draws
    )
  },
  error = function(e) cat("  BF fit failed:", conditionMessage(e), "\n")
)
t_bi_elapsed <- (proc.time() - t_bi_start)[["elapsed"]]

# --- Save outputs ---
save_multipanel_svg(test_name, stan_draws, BF_draws, out_dir)
save_combination_log(test_name, stan_draws, BF_draws, out_dir)

# --- Summary stats ---
cat("\n=== Summary KL stats ===\n")
all_params <- intersect(names(stan_draws), names(BF_draws))
kls <- sapply(all_params, function(p) {
  s <- stan_draws[[p]]
  b <- BF_draws[[p]]
  if (length(s) > 1 && length(b) > 1) kl_divergence(s, b) else NA
})
kls <- kls[!is.na(kls)]
cat(sprintf("  N params:  %d\n", length(kls)))
cat(sprintf("  Mean KL:   %.4f\n", mean(kls)))
cat(sprintf("  Max KL:    %.4f\n", max(kls)))
cat(sprintf("  KL > 0.10: %d\n", sum(kls > 0.10)))
cat(sprintf("  KL > 0.20: %d\n", sum(kls > 0.20)))
cat(sprintf("  KL > 0.50: %d\n", sum(kls > 0.50)))

edge_kls <- kls[grep("^edge_weight", names(kls))]
rand_kls <- kls[grep("^beta_random", names(kls))]
other_kls <- kls[!grepl("^edge_weight|^beta_random", names(kls))]
cat(sprintf("\n  edge_weight:   mean=%.4f  max=%.4f\n", mean(edge_kls), max(edge_kls)))
cat(sprintf("  beta_random:   mean=%.4f  max=%.4f\n", mean(rand_kls), max(rand_kls)))
cat(sprintf("  other params:  mean=%.4f  max=%.4f\n", mean(other_kls), max(other_kls)))
cat(sprintf(
  "\n  random_group_mu  Stan: %.4f / %.4f   BF: %.4f / %.4f\n",
  mean(stan_draws[["random_group_mu[1]"]]),
  mean(stan_draws[["random_group_mu[2]"]]),
  mean(BF_draws[["random_group_mu[1]"]]),
  mean(BF_draws[["random_group_mu[2]"]])
))


cat("\n=== Timing ===\n")
cat(sprintf("  Stan: %6.1f s  (%4.1f min)\n", t_stan_elapsed, t_stan_elapsed / 60))
cat(sprintf("  BF:   %6.1f s  (%4.1f min)\n", t_bi_elapsed, t_bi_elapsed / 60))
cat(sprintf("  Stan/BF ratio: %.1fx\n", t_stan_elapsed / max(t_bi_elapsed, 0.1)))

timing_csv <- file.path(RESULTS_DIR, "timing_summary.csv")
timing_row <- data.frame(
  model = test_name,
  num_chains = NUM_CHAINS,
  iter_warmup = ITER_WARMUP,
  iter_samples = ITER_SAMPLES,
  stan_sec = round(t_stan_elapsed, 1),
  BF_sec = round(t_bi_elapsed, 1),
  stan_bi_ratio = round(t_stan_elapsed / max(t_bi_elapsed, 0.1), 1),
  timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  stringsAsFactors = FALSE
)
write.table(timing_row, timing_csv,
  append = file.exists(timing_csv), sep = ",",
  row.names = FALSE, col.names = !file.exists(timing_csv), quote = TRUE
)

# --- Cleanup Stan CSV output ---
unlink(list.files(stan_out_dir, full.names = TRUE, pattern = "\\.csv$"))

cat("\n  Completed:", test_name, "\n")
