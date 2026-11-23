get_bi_draws <- function(raw_post) {
  np <- import("numpy")
  out <- list()
  for (nm in names(raw_post)) {
    if (grepl("_lik$", nm) || nm == "event" || nm == "event_count") next
    r_arr <- py_to_r(np$asarray(raw_post[[nm]]))
    dims   <- dim(r_arr)
    if (!is.null(dims) && length(dims) == 3) {
      r_arr <- matrix(r_arr, nrow = dims[1] * dims[2], ncol = dims[3])
      dims  <- dim(r_arr)
    }
    if (is.null(dims) || length(dims) == 1) {
      out[[nm]] <- as.numeric(r_arr)
    } else if (length(dims) == 2) {
      for (i in seq_len(dims[2]))
        out[[paste0(nm, "[", i, "]")]] <- as.numeric(r_arr[, i])
    }
  }
  out
}

get_stan_draws <- function(fit, param_names) {
  out <- list()
  for (p in param_names) {
    tryCatch({
      draws <- fit$draws(p, format = "matrix")
      if (ncol(draws) == 1) {
        out[[colnames(draws)[1]]] <- as.numeric(draws[, 1])
      } else {
        for (i in seq_len(ncol(draws)))
          out[[colnames(draws)[i]]] <- as.numeric(draws[, i])
      }
    }, error = function(e) NULL)
  }
  out
}

normalize_bi_names <- function(BF_draws, stan_draws) {
  out <- BF_draws
  for (nm in names(BF_draws)) {
    idx_nm <- paste0(nm, "[1]")
    if (idx_nm %in% names(stan_draws) && !(nm %in% names(stan_draws))) {
      out[[idx_nm]] <- out[[nm]]; out[[nm]] <- NULL
    }
  }
  out
}

# Undo the non-centered parameterisation used in BF_model_binary:
#   edge_weight_actual  = prior_mu + edge_sigma       * edge_weight_raw   (N(0,1))
#   beta_random_actual  = group_mu + group_sigma[grp] * beta_random_raw   (N(0,1))
# Each transform is applied sample-wise (all vectors have length chains*samples).
apply_non_centered_transform <- function(BF_draws, BF_data) {
  # --- edge_weight ---
  if (as.numeric(BF_data$partial_pooling) == 1) {
    sig_key <- if ("edge_sigma"    %in% names(BF_draws)) "edge_sigma"
               else if ("edge_sigma[1]" %in% names(BF_draws)) "edge_sigma[1]"
               else NULL
    if (!is.null(sig_key)) {
      sigma_s <- BF_draws[[sig_key]]
      mu_val  <- as.numeric(BF_data$prior_edge_mu)
      for (i in seq_len(as.integer(BF_data$num_edges))) {
        k <- paste0("edge_weight[", i, "]")
        if (k %in% names(BF_draws))
          BF_draws[[k]] <- mu_val + sigma_s * BF_draws[[k]]
      }
    }
  }
  # --- beta_random ---
  if (as.numeric(BF_data$num_random) > 0) {
    grp_idx <- BF_data$random_group_index   # 1-based R vector
    for (r in seq_along(grp_idx)) {
      g     <- grp_idx[r]
      mu_k  <- paste0("random_group_mu[",    g, "]")
      sig_k <- paste0("random_group_sigma[", g, "]")
      br_k  <- paste0("beta_random[",        r, "]")
      if (all(c(mu_k, sig_k, br_k) %in% names(BF_draws)))
        BF_draws[[br_k]] <- BF_draws[[mu_k]] + BF_draws[[sig_k]] * BF_draws[[br_k]]
    }
  }
  BF_draws
}

kl_divergence <- function(s, b, n_grid = 512) {
  tryCatch({
    all_v <- c(as.numeric(s), as.numeric(b))
    rng   <- range(all_v)
    pad   <- diff(rng) * 0.15 + 1e-6
    from  <- rng[1] - pad; to <- rng[2] + pad
    d_s   <- density(as.numeric(s), from = from, to = to, n = n_grid)
    d_b   <- density(as.numeric(b), from = from, to = to, n = n_grid)
    eps   <- 1e-10
    p     <- d_s$y + eps;  p <- p / sum(p)
    q     <- d_b$y + eps;  q <- q / sum(q)
    sum(p * log(p / q))
  }, error = function(e) NA_real_)
}

save_multipanel_svg <- function(test_name, stan_draws, BF_draws, out_dir) {
  params <- intersect(names(stan_draws), names(BF_draws))
  params <- params[sapply(params, function(p)
    length(stan_draws[[p]]) > 1 && length(BF_draws[[p]]) > 1)]
  if (!length(params)) return(invisible(NULL))
  ncols <- min(4L, length(params))
  nrows <- ceiling(length(params) / ncols)
  svg_path <- file.path(out_dir, paste0(test_name, ".svg"))
  tryCatch({
    svg(svg_path, width = ncols * 4, height = nrows * 3)
    par(mfrow = c(nrows, ncols), mar = c(3, 3, 2, 1))
    for (param in params) {
      s <- as.numeric(stan_draws[[param]])
      b <- as.numeric(BF_draws[[param]])
      d_s <- density(s); d_b <- density(b)
      xlim <- range(c(d_s$x, d_b$x)); ylim <- range(c(d_s$y, d_b$y))
      plot(d_s, col = "red", lwd = 1.5, main = param,
           xlim = xlim, ylim = ylim, xlab = "", ylab = "", cex.main = 0.7)
      lines(d_b, col = "blue", lwd = 1.5, lty = 2)
    }
    legend("topright", legend = c("Stan", "BF"), col = c("red", "blue"),
           lwd = 1.5, lty = c(1, 2), cex = 0.7, bty = "n")
    dev.off()
    cat("  SVG:", svg_path, "\n")
  }, error = function(e) { try(dev.off(), silent = TRUE) })
}

save_combination_log <- function(test_name, stan_draws, BF_draws, out_dir) {
  all_params <- union(names(stan_draws), names(BF_draws))
  log_path   <- file.path(out_dir, paste0(test_name, "_log.txt"))
  header <- sprintf("%-40s %12s %12s %12s %14s",
                    "Parameter", "Stan_mean", "BF_mean", "Diff", "KL(Stan||BF)")
  sep    <- paste(rep("-", 95), collapse = "")
  rows   <- c(paste0("=== ", test_name, " ==="), header, sep)
  for (param in all_params) {
    s    <- stan_draws[[param]]; b <- BF_draws[[param]]
    s_m  <- if (!is.null(s) && length(s) > 0) mean(as.numeric(s)) else NA
    b_m  <- if (!is.null(b) && length(b) > 0) mean(as.numeric(b)) else NA
    diff <- if (!is.na(s_m) && !is.na(b_m)) s_m - b_m else NA
    kl   <- if (!is.null(s) && !is.null(b) && length(s) > 1 && length(b) > 1)
              kl_divergence(s, b) else NA
    rows <- c(rows, sprintf("%-40s %12.6f %12.6f %12.6f %14.6f",
                            param,
                            ifelse(is.na(s_m), NaN, s_m),
                            ifelse(is.na(b_m), NaN, b_m),
                            ifelse(is.na(diff), NaN, diff),
                            ifelse(is.na(kl),   NaN, kl)))
  }
  cat(paste(rows, collapse = "\n"), "\n", file = log_path)
  cat("  Log:", log_path, "\n")
}

build_stan_data_bison <- function(stan_data, model_type, partial_pooling, zero_inflated, directed = TRUE) {
  BF <- list(
    num_rows = as.integer(stan_data$num_rows), event = stan_data$event,
    divisor = stan_data$divisor, dyad_ids = stan_data$dyad_ids,
    num_edges = as.integer(stan_data$num_edges), num_fixed = as.integer(stan_data$num_fixed),
    num_random = as.integer(stan_data$num_random),
    num_random_groups = as.integer(stan_data$num_random_groups),
    random_group_index = stan_data$random_group_index,
    design_fixed = stan_data$design_fixed, design_random = stan_data$design_random,
    partial_pooling = as.integer(stan_data$partial_pooling),
    zero_inflated = as.integer(stan_data$zero_inflated),
    directed = as.integer(directed),
    prior_edge_mu = as.numeric(stan_data$prior_edge_mu),
    prior_edge_sigma = as.numeric(stan_data$prior_edge_sigma),
    prior_fixed_mu = as.numeric(stan_data$prior_fixed_mu),
    prior_fixed_sigma = as.numeric(stan_data$prior_fixed_sigma),
    prior_rate_sigma = as.numeric(stan_data$prior_rate_sigma),
    prior_random_mean_mu = as.numeric(stan_data$prior_random_mean_mu),
    prior_random_mean_sigma = as.numeric(stan_data$prior_random_mean_sigma),
    prior_random_std_sigma = as.numeric(stan_data$prior_random_std_sigma),
    prior_zero_prob_alpha = as.numeric(stan_data$prior_zero_prob_alpha),
    prior_zero_prob_beta = as.numeric(stan_data$prior_zero_prob_beta)
  )
  defaults <- list(prior_rate_sigma=1.0, prior_fixed_mu=0.0, prior_fixed_sigma=2.5,
    num_fixed=0L, num_random=0L, num_random_groups=0L, partial_pooling=0L, zero_inflated=0L,
    prior_edge_mu=0.0, prior_edge_sigma=2.5, prior_random_mean_mu=0.0,
    prior_random_mean_sigma=1.0, prior_random_std_sigma=1.0,
    prior_zero_prob_alpha=1.0, prior_zero_prob_beta=1.0)
  for (nm in names(defaults))
    if (length(BF[[nm]]) == 0 || any(is.na(BF[[nm]]))) BF[[nm]] <- defaults[[nm]]
  BF
}

make_jax_data <- function(BF_data, model_type) {
  jd <- BF_data
  jd$dyad_ids <- jnp$array(as.integer(BF_data$dyad_ids), dtype = jnp$int32)
  jd$event    <- jnp$array(as.integer(BF_data$event),    dtype = jnp$int32)
  if (!is.null(BF_data$divisor) && length(BF_data$divisor) > 0)
    jd$divisor <- jnp$array(as.numeric(BF_data$divisor), dtype = jnp$float64)
  if (as.numeric(BF_data$num_fixed) > 0)
    jd$design_fixed <- jnp$array(as.matrix(BF_data$design_fixed), dtype = jnp$float64)
  else
    jd$design_fixed <- jnp$zeros(as.integer(c(BF_data$num_rows, 0L)))
  if (as.numeric(BF_data$num_random) > 0) {
    jd$design_random      <- jnp$array(as.matrix(BF_data$design_random), dtype = jnp$float64)
    jd$random_group_index <- jnp$array(as.integer(BF_data$random_group_index), dtype = jnp$int32)
  } else {
    jd$design_random      <- jnp$zeros(as.integer(c(BF_data$num_rows, 0L)))
    jd$random_group_index <- jnp$zeros(0L, dtype = jnp$int32)
  }
  jd
}

# Print actual prior values used (diagnostic)
print_priors <- function(BF_data) {
  cat("  Prior values used by BF model:\n")
  cat("    prior_edge_mu          =", BF_data$prior_edge_mu, "\n")
  cat("    prior_edge_sigma       =", BF_data$prior_edge_sigma, "\n")
  cat("    prior_fixed_mu         =", BF_data$prior_fixed_mu, "\n")
  cat("    prior_fixed_sigma      =", BF_data$prior_fixed_sigma, "\n")
  cat("    prior_random_mean_mu   =", BF_data$prior_random_mean_mu, "\n")
  cat("    prior_random_mean_sigma=", BF_data$prior_random_mean_sigma, "\n")
  cat("    prior_random_std_sigma =", BF_data$prior_random_std_sigma, "\n")
}

# Restored from ff68bb35^ (dropped in test refactor)
center_undirected_random_effects <- function(draws, BF_data) {
  if (as.numeric(BF_data$directed) == 1) return(draws)
  if (as.numeric(BF_data$num_random_groups) < 2) return(draws)
  g1_key <- "random_group_mu[1]"
  g2_key <- "random_group_mu[2]"
  if (!all(c(g1_key, g2_key) %in% names(draws))) return(draws)
  offset <- (draws[[g1_key]] - draws[[g2_key]]) / 2
  grp_idx <- BF_data$random_group_index
  for (r in seq_along(grp_idx)) {
    br_k <- paste0("beta_random[", r, "]")
    if (br_k %in% names(draws)) {
      if (grp_idx[r] == 1) draws[[br_k]] <- draws[[br_k]] - offset
      else                  draws[[br_k]] <- draws[[br_k]] + offset
    }
  }
  mu_shared <- (draws[[g1_key]] + draws[[g2_key]]) / 2
  draws[[g1_key]] <- mu_shared
  draws[[g2_key]] <- mu_shared
  draws
}

# cmdstanr's $sample() sometimes fails right after all chains finish, while reading
# the chain CSVs back ("abind ... dims=0"): one chain is read with zero draws.
# The cause is intermittent and not reproducible in isolation, so retry the fit.
sample_with_retry <- function(model, ..., max_tries = 3) {
  for (k in seq_len(max_tries)) {
    res <- tryCatch(model$sample(...), error = function(e) e)
    if (!inherits(res, "error")) return(res)
    cat(sprintf("  Stan sample failed (try %d/%d): %s\n", k, max_tries, conditionMessage(res)))
  }
  stop(res)
}
