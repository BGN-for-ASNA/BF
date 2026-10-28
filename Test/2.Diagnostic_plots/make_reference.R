#!/usr/bin/env Rscript
# make_reference.R -- generate fixtures/reference.json for the bayesplot
# cross-check of BayesForge's PPC diagnostic plots (Test/2.Diagnostic_plots).
#
# Usage:
#   cd /home/sosa/work/BF && Rscript Test/2.Diagnostic_plots/make_reference.R .
# The first (optional) argument is the repo base directory (default: getwd());
# fixtures are read/written under <base>/Test/2.Diagnostic_plots/fixtures/.

suppressMessages({
  for (pkg in c("jsonlite", "bayesplot", "loo", "posterior", "rstantools", "ggplot2", "survival")) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      stop(sprintf("Required package '%s' is not installed (no installs allowed).", pkg))
    }
  }
  library(jsonlite)
  library(bayesplot)   # 1.16.0
  library(loo)         # 2.10.1
})

args <- commandArgs(trailingOnly = TRUE)
base <- if (length(args) >= 1 && dir.exists(args[1])) args[1] else getwd()
fx_dir <- file.path(base, "Test", "2.Diagnostic_plots", "fixtures")
in_path <- file.path(fx_dir, "input.json")
out_path <- file.path(fx_dir, "reference.json")
if (!file.exists(in_path)) stop("missing input fixture: ", in_path)

d <- jsonlite::fromJSON(in_path, simplifyVector = TRUE)

## ---- read + dimension checks ------------------------------------------------
y <- as.numeric(d$continuous$y)
yrep <- as.matrix(d$continuous$yrep)      # rows = draws
log_lik <- as.matrix(d$continuous$log_lik)
x <- as.numeric(d$x)
stopifnot(is.matrix(yrep), is.matrix(log_lik),
          length(y) == 30L, nrow(yrep) == 200L, ncol(yrep) == 30L,
          identical(dim(log_lik), dim(yrep)), length(x) == 30L)

cy <- as.integer(d$count$y)
cyrep <- as.matrix(d$count$yrep)          # rows = draws
stopifnot(length(cy) == 30L, nrow(cyrep) == 200L, ncol(cyrep) == 30L)

by <- as.integer(y > stats::median(y))        # binary outcome (30)
bp <- stats::plogis(yrep)                     # expected values in (0,1), 200 x 30
eb <- as.data.frame(bayesplot:::ppc_error_binnned_data(by, bp[1:3, , drop = FALSE]))
error_binned <- list(rep_id = as.integer(eb$rep_id), bin = as.integer(eb$bin),
                     ey_bar = eb$ey_bar, err_bar = eb$err_bar, se2 = eb$se2)

## ---- survival: Kaplan-Meier overlay (bayesplot ppc_km_overlay) ----------------
# bayesplot::ppc_km_overlay needs ggfortify (not installed); it only wraps
# survival::survfit, which we call directly. Observed: (ty, st); each
# replication: (tyrep[s, ], all events -- predicted data carry no censoring).
st <- rep(c(1L, 1L, 0L), length.out = length(y))     # status: every 3rd censored
ty <- exp(y / 2)                                      # positive times (30)
tyrep <- exp(yrep[1:3, , drop = FALSE] / 2)           # 3 x 30
km_obs <- survival::survfit(survival::Surv(ty, st) ~ 1)
km_rep <- lapply(1:3, function(s) survival::survfit(survival::Surv(tyrep[s, ], rep(1L, 30)) ~ 1))
km <- list(obs = list(time = km_obs$time, surv = km_obs$surv, n_event = km_obs$n.event),
           rep = lapply(km_rep, function(f) list(time = f$time, surv = f$surv)))

q7 <- function(v, probs) unname(stats::quantile(v, probs = probs, type = 7))

## ---- continuous: intervals / ribbon ----------------------------------------
# bayesplot returns the data frame in y_id order (1..30); it does not reorder
# by x. Store as returned + obs_index (= y_id) so the test can align rows.
iv <- as.data.frame(bayesplot:::ppc_intervals_data(y, yrep, x = x, prob = 0.5, prob_outer = 0.9))
intervals <- list(ll = iv$ll, l = iv$l, m = iv$m, h = iv$h, hh = iv$hh,
                  obs_index = as.integer(iv$y_id))

ord <- order(y)                            # sort observations by y
ivs <- as.data.frame(bayesplot:::ppc_intervals_data(y[ord], yrep[, ord, drop = FALSE],
                                                    x = x[ord], prob = 0.9, prob_outer = 0.9))
intervals_sorted <- list(ll = ivs$ll, l = ivs$l, m = ivs$m, h = ivs$h, hh = ivs$hh,
                         order = as.integer(ord))

ribbon <- intervals                        # fallback
if (exists("ppc_ribbon_data", envir = asNamespace("bayesplot"), inherits = FALSE)) {
  rb <- as.data.frame(bayesplot:::ppc_ribbon_data(y, yrep, x = x, prob = 0.5, prob_outer = 0.9))
  ribbon <- list(ll = rb$ll, l = rb$l, m = rb$m, h = rb$h, hh = rb$hh,
                 obs_index = as.integer(rb$y_id))
}

## ---- continuous: stats, error scatter/hist, boxplot --------------------------
stat_mean_y    <- mean(y)
stat_mean_yrep <- rowMeans(yrep)           # 200
stat_sd_y      <- stats::sd(y)
stat_sd_yrep   <- apply(yrep, 1L, stats::sd)  # 200
error_scatter_x <- colMeans(yrep)          # 30
error_scatter_y <- y                       # 30
# bayesplot ppc_error_scatter_avg: x = average predictive error y - colMeans(yrep),
# y = observed y (data columns: value = average error, y_obs = observed).
esa <- as.data.frame(bayesplot::ppc_error_scatter_avg(y, yrep)$data)
stopifnot(nrow(esa) == length(y),
          max(abs(esa$value - (y - colMeans(yrep)))) < 1e-10,
          max(abs(esa$y_obs - y)) == 0)
error_scatter_avg <- list(error = as.numeric(esa$value), y_obs = as.numeric(esa$y_obs))

resid <- matrix(y, nrow = nrow(yrep), ncol = ncol(yrep), byrow = TRUE) - yrep  # y[j] - yrep[s, j]
stopifnot(identical(dim(resid), dim(yrep)))
error_hist_resid <- lapply(seq_len(nrow(resid)), function(s) as.numeric(resid[s, ]))  # 200 x 30 row-major

boxplot_q <- c(list(y = q7(y, c(0.25, 0.5, 0.75))),
               stats::setNames(lapply(1:10, function(s) q7(yrep[s, ], c(0.25, 0.5, 0.75))),
                               paste0("yrep_", 1:10)))

## ---- continuous: grouped (bayesplot ppc_stat_grouped / ppc_intervals_grouped) --
# group = rep(1:3, length.out = length(y)): deterministic, derived from the
# fixture only (the Python test uses np.arange(30) %% 3 + 1, the same vector).
group <- rep(1:3, length.out = length(y))
stopifnot(length(group) == length(y))

sd_g <- as.data.frame(bayesplot:::ppc_stat_data(y, yrep, group = group, stat = "mean"))
stopifnot(all(c("group", "variable", "value") %in% names(sd_g)))
stat_grouped <- lapply(sort(unique(group)), function(g) {
  sub  <- sd_g[sd_g$group == g, , drop = FALSE]
  reps <- sub[sub$variable != "y", , drop = FALSE]
  reps <- reps[order(as.integer(sub("yrep_", "", reps$variable))), , drop = FALSE]
  list(y_obs = sub$value[sub$variable == "y"], yrep = reps$value)
})
names(stat_grouped) <- paste0("g", sort(unique(group)))
stopifnot(all(vapply(stat_grouped, function(s) length(s$yrep) == nrow(yrep) &&
                                             length(s$y_obs) == 1L, logical(1))))

# cross-check ppc_stat_data(group=) against direct mean(y[g]) / rowMeans(yrep[, g])
diff_stat_g <- max(unlist(lapply(sort(unique(group)), function(g) {
  cols <- which(group == g)
  s <- stat_grouped[[paste0("g", g)]]
  c(abs(s$y_obs - mean(y[cols])), max(abs(s$yrep - rowMeans(yrep[, cols, drop = FALSE]))))
})))

ig <- as.data.frame(bayesplot:::ppc_intervals_data(y, yrep, group = group,
                                                   prob = 0.5, prob_outer = 0.9))
stopifnot(nrow(ig) == length(y),
          all(c("y_id", "group", "ll", "l", "m", "h", "hh", "y_obs") %in% names(ig)))
ig <- ig[order(ig$group, ig$y_id), , drop = FALSE]   # rows: group, then observation
intervals_grouped <- list(group = as.integer(ig$group), ll = ig$ll, l = ig$l,
                          m = ig$m, h = ig$h, hh = ig$hh, y_obs = ig$y_obs)

grouped <- list(stat_grouped = stat_grouped, intervals_grouped = intervals_grouped)

## ---- continuous: LOO-PIT ------------------------------------------------------
psis <- loo::psis(-log_lik, r_eff = 1)
lw_norm <- stats::weights(psis)            # loo S3 method: normalised log weights (log = TRUE default)
stopifnot(identical(dim(lw_norm), dim(yrep)))
# a PIT is a weighted CDF and must lie in [0, 1], so it needs weights that sum
# to 1 per observation: psis$log_weights are unnormalised (and w / colSums(w)
# would not normalise per column either -- R recycles down the columns).
pit_loo <- vapply(seq_len(ncol(yrep)),
                  function(i) sum(exp(lw_norm[, i]) * (yrep[, i] <= y[i])), numeric(1))
stopifnot(all(pit_loo >= 0 & pit_loo <= 1 + 1e-12))

# bayesplot:::ppc_loo_pit_data() exists but returns the KDE data frame
# (rep_id / is_rep / x / density); it has no $pit field -> use rstantools.
lp <- try(bayesplot:::ppc_loo_pit_data(y, yrep, lw = lw_norm), silent = TRUE)
pit_bayesplot <- if (!inherits(lp, "try-error") && "pit" %in% names(lp)) lp$pit else NULL

set.seed(1)                                # rstantools breaks ties with runif()
pit_rstantools <- rstantools::loo_pit(object = yrep, y = y, lw = lw_norm)

pit_plain <- colMeans(yrep <= matrix(y, nrow = nrow(yrep), ncol = length(y), byrow = TRUE))

loo_pit <- if (is.null(pit_bayesplot)) {
  list(pit_loo = pit_loo, pit_rstantools = as.numeric(pit_rstantools), pit_plain = pit_plain)
} else {
  list(pit_loo = pit_loo, pit_bayesplot = as.numeric(pit_bayesplot), pit_plain = pit_plain)
}

## ---- count: rootogram ----------------------------------------------------------
cx <- 0:max(c(cy, as.vector(cyrep)))
K <- length(cx)
obs_ct <- tabulate(match(cy, cx), nbins = K)
counts_draws <- t(apply(cyrep, 1L, function(r) tabulate(match(r, cx), nbins = K)))  # 200 x K
stopifnot(nrow(counts_draws) == 200L)
expected <- colMeans(counts_draws)
rootogram <- list(xpos = cx, observed = obs_ct, expected = expected,
                  sqrt_observed = sqrt(obs_ct), sqrt_expected = sqrt(expected))

## ---- count: bars ----------------------------------------------------------------
bx <- sort(unique(c(cy, as.vector(cyrep))))
Kb <- length(bx)
obs_b <- tabulate(match(cy, bx), nbins = Kb)
cd_b <- t(apply(cyrep, 1L, function(r) tabulate(match(r, bx), nbins = Kb)))  # 200 x Kb
cm <- t(cd_b)                                  # rows = xpos, cols = draws
yrep_median <- apply(cm, 1L, stats::median)
yrep_q05    <- apply(cm, 1L, q7, probs = 0.05)
yrep_q95    <- apply(cm, 1L, q7, probs = 0.95)
yrep_mean   <- rowMeans(cm)
bars <- list(xpos = bx, observed = obs_b, yrep_median = yrep_median,
             yrep_q05 = yrep_q05, yrep_q95 = yrep_q95, yrep_mean = yrep_mean)

## ---- cross-checks against bayesplot ----------------------------------------------
# ppc_bars_data: l/m/h = quantile(counts across draws, .05/.5/.95), type 7
bd <- as.data.frame(bayesplot:::ppc_bars_data(cy, cyrep, prob = 0.9, freq = TRUE))
stopifnot(identical(as.numeric(bd$x), as.numeric(bx)))
diff_bars <- max(abs(c(bd$l - yrep_q05, bd$m - yrep_median, bd$h - yrep_q95)))
diff_bars_obs <- max(abs(bd$y_obs - obs_b))

# ppc_intervals_data: ll/l/m/h/hh = quantile(type 7) at
# (1-prob_outer)/2, (1-prob)/2, 0.5, (1+prob)/2, (1+prob_outer)/2  (prob=0.5, prob_outer=0.9)
mine_iv <- t(apply(yrep, 2L, q7, probs = c((1 - 0.9) / 2, (1 - 0.5) / 2, 0.5, (1 + 0.5) / 2, (1 + 0.9) / 2)))
diff_iv <- max(abs(mine_iv - as.matrix(iv[, c("ll", "l", "m", "h", "hh")])))

## ---- assemble + write --------------------------------------------------------------
ref <- list(
  meta = list(bayesplot = as.character(packageVersion("bayesplot")),
              loo = as.character(packageVersion("loo")),
              R = R.version.string),
  continuous = list(
    intervals = intervals, intervals_sorted = intervals_sorted, ribbon = ribbon,
    grouped = grouped,
    stat_mean_y = stat_mean_y, stat_mean_yrep = stat_mean_yrep,
    stat_sd_y = stat_sd_y, stat_sd_yrep = stat_sd_yrep,
    error_scatter_x = error_scatter_x, error_scatter_y = error_scatter_y,
    error_scatter_avg = error_scatter_avg,
    error_hist_resid = error_hist_resid, boxplot = boxplot_q, loo_pit = loo_pit
  ),
  count = list(rootogram = rootogram, bars = bars),
  binary = list(error_binned = error_binned),
  survival = list(km = km)
)

jsonlite::write_json(ref, out_path, digits = NA, auto_unbox = TRUE, pretty = FALSE)

## ---- report ------------------------------------------------------------------------
field_len <- function(obj, prefix = "") {
  if (is.list(obj) && !is.data.frame(obj)) {
    for (nm in names(obj)) {
      v <- obj[[nm]]
      if (is.list(v) && any(vapply(v, is.list, logical(1)))) field_len(v, paste0(prefix, nm, "."))
      else if (is.list(v)) cat(sprintf("%s%s: %d x %d\n", prefix, nm, length(v), length(v[[1]])))
      else cat(sprintf("%s%s: %s\n", prefix, nm,
                       if (is.matrix(v)) paste(dim(v), collapse = "x") else length(v)))
    }
  }
}
cat("top-level keys:", paste(names(ref), collapse = ", "), "\n")
field_len(ref$continuous, "continuous.")
field_len(ref$count, "count.")
cat("meta: bayesplot", ref$meta$bayesplot, "| loo", ref$meta$loo, "|", ref$meta$R, "\n")
cat(sprintf("cross-check ppc_bars_data  max|diff| l/m/h: %.3e (obs counts: %.3e) -> %s\n",
            diff_bars, diff_bars_obs, if (diff_bars == 0 && diff_bars_obs == 0) "MATCH" else "MISMATCH"))
cat(sprintf("cross-check ppc_intervals_data max|diff| ll..hh: %.3e -> %s\n",
            diff_iv, if (diff_iv == 0) "MATCH" else "MISMATCH"))
cat(sprintf("cross-check ppc_stat_data(group=) vs mean/rowMeans max|diff|: %.3e -> %s\n",
            diff_stat_g, if (diff_stat_g == 0) "MATCH" else "MISMATCH"))
cat("grouped: levels", paste(sort(unique(group)), collapse = ","),
    "| intervals_grouped rows:", nrow(ig), "\n")
cat("ppc_loo_pit_data $pit field present:", !is.null(pit_bayesplot),
    "(used rstantools::loo_pit instead)\n")
cat("wrote", out_path, "\n")
