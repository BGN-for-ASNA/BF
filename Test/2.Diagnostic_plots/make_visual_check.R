#!/usr/bin/env Rscript
# make_visual_check.R -- render the bayesplot plots of the visual gallery
# (visual_check/index.html, built by make_visual_check.py) to PNG.
#
# Usage:
#   cd /home/sosa/work/BF-ppc && Rscript Test/2.Diagnostic_plots/make_visual_check.R .
# The first (optional) argument is the repo base directory (default: getwd()).
# Reads fixtures/input.json (same arrays as make_reference.R and the tests),
# writes visual_check/r_<key>.png, one per plot in the gallery.
#
# ppc_km_overlay needs ggfortify (not installed): the KM overlay is rebuilt
# directly from survival::survfit + geom_step, mirroring make_reference.R and
# BayesForge's extrapolation cut (1.2 x max observed time).

suppressMessages({
  library(jsonlite)
  library(bayesplot)
  library(loo)
  library(ggplot2)
  library(survival)
})

args <- commandArgs(trailingOnly = TRUE)
base <- if (length(args) >= 1 && dir.exists(args[1])) args[1] else getwd()
fx_dir <- file.path(base, "Test", "2.Diagnostic_plots", "fixtures")
out_dir <- file.path(base, "Test", "2.Diagnostic_plots", "visual_check")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

d <- jsonlite::fromJSON(file.path(fx_dir, "input.json"), simplifyVector = TRUE)
y <- as.numeric(d$continuous$y)
yrep <- as.matrix(d$continuous$yrep)
log_lik <- as.matrix(d$continuous$log_lik)
x <- as.numeric(d$x)
cy <- as.integer(d$count$y)
cyrep <- as.matrix(d$count$yrep)

by <- as.integer(y > stats::median(y))          # binary outcome (parity with tests)
bp <- stats::plogis(yrep)                       # expected values in (0,1)
psis <- loo::psis(-log_lik, r_eff = 1)
lw_norm <- stats::weights(psis)                 # normalized log weights for ppc_loo_pit

save_png <- function(p, key, w = 7, h = 5) {
  f <- file.path(out_dir, paste0("r_", key, ".png"))
  ragg::agg_png(f, width = w, height = h, units = "in", res = 110)
  print(p)
  invisible(grDevices::dev.off())
  cat("wrote", f, "\n")
}

# ---- continuous ------------------------------------------------------------
save_png(ppc_intervals(y, yrep, x = x, prob = 0.5, prob_outer = 0.9), "intervals")

save_png(ppc_ribbon(y, yrep, x = x, prob = 0.5, prob_outer = 0.9), "ribbon")

# ppc_intervals_sorted has no bayesplot counterpart: rows sorted by y,
# prob = prob_outer = 0.9 (same recipe as make_reference.R).
ord <- order(y)
save_png(ppc_intervals(y[ord], yrep[, ord, drop = FALSE], x = x[ord],
                       prob = 0.9, prob_outer = 0.9), "intervals_sorted")

save_png(ppc_stat(y, yrep, stat = "mean"), "stat_mean")
save_png(ppc_stat(y, yrep, stat = "sd"), "stat_sd")
save_png(ppc_error_scatter_avg(y, yrep), "error_scatter")
save_png(ppc_error_hist(y, yrep), "error_hist")     # per-draw, faceted (BF differs!)
save_png(ppc_dens_overlay(y, yrep[1:50, , drop = FALSE]), "density")  # BF: Scott, seed-0 sample of 50
save_png(ppc_hist(y, yrep, n = 8, bins = 30), "hist")
save_png(ppc_boxplot(y, yrep, n = 20), "boxplot")
save_png(ppc_scatter(y, yrep), "scatter")   # y vs average yrep (BF differs!)

# ---- LOO-PIT -----------------------------------------------------------------
save_png(ppc_loo_pit_overlay(y, yrep, lw = lw_norm), "loo_pit")

# ---- count -------------------------------------------------------------------
save_png(ppc_bars(cy, cyrep), "bars")
save_png(ppc_rootogram(cy, cyrep), "rootogram")

# ---- binary: binned error ------------------------------------------------------
save_png(ppc_error_binned(by, bp[1:3, , drop = FALSE]), "error_binned")

# ---- survival: KM overlay (no ggfortify -> manual survfit steps) ---------------
st <- rep(c(1L, 1L, 0L), length.out = length(y))
ty <- exp(y / 2)
tyrep <- exp(yrep[1:3, , drop = FALSE] / 2)
km_df <- function(f) data.frame(time = f$time, surv = as.vector(f$surv))
km_obs <- km_df(survival::survfit(survival::Surv(ty, st) ~ 1))
km_rep <- do.call(rbind, lapply(1:3, function(s) {
  f <- survival::survfit(survival::Surv(tyrep[s, ], rep(1L, 30)) ~ 1)
  dd <- km_df(f); dd$rep <- s; dd
}))
km_rep <- km_rep[km_rep$time <= 1.2 * max(ty), ]   # BF extrapolation_factor = 1.2
save_png(ggplot() +
  geom_step(data = km_rep, aes(x = time, y = surv, group = rep),
            color = "gray50", linewidth = 0.3) +
  geom_step(data = km_obs, aes(x = time, y = surv),
            color = "#1a6fb4", linewidth = 1) +
  labs(x = "Time", y = "Survival probability",
       title = "PPC: Kaplan-Meier overlay of y (survfit, 3 reps)") +
  theme_default(), "km_overlay")

cat("done:", length(list.files(out_dir, "^r_.*\\.png$")), "PNGs in", out_dir, "\n")
