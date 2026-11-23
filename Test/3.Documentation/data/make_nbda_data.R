# Data of the examples of "31. Network-Based Diffusion Analysis" (cTADA, OADA, ILV, veff,
# complex_f, dynamic_tweights, posterior_edges).
# Writes nbda_data.json next to this script: STbayes' example diffusion data (event_data,
# edge_list) imported with import_user_STb and prepared as in Test/12.NBDA/density_plots.R
# (build_data_list), plus one continuous individual-level variable (ILV) for the ILV models.
# Needs the R packages STbayes and jsonlite. Run from this folder: Rscript make_nbda_data.R
suppressMessages(library(STbayes))

dl <- import_user_STb(STbayes::event_data, STbayes::edge_list, network_type = "undirected")
K <- dl$K; P <- dl$P; T_max <- max(dl$T)

# Time of acquisition (or censoring) of each individual, as in build_data_list
obs_end_time  <- matrix(0L, K, P)
is_event      <- matrix(FALSE, K, P)
valid_ind     <- matrix(1, K, P)
is_event_3d   <- array(0, c(K, T_max, P))
event_at_time <- array(FALSE, c(K, T_max))
for (k in 1:K) {
  for (n in 1:dl$N[k]) {
    id <- dl$ind_id[k, n]
    if (dl$t[k, id] > 0) {
      obs_end_time[k, id] <- dl$t[k, id]; is_event[k, id] <- TRUE
      is_event_3d[k, dl$t[k, id], id] <- 1; event_at_time[k, dl$t[k, id]] <- TRUE
    } else valid_ind[k, id] <- 0
  }
  if (dl$N_c[k] > 0) for (c in 1:dl$N_c[k]) {
    id <- dl$ind_id[k, dl$N[k] + c]; obs_end_time[k, id] <- dl$T[k]
  }
}

# Individual-level variable for the ILV and complex_f examples (as inject_model_data)
set.seed(1)
ILV <- rnorm(P)

out <- list(
  K = jsonlite::unbox(as.integer(K)), P = jsonlite::unbox(as.integer(P)),
  T_max = jsonlite::unbox(as.integer(T_max)),
  N = as.integer(dl$N), N_c = as.integer(dl$N_c), T = as.integer(dl$T),
  N_networks = jsonlite::unbox(as.integer(dl$N_networks)), Q = jsonlite::unbox(as.integer(dl$Q)),
  t = dl$t, time = dl$time, D = dl$D, ind_id = dl$ind_id,
  Z = dl$Z, A = dl$A,
  obs_end_time = obs_end_time, is_event = is_event, valid_ind = valid_ind,
  is_event_3d = is_event_3d, event_at_time = event_at_time,
  ILV = ILV
)
# Sizes are scalars; the 1-trial dimensions (K = 1) stay arrays
jsonlite::write_json(out, "nbda_data.json", auto_unbox = FALSE, digits = NA, matrix = "rowmajor")
cat("wrote nbda_data.json\n")
