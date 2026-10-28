# Data of the examples of "29. Network Edges Inference" (binary, count and duration edge models).
# Writes network_edges_binary.json, network_edges_count.json and network_edges_duration.json next
# to this script: the `data` list of BF_model_binary, BF_model_count and BF_model_duration.
# Simulated with bisonR, as in the validation tests of Test/6.Network/Network edges (20 nodes,
# seed 42); the model data come from bisonR's get_bison_model_data, so no Stan fit is needed.
# Needs the R packages bisonR and jsonlite. Run from this folder: Rscript make_network_edges_data.R
library(bisonR)

source(file.path("..", "..", "6.Network", "Network edges", "R_scripts", "Modified simulate_bison_model.R"))
assignInNamespace("simulate_bison_model", simulate_bison_model, ns = "bisonR")
get_bison_model_data <- get("get_bison_model_data", envir = getNamespace("bisonR"))

formula <- as.formula(
  "(event | duration) ~ dyad(node_1_id, node_2_id) + age_diff + (1 | node_1_id) + (1 | node_2_id)")

simulate <- function(model_type) {
  set.seed(42)
  simulate_bison_model(model_type, aggregated = TRUE,
    location_effect = TRUE, age_diff_effect = TRUE,
    num_nodes = 20L, num_locations = 5, max_obs = 10)$df_sim
}

# Priors and design of the binary and count models (zero-inflated, partial pooling),
# as build_stan_data_bison in Test/6.Network/Network edges/R_scripts/util.R
edge_data <- function(model_type) {
  md <- get_bison_model_data(formula = formula, observations = simulate(model_type),
                             directed = TRUE, model_type = model_type)$model_data
  out <- list(
    num_rows = as.integer(md$num_rows), event = md$event,
    divisor = md$divisor, dyad_ids = as.integer(md$dyad_ids),
    num_edges = as.integer(md$num_edges), num_fixed = as.integer(md$num_fixed),
    num_random = as.integer(md$num_random),
    num_random_groups = as.integer(md$num_random_groups),
    random_group_index = as.integer(md$random_group_index),
    design_fixed = data.matrix(md$design_fixed), design_random = data.matrix(md$design_random),
    partial_pooling = 1L, zero_inflated = 1L, directed = 1L,
    prior_edge_mu = md$prior_edge_mu, prior_edge_sigma = md$prior_edge_sigma,
    prior_fixed_mu = md$prior_fixed_mu, prior_fixed_sigma = md$prior_fixed_sigma,
    prior_rate_sigma = md$prior_rate_sigma,
    prior_random_mean_mu = md$prior_random_mean_mu,
    prior_random_mean_sigma = md$prior_random_mean_sigma,
    prior_random_std_sigma = md$prior_random_std_sigma,
    prior_zero_prob_alpha = md$prior_zero_prob_alpha,
    prior_zero_prob_beta = md$prior_zero_prob_beta
  )
  defaults <- list(prior_rate_sigma = 1.0, prior_fixed_mu = 0.0, prior_fixed_sigma = 2.5,
    prior_edge_mu = 0.0, prior_edge_sigma = 2.5, prior_random_mean_mu = 0.0,
    prior_random_mean_sigma = 1.0, prior_random_std_sigma = 1.0,
    prior_zero_prob_alpha = 1.0, prior_zero_prob_beta = 1.0)
  for (nm in names(defaults))
    if (length(out[[nm]]) == 0 || any(is.na(out[[nm]]))) out[[nm]] <- defaults[[nm]]
  out
}

# Duration model: durations with the number of events of each edge (event_count),
# as build_stan_data_duration in the duration validation tests
duration_data <- function() {
  df <- simulate("duration")
  md <- get_bison_model_data(formula = formula, observations = df,
                             directed = TRUE, model_type = "count")$model_data
  ne <- as.integer(md$num_edges); nr <- as.integer(md$num_rows)
  ec <- integer(ne)
  dfc <- if (!is.null(df$event_count)) as.integer(df$event_count) else rep(1L, nrow(df))
  for (i in seq_len(nr)) ec[as.integer(md$dyad_ids[i])] <- ec[as.integer(md$dyad_ids[i])] + dfc[i]
  list(
    num_rows = nr, num_edges = ne,
    num_fixed = as.integer(md$num_fixed), num_random = as.integer(md$num_random),
    num_random_groups = as.integer(md$num_random_groups),
    event = as.numeric(md$event),
    event_count = pmax(1L, ec), divisor = as.numeric(md$divisor),
    dyad_ids = as.integer(md$dyad_ids),
    design_fixed = data.matrix(md$design_fixed), design_random = data.matrix(md$design_random),
    random_group_index = as.integer(md$random_group_index),
    partial_pooling = 1L, zero_inflated = 0L, directed = 1L,
    prior_edge_mu = 0.0, prior_edge_sigma = 2.0, prior_fixed_mu = 0.0, prior_fixed_sigma = 1.0,
    prior_rate_sigma = 1.0, prior_random_mean_mu = 0.0, prior_random_mean_sigma = 1.0,
    prior_random_std_sigma = 1.0, prior_zero_prob_alpha = 1.0, prior_zero_prob_beta = 1.0
  )
}

write <- function(lst, path) {
  jsonlite::write_json(lst, path, auto_unbox = TRUE, digits = NA, matrix = "rowmajor")
  cat("wrote", path, "\n")
}
write(edge_data("binary"), "network_edges_binary.json")
write(edge_data("count"), "network_edges_count.json")
write(duration_data(), "network_edges_duration.json")
