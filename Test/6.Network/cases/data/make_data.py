"""Simulate the SRM case data (same simulation as Documentation/26. Network model.qmd)."""
from BayesForge import bf
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pathlib import Path

m = bf(platform='cpu')
N = 50
individual_predictor = m.dist.normal(0, 1, shape=(N, 1), sample=True)        # (50, 1) float
kinship = m.dist.bernoulli(0.3, shape=(N, N), sample=True)                    # (50, 50)
kinship = kinship.at[jnp.diag_indices(N)].set(0)
alpha = m.dist.normal(0, 1, sample=True)
sr = m.net.sender_receiver(individual_predictor, individual_predictor, s_mu=0.4, r_mu=-0.4, sample=True)
DR = m.net.dyadic_effect(m.net.mat_to_edgl(kinship), d_sd=2.5, sample=True)
network = m.dist.bernoulli(logits=alpha + sr + DR, sample=True)               # (1225, 2) edge list

here = Path(__file__).parent
np.savetxt(here / 'srm_individual_predictor.csv', np.asarray(individual_predictor), delimiter=',', fmt='%.17g')
np.savetxt(here / 'srm_kinship.csv', np.asarray(kinship).astype(int), delimiter=',', fmt='%d')
np.savetxt(here / 'srm_network.csv', np.asarray(network).astype(int), delimiter=',', fmt='%d')
