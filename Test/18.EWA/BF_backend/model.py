import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "3.Documentation"
    ),
)
from doc_blocks import doc_model

# The model as published on the documentation page (EWA-ILV, 8 varying effects:
# phi, gamma, fconf + payoff/kin/rank/cohort/YOB biases, age slopes on phi and
# gamma, Categorical likelihood over the K techniques).
# Priors are identical to the script's old copy: lambda ~ exponential(rate=1),
# mu ~ normal(0, 1)[8], sigma ~ exponential(rate=3)[8],
# L_Rho ~ lkj_cholesky(dimension=8, concentration=3), z ~ normal(0, 1)(8, J),
# b_age ~ normal(0, 1)[2]. Only the internal names of the bias coefficients
# differ (page B_pay/B_kin/B_pres/B_coho/B_yob vs old beta_payoff/beta_kin/
# beta_rank/beta_cohort/beta_age_bias); the sampled site names (lambda, mu,
# sigma, L_Rho, z, b_age, obs) and the data keys are unchanged.
PAGE_MODEL = doc_model("32. EWA.qmd", 1, name="model")


def make_model(m):
    """Return the published EWA-ILV model (Documentation/32. EWA.qmd, block 2).

    The published model closes over the bf instance built by the doc page, so it
    is fitted with the caller's own bf object (``m.fit(model=PAGE_MODEL, ...)``);
    the ``m`` argument is kept for backwards compatibility with callers.
    """

    return PAGE_MODEL

