# BayesForge — tests en échec et pourquoi

Suivi des échecs pendant la correction de `RELEASE_AUDIT.md` et des vagues de tests
(`run_wave.sh`, snapshots `git archive HEAD` sur Server51, `~/bf_audit/waves/<vague>-<commit>`).
Chaque entrée indique si l'échec existe déjà sur `head` (copie d'avant l'audit, commit `87ce17a6`)
ou s'il est apparu pendant l'audit.

Légende du statut : **pré-existant** (échoue aussi sur head), **régression** (introduit par
l'audit), **environnement** (dépendance absente du serveur), **instable** (dépend du hasard).

Dernière mise à jour : 2026-11-14.

## Vague 1 — commit `c477610e`

Dossiers : `1.Distributions`, `12.NBDA`, `6.Network` (hors `SRM/`), `Causal & DAG (WIP)`.
Pytest : 50 + 43 + 49 + 66 passed, 0 échec.

| Test | Statut | Cause |
|---|---|---|
| `Test/6.Network/Multiplex (WIP)/SIM.py` | environnement | `ModuleNotFoundError: No module named 'rpy2'` dans le venv. `rpy2` a été installé depuis (2026-11-13) ; à relancer. |
| `Test/6.Network/Network metrics/network_metrics.py` | pré-existant | Le script se termine en rc=120 : à la fermeture de l'interpréteur, sa classe `Tee` appelle `flush()` sur un fichier déjà fermé (`ValueError: I/O operation on closed file`). Bug du script de test, pas de BayesForge ; identique sur head. |

## Test/15.Diagnostics — vérification du commit `4fd19f24` (mineures Diagnostic)

Sur dev4 : 118 passed, **10 failed, 68 errors**. Exactement les mêmes échecs sur head :
aucun n'est causé par l'audit. Cause commune : **les tests de parité sont écrits pour l'API
d'ArviZ 0.x, alors que le serveur a ArviZ 1.3.0**. Les comparaisons ArviZ de ce dossier
ne vérifient donc plus rien depuis le passage à ArviZ 1.x.

| Test | Statut | Cause |
|---|---|---|
| `Parity_Tests/test_jax_diagnostics_parity.py` : 68 erreurs (`test_ess_*`, `test_rhat_*`, `test_mcse_*`, `test_summary_column_matches_arviz`, `test_waic_*`, `test_loo_*`, `test_relative_eff_is_a_fraction`, `test_multidim_parameters_are_expanded`) | pré-existant | Fixture : `az.from_dict(posterior=...)` → `TypeError: from_dict() got an unexpected keyword argument 'posterior'` (ArviZ 1.x : `from_dict({"posterior": ...})`). |
| `Parity_Tests/test_jax_diagnostics_parity.py::test_hdi_matches_arviz` (5) | pré-existant | `az.hdi(..., hdi_prob=)` : argument renommé `prob` dans arviz-stats 1.0. |
| `Parity_Tests/test_diag_parity.py::test_manual_rhat_detects_variance_nonconvergence`, `::test_summary_mean_sd_match` | pré-existant | Même `from_dict(posterior=...)`. |
| `Parity_Tests/test_diag_parity.py::test_waic_matches` | pré-existant | `az.waic` n'existe plus dans ArviZ 1.x. |
| `API_Routing/test_diag_backend.py::test_backends_agree_on_summary` | pré-existant | `ess_tail` : 2286 (BF, quantiles 5 %/95 %) contre 2452 (ArviZ 1.x, `ci_prob` 0,89 → 11 %/89 %). Même estimateur, valeur par défaut différente ; se résout avec la bascule de `summary` sur ArviZ. |
| `API_Routing/test_diag_backend.py::test_diag_metrics_use_arviz` | pré-existant | Le test attend un `xarray.Dataset` ; `az.rhat` renvoie un `DataTree` en ArviZ 1.x. |

**Mise à jour (commit `d5efd224`, tests portés vers ArviZ 1.x) :** 195 passed, 1 failed.

| Test | Statut | Cause |
|---|---|---|
| `Parity_Tests/test_jax_diagnostics_parity.py::test_loo_pareto_k_matches_arviz_when_reff_is_supplied` | pré-existant, révélé par le portage | Vrai écart numérique : les k̂ de Pareto PSIS diffèrent jusqu'à 0,0094 (tolérance 1e-6), sur les 60 observations. La cause est la taille de la queue : `jax_diagnostics._psis_weights` prend `ceil(min(S/5, 3·sqrt(S/reff)))` (règle de Vehtari et al. et d'ArviZ 0.x, 562 tirages), arviz-stats 1.3 prend `floor(3·sqrt(S/reff))` (561). Avec la même queue, l'ajustement GPD est identique. **Résolu en `a7a3c4ef`** : BF suit la règle d'arviz-stats 1.3 (décision du 2026-11-13 : version la plus récente d'ArviZ). Test/15.Diagnostics : 241 passed, 0 échec. |

Remarque : ce dossier doit être lancé avec BayesForge importé avant arviz
(`~/bf_audit/run_diag_audit.py`). Sinon le shim `jax.interpreters.pxla` de
`BayesForge/__init__.py` échoue (`AttributeError: module 'jax.interpreters' has no attribute 'pxla'`).
C'est un défaut d'ordre d'import, à traiter avec X9 (intégration).

## Comparaison SRM — `Test/6.Network/SRM/fit_comparison.py`

| Test | Statut | Cause |
|---|---|---|
| Étape 3 (Stan `STAN2.stan`) | environnement | Les exécutables Stan compilés en local (`STAN2`, `SRM`, `SRM2`, `original_srm`) sont commités dans le dépôt. cmdstanpy les réutilise au lieu de recompiler, et ils plantent sur Server51 (`RuntimeError: Error during sampling`, retcode −1). Correctif côté test : les supprimer avant le lancement (ou ne plus les commiter). Relancé avec une recompilation. |

## Connus avant les vagues

| Test | Statut | Cause |
|---|---|---|
| `Test/4. Rethinking/TFP backend/13.Gaussian processes.py` | pré-existant | `AssertionError`, sans ligne `RESULT` ; identique sur head. |
| `Test/11.MGM and DPMM/Similarity_Test` (ARI du GMM) | instable | Script sans graine : l'ARI varie d'un lancement à l'autre. |
| tests HDI à tolérance serrée (`Test/0.Core`) | instable | Bruit de Monte-Carlo ; voir les mineures de rigueur m2 à m4 de l'audit. |

## Campagne finale — commit `fb1f0323` (Server51, `~/bf_audit/waves/final-fb1f0323`)

Pytest : 10.BNN 46, 7.Survival 12, 8.PCA 11, 11.MGM and DPMM 14, 13.Parallel 38, 0.Core 718 (+2 skipped),
1.Distributions 51, 12.NBDA 43, 6.Network 60, Causal & DAG 66, 15.Diagnostics 205, 16.SBI 34 passed, 0 échec.
4. Rethinking : tous les scripts `RESULT: PASS` sauf TFP 13 (déjà connu, pré-existant).

| Test | Statut | Cause |
|---|---|---|
| `Test/6.Network/Multiplex (WIP)/SIM.py`, `Test/9.Phylogenic analysis/run_all.py` | environnement | Le lanceur de la campagne n'avait pas `/opt/R-4.6.0/bin` dans `PATH` (`FileNotFoundError: 'Rscript'`). La comparaison BF/brms du dossier 9 a été vérifiée à part (5 modèles concordants). |
| `Test/7.Survival analysis/Survival_analysis_verify.py` | environnement | `pymc` absent du venv. |
| `Test/10.BNN/code/BNN covariance.py`, `BNN regressions.py` | pré-existant | Les scripts écrivent dans `results/` et lisent `data/BNN.json`, absents du dépôt. Avec `results/` créé, `BNN covariance.py` passe. |
| `Test/14.SampledData.py` | pré-existant | Nom de fichier non importable par pytest (`No module named '14'`). Lancé sous un autre nom : 39 passed, 2 failed (`test_setitem`, `test_isinstance_reports_as_jax_array`), identiques sur head. |
| `Test/14.Jax extension (WIP)/test_sampleddata.py` | pré-existant | `from jax.experimental import enable_x64` n'existe plus dans jax 0.11. |
| `Test/17.Repertoire/run_all.py`, `scripts/benchmark.py` | environnement | Exécutable Stan `legacy/cg_vocal_repertoires/model0` commité, sans bit exécutable dans le snapshot (`retcodes=[-1, -1]`). Après sa suppression (recompilation), les deux passent, comme sur head. |
| `Test/17.Repertoire/scripts/save_stan_samples.py` | pré-existant | Cherche `scripts/cg_vocal_repertoires/model0.stan`, qui est dans `legacy/`. |
| `Test/6.Network/Network metrics/network_metrics.py` | pré-existant | rc=120 (classe `Tee`), déjà décrit en vague 1. |

## Comparaison SRM sur le code corrigé — commit `24094d0e`

BF (format large), BF (matrice), Stan bloc, Stan original et STRAND concordent. Écart maximal
|Δmoyenne| / SD moyen de BF : 0,059 (Stan original) à 0,136 (STRAND) ; Stan contre STRAND : 0,099 ;
BF matrice contre BF large : 0,193 (bruit MCMC). Pas de divergence inter-librairies.

## Exemples de la documentation — `Test/3.Documentation` (Python, R, Julia)

Avant le 2026-11-14, les tests R et Julia étaient toujours sautés sur Server51 (pas de package R
BayesForge, pas de `julia`). Installés : package R depuis `BFR/` dans `~/R/lib-4.6`, `tidyverse`,
Julia 1.13.4 (juliaup) avec le projet `~/bf_audit/julia_docs` (PyBayesForge, PythonCall, DataFrames,
CSV, MCMCChains, Distributions).

Piège d'environnement : `~/.Renviron` de Server51 fixe `PYTHONPATH=/home/sebastian_sosa/BF` (ancienne
copie, commit `7b5f97a`), qui écrase le `PYTHONPATH` des tests : le premier run R testait donc cette
ancienne copie (15 faux échecs). Les lanceurs utilisent maintenant `R_ENVIRON_USER=~/bf_audit/empty.Renviron`.

| Page | Statut | Cause et correctif |
|---|---|---|
| 24. GMM, Build in models (Python, Julia) | régression (audit) | `plot_gmm` associait 300 tirages de responsabilités à 4000 tirages de `mu`/`cov`. Corrigé en `dbd1a8a5`. |
| 27, 28. Network block model (Python, Julia) | régression (audit) | Contrôles d'entrée en `jnp` évalués pendant le traçage de `m.fit` (`ConcretizationTypeError`). Passés en numpy en `dbd1a8a5`. |
| 4. Categorical variable (Python, Julia) | régression (audit) | `m.index()` ne renomme plus toutes les colonnes (M5) ; `kcal_per_g` n'existait plus. `m.data()` normalise les noms au chargement en `895bde2b`. |
| Sampling & plots (Python, Julia) | régression (audit) | `hdi()` refusait les tableaux 2-D. Intervalle par colonne en `78fa0780`. |
| 38. Approximate Bayesian Computation (Python) | pré-existant | La boucle PMC simulait des propositions avec `s < 0`. Exemple corrigé en `4eab5bae`. |
| 6. Beta binomial, 20. Measuring error (3 langages) | régression (ce jour) | Accès direct `m.df["applicant.gender"]`, `m.df["Divorce SE"]` après la normalisation des noms. Pages corrigées en `d86ae999`. |
| 11, 12, 8, gaussian_copula (Julia) ; gaussian_copula (R) | environnement | `Distributions.jl` et `tidyverse` absents ; installés. |

Note de comportement : depuis `895bde2b`, `m.df` contient les noms normalisés (`applicant_gender`).
Les méthodes de `m` acceptent encore l'ancien nom, mais un accès pandas direct `m.df["applicant.gender"]`
lève `KeyError`.

## Validation finale — commit `d86ae999` (2026-11-14)

- `Test/3.Documentation` : Python 111 passed, R 55 passed, Julia 55 passed, 0 échec. Le seul skip
  de chaque suite est une page sans bloc de code dans ce langage.
- `Test/0.Core`, `6.Network`, `12.NBDA`, `11.MGM and DPMM`, `5. Nested effect` : 852 passed, 2 skipped.
  Les 2 skips sont dans `Test/0.Core/test_run_simulations_languages.py` : `Rscript not found` (R hors
  du `PATH` de ce run) et `PyBayesForge.jl with run_simulations not installed` (Julia trouvé, mais
  l'environnement Julia par défaut n'a pas PyBayesForge). Relancés avec R et `BF_JULIA_PROJECT` : 2 passed.
