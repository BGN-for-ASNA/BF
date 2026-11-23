# BayesForge — Audit de release

**Périmètre** : package `BayesForge` v0.0.60, racine `/home/sosa/work/BF` — 78 fichiers `.py`, ≈ 42 500 lignes. Un rapport dédié par module/sous-package (19 rapports, section 1), audit des interfaces inter-modules (section 2), validation empirique et couverture par `Test/` (section 3).
**Méthode** (spec `/home/sosa/AUDIT_SPEC.md`, Volet A) : un relecteur dédié par module, lecture **intégrale ligne par ligne** de chaque fichier, vérification des formules mathématiques, des opérations matricielles/tensorielles (broadcasting, shapes), du maniement des PRNG, de la gestion d'erreurs et des cas limites. Chaque constat est noté Sévérité (Critique / Majeure / Mineure) avec `fichier:ligne` exact, explication du risque et scénario déclencheur concret. Les constats marqués ✅/« vérifié » ont été reproduits par exécution en lecture seule.
**Environnement de vérification** : Python 3.14, JAX 0.11.2 (`jax_enable_x64=False`), numpyro 0.22.0, pandas 3.0.6, numpy 2.3.5, ArviZ 1.3.0, CPU ; `tensorflow_probability` **non installé** (volet TFP audité statiquement).
**Aucun code source n'a été modifié** ; seuls les rapports `/tmp/opencode/audit-reports/*.md` ont été écrits.

---

## Verdict d'ensemble

**Volet 1 (modules) : 540 constats — 68 critiques, 158 majeures, 314 mineures.**
**Volet 2 (interfaces inter-modules, section 2) : 75 vulnérabilités d'intégration — 14 critiques, 35 majeures, 26 mineures — sur 102 coutures (32 cassées).**
**Volet 3 (validation empirique & couverture test, section 3) : 26 constats de rigueur des tests (4 critiques) + matrice 82 critiques × tests (0 ne ferait rougir un test ; 87,8 % non couverts) + 22 sondes de torture exécutées sur le cluster (15 confirmées, 1 infirmée, 3 saines, X1 reclassé, 1 découverte).**
**Total audit : 641 constats, 86 critiques.**

| # | Module | Lignes | Critique | Majeure | Mineure | Total | Verdict de release |
|---|--------|-------:|---------:|--------:|--------:|------:|--------------------|
| 01 | Distributions/np_dists.py | 8 794 | 2 | 6 | 10 | 18 | 2 corrections bloquantes (contrat `seed`) |
| 02 | Distributions/tfp_dists.py | 4 729 | 8 | 11 | 17 | 36 | **NON PRÊT** — 2 fonctions plantent, 3 lois fausses |
| 03 | Samplers | 1 630 | 3 | 12 | 34 | 49 | numpyro/SBI publiable avec correctifs ; TFP expérimental |
| 04 | Models | 3 701 | 5 | 19 | 43 | 67 | **SRM bloqué** ; surv.py:1160 à corriger |
| 05 | Main | 2 247 | 3 | 11 | 29 | 43 | **Non prêt** (log_prob, mutation X, no-op backend) |
| 06 | Diagnostic | 8 034 | 2 | 11 | 13 | 26 | Surface de plots ArviZ 1.x morte ; stats TFP fausses |
| 07 | Utils | 2 391 | 5 | 8 | 24 | 37 | 5 bloquants (2 NameError, LKJcor, hdi, covariance) |
| 08 | PostModel | 1 985 | 1 | 5 | 20 | 26 | **Bloqueur** (hyperpriors fabriqués affichés) |
| 09 | Network | 4 286 | 4 | 9 | 18 | 31 | C1–C4 + M7 bloquants (chemins SRM/priors) |
| 10 | Parallel | 943 | 1 | 7 | 12 | 20 | Perte totale de résultats sur erreur worker |
| 11 | NBDA | 735 | 7 | 13 | 15 | 35 | **NON PUBliable** (priors fausses, i/j inversés) |
| 12 | BNN | 600 | 2 | 8 | 16 | 26 | `__call__` et broadcasting `cov` cassés |
| 13 | Build | 424 | 7 | 9 | 7 | 23 | **Classer non livrables** (générateurs périmés) |
| 14 | Causal | 382 | 2 | 5 | 12 | 19 | Outil pédagogique ; non libérable en API publique |
| 15 | Resources | 667 | 3 | 7 | 11 | 21 | **Bloquant** (3 chargeurs cassés par défaut) |
| 16 | ML | 357 | 2 | 4 | 8 | 14 | **Bloquant** (seed factice, `fit()` mort) |
| 17 | Data | 299 | 8 | 6 | 12 | 26 | **NON PRÊT** (OHE corrompt `m.df`, NaN→index −1) |
| 18 | SetDevice | 199 | 1 | 3 | 8 | 12 | Cas nominal seul ; no-op silencieux dès le 2ᵉ `bf()` |
| 19 | R | 64 | 2 | 4 | 5 | 11 | Sortir du paquet ou refondre le pipeline |
| — | **Total** | **42 467** | **68** | **158** | **314** | **540** | |

### Thèmes transverses (constats confirmés dans plusieurs modules)

1. **Reproductibilité brisée par les graines** — `seed=True → time.time_ns()` (np_dists), graine `0` fixe par défaut (tfp_dists), `random_state` totalement ignoré (ML/KMeans), `PRNGKey(0)` en dur (Samplers/simulate.py:268), `seed_globals` qui re-seed `np.random`/`random` globaux (Utils/rng.py:98-108 + Main), configuration matérielle dépendante de l'ordre des `bf()` (SetDevice C1). Une étude de sensibilité à la graine ne mesure rien dans plusieurs chemins.
2. **Échecs silencieux — le mode de défaillance le plus dangereux** — backend inconnu = no-op (main.py:815-836), `log_prob` sans données (main.py:1783), `pd_to_jax` qui retourne une chaîne d'erreur injectée dans `**data_on_model` (Data C6), `OHE` qui duplique les colonnes de `m.df` (Data C3), NaN encodés en index `-1` de groupe (Data C8), betweenness faux sur graphe déconnecté (Network C1), plots ArviZ morts (Diagnostic 22/23), postérieurs TFP mal étiquetés (Samplers). Beaucoup produiraient des **postérieurs faux sans message d'erreur**.
3. **Backend TFP à garder expérimental** — critiques dans les trois modules qui le touchent (tfp_dists 8, mcmc_tfp 3, Diagnostic stats TFP décalées) + x64 forcé à False (main.py:297) + TFP absent de l'environnement de test : le contrat « changer de backend sans changer de loi » n'est pas tenu.
4. **Code généré déconnecté de son générateur** — `Build/` produit un fichier non compilable corrigé à la main (Ctrl+H documenté), les correctifs manuels ne sont pas tous appliqués (tfp_dists C1/C2), et la production a quitté le générateur : toute régénération écraserait des correctifs ou régénérerait les bugs connus.
5. **float32 silencieux** — cartes de dtype promettant float64/int64 tronquées sans x64 (Data M3), float64→float32 muet (ML m4, Resources M-3), SVI qui force la descente de tout le processus (Main M-07), x64 changé « à chaud » par `float_precision` (SetDevice m5).
6. **Packaging** — la roue 0.0.60 embarque 681 fichiers de `worktrees/` (~9,5 Mo) et le module mort `R/BF doc.py` ; `MANIFEST.in` périmé ; `Data/` sans `__init__.py` ; données de test orphelines (NBDA_data.json 24 Mo).
7. **ArviZ 1.3.0** — la surface de tracé legacy de `Diag.py` est morte (calls `plot_*` rejetés), `plot_compare` plante systématiquement : vérifié en exécution.

### Priorité de correction proposée

- **P0 (postérieurs faux silencieux)** : Main M-01/M-02/M-03 ; Data C3/C6/C8 ; NBDA C1/C2/C4 ; Network C1/C4 + M7 ; Models SRM C1–C4 + surv.py:1160 ; tfp_dists C3–C8 ; Diagnostic 1 ; PostModel C1 ; Utils LKJcor/hdi/covariance.
- **P1 (crashs garantis sur API publiées)** : Data C1/C2/C4/C5/C7 ; Resources C-1/C-2/C-3 ; ML C1/C2 ; Utils NameError ; BNN CRIT-1/2 ; Diagnostic 22/23 ; Parallel C1.
- **P2 (reproductibilité)** : graines (np/tfp/ML/SetDevice/seed_globals) ; packaging (worktrees, R/, MANIFEST).

---

## 1. Audit des Modules

Rapports complets par module, dans l'ordre d'exécution des passes. Chaque rapport : constats avec sévérité/`fichier:ligne`/risque/scénario, puis résumé, fragilités architecturales et décompte.

### Module Distributions — np_dists.py

**Périmètre** : `/home/sosa/work/BF/BayesForge/Distributions/np_dists.py` (8 794 lignes), lecture intégrale ligne par ligne (tranches de ~1 000 lignes), complétée par des passes structurelles automatisées (AST, grep) et des vérifications JAX/NumPyro exécutées localement (jax 0.11.2, numpyro 0.22.0). Aucune modification du code source.

**Contexte** : le module expose la classe `UnifiedDist` (L175) et son proxy `DistProxy` (L122), qui encapsulent ~90 méthodes de distribution quasi identiques (constructeur numpyro → branche `sample=True` avec clé PRNG + masque NaN → branche modèle avec `expand`/`to_event`/`create_obj` → `numpyro.sample` sous `handlers.mask`). Les méthodes vérifiées par AST forwardent toutes correctement leurs paramètres au constructeur numpyro sous-jacent, `validate_args` est propagé partout, et le raccourci `enum=True` est réservé à bon escient aux 16 distributions discrètes à support fini (les 14 discrètes qui ne l'ont pas sont à support infini ou continues — ce n'est pas un défaut). Les bugs relevés sont donc majoritairement **systémiques** (dupliqués ~90 fois) ou documentaires.

---

#### Constatations

## Sévérité : Critique

**C1 — `seed=False` produit une clé PRNG fixe : tous les tirages sont identiques** `[RÉSOLU 87ce17a6]`
- Emplacement : `BayesForge/Distributions/np_dists.py:L211-L224` (`normalize_seed`), utilisé par `_draw_key` L198-L209.
- Risque : dans `normalize_seed`, un booléen `False` tombe dans la branche `else` (L218-L219) et retourne `0`, donc `random.PRNGKey(0)`. Un utilisateur qui passe `seed=False` (en pensant « pas de graine / aléatoire ») obtient au contraire la clé fixe 0 : **chaque appel avec `seed=False` renvoie exactement les mêmes valeurs**, y compris à travers les méthodes et les sessions. C'est une erreur statistique silencieuse — rien n'avertit que le tirage est dégénéré.
- Scénario déclencheur : `for i in range(100): draws.append(m.dist.normal(0, 1, sample=True, seed=False))` → 100 valeurs identiques ; ou deux variables différentes `m.dist.normal(0,1,sample=True,seed=False)` et `m.dist.exponential(1,sample=True,seed=False)` tirent depuis la même clé 0 et sont donc corrélées de façon inattendue.

**C2 — Tirage par défaut non reproductible : `rand_seed=True` → `time.time_ns()`** `[RÉSOLU 87ce17a6]`
- Emplacement : `BayesForge/Distributions/np_dists.py:L178-L179` (`__init__(self, rand_seed=True, ...)`), `L216-L217` (`return int(time.time_ns())`), `L207-L209` (`_draw_key`).
- Risque : quand aucune graine explicite n'est passée et qu'aucun gestionnaire RNG central n'est branché (`self._rng is None`, défaut du constructeur L180), la graine vient de l'horloge. Le docstring de la méthode `seed` (copié dans les ~90 méthodes, ex. L8651) promet pourtant : « By default (`None`) the draw comes from the session's random stream … so the whole sequence is reproducible ». Cette promesse n'est tenue que si le noyau `bf` injecte un `rng` ; au niveau du module, le comportement par défaut est **non reproductible**, et un même script exécuté deux fois donne des tirages différents.
- Scénario déclencheur : `m = bf()` sans gestionnaire RNG attaché à `UnifiedDist` ; `m.dist.normal(0, 1, sample=True, shape=(1000,))` exécuté deux fois → deux jeux de draws différents, résultats de noteboook impossibles à republier, alors que la documentation du paramètre `seed` affirme le contraire.

## Sévérité : Majeure

**M1 — Masquage d'échantillons discrets par NaN : promotion de dtype int→float32 et valeurs invalides (dupliqué ~90 fois)** `[RÉSOLU d47ba577]`
- Emplacement : `BayesForge/Distributions/np_dists.py:L8768-L8769` (exemple, `negative_binomial` ; motif `jnp.where(mask, samples, jnp.nan)` présent 90 fois, ex. L5954 `poisson`, L8582, L8683).
- Risque : vérifié empiriquement, `jnp.where(mask, samples_int, jnp.nan)` promeut les échantillons entiers des lois discrètes en `float32` et injecte `NaN` — une valeur qui n'appartient pas au support d'une loi de comptage. L'utilisateur qui masque des observations en mode `sample=True` récupère un tableau de flottants avec des `NaN` là où il attend des comptes entiers.
- Scénario déclencheur : `m.dist.poisson(rate=5, sample=True, shape=(100,), mask=keep)` puis passage du résultat comme `obs` d'un modèle ou comme `total_count` d'une loi binomiale → échec de validation du support ou, pire, acceptation silencieuse de flottants ; les dtype `int32` d'origine sont perdus.

**M2 — Deux sémantiques incompatibles du paramètre `mask` selon le mode** `[RÉSOLU d47ba577]`
- Emplacement : `BayesForge/Distributions/np_dists.py:L8784-L8791` (mode modèle : `numpyro.handlers.mask`) vs `L8768-L8769` (mode `sample=True` : NaN dans les draws) ; `mask` statique L243-L257.
- Risque : en mode modèle, `mask` met à zéro la contribution log-prob (comportement numpyro correct) ; en mode `sample=True`, il remplace des tirages par `NaN`. Le même argument documenté de façon identique (« Optional boolean array to mask observations ») produit donc deux comportements sans rapport. De plus, `handlers.mask` appliqué à un site **latent** (sans `obs`) annule aussi le terme de prior de ce site — le docstring du `mask` statique (L246-L249) ne parle pourtant que d'« exclure des observations de la likelihood ».
- Scénario déclencheur : un utilisateur masque des points manquants avec `mask` sur le site latent d'un effet aléatoire pour « ignorer » certaines coordonnées ; en réalité tout le terme a priori du site disparaît de la densité, et le posterior des paramètres globaux change sans avertissement.

**M3 — `sine_bivariate_vonmises` hors gabarit : `mask` documenté mais ignoré en mode `sample=True`, pas de `to_jax`** `[RÉSOLU d47ba577]`
- Emplacement : `BayesForge/Distributions/np_dists.py:L6321` (signature, seule des 90 méthodes sans `to_jax`), `L6347` (le docstring documente `mask`), `L6405-L6407` (branche `sample` : `return d.sample(seed_key, sample_shape=shape)` sans appliquer le masque ni `SampledData`).
- Risque : le paramètre `mask` est accepté silencieusement et jeté quand `sample=True` ; `to_jax=False` n'existe pas, donc le retour `SampledData` est impossible pour cette méthode. C'est une divergence d'API invisible à l'exécution.
- Scénario déclencheur : `m.dist.sine_bivariate_vonmises(..., sample=True, mask=keep, to_jax=False)` → `TypeError` inattendu pour `to_jax` ; avec `mask=keep` passé correctement, l'utilisateur reçoit des draws complets non masqués sans le moindre avertissement.

**M4 — `improper_uniform(sample=True)` lève toujours `NotImplementedError` alors que le docstring le promet** `[RÉSOLU d47ba577]`
- Emplacement : `BayesForge/Distributions/np_dists.py:L3510` (signature), `L3541` (Returns : « A JAX array of samples drawn from the ImproperUniform distribution »), `L3559-L3562` (`d.sample(...)` sur `ImproperUniform`).
- Risque : vérifié avec numpyro 0.22, `ImproperUniform.sample` lève `NotImplementedError` (message vide). La branche `sample=True` est du code mort qui ne peut que crasher, et le docstring vend explicitement ce mode.
- Scénario déclencheur : `m.dist.improper_uniform(constraints.interval(0, 1), (), (), sample=True)` → `NotImplementedError` avec message vide, aucune piste sur la cause (la loi est impropre par construction, donc non échantillonnable).

**M5 — Inférence de nom par inspection de bytecode et frames : fragile, dépendante de CPython, avec cache global mutable** `[RÉSOLU d47ba577]`
- Emplacement : `BayesForge/Distributions/np_dists.py:L27-L63` (`_infer_caller_name`), `L67` (`_OBS_NAMES` global), `L70-L102` (`_infer_obs_name`), `L122-L173` (`DistProxy`).
- Risque : l'inférence suppose une profondeur de frame exacte (`sys._getframe(2)`) et s'appuie sur les offsets du bytecode (`f_lasti`, opcode suivant le `CALL`) ; le docstring lui-même (L37-L40) admet que le comportement change entre versions de CPython (inline-cache 3.12). Sous tout autre wrapper, compilateur ou exécution optimisée, l'inférence se trompe silencieusement ou échoue. Le cache `_OBS_NAMES` est un dict **module** (L67) partagé par toutes les instances `bf`, indexé par `(code, f_lasti)` : sans bornes, il conserve les objets `code` indéfiniment et peut attribuer le nom d'un site d'une instance à une autre.
- Scénario déclencheur : l'utilisateur appelle `m.dist.normal(...)` dans un helper décoré ou un lambda → le nom inféré est celui d'une variable du wrapper, deux sites différents portent le même nom et numpyro lève une collision, ou pis, un site est écrasé silencieusement ; en service long avec modèles générés dynamiquement, `_OBS_NAMES` croît sans limite.

**M6 — Docstring `half_normal` : la formule affichée est celle de la loi de Cauchy, pas de la HalfNormal** `[RÉSOLU 1bc8055b]`
- Emplacement : `BayesForge/Distributions/np_dists.py:L3439-L3441` : `f(x; σ) = 2/(π σ) · 1/(1+(x/σ)²), x ≥ 0`.
- Risque : la vraie densité est `√(2/π)/σ · exp(−x²/2σ²)`. La formule montrée est la demi-Cauchy normalisée. Dans une bibliothèque dont la valeur ajoutée est précisément la documentation mathématique embarquée, une formule fausse est une erreur de spécification : l'utilisateur qui implémente un facteur custom (`dist.factor`) d'après le docstring introduit une erreur statistique réelle.
- Scénario déclencheur : l'utilisateur code un prior half-normal « custom » via `m.dist.factor` en recopiant la formule du docstring → densité incorrecte, posterior biaisé, sans aucun lien visible avec la doc.

## Sévérité : Mineure

**m1 — LKJ : nom des auteurs inventé** [RÉSOLU 990053cc]
- Emplacement : `BayesForge/Distributions/np_dists.py:L3849` : « The LKJ (Leonard-Kjærgaard-Jørgensen) Cholesky distribution ».
- Risque : LKJ = **Lewandowski–Kurowicka–Joe**. Le docstring oriente mal toute recherche bibliographique et décrédibilise le reste de la documentation.
- Scénario déclencheur : l'utilisateur cite la référence dans un papier et ne la trouve nulle part.

**m2 — `lower_truncated_power_law` : contrainte sur `alpha` contradictoire avec la formule** [RÉSOLU 990053cc]
- Emplacement : `BayesForge/Distributions/np_dists.py:L4620` (« where α > 1 ») vs `L4624` (« `alpha` … Must be less than -1 »).
- Risque : les deux énoncés s'excluent ; numpyro exige `alpha > 1`. Un utilisateur qui suit la ligne Args passe un alpha négatif et obtient une erreur de validation obscure.
- Scénario déclencheur : `m.dist.lower_truncated_power_law(alpha=-2, low=1.0, ...)` → échec `validate_args` ou NaN, avec la doc comme alibi.

**m3 — Paramètre `scale` documenté sous le nom `*sample*`** [RÉSOLU 990053cc]
- Emplacement : `BayesForge/Distributions/np_dists.py:L440` (`asymmetric_laplace_quantile`) et `L8441` (`truncated_normal`) : « - *sample* (float): The scale parameter ».
- Risque : le nom du paramètre documenté (`sample`) est celui d'un autre paramètre existant (booléen de contrôle) ; la doc des Args est silencieusement fausse pour `scale`, qui n'apparaît nulle part.
- Scénario déclencheur : l'utilisateur cherche le paramètre d'échelle de `truncated_normal` dans la doc générée, voit `sample (float)` et passe `sample=2.0` → bascule en mode tirage direct au lieu de fixer l'échelle.

**m4 — Docstring `improper_uniform` pollué par la documentation de TruncatedDistribution** [RÉSOLU 990053cc]
- Emplacement : `BayesForge/Distributions/np_dists.py:L3531-L3533` : les références « [1] Truncated distribution », « [2] Greene », « [3] LaplacesDemon dist.Truncated » apparaissent dans le docstring d'`improper_uniform`, et la section Args est tronquée.
- Risque : copier-coller non nettoyé ; la doc rendue pour cette méthode mélange deux distributions.
- Scénario déclencheur : lecture de `help(m.dist.improper_uniform)` → références sans rapport, Args amputés (`obs`, `name`, `seed` manquants avant la pollution).

**m5 — Mentions résiduelles « BernoulliLogits » dans les Returns d'autres méthodes (11 occurrences)** [RÉSOLU 990053cc]
- Emplacement : `BayesForge/Distributions/np_dists.py` — 11 fois « JAX array of samples drawn from the BernoulliLogits distribution » dans les sections Returns de méthodes qui ne sont pas `bernoulli_logits`.
- Risque : contamination de gabarit ; la doc générée affiche le mauvais nom de distribution.
- Scénario déclencheur : doc HTML de `geometric` ou `poisson` affichant « drawn from the BernoulliLogits distribution ».

**m6 — `gaussian_copula_beta` : seul défaut `validate_args=False` (les 90 autres sont `None`)** [RÉSOLU 990053cc]
- Emplacement : `BayesForge/Distributions/np_dists.py:L2755`.
- Risque : incohérence de gabarit ; cette méthode désactive la validation par défaut alors que partout ailleurs elle suit le réglage global numpyro. Un `concentration` négatif passe sans erreur ici mais est attrapé ailleurs.
- Scénario déclencheur : `m.dist.gaussian_copula_beta(concentration1=-1, ...)` construit un objet invalide sans avertissement, l'erreur n'apparaît que plus loin dans l'échantillonneur.

**m7 — Imports inutilisés en tête de fichier** [RÉSOLU 990053cc]
- Emplacement : `BayesForge/Distributions/np_dists.py:L1-L2` : `import numpyro.distributions as dist` et `import matplotlib.pyplot as plt` — `plt` n'est jamais utilisé (0 occurrence de `plt.`), `dist.` jamais utilisé hors docstrings.
- Risque : `matplotlib` est une dépendance lourde importée à chaque chargement du module (coût au warm-up JAX, environnement headless), sans raison.
- Scénario déclencheur : import de `BayesForge.Distributions` dans un conteneur sans backend matplotlib ou avec un `matplotlib.use()` tardif → configuration du backend déclenchée trop tôt.

**m8 — Code mort dans `_infer_caller_name`** [RÉSOLU 990053cc]
- Emplacement : `BayesForge/Distributions/np_dists.py:L63` : `return None` unreachable après le `try/except` qui retourne dans toutes les branches.
- Risque : cosmétique ; signale que la fonction a été modifiée sans relecture.
- Scénario déclencheur : aucun à l'exécution ; entretien seulement.

**m9 — Troncature silencieuse 32 bits des graines entières** [RÉSOLU 990053cc]
- Emplacement : `BayesForge/Distributions/np_dists.py:L209` (`random.PRNGKey(self.normalize_seed(seed))`) avec `normalize_seed` L221-L222 qui accepte tout entier.
- Risque : vérifié empiriquement, `random.PRNGKey` réduit la graine modulo 2³² sans avertissement : `PRNGKey(2**40)` et `PRNGKey(2**40 + 2**32)` donnent la **même** clé. Les graines issues de hachages ou d'horodatages ns dépassent couramment 2³².
- Scénario déclencheur : l'utilisateur graine chaque chaîne d'un study par `hash(tag) % 2**63` ou `time.time_ns()` ; deux tags différents partagent la même clé 32 bits → chaînes de draws identiques, « indépendance » des graines brisée silencieusement.

**m10 — `infer={}` (dictionnaire vide) est traité comme `None`** [RÉSOLU 990053cc]
- Emplacement : `BayesForge/Distributions/np_dists.py:L8596` (exemple, motif `dict(infer) if infer else None` dupliqué ~90 fois).
- Risque : un `infer` vide est converti en `None` — inoffensif en pratique, mais empêche de passer un `infer={}` explicite pour écraser une valeur par défaut, et masque la différence entre « absent » et « vide ».
- Scénario déclencheur : code générique qui fait `infer=cfg.get("infer", {})` ne peut pas distinguer les deux cas en aval.

---

#### (a) Résumé du module

`np_dists.py` est un wrapper de façade au-dessus de `numpyro.distributions` : ~90 méthodes homothétiques qui normalisent l'accès aux distributions (construction, tirage direct avec graine, création de site de modèle, masquage, façonnage `shape`/`event`, sortie brute JAX ou `SampledData`), plus trois primitives (`track`, `plate`, `factor`) et un proxy `DistProxy` qui infère automatiquement le nom des sites. Le contrôle AST confirme que **toutes** les méthodes forwardent correctement leurs paramètres au constructeur numpyro correspondant (aucun bug de copie de paramètre, `validate_args` propagé partout, aucun nom non défini) et que le raccourci `enum=True` est cantonné aux 16 lois discrètes à support fini, ce qui est correct. Le module est donc fonctionnellement solide dans son chemin nominal ; ses faiblesses sont (1) la chaîne de gestion des graines (`seed=False` dégénéré, défaut horloge non reproductible, troncature 32 bits), (2) la sémantique double du `mask` et sa version NaN qui corrompt dtype et support des lois discrètes, (3) une documentation massive mais partiellement erronée (formule HalfNormal fausse, noms et contraintes contradictoires, contaminations de gabarit), et (4) l'inférence de nom par bytecode, mécaniquement fragile.

#### (b) Fragilités architecturales

1. **Duplication systématique (~90 copies)** : le corps de chaque méthode est un gabarit copié — le commentaire sur `handlers.mask` apparaît 91 fois, le motif `jnp.where(mask, samples, jnp.nan)` 90 fois. Tout correctif sémantique (M1, M2, C1) exige 90 éditions cohérentes ; la dérive constatée sur `sine_bivariate_vonmises` (M3) et `gaussian_copula_beta` (m6) montre que la dérive a déjà commencé. Une fabrique de méthodes (fermure générique paramétrée par la classe numpyro et ses arguments) supprimerait la classe entière de bugs de copie.
2. **Le paramètre `shape` à double sens** : `expand(shape)` en mode modèle, `sample_shape=shape` en mode tirage. Documenté, mais c'est un pièce d'API : la même valeur change de signification selon un booléen de contrôle (`sample`), et `sine_bivariate_vonmises` ajoute un troisième sens avec son `sample_shape` positionnel.
3. **Deux sémantiques de masque non réconciliables** (log-prob zéro vs NaN dans les draws) exposées sous le même nom — voir M2 ; un design plus sûr distinguerait `mask` (modèle) de `where`/`nan_out` (tirage).
4. **La chaîne de graines est un contrat non tenu au niveau du module** : `UnifiedDist` promet la reproductibilité de session mais ne la garantit que si le noyau `bf` lui injecte un `rng` ; sinon le défaut est l'horloge (C2) et `False` est la clé 0 (C1). La politique de graine devrait vivre à un seul endroit, avec validation (rejet des booléens, avertissement de troncature 32 bits).
5. **`DistProxy` et l'inspection de frames** : la magie de nommage (bytecode, `sys._getframe(2)`, cache global `_OBS_NAMES`) crée un couplage avec le CPython exact et avec la profondeur d'appel ; elle échoue silencieusement plutôt que de tomber sur le défaut `name='x'`. Un cache global par module, non borné et partagé entre instances, est en outre un état caché.
6. **Documentation embarquée non vérifiée** : ~90 docstrings générés par gabarit avec formules LaTeX recopiées ; rien dans le module (ni test visible) ne relie une formule documentée à la distribution réelle — d'où M6, m1–m5.

#### (c) Décompte par sévérité

| Sévérité | Nombre |
|---|---|
| Critique | 2 |
| Majeure | 6 |
| Mineure | 10 |
| **Total** | **18** |

---

### Module Distributions — tfp_dists.py

**Audit de release — Volet A (façade TFP)**
Fichiers audités : `BayesForge/Distributions/tfp_dists.py` (4729 lignes, lues intégralement), `BayesForge/Distributions/__init__.py` (0 octet).
Références croisées (sans audit en profondeur) : `BayesForge/Distributions/np_dists.py`, signatures numpyro installées, `Test/0.Distributions/distribution_tfp.py`, `BayesForge/Main/main.py` (L247-250 : `self.dist = tfp_dists`, la **classe** est utilisée comme singleton, contrairement au backend numpyro qui instancie un objet avec gestionnaire RNG).

Contexte structurel : le fichier est **généré** par `Build_TFP_light.py` et son propre en-tête (L1-L15) liste des correctifs manuels à appliquer (« Replace `args=args, kwargs=kwargs` with `*args, **kwargs` », « Replace `k_k_j` with `lkj` », avertissement « Not all of the functions have been tested »). Plusieurs de ces correctifs **n'ont jamais été appliqués**, ce qui explique une partie des bugs critiques ci-dessous.

---

## 1. Bugs critiques

### C1. `zero_inflated_negative_binomial` — toujours cassé (correctif du générateur non appliqué) `[RÉSOLU 674c285a]`
- **Sévérité : Critique** · `BayesForge/Distributions/tfp_dists.py:L4584`
- `tfd.ZeroInflatedNegativeBinomial(inflated_loc_logits=..., inflated_loc_probs=..., name=name, kwargs=kwargs)` passe le dictionnaire sous le mot-clé littéral `kwargs` au lieu de le dépaqueter (`**kwargs`). C'est exactement le motif que l'en-tête L4 demande de corriger, oublié ici.
- **Scénario** : `UnifiedDist.zero_inflated_negative_binomial(total_count=10., probs=0.5, inflated_loc_probs=0.2)` → `total_count`/`probs` atterrissent dans `kwargs`, jamais transmis → `TypeError: unexpected keyword argument 'kwargs'` + `missing required argument 'total_count'`. **Tout appel plante.**

### C2. `gamma_poisson` — paramètre `rate` calculé puis ignoré, appel toujours invalide `[RÉSOLU 674c285a]`
- **Sévérité : Critique** · `BayesForge/Distributions/tfp_dists.py:L4616-L4623`
- L4621-4622 convertit `scale` en `rate`… et ne s'en sert jamais : la délégation passe `probs=None, logits=None` à `negative_binomial` (L4623). Or `tfd.NegativeBinomial` exige `logits` **ou** `probs`. De plus, la conversion correcte Gamma-Poisson → NB (`probs = rate/(rate+1)`) est totalement absente.
- **Scénario** : `UnifiedDist.gamma_poisson(concentration=3., rate=2.)` → `TypeError` (ni logits ni probs). Le fichier de test du projet le confirme lui-même : `Test/0.Distributions/distribution_tfp.py` L50, commentaire « Broken in tfp_dists.py (passes probs=None, logits=None) ». Côté np, `np_dists.py:L2582` (`numpyro GammaPoisson(concentration, rate=1.0)`) fonctionne : divergence totale.

### C3. `discrete_uniform` — loi continue déguisée en loi discrète dans les modèles `[RÉSOLU 674c285a]`
- **Sévérité : Critique** · `BayesForge/Distributions/tfp_dists.py:L4651-L4667`
- L4656 construit `tfd.Uniform(low, high + 1.0)` — une densité **continue**. Le chemin `sample=True` (L4659) corrige avec `floor(...).astype(int32)`, mais le chemin modèle (`wrap=True`, L4664-4665) retourne la loi continue : son `log_prob` vaut `−log(high−low+1)` en continu (mesure de Lebesgue) au lieu de la pmf `log(1/(high−low+1))`, et l'échantillonnage du modèle (via `root(final_dist)`) produira des **flottants continus**, pas des entiers.
- **Scénario** : `UnifiedDist.discrete_uniform(low=1, high=10, obs=y)` dans un modèle TFP → log-vrasemblance fausse (mauvaise mesure), valeurs latentes non entières. Côté np (`np_dists.py:L2046`, `numpyro DiscreteUniform`) c'est une vraie pmf. Deux backends, deux lois différentes.

### C4. `unit` — un point en 1.0 au lieu d'une loi uniforme [0,1] `[RÉSOLU 674c285a — prémisse corrigée : numpyro Unit est un facteur log, implémenté ainsi]`
- **Sévérité : Critique** · `BayesForge/Distributions/tfp_dists.py:L4669-L4674`
- Retourne `deterministic(loc=1.0)`. Or la « Unit distribution » de numpyro (`np_dists.py:L6926`, `numpyro.distributions.Unit(log_factor)`) est la **loi uniforme continue sur [0,1]** (densité 1 sur [0,1]), avec un paramètre `log_factor` qui n'existe même pas côté TFP.
- **Scénario** : `m.dist.unit()` — backend numpyro : échantillons dans [0,1], log_prob = 0 sur [0,1] ; backend TFP : masse 1 en x=1.0, log_prob = −∞ ailleurs. Lois radicalement différentes, mêmes appels.

### C5. `inverse_gamma` — `rate` traité comme `scale` sans inversion (taux ≠ échelle) `[NON REPRODUIT, test de non-régression 674c285a]`
- **Sévérité : Critique** · `BayesForge/Distributions/tfp_dists.py:L1847-L1848`
- `if scale is None and rate is not None: scale = rate`. TFP `InverseGamma` est paramétré en **échelle** ; numpyro (`InverseGamma(concentration, rate=1.0)`, vérifié dans la source installée) en **taux**, avec `scale = 1/rate`. La façade TFP affecte le taux dans l'échelle.
- **Scénario** : `m.dist.inverse_gamma(2.0, 4.0)` — np : 2ᵉ positionnel = `rate=4` → scale 0.25 ; tfp : 2ᵉ positionnel = `scale=4` → loi dont la moyenne est 16× trop grande. Même pour `rate=2.0` explicite : tfp produit la loi `InvGamma(α=2, scale=2)` au lieu de `scale=0.5`. Erreur silencieuse, aucun message.

### C6. `pareto` — paramètres positionnels inversés vs la façade numpyro `[RÉSOLU 674c285a]`
- **Sévérité : Critique** · `BayesForge/Distributions/tfp_dists.py:L3342`
- Signature TFP : `pareto(concentration=None, scale=1.0, alpha=None)` (1er positionnel = α). Signature np (`np_dists.py:L5853`) et numpyro : `pareto(scale, alpha)` (1er positionnel = scale). Pire : L3348-3349, si ni `concentration` ni `alpha` ne sont fournis, **fallback silencieux `concentration = 1.0`** au lieu d'une erreur.
- **Scénario** : `m.dist.pareto(2.0, 3.0)` — np : scale=2, α=3 ; tfp : α=2, scale=3. Deux lois différentes, zéro erreur.

### C7. `asymmetric_laplace_quantile` — conversion d'échelle manquante (mauvaise loi vs numpyro) `[RÉSOLU 674c285a]`
- **Sévérité : Critique** · `BayesForge/Distributions/tfp_dists.py:L4685-L4691`
- Le calcul `asymmetry = sqrt(q/(1-q))` (L4690) est correct, mais numpyro (`AsymmetricLaplaceQuantile.__init__`, source installée vérifiée) fait en plus `scale_classic = scale * asymmetry / quantile`. La façade TFP passe `scale` **brut** (L4691). Le facteur manquant vaut `1/sqrt(q(1-q))` (= 2 pour q=0.5).
- **Scénario** : `m.dist.asymmetric_laplace_quantile(loc=0., scale=1., quantile=0.5)` — np : `Laplace(0, 2)` (conformément à la doc numpyro « AsymmetricLaplaceQuantile(0,1,0.5) is equivalent to Laplace(0,2) ») ; tfp : `Laplace(0, 1)`. En régression quantile, les intervalles crédibles sont faux d'un facteur 2. Bonus : `quantile=1.0` → division par zéro → `asymmetry=inf` sans validation (L4690).

### C8. Sémantique de `seed` — draws identiques par défaut, crash sur `seed=None`, pas de flux RNG de session `[RÉSOLU 674c285a]`
- **Sévérité : Critique** (reproductibilité / contrat inter-backends) · `BayesForge/Distributions/tfp_dists.py:L54-L55` et **151 occurrences** de `seed=0` / `jax.random.PRNGKey(seed)` (motif L53-55 répété partout)
- Côté np : `seed=None` par défaut, résolu par `_draw_key` (`np_dists.py:L198-L209`) qui tire une sous-clé fraîche du flux de la session (tirages consécutifs **différents**, séquence reproductible) ; `normalize_seed` (`np_dists.py:L211-L227`) accepte `None`, `bool` (True → `time_ns`), `int`.
- Côté tfp : `seed=0` par défaut → `jax.random.PRNGKey(0)` **à chaque appel**.
- **Scénarios** :
  1. `UnifiedDist.normal(0., 1., shape=(5,), sample=True)` appelé deux fois → **mêmes nombres** (clé 0 fixe) ; côté np, tirages différents. Toute simulation Monte-Carlo naïve avec les défauts produit des répliques.
  2. `seed=None` (valeur par défaut de la façade np) → `TypeError` dans `jax.random.PRNGKey(None)`.
  3. `seed=-1` ou `seed=2**32` → `ValueError` (PRNGKey exige uint32), là où np normalise.

---

## 2. Bugs majeurs

### M1. `mask` inexistant dans toute la façade TFP `[RÉSOLU c0b5d16d]`
- **Sévérité : Majeure** · global (aucune signature de `tfp_dists.py` n'expose `mask`, hors `mask` natif TFP de `hidden_markov_model` L1689 et `linear_gaussian_state_space_model` L2358)
- `np_dists.py` a `mask=None` dans **91 signatures** + un contexte manager `mask` (`np_dists.py:L243`) + `jnp.where(mask, samples, nan)` au tirage + `numpyro.handlers.mask` en mode modèle. Côté tfp : `UnifiedDist.normal(0., 1., mask=m)` → `TypeError: unexpected keyword argument 'mask'`. Le code utilisateur écrit pour le backend numpyro ne **compile même pas** sur le backend TFP.

### M2. `obs` ne conditionne rien — il ne fait que relooker l'event_shape (et plante ou casse les lois multivariées) `[RÉSOLU c0b5d16d]`
- **Sévérité : Majeure** · motif L56-59 (L57 : `tfd.Independent(dist, reinterpreted_batch_ndims=len(jnp.shape(obs)))`), répété ~151 fois
- Côté np, `obs=y` est transmis à `numpyro.sample(name, d, obs=y)` → le site est **observé/conditionné**. Côté tfp, la valeur `obs` n'est jamais attachée à quoi que ce soit : seul le nombre de dimensions sert à empaqueter dans `Independent`.
- **Scénarios** :
  1. `UnifiedDist.normal(loc=0., scale=1., obs=jnp.zeros(5))` (loc scalaire, obs vecteur) → `batch_shape=()` et `reinterpreted_batch_ndims=1` → `ValueError` de TFP (ndims > rang du batch_shape).
  2. `UnifiedDist.multivariate_normal(loc=mu, scale_tril=L, obs=Y)` avec `Y` de shape `(N, K)` → `Independent(2)` reinterpret **aussi la dimension événement K** : batch_shape vidé, toute vectorisation (carte par chaîne, log_prob vectorisé) est cassée silencieusement.
  3. `sample=True` **ignore obs** sans avertissement (L53-55 : la branche `sample` retourne avant d'examiner `obs`).

### M3. `name` : nom de site (np) vs étiquette interne TFP — même mot-clé, sens opposés `[RÉSOLU c0b5d16d]`
- **Sévérité : Majeure** · toutes les signatures, ex. L67, L3159
- Côté np, `name='mu'` nomme le site d'échantillonnage du modèle. Côté tfp, `name` est passé au constructeur TFP comme nom de distribution (label de debug) ; le site du modèle n'est jamais nommé. Un modèle écrit avec `name=` produit des traces correctement nommées sur numpyro et **anonymes** sur TFP.

### M4. `create_obj` accepté dans 43 signatures, jamais utilisé `[RÉSOLU c0b5d16d]`
- **Sévérité : Majeure** · ex. L253 (`bernoulli`), L297 (`beta`), L3159 (`normal`) — vérifié : aucune occurrence de `if create_obj` dans le corps des méthodes
- Côté np, `create_obj=True` retourne l'objet brut (essentiel pour construire des mélanges). Côté tfp, le paramètre est **silencieusement ignoré** : l'utilisateur qui passe `create_obj=True` (sans `wrap=False`) reçoit un `root(...)` au lieu de l'objet brut, et aucune erreur ne le signale.

### M5. Clamping silencieux `jnp.where(x <= 0, 1e-6, x)` — erreurs d'entrée masquées, NaN non filtrés, gradients plats `[RÉSOLU c0b5d16d]`
- **Sévérité : Majeure** · L319-320 (`beta`), L343-344 (`beta_binomial`), L831 (`dirichlet`), L1265-1267 (`gamma`), L1850-1852 (`inverse_gamma`), L2456 (`log_normal`)
- Une concentration négative ou un taux nul est remplacé par `1e-6` **sans erreur ni warning** : la loi retournée est quasi dégénérée (`Gamma(1e-6)`), le log_prob explose, et l'utilisateur croit avoir passé son paramètre. Pire : `jnp.where(NaN <= 0, 1e-6, NaN)` → **NaN passe** (comparaison fausse), donc la propagation silencieuse de NaN n'est même pas garantie d'être bloquée. Sous HMC/NUTS, le gradient de `where` est **nul dans la zone clampée** : un paramètre qui traverse 0 pendant le warmup se retrouve sur un plateau plat, sans diagnostic.
- Incohérence interne : `log_normal` clamp son `scale` (L2456) mais `normal` (L3179), `student_t` (L3947), `cauchy` (L560), `laplace` (L3343) ne clampent rien — deux politiques de validation coexistent dans le même fichier. Et `validate_args=False` partout par défaut (vs `None` côté np) désactive la dernière ligne de défense TFP.

### M6. `multivariate_normal` accepte `precision_matrix` et `scale_identity_multiplier` puis les ignore `[RÉSOLU c0b5d16d]`
- **Sévérité : Majeure** · `BayesForge/Distributions/tfp_dists.py:L2928-L2939`
- La signature (L2928) expose les deux paramètres ; le corps ne regarde que `scale_tril`, `scale_diag`, `covariance_matrix`. Côté np (`np_dists.py:L5183`), `precision_matrix` est réellement supporté.
- **Scénario** : `m.dist.multivariate_normal(loc=mu, precision_matrix=P)` → backend np : correct ; backend tfp : branche `else` (L2939) → `MultivariateNormalFullCovariance(covariance_matrix=None)` → crash obscur (ou loi identité si loc aussi omis), le mot-clé `precision_matrix` n'ayant jamais été mentionné dans l'erreur.

### M7. Ordre positionnel `probs`/`logits` inversé vs la façade numpyro (Bernoulli, Binomial, Categorical, Multinomial, NegativeBinomial) `[RÉSOLU c0b5d16d]`
- **Sévérité : Majeure** · L253 (`bernoulli(logits, probs)`), L419 (`binomial(total_count, logits, probs)`), L495 (`categorical(logits, probs)`), L2792 (`multinomial(total_count, logits, probs)`), L3097 (`negative_binomial(total_count, logits, probs)`)
- Signatures np : `bernoulli(probs=None, logits=None)` (`np_dists.py:L7502`), `binomial(total_count=1, probs=None, logits=None)` (L7615), `categorical(probs=None, logits=None)` (L7723), `multinomial(total_count=1, probs=None, ...)` (L8031), `negative_binomial(total_count, probs=None, logits=None)` (L8708).
- **Scénario** : `m.dist.bernoulli(0.3)` → np : p=0.3 ; tfp : **logit**=0.3 → p≈0.574. `m.dist.categorical([1., 1.])` → np : probs non normalisées (comportement numpyro) ; tfp : logits → uniforme… et pour des probs non normalisées, TFP **renormalise** alors que numpyro ne renormalise pas : même entrée, lois différentes même passées en mots-clés (`categorical_probs(probs=[2., 2.])`).

### M8. `gaussian_random_walk` — imports privés TF-dépendants + test booléen sur `loc` non traceable `[RÉSOLU c0b5d16d]`
- **Sévérité : Majeure** · `BayesForge/Distributions/tfp_dists.py:L4631`, `L4635-L4637`
- L4631/L4636 importent `tensorflow_probability.python.bijectors.*` (chemin privé, backends **TensorFlow**, pas le substrate JAX utilisé partout ailleurs L20-21) → `ImportError`/`ModuleNotFoundError` « Missing TF dependency » dans un environnement substrate-only (confirmé par le commentaire du test, `distribution_tfp.py` L53).
- L4635 `if loc != 0.:` — si `loc` est un tableau JAX ou une valeur tracée (jit), l'évaluation booléenne lève `TypeError`/`TracerBoolConversionError`. Côté np (`np_dists.py:L2842`), pas de paramètre `loc` du tout : divergence d'API en plus.

### M9. `zero_inflated_poisson` — composante `Poisson(rate=0)` : risque NaN dans le log_prob, `gate` non validé `[RÉSOLU c0b5d16d — NaN de Poisson(0) non reproduit ; rate=1 par défaut, gate documenté]`
- **Sévérité : Majeure** · `BayesForge/Distributions/tfp_dists.py:L363-L367`
- La composante « zéro » est `tfd.Poisson(rate=0)` (L366). Le log_prob TFP d'un Poisson est `x·log(rate) − rate − lgamma(1+x)` ; avec `rate=0` et `x=0`, `0·(−∞) = NaN` en arithmétique flottante — à moins que le substrate ne garde-foue ce cas (à vérifier au runtime, TFP non installé dans l'environnement d'audit ; incident historique connu côté TFP). Si non gardé-foué, **tout le log_prob du mélange devient NaN** (logsumexp contamine).
- `gate` hors de [0,1] : `probs = stack([gate, 1−gate])` (L363) avec `gate=1.5` → probabilité négative, silencieuse car `validate_args=False`. Côté np, `zero_inflated_poisson(gate, rate=1.0)` a un défaut pour `rate` ; côté tfp `rate` est requis (divergence mineure associée).

### M10. `wishart` — nom du paramètre d'incompatibilité (`df` vs `concentration`) et options np manquantes `[RÉSOLU c0b5d16d]`
- **Sévérité : Majeure** · `BayesForge/Distributions/tfp_dists.py:L4694` vs `np_dists.py:L7104`
- tfp : `wishart(df, scale_tril)` ; np : `wishart(concentration, scale_matrix=None, rate_matrix=None, scale_tril=None)`. `m.dist.wishart(concentration=3, scale_tril=L)` plante côté tfp (`unexpected keyword 'concentration'`) et l'inverse plante côté np. Les paramétrisations matricielles np (`scale_matrix`, `rate_matrix`) n'existent pas côté tfp.

### M11. `gamma` sans `rate` : crash côté tfp, loi par défaut côté np `[RÉSOLU c0b5d16d]`
- **Sévérité : Majeure** · `BayesForge/Distributions/tfp_dists.py:L1242` vs `np_dists.py:L2487`
- np : `gamma(concentration, rate=1.0)` → `UnifiedDist.gamma(2.)` fonctionne (Exp(1) décalée). tfp : `rate=None, log_rate=None` → TFP exige exactement l'un des deux → `ValueError`. Même appel, un backend marche, l'autre plante. (Même famille de divergence, mineure cette fois : `normal`, `cauchy`, `laplace`, `student_t`, `log_normal`, `exponential`, `von_mises` sont requis côté tfp et ont des défauts côté np — cf. §3.)

---

## 3. Bugs mineurs / fragilités

1. **Mineure** · L643-658 vs L766-771 — **deux définitions de `delta`** dans la classe ; la première (`delta(loc, atol, rtol, ..., create_obj)`) est **morte** (masquée par la seconde `delta(v, ...)`). Code mort + contrat ambigu. [RÉSOLU f5a85d92]
2. **Mineure** · L2455-2477 — `log_normal` construit `tfd.LogNormal` **deux fois** (L2457 puis L2477) avec une docstring orpheline au milieu (L2458-2476) : construction gaspillée à chaque appel, trace d'un merge manuel raté. [RÉSOLU f5a85d92]
3. **Mineure** · L30-63, L882-915, L3576-3604, L3680-3707 — wrappers de classes **non échantillonnables** (`auto_composite_tensor_distribution`, `distribution` (classe de base abstraite), `register_k_l` (fabrique de décorateurs KL), `reparameterization_type` (enum)) : surface d'API morte qui plante à l'usage (`NotImplementedError`/`AttributeError`) au lieu de ne pas exister. [RÉSOLU f5a85d92]
4. **Mineure** · L661 (`cholesky_l_k_j`), L3305 (`p_e_r_t`) — correctif n°2 de l'en-tête (« Replace `k_k_j` with `lkj` ») appliqué seulement pour `lkj` (L2214) : `m.dist.cholesky_lkj` / `m.dist.pert` → `AttributeError`. [RÉSOLU f5a85d92]
5. **Mineure** · L8-9 — l'en-tête documente le paramètre `warp=True` alors que le code utilise `wrap` : faute du générateur, doc trompeuse. L13 admet « Not all of the functions have been tested ». [RÉSOLU f5a85d92]
6. **Mineure** · L2819-2840 — `multivariate_normal_diag(loc=None, scale_diag=None)` : aucun argument requis → `MVN(0, I)` silencieux si l'utilisateur oublie les paramètres. [RÉSOLU f5a85d92]
7. **Mineure** · L696, L1206, L3854 — `dtype=jax.numpy.float32` codé en dur (`continuous_bernoulli`, `flat_dirichlet`, `spherical_uniform`) : en mode `jax_enable_x64`, divergence de précision avec le backend np qui suit le dtype global. [RÉSOLU f5a85d92]
8. **Mineure** · L4712 (`truncated_distribution(distribution=...)`) vs `np_dists.py:L8322` (`truncated_distribution(base_dist=...)`) — mot-clé incompatible entre backends. [RÉSOLU f5a85d92]
9. **Mineure** · L1584 — `half_cauchy(scale, loc=0.0)` accepte `loc` ; np (`np_dists.py:L3354`) n'a que `scale` (numpyro n'a pas de `loc` sur HalfCauchy). Code avec `loc=` ne passe pas côté np. [RÉSOLU f5a85d92]
10. **Mineure** · L767-771 — `delta(v)` côté tfp n'expose pas `log_density`/`event_dim` (présents côté np, `np_dists.py:L1742`) : les Delta non normalisés (offsets de mesure) sont inexprimables côté TFP. [RÉSOLU f5a85d92]
11. **Mineure** · L341 — `beta_binomial` : `total_count` casté vers `result_type(concentration1, concentration0)` ; avec concentrations float32 et `total_count` grand (>2²⁴), perte de précision silencieuse. [RÉSOLU f5a85d92]
12. **Mineure** · L4676-4683 — `beta_proportion` : aucune validation `mean ∈ (0,1)` ; un `mean=1.2` donne `concentration0 < 0` qui sera **clampé** en 1e-6 par `beta` (L320) — double masquage. [RÉSOLU f5a85d92]
13. **Mineure** · L359-378 — `zero_inflated_poisson` : `rate` requis côté tfp, défaut 1.0 côté np (`np_dists.py:L7313`). [RÉSOLU f5a85d92]
14. **Mineure** · L23 — alias `tfb = tfp.bijectors` défini et jamais utilisé (le seul usage de bijecteurs, L4631/L4636, passe par des imports privés). [RÉSOLU f5a85d92]
15. **Mineure** · `__init__.py` (0 octet) — le paquet `BayesForge.Distributions` n'expose rien : pas de `__all__`, pas de ré-export, pas de docstring ; tout passe par des imports de modules internes (`from BayesForge.Distributions.tfp_dists import UnifiedDist`, cf. `Main/main.py:L247`). Surface publique non définie. [RÉSOLU f5a85d92]
16. **Mineure** · L919 (`doublesided_maxwell`, `name='doublesided_maxwell'`), L419/L1477/L1903 (`name=None`) — conventions de nommage TFP incohérentes (minuscules/None) ; cosmétique mais casse les recherches par nom. [RÉSOLU f5a85d92]
17. **Mineure** · défauts requis vs np : `normal(loc, scale)` (L3159), `student_t(df, loc, scale)` (L3926), `cauchy` (L540), `laplace` (L2323), `log_normal` (L2435), `exponential(rate)` (L1097), `von_mises` (L4388) — np a des défauts (0.0/1.0) partout ; les appels sans arguments explicites divergent (crash tfp vs loi par défaut np). [RÉSOLU f5a85d92]

---

## (a) Résumé du module

`tfp_dists.py` est une façade générée (~151 méthodes statiques de la classe `UnifiedDist`) qui miroite TFP-JAX sur le même gabarit copié-collé : construction de la distribution TFP, tirage optionnel via `jax.random.PRNGKey(seed)`, reinterpretation de `obs` en event-dims via `tfd.Independent`, ou extension par `tfd.Sample`, puis encapsulation dans `JointDistributionCoroutine.Root` (`wrap=True`). Le cœur « passe-plat » (Normal, Gamma, MVN, LKJ, Wishart, GMM/Process…) est structurellement correct et fidèle à TFP.

Mais la façade **n'est pas un miroir de `np_dists.py`** : les sept arguments de service (`seed`, `obs`, `mask`, `name`, `event`, `create_obj`, `infer`/`to_jax`) ont une sémantique différente ou absente, et six lois « maison » (`unit`, `discrete_uniform`, `gamma_poisson`, `zero_inflated_negative_binomial`, `gaussian_random_walk`, `asymmetric_laplace_quantile`) sont cassées ou mathématiquement fausses. L'utilisateur qui change de backend n'obtient **pas la même loi** dans plusieurs cas silencieux (inverse_gamma, pareto, bernoulli/categorical positionnels, ALQ) et un crash dans d'autres. L'état de release du volet TFP est **bloquant**.

## (b) Fragilités architecturales

1. **Générateur + correctifs manuels = classe systématique de bugs.** L'en-tête (L1-15) assume un post-traitement manuel (`kwargs=kwargs`, `k_k_j`, `warp`) partiellement appliqué ; rien dans le CI ne vérifie que les correctifs sont passés. Les wrappers non-distributions (L30, L882, L3576, L3680) sont générés sans filtre.
2. **Aucun contrat partagé entre les deux façades.** Pas de spécification commune de `seed`/`obs`/`mask`/`name`/`event`, pas de suite de tests d'équivalence inter-backends (le seul test existant, `distribution_tfp.py`, commente les cas cassés au lieu de les corriger). La promesse « même loi depuis les deux backends » n'est testée nulle part.
3. **Copie-collé du tronc de 151 méthodes** (branches sample/obs/wrap identiques) : toute correction de sémantique devra être appliquée 151 fois — le `log_normal` dupliqué (L2455-2477) et le `delta` masqué (L643/L767) montrent que la dérive est déjà là. Un helper unique `_finalize(dist, shape, sample, seed, obs, wrap)` s'impose.
4. **Politique de validation incohérente** : clamping `1e-6` sur 6 lois, rien sur les autres, `validate_args=False` partout par défaut (vs `None` côté np) — les erreurs d'entrée arrivent tard (NaN en pleine inférence) ou jamais.
5. **PRNG non découplé de la session** : la façade TFP est une classe statique (cf. `main.py:L250`) sans flux RNG, donc ne peut pas honorer le `rand_seed` de la session `bf()` que np honore via `_draw_key`. Reproductibilité hétérogène par construction.
6. **Conditionnement numérique non couvert pour les lois matricielles** : les GP/StudentT-process exposent `jitter=1e-6` (L1319, L3962), mais `MultivariateNormalFullCovariance` (L2974), `MatrixNormalLinearOperator`, `Wishart*` n'offrent ni jitter ni `cholesky_fn` — une covariance presque singulière produit des NaN silencieux (et `allow_nan_stats=True` les accepte).
7. **`__init__.py` vide** : pas de frontière de paquets, la façade TFP n'est pas déclarée comme API publique alors que le code du projet l'importe en interne partout.

## (c) Décompte par sévérité

| Sévérité | Nombre |
|---|---|
| **Critique** | **8** (C1-C8) |
| **Majeure** | **11** (M1-M11) |
| **Mineure** | **17** (§3.1-3.17) |
| **Total** | **36** |

**Verdict de release — volet TFP : NON PRÊT.** Les 8 critiques incluent 2 fonctions qui plantent à chaque appel (C1, C2), 3 lois silencieusement fausses vs numpyro (C4, C5, C7), 1 inversion de paramètres positionnels (C6), 1 loi discrète rendue continue dans les modèles (C3) et un contrat de reproductibilité qui donne des tirages dupliqués par défaut (C8). Minimum bloquant avant release : corriger C1-C8, ajouter `mask`/`create_obj` réels (M1, M4), aligner l'ordre `probs`/`logits` (M7) et `rate`→`scale` (C5), et instaurer une suite de tests d'équivalence par loi `np_dists` vs `tfp_dists` (mêmes paramètres → mêmes log_prob/quantiles dans les deux backends).

---

### Module Samplers

**Audit de release — `BayesForge/Samplers/`** (racine `/home/sosa/work/BF`)
Périmètre lu intégralement : `Model_handler.py` (124 l.), `mcmc_numpyro.py` (125 l.), `mcmc_tfp.py` (182 l.), `SVI/svi_numpyro.py` (71 l.), `SBI/abc.py` (246 l.), `SBI/abc_rf.py` (181 l.), `SBI/npe.py` (363 l.), `SBI/simulate.py` (338 l.), les trois `__init__.py` (0 octet chacun). Vérifications croisées hors module : `Main/main.py` (appelants `fit`/`svi`/`abc`/`abc_rf`/`npe`, `_resolve_key`, `_as_count`), `Diagnostic/Diag2.py` (`az_from_dict`, chemin TFP), `BNN/bnn.py` (`layer_mixture_normal`, Cholesky).

---

## 1. Bugs et fragilités

### 🔴 Critiques

**C1 — `mcmc_tfp.py:71` — `num_adaptation_steps` codé en dur à 400 au lieu de suivre `num_burnin_steps`** `[RÉSOLU 6d7d0ba0]`
`DualAveragingStepSizeAdaptation(..., num_adaptation_steps=int(0.8*500))` : le 500 est une constante, pas l'argument `num_burnin_steps` (paramètre de `NUTS`, l.53). Le dual averaging s'arrête toujours à l'étape 400 de la chaîne.
*Risque mathématique* : (a) si `num_burnin_steps < 400`, l'adaptation **déborde sur la phase d'échantillonnage** — le noyau n'est pas stationnaire pendant les `400 − num_burnin_steps` premiers « draws », qui sont pourtant conservés dans le postérieur (échantillons biaisés, non réversibles) ; (b) si `num_burnin_steps = 2000`, l'adaptation s'arrête à 400 et les 1600 pas restants sont gaspillés.
*Scénario* : `m.fit(model, backend="tfp", num_warmup=100, num_samples=1000)` → 300 échantillons produits pendant l'adaptation de `step_size` entrent dans `m.posteriors`. Avec `num_warmup=0` (autorisé par `_as_count`, main.py:75-84), c'est tout l'échantillonnage qui est sous dual averaging.

**C2 — `mcmc_tfp.py:169` — `self.target_log_prob_fn` n'existe jamais → `AttributeError` garanti** `[RÉSOLU 6d7d0ba0]`
Dans `run()`, la branche `else` défère à `self.target_log_prob_fn(params + obs_tuple)`, mais l'argument `target_log_prob_fn` n'est jamais affecté à `self` (comparer l.164-169). Dès qu'un utilisateur passe `m.fit(..., backend="tfp", target_log_prob_fn=fn)`, la fermeture appelle un attribut inexistant.
*Scénario* : tout appel avec `target_log_prob_fn` non-`None` sur le backend TFP → `AttributeError: 'mcmc' object has no attribute 'target_log_prob_fn'` au premier appel du noyau, après compilation.

**C3 — `mcmc_tfp.py:81` (+ `:38`) — état initial contraint passé comme état non contraint à `TransformedTransitionKernel`** `[NON REPRODUIT, test de non-régression 6d7d0ba0]`
`sample_chain(current_state=self.init_params, kernel=TransformedTransitionKernel(...))` : TFP interprète `current_state` comme vivant dans l'espace **non contraint** et applique `bijector.forward` dessus. Or `self.init_params` (l.32-33) est un tirage du **prior**, donc dans l'espace contraint. Conséquences :
- paramètre positif (`Exp`) : un tirage `sigma=2.0` est traité comme `z=2.0` → point de départ réel `σ=e²≈7.39` (double transformation, init sémantiquement fausse) ;
- paramètre matriciel contraint (LKJ/`CorrelationCholesky`, Dirichlet…) : la dimension contrainte `(K,K)` ne correspond pas à la dimension non contrainte `K(K−1)/2` → **crash** dans `sample_chain`.
L'intention correcte est pourtant là : `initialise()` calcule `init_params2` (identité pour LKJ, `ones` pour positifs, `zeros` sinon, Model_handler.py:105-120) mais `init_Model` **jette ce résultat** (`_, self.bijectors = self.initialise(...)`, l.38).
*Scénario* : `m.fit(modele_avec_LKJ, backend="tfp")` → `InvalidArgumentError` sur les shapes dans `CorrelationCholesky.forward` ; modèle scalaire avec `HalfNormal` → démarre silencieusement au mauvais point.

### 🟠 Majeures

**M1 — `mcmc_tfp.py:33` — `list(init_params)[:-num_obs]` : hypothèses non vérifiées sur l'ordre des sites** `[RÉSOLU 19f93e5d]`
On suppose que les `num_obs` derniers sites du `JointDistributionCoroutine` sont les observés, dans le même ordre que `obs`. Si un site `obs=` est yieldé au milieu du modèle, ce sont des **paramètres légitimes qui sont tronqués** et des obs qui restent comme paramètres. De plus `num_obs=0` (modèle sans `obs=`, `find_obs_in_model` renvoie `[]`, l.147) donne `[:-0] == []` : `init_params` devient vide et le fit échoue avec une erreur incompréhensible au lieu d'un message explicite.

**M2 — `mcmc_tfp.py:162` — la graine de l'utilisateur n'atteint jamais l'initialisation** `[RÉSOLU 19f93e5d]`
`run()` reçoit `seed` (l.134) et le propage à `NUTS` (l.177) mais appelle `self.init_Model(..., seed=0)` en dur : l'échantillon du prior qui sert de point de départ est **toujours le même**, quel que soit le `seed=` de `m.fit`. Deux fits avec des graines différentes partagent leur point d'init → dépendance cachée entre « runs indépendants », et fausse idée de la reproductibilité.

**M3 — `mcmc_tfp.py:120-132` — `get_samples(group_by_chain=...)` ignore son argument : contrat inter-backends rompu** `[RÉSOLU 19f93e5d]`
Le flag est documenté mais inappliqué (le docstring l'avoue). Résultat côté `main.py:707-708` : `m.posteriors` (censé être plat, dim `(n_draws, …)`) contient la dimension chaîne en plus sous TFP, alors qu'elle est absente sous numpyro. Tout code utilisateur ou downstream écrit pour `m.posteriors` du backend numpyro se trompe de rang sous TFP, **sans erreur**.

**M4 — `mcmc_tfp.py:52-86` vs `mcmc_numpyro.py:59-89` — contrats d'arguments divergents, silencieusement** `[RÉSOLU 19f93e5d]`
Le TFP n'expose ni `thinning` (son équivalent `num_steps_between_results` n'est jamais câblé depuis `fit`, main.py:703-710 → `m.fit(backend="tfp", thinning=5)` est **silencieusement ignoré**), ni `target_accept_prob` (codé `.8`, l.70), ni `step_size` initial (codé `1e-3`, l.60), ni `extra_fields`. `NUTS` accepte par ailleurs `model` et `obs` sans jamais les lire (l.52). Un utilisateur qui change de backend obtient d'autres garanties sans le moindre avertissement.

**M5 — `Model_handler.py:114` — branches `elif` inachevées livrées en l'état** `[RÉSOLU 19f93e5d]`
Le commentaire `# (... rest of your elif conditions for beta, dirichlet, etc. ...)` est du code non fini : tout paramètre `Beta`, `Dirichlet`, ordonné, etc. tombe dans le `else` (l.117-120) → bijecteur `Identity` sur un support borné. L'échantillonneur explore `p<0` ou `p>1` → `log_prob = -inf/NaN` → divergences massives ou NaN dans le postérieur. Le `print` WARNING (l.118) est le seul indice, et il dit « Assuming Unconstrained » — un choix **mathématiquement faux**, pas un repli sûr.

**M6 — `Model_handler.py:44` — heuristique sous-chaîne `'obs' in line.lower()` pour détecter l'observation** `[RÉSOLU 19f93e5d]`
Tout paramètre dont la ligne de `yield` contient « obs » (`obs_rate`, `obstacle`, `x_observed`, un commentaire…) est exclu de `model_info`. Décalages en cascade : `initialise` (l.100, index `i` désynchronisé de `init_params`), et surtout `Diag2.py:392` qui `zip`pe `model_info.keys()` avec `posterior` **sans `strict`** → variables du postérieur **étiquetées sous les mauvais noms**, silencieusement.

**M7 — `mcmc_numpyro.py:84` — paramètre `seed=0` mort** `[RÉSOLU 19f93e5d]`
`seed` n'est référencé nulle part dans le corps : le `MCMC` retourné (l.114-124) ne le reçoit pas. L'appelant `main.py:661` passe sa propre clé à `.run()`, donc le chemin BF est correct — mais l'API publique du module promet une graine qui n'a aucun effet. `mcmc_numpyro(model=..., seed=42)` puis `sampler.run(jax.random.PRNGKey(0))` : la « reproductibilité » choisie par l'utilisateur est jetée à la poubelle sans un mot.

**M8 — `SBI/abc.py:191-193` — jitter de Cholesky invisible en float32** `[RÉSOLU 19f93e5d]`
`cov = cov + 1e-9 * jnp.eye(D)` : en float32 (défaut JAX ; `main.py:297` force même `jax_enable_x64=False` pour le TFP), `1e-9 < eps(1.0) ≈ 1.2e-7` → le jitter est **absorbé à l'addition**, la régularisation est nulle. Quand les particules s'effondrent (tolérance `eps_t` petite, particules quasi identiques → cov quasi singulier), `jnp.linalg.cholesky` (l.192) lève `NotPositiveDefinite` au cœur de la round `r`, et `jnp.linalg.inv` (l.193) produit des valeurs énormes → `log_kernel_mixture` en NaN. C'est exactement le plantage « au fond du solveur » à éviter ; jitter adapté (`1e-6·trace(D)·I`) ou `pinv`/eigen-clipping requis.

**M9 — `SBI/abc.py:200-208` — comptage du budget de simulation faux par défaut** `[RÉSOLU 19f93e5d]`
`n_prop += int(ok.sum())` ne compte que les propositions à densité de prior finie, mais `sim.simulate_given(k_s, u_prop)` (l.201) a déjà exécuté **les B simulations** du batch. `used += n_prop` (l.208) sous-estime donc le nombre réel d'appels au simulateur de `(B − ok.sum())` par batch, et `acceptance = n_acc / max(n_prop, 1)` (l.209) est **surévaluée** (le dénominateur ne devrait pas dépendre de `ok`). `num_simulations` cesse d'être un plafond garanti du coût, et l'arrêt `acceptance < min_acceptance` (l.226) se déclenche trop tard.

**M10 — `SBI/npe.py:211-214, 245` — aucune validation `num_simulations` / `val_frac` / `eval_every`** `[RÉSOLU 19f93e5d]`
`n_val = max(int(val_frac * n), 1)` puis `tr = perm[n_val:]` : si `val_frac ≥ 1` ou si le jeu filtré est minuscule (ex. 3 simulations valides, `val_frac=0.5`), `u_tr` est vide → `bs = min(batch_size, 0) = 0` → `jax.random.randint(k, (0,), 0, 0)` (l.237, `high=0`) ou pertes NaN, sans message. `eval_every=0` → `ZeroDivisionError` à la ligne 245. Le filtre `ok` (l.195) peut aussi vider `u`/`s` sans le moindre diagnostic.

**M11 — `SBI/simulate.py:268` — `jax.random.PRNGKey(0)` en dur dans `prior_log_density_u`** `[RÉSOLU 19f93e5d]`
La trace numpyro qui évalue la densité du prior est graine 0 fixe. Pour un modèle où la densité d'un paramètre inféré dépend d'un **latent nuisance** (le docstring, l.263-265, ne couvre que les dépendances paramètre→paramètre), la « densité » retournée est un tirage unique et déterministe de ce nuisance → poids d'importance de l'ABC-SMC biaisés d'une **dérive systématique invisible** (le même tirage est réutilisé pour toutes les particules, round après round).

**M12 — `SVI/svi_numpyro.py:40` — `except Exception` nu** `[RÉSOLU 19f93e5d]`
Toute exception de `guide.sample_posterior` — y compris un vrai bug (mauvaise shape, paramètre manquant dans `result.params`) — est avalée et remplace silencieusement le chemin de tirage par `Predictive` (l.42-43), qui n'a pas la même sémantique (sites déterministiques/re-simulés). L'utilisateur ne saura jamais que la voie de secours a été prise ; si `run()` n'a jamais été appelé, `self.result` est `None` (l.39), l'`AttributeError` est avalé, et le second crash (l.42) masque la vraie cause.

### 🟡 Mineures

**`Model_handler.py`**
1. L.47/L.51 — `line.split('=', 1)` lève `ValueError` sur une ligne `yield` sans affectation (`yield tfd.Normal(...)` final) et `words[0]` lève `IndexError` si aucun `(` n'est trouvé : crash sur des modèles valides. [RÉSOLU antérieurement (19f93e5d) — vérifié, test 5ec92df2]
2. L.34 — ligne morte : `var_model` est écrasé par la comprehension de la ligne 35. [RÉSOLU 5ec92df2]
3. L.92, 104, 110, 118 — `print()` INFO/WARNING dans une bibliothèque (pas de `logging`, pas muet, émis à chaque fit). [RÉSOLU 5ec92df2]
4. L.100 — `init_params[i]` : `IndexError` si `model_info` (parsing) est plus long que `init_params` (ex. ligne avec `=` et `yield` qui n'est pas une distribution). [RÉSOLU 5ec92df2]

**`mcmc_numpyro.py`**
5. L.5 — import `manip` inutilisé : couplage inutile vers `BayesForge.Data` (risque d'import circulaire). [RÉSOLU 5ec92df2]
6. L.76-88 — aucune validation précoce : `num_warmup=0` avec `adapt_step_size=True`, `num_chains>1` avec `chain_method="parallel"` sur 1 device → l'erreur (ou le repli) vient de numpyro au `run()`, pas du module (point 4 de la grille). [RÉSOLU 5ec92df2]

**`mcmc_tfp.py`**
7. L.38 — `self.obs_names` n'existe que si `run()` a été appelé : `AttributeError` pour qui utilise `init_Model` directement (méthode publique). [RÉSOLU 5ec92df2]
8. L.177-181 — `self.sampler = self.NUTS(...)` : `NUTS` ne retourne rien → `self.sampler = None` et `run()` retourne `None` (incohérent avec `SVI_BF.sampler = self`). [RÉSOLU 5ec92df2]
9. L.52 — arguments `model` et `obs` de `NUTS` jamais lus (morts). [RÉSOLU 5ec92df2]
10. L.107-114 vs L.118 — `self.posterior`/`sample_stats` : `list` dans la branche multi-run, `tuple` dans la branche simple → types incohérents pour les consumers. [RÉSOLU 5ec92df2]
11. L.93-101 — `n_chains > Ndevices` : boucle de `pmap` séquentielles correcte mais sans avertissement de coût ; `n_chains ≤ 0` non validé. [RÉSOLU 5ec92df2]

**`SVI/svi_numpyro.py`**
12. L.21 — `run(self, rng_key, num_steps=None, *args, ...)` : le premier argument positionnel du modèle est avalé comme `num_steps` (`run(key, Y)` → `num_steps=Y`). [RÉSOLU 5ec92df2]
13. L.6 — défaut mutable `loss=Trace_ELBO()` partagé par toutes les instances (inoffensif aujourd'hui, piège classique). [RÉSOLU 5ec92df2]
14. L.34 — `get_samples(seed=0)` par défaut : tirages identiques d'un appel à l'autre sans seed explicite (main.py passe bien une seed, l'API brute non).

**`SBI/abc.py`**
15. L.123-130 — pas de garde NaN sur les distances : si le simulateur retourne des stats NaN, `argsort` les repousse en fin de liste (exclusion implicite) mais si la majorité est NaN, `eps = NaN` et `keep` devient absurde — aucun avertissement (le point 4 de la grille demande une erreur explicite sur « simulateur qui retourne des NaN »). [RÉSOLU 93a0ab05]
16. L.54/L.137 — `eps = 0` (simulateur déterministe ou correspondance exacte) → `d / eps` = division par zéro dans le noyau d'Epanechnikov → particules ajustées à NaN. [RÉSOLU 93a0ab05]
17. L.151-152 — `n0 = min(n0, max(budget // 4, N))` : si `budget < 4N`, la round 0 simule `n0 > budget` → dépassement du budget déclaré dès le départ. [RÉSOLU 93a0ab05]
18. Absence de validation : `quantile ∈ (0,1]` (`quantile>1` → tout accepté, postérieur = prior), `alpha ∈ (0,1)` (l.185), `epsilon > 0` (négatif → `RuntimeError` au message trompeur, l.132-134), `num_rounds ≥ 1`. [RÉSOLU 93a0ab05]
19. L.41 — distance callables : `jax.vmap` impose une fonction jnp-compatible retournant un scalaire ; une fonction numpy ou retournant un tableau → erreur obscures « inside vmap ». [RÉSOLU 93a0ab05]
20. L.46-47 — `jnp.cov` sur stats dégénérées (1 simulation, stat constante) → NaN/singulier ; `pinv` amortit mais sans jitter ni garde. [RÉSOLU 93a0ab05]

**`SBI/abc_rf.py`**
21. L.121 — double système de RNG : `np.random.default_rng(random_state=0)` indépendant de la clé JAX de `run()` ; avec `random_state=0` par défaut, deux `m.abc_rf(seed=1)` / `m.abc_rf(seed=2)` partagent la même graine de forêt — reproductibilité à deux sources, trompeuse. [RÉSOLU 93a0ab05]
22. L.112 — `s_obs` jamais testé fini : un NaN dans les stats observées produit des poids QRF arbitraires et un postérieur silencieusement faux (le tableau de référence, lui, est bien filtré l.110). [RÉSOLU 93a0ab05]
23. L.164 — paramètre constant dans la table → dénominateur du R² OOB nul → `inf`/`NaN` dans `self.oob`. [RÉSOLU 93a0ab05]
24. L.44 — `forest.apply(S)` matérialise `(N, T)` en int32 (~200 Mo pour N=100 k, T=500), réalloué pour chaque forêt (une par colonne de paramètre) : pic mémoire quadratique en pratique, sans découpage. [RÉSOLU 93a0ab05]

**`SBI/npe.py`**
25. L.212/L.214 — même clé `k_init` réutilisée pour `est.init()` et `permutation()` : réutilisation de clé (à splitter). [RÉSOLU 93a0ab05]
26. L.159/L.178 — `prior_scale`, `lr`, `batch_size` non validés : `prior_scale < 0` → erreur `Normal` au cœur de la SVI (l.271). [RÉSOLU 93a0ab05]
27. L.296 — `broadcast_to(self._ctx(s), (un.shape[0],) + s_mean.shape)` : le docstring promet `s (n,S) ou (S,)` mais un `(n,S)` avec `n ≠ un.shape[0]` échoue en `broadcast_to` avec un message bas niveau. [RÉSOLU 93a0ab05]
28. L.309/L.312 — `fold_in(key, 7)` dérivé de la même clé que l'échantillonnage du mixture : décorrélation faible mais acceptable ; à documenter. [RÉSOLU 93a0ab05]

**`SBI/simulate.py`**
29. L.284 — `jnp.where(isfinite(lp), lp, -inf)` convertit aussi `+inf → -inf` (densités dégénérées inversées). [RÉSOLU 93a0ab05]
30. L.208-213 — cache `_prior_fn` indexé sur la seule taille de batch : la closure jit fige `data = self.data` ; muter `sim.data` après le premier appel laisse un simulateur aux anciennes données. [RÉSOLU 93a0ab05]
31. L.216/L.252 — `n = 0` → liste vide → `jnp.concatenate([])` lève une `ValueError` incompréhensible au lieu d'une erreur explicite. [RÉSOLU 93a0ab05]
32. L.45 — flag `_sbi_observed` écrit dans `infer` et lu nulle part (mort). [RÉSOLU 93a0ab05]
33. L.302 — `weighted_resample` avec poids de somme nulle → `p` NaN → erreur `jax.random.choice` bas niveau. [RÉSOLU 93a0ab05]
34. L.328-329 — `num_chains` pseudo-4 seulement si `n % 4 == 0` : `num_samples=999` → 1 pseudo-chaîne, `1000` → 4 ; les diagnostics changent de granularity selon un détail de l'arrondi. [RÉSOLU 93a0ab05]

---

## 2. Ce qui est solide (pour l'équilibre de l'audit)

- **SBI, pureté PRNG** : `abc.py`, `abc_rf.py`, `npe.py`, `simulate.py` ne touchent **jamais** `np.random`/`jax.random.PRNGKey()` dans les boucles de simulation — toute l'aléa découle de la clé `rng_key` propagée (split à chaque étape, ex. abc.py:156, 198 ; npe.py:191). Le seul entachement est le `PRNGKey(0)` en dur de `simulate.py:268` (M11) et le `default_rng` d'`abc_rf` (mineure 21).
- **Math ABC** : noyau d'Epanechnikov et moindres carrés pondérés de `_regression_adjust` (abc.py:52-59) corrects (équations normales `XᵀWX β = XᵀWu` avec ridge) ; mélange de noyaux de la proposition PMC avec `logsumexp` stabilisé (abc.py:176-180, 219) correct ; poids `prior/proposal` conformes à Beaumont 2009 ; exclusion correcte des NaN de distance dans le SMC (l.204).
- **Math NPE** : RealNVP aller/retour inverse l'un de l'autre (npe.py:124-143, masques et `(1-m)` aux lignes 120-121 corrects) ; Jacobien de normalisation `−Σlog u_std` (l.297) correct ; mixture bayésienne `logsumexp(lp, 0) − log M` (l.300) correcte ; scaling de sous-échantillonnage du `plate`+`factor` (l.274-275) correct ; early stopping avec meilleur checkpoint (l.249-255) ; test de couverture HPD (l.323-350) conforme à Hermans 2022.
- **`simulate.py`** : expansion de batch de `_uncondition` (l.38-42) juste ; Jacobien `log|dθ/du|` de `prior_log_density_u` (l.279) correct ; padding/troncature de `simulate_given` (l.248-259) correct ; bucketing des shapes en puissances de deux pour limiter la recompilation jit ; rejet précoce explicite des paramètres discrets (l.140-144) et de l'absence de site observé (l.124-128).
- **`mcmc_numpyro.build_kernel`** : repli `trajectory_length` pour HMC (l.39-42) et validation `hmc_gibbs` (l.49-54) soignés.
- Pas de mutation d'arrays utilisateur détectée dans les chemins SBI (les filtres/tri produisent des copies) ; `main.py` ne mute pas `data_on_model` pour le sharding.

---

## (a) Résumé du module

Le module couvre quatre familles (MCMC numpyro, MCMC TFP, SVI, SBI = ABC / ABC-RF / NPE) derrière un canard typé commun (`SBIPosterior`, `get_samples`/`get_extra_fields`/`num_chains`) que `Diag2` sait consommer via `az_from_dict`. La partie **SBI est de loin la plus aboutie** : PRNG pur, math correcte (PMC, QRF, NPE/BNPE, couverture HPD), erreurs précoces sur les entrées structurelles. La partie **numpyro (MCMC/SVI) est mince mais saine**, hormis un paramètre `seed` mort et un `except Exception` trop large. La partie **TFP (`mcmc_tfp.py` + `Model_handler.py`) n'est pas au niveau d'une release** : trois bugs critiques (adaptation mal calibrée, attribut inexistant, espace d'init confondu), un parsing de modèle par regex fragile qui peut **étiqueter les paramètres sous les mauvais noms**, et des branches `elif` manifestement inachevées. Le backend est marqué « in development » à l'usage (main.py:691) — l'audit confirme qu'il ne doit pas sortir de cet état.

## (b) Fragilités architecturales

1. **Deux contrats MCMC qui ne convergent pas** : `mcmc_numpyro` retourne un `MCMC` numpyro (graine au `run()`, `thinning`, `extra_fields`, `group_by_chain` réel) ; `mcmc_tfp` est une classe d'état avec `thinning` ignoré, `group_by_chain` ignoré, graines en dur, retour `None`. `main.py` doit maintenir deux chemins et l'utilisateur change de sémantique sans signal. Un adaptateur commun (ou l'alignement du TFP sur le contrat numpyro) est nécessaire avant d'exposer les deux backends.
2. **Le parsing textuel de modèle (`Model_handler.get_model_distributions`)** — regex sur les lignes `yield`, sous-chaîne `'obs'` — est le talon d'Achille de tout le chemin TFP : ses erreurs se propagent en décalages de bijecteurs (C3, M5) et en **mauvais noms de variables** dans l'`InferenceData` (`Diag2.py:392`, `zip` non strict). Une introspection par trace (comme le fait déjà `Simulator.__init__`, simulate.py:117-121) éliminerait la classe entière de bugs.
3. **Trois sources de RNG coexistent** : la clé JAX propagée (SBI, correcte), `np.random.default_rng(random_state)` (abc_rf), et des `PRNGKey(0)`/`seed=0` en dur (mcmc_tfp:162, simulate.py:268). La reproductibilité globale n'est garantie que pour le premier système.
4. **Les pseudo-chaînes de `SBIPosterior.num_chains`** (simulate.py:324-329) transforment R-hat/ESS en diagnostics **toujours rassurants** (~1/~n par construction) : ils ne peuvent pas détecter la non-convergence d'ABC-SMC ni le biais de NPE, mais s'affichent comme les diagnostics MCMC habituels. Risque de faux sentiment de sécurité ; un avertissement dans les plots de diagnostic s'impose.
5. **Validations d'API en amont manquantes** : le module fait confiance à `main.py:_as_count` (qui n'accepte que les entiers, pas les bornes) et ne valide ni `num_warmup=0`, ni `num_chains` vs devices, ni `quantile/alpha/epsilon/val_frac/eval_every/prior_scale`. Les erreurs surgissent au cœur des solveurs (jit, Cholesky, SVI) au lieu de la surface publique.
6. **`__init__.py` vides** (les trois) : aucune surface d'export ; le contrat `SBIPosterior` n'est documenté que dans une docstring interne, ce qui rend la dérive entre samplers invisible (cf. point 1).

## (c) Décompte par sévérité

| Sévérité | Nombre |
|---|---|
| 🔴 Critique | **3** (toutes dans `mcmc_tfp.py` : L71, L169, L81) |
| 🟠 Majeure | **12** (`mcmc_tfp` ×4, `Model_handler` ×2, `mcmc_numpyro` ×1, `svi_numpyro` ×1, `abc` ×2, `npe` ×1, `simulate` ×1) |
| 🟡 Mineure | **34** (`Model_handler` 4, `mcmc_numpyro` 2, `mcmc_tfp` 5, `svi_numpyro` 3, `abc` 6, `abc_rf` 4, `npe` 4, `simulate` 6) |
| **Total** | **49** |

**Verdict de release** : SBI et le chemin numpyro sont publiableux avec les correctifs M7-M12 (jitter float32, comptage du budget, validations d'entrée, `except` resserré, graine morte supprimée ou câblée). Le backend TFP (`mcmc_tfp.py` + `Model_handler.py`) doit rester marqué expérimental : ses trois bugs critiques produisent des **postérieurs faux ou mal étiquetés sans erreur**, ce qui est le pire mode de défaillance pour une bibliothèque d'inférence bayésienne.

---

### Module Models

Audit de release — `BayesForge/Models/` (racine `/home/sosa/work/BF`). Lecture intégrale de `models.py` (37 l.), `GMM.py` (322), `DPMM.py` (639), `BNNC.py` (216), `SRM.py` (517), `PCA.py` (769), `surv.py` (1206), `__init__.py` (vide). Vérifications croisées : conventions de `UnifiedDist` (`Distributions/np_dists.py` : `gamma(concentration, rate)`, `exponential(rate)`, `normal(loc, scale)`, `student_t(df, loc, scale)`, `truncated_normal(loc, scale, low, high)` — confirmées) et chemin d'appel `Main/main.py:572-590` (transposition de `X` pour PCA).

Note de périmètre : la spec annonce « survie (Cox/Weibull/AFT dans surv.py) ». `surv.py` n'implémente **ni partial likelihood de Cox, ni Weibull, ni AFT** : c'est un modèle à temps discret piecewise-exponentiel en vraisemblance de Poisson (`surv.py:418-431`, `817-883`). La gestion des ties Cox est donc sans objet ici ; l'écart spec/code doit être corrigé quelque part avant release.

---

## 1. Bugs critiques (math silencieusement faux ou modèle sans vraisemblance)

### C1 — `SRM.py:276` + `SRM.py:289-290` — `index1=True` produit une matrice de blocs silencieusement fausse `[RÉSOLU a22aeae6]`
`import_block_predictors` construit `dat_B` avec des indices de groupes `0..g-1` (L238-246), puis appelle `build_block_tensor(dat_B, block_predictors)` (L276) avec les **labels bruts**. Si `index1=True` (labels R 1-based `1..g`), la comparaison `block_preds_indexed == dat_B2_col1` (L268) associe le label `i` au paramètre `i` : le paramètre d'indice 0 ne capture jamais de nœud (aucun label 0), et les nœuds de label `g` ne sont jamais modélisés (jamais égaux à un `j ≤ g-1`). Ensuite L289-290 stocke `block_predictors - 1`, décalé par rapport au tenseur déjà construit.
**Scénario** : `import_block_predictors(np.array([[1],[1],[2],[2]]), index1=True)` sur 4 nœuds → le groupe 2 est absent du design, l'effet de bloc du groupe 2 est muet dans la log-vraiesemblance (L296-297, L310), sans aucune erreur. Inférence fausse.

### C2 — `SRM.py:139-143` — `import_block_predictors2` : `index1=True` détruit l'intercept qui vient d'être ajouté `[RÉSOLU a22aeae6]`
L139-141 concatène la colonne d'intercept, puis L142-143, si `index1`, **réécrit** `self.block_predictors = block_predictors - 1` depuis la variable brute : l'intercept est perdu, et les compteurs de groupes de `prepare_block_base_rate` (L151-167) tournent sur une matrice sans intercept. De plus L140 dépend de `self.focal_predictors` déjà importé (`AttributeError` sinon) et L143 soustrait 1 aussi aux colonnes codées 0 (cas mixte). Deux implémentations divergentes du même import (`import_block_predictors2` L125 vs `import_block_predictors` L224) aggravent le risque de mauvais choix.
**Scénario** : `srm.import_focal_predictors(F); srm.import_block_predictors2(b, index1=True)` → modèle sans intercept de bloc, base rate décalée, aucun avertissement.

### C3 — `SRM.py:344-345` — `numpyro_bsrm_model` : structure de blocs codée en dur à 2 groupes `[RÉSOLU a22aeae6]`
```python
N_blocks = block_predictors.shape[1] * 2   # calculé…
N_blocks = 2  # For simplicity, assume 2 blocks total (groups 0 and 1)
```
La première ligne est écrasée. Seule la colonne 0 de `block_predictors` est lue (L353-354) et les indices de groupe sont supposés ∈ {0,1}. Docstring (L314-316) : « matches the reference implementation exactly » — faux.
**Scénario** : `block_predictors` avec 3 groupes → `long_block_set_np.at[d, 2].set(1)` (L356) → `IndexError` ; avec 2 variables de blocs → la 2ᵉ est silencieusement ignorée.

### C4 — `SRM.py:490-512` — `outcome_mode` non validé : modèle sans site d'observation, échantillonnage de prior silencieux `[RÉSOLU a22aeae6]`
Les branches couvrent `outcome_mode ∈ {1,2,3,4}` ; toute autre valeur (0, 5, `"bernoulli"`, `None`) ne lève **aucune** erreur et n'ajoute aucune vraisemblance. Le modèle contient pourtant des sites `numpyro.sample` (`dr_L` L454, `sr_L` L470, …), donc NUTS « fit » en échantillonnant le prior.
**Scénario** : `_numpyro_bsrm_model_core(..., outcome_mode=0)` → chaînes qui tournent, trace = prior, l'utilisateur croit ajuster un réseau.

### C5 — `surv.py:1160` (et hérité `surv.py:367`) — `plot_surv` : hazard covariable calculé `λ0 · β` au lieu de `λ0 · exp(β)` `[RÉSOLU a22aeae6]`
```python
met_hazard = self.parent.posteriors[lambda0] * self.parent.posteriors[beta]
```
Le modèle (L754/L760/L766, `calculate_hazard_rate_*`) définit `λ = λ0 · exp(cov·β)`. La méthode de tracé multiplie par `β` **sans exponentielle** : les courbes « Metastasized » de hazard cumulé et de survie (L1184-1199) sont mathématiquement fausses (β<0 → hazard négatif → survie > 1 et explose). Qui plus est le défaut `beta='beta'` (L1139) ne correspond à aucun site produit par `priors()` (les sites sont `Hazard_rate_<cov>`, L802-812) → `KeyError` par défaut, ou courbes fausses si un site `beta` existe. Contraste avec `hazards()` de la classe legacy (L355-357) qui, lui, applique bien `exp(β)`.
**Scénario** : fit univarié (`Hazard_rate_treated` ≈ −0.7) puis `m.models.survival.plot_surv(beta='Hazard_rate_treated')` → bande « Metastasized » avec hazard cumulé négatif et S(t) > 1.

---

## 2. Bugs majeurs

### GMM.py

**G1 — `GMM.py:134-141` — moyennes a posteriori de paramètres de mélange soumis au label switching.** `[RÉSOLU 4a318ce5]` `post_mean_w/mu/Lcorr = jnp.mean(..., axis=0)` puis reconstitution `post_mean_cov` (L140-141). Sous label switching, la moyenne de `mu` à travers des draws permutés converge vers le centre de masse des clusters — quantité vide de sens ; de plus la moyenne de facteurs de Cholesky n'est pas le facteur de Cholesky de la covariance moyenne. `DPMM.py:376-377` l'énonce explicitement (« We return the RAW samples, not the means. Averaging parameters destroys information if chains swapped labels ») — GMM fait exactement l'inverse et retourne ces moyennes comme résultats principaux (L166-168). Seuls les `final_labels` (PSM, L151-163) sont switching-safe. **Scénario** : 2 clusters bien séparés, chaînes échangées → `post_mean_mu` ≈ milieu des deux nuages, `post_mean_cov` géante ; tout calcul downstream (ellipses de `predict_gmm`, comparaison sklearn) est faux.

**G2 — `GMM.py:151-154` — pas de sous-échantillonnage des draws : OOM.** `[RÉSOLU 4a318ce5]` `cluster_probs` (S,N,K) puis `cluster_probs @ cluster_probsᵀ` → (S,N,N) sur **tous** les draws. DPMM (`DPMM.py:340-351`) et BNNC (`BNNC.py:124-133`) plafonnent à `MAX_DRAWS=300` en documentant le kernel tué par l'OOM ; GMM, le modèle jumeau, ne le fait pas. **Scénario** : 4 chaînes × 1000 draws, N=1000 → ~16 Go pour la matrice de similarité → kernel killé.

**G3 — `GMM.py:44-45` — échelles de prior codées en dur, dépendantes de l'échelle des données.** `[RÉSOLU 4a318ce5]` `covariance_matrix=0.1*jnp.eye(D)` sur `mu` et `half_cauchy(1)` sur `sigma` : sur des données non standardisées (valeurs ~10²), le prior force `mu` à ±0.3 autour des centroïdes KMeans et `sigma` vers 0. **Scénario** : `gmm(100*data, 2)` → chaînes collées, variances sous-estimées, sans avertissement.

### DPMM.py

**D1 — `DPMM.py:80/90/92` vs `117/125/126` — le même modèle a deux jeux de priors selon `method`.** `[RÉSOLU 4a318ce5]` `latent` : `mu ~ MVN(data_mean, diag(2·std))` (L75+L90 — l'**écart-type** est utilisé comme **variance**, incohérence dimensionnelle), `sigma ~ log_normal(0,1)` (L92), `alpha ~ Gamma(1,10)` (L80). `marginal` : `mu ~ MVN(data_mean, 5.0·I)` fixe (L125 — pas adaptatif à l'échelle des données), `sigma ~ half_cauchy(1)` (L126), `alpha ~ Gamma(1,15)` (L117). Deux inférences du « même » dpmm ne sont pas comparables, et aucune n'est documentée comme différente. **Scénario** : `m(dpmm, method='latent')` puis `method='marginal')` sur données à échelle 10³ → priors de `mu` divergents d'un facteur ~10⁶, conclusions différentes.

**D2 — `DPMM.py:75/90` — colonne constante → covariance de prior singulière → crash Cholesky.** `[RÉSOLU 4a318ce5]` `data_std = jnp.std(data, axis=0)*2` vaut 0 sur une colonne constante ; `covariance_matrix=data_std*jnp.eye(D)` n'est pas définie positive → erreur de factorisation sans message explicite. **Scénario** : feature binaire majoritairement 0 dans un sous-échantillon d'init.

### BNNC.py

**B1 — `BNNC.py:70` + `BNNC.py:120` — `theta` deterministic de shape (N,K) stocké par draw.** `[RÉSOLU 4a318ce5]` Pour N=10⁴, K=11, 4×1000 draws : ~1,8 Go de draws pour seul `theta`, et `predict` le recharge intégralement. **Scénario** : BNNC sur 10 000 points → OOM en post-processing.

**B2 — `BNNC.py:166-169, 187, 202` — `plot` ignore le gating par point `w2` : incohérent avec le modèle et avec `predict`.** `[RÉSOLU 4a318ce5]` Le modèle définit les poids par point `theta = softmax(w2 + log π)` (L67-70) ; `plot` reconstruit la prédictive avec `global_pi` seul (L187) et étiquette les points par `argmax` de cette quantité (L202). Les labels affichés peuvent contredire `predict()` (qui, lui, utilise `theta`, L120). **Scénario** : gating fort (`w2` dominateur) → le contour « Posterior Predictive Mean » et les couleurs des points ne correspondent à aucune quantité du modèle.

**B3 — `BNNC.py:44-48, 73, 78` — priors « non informatifs » par défaut qui supposent des données standardisées.** `[RÉSOLU 4a318ce5]` `mu ~ N(0, 5)`, `sigma ~ Exponential(rate=1)` (moyenne 1) : sur données brutes à échelle 100, le prior écrase la vraisemblance. Aucun test de scale, aucun avertissement. **Scénario** : `bnnc(raw_features)` → clusters artificiellement serrés autour de 0.

**B4 — `BNNC.py:183-191` (+ L195-197) — materialisation (n_grid, S, K, D) en float64.** `[RÉSOLU 4a318ce5]` Grille 200×200 = 40 000 points, S=200, K=11, D=2 → ~1,4 Go pour `logn` seul. **Scénario** : `bnnc.plot` sur 200 draws → OOM.

### SRM.py

**S1 — `SRM.py:182` + `199` — padding NaN dans les moyennes de prior → échantillons NaN.** `[RÉSOLU 4a318ce5]` `blocks_mu_ij_padded` est paddé avec `jnp.nan` (L182) ; `update_block_effect` échantillonne `b ~ normal(logit(current_block_mu), 0.1)` (L199) sur la matrice **entièrement paddée** : les entrées NaN donnent des draws NaN sur le site `b{i}` → énergie potentielle NaN / posteriors NaN, même si `Neteffect.block_prior_to_edglelist` n'indexe jamais les lignes paddées. **Scénario** : deux variables de blocs avec 2 et 5 groupes → padding → NUTS « Invalid potential energy ».

**S2 — `SRM.py:82` — validation des prédicteurs dyadiques comparée à la shape de la **liste d'arêtes**, pas du réseau.** `[RÉSOLU 4a318ce5]` `self.network` a été converti par `Neteffect.mat_to_edgl` (L23) en (N_dyad, 2) ; le test `shape[0] != self.network.shape[0] and shape[1] != self.network.shape[1]` compare donc (N,N) à (N_dyad,2) : une entrée valide (N,N) est rejetée (message « Error » imprimé seulement, `self.dyadic_predictors` reste `None` → crash bien plus loin dans `model` L308), et une entrée invalide de shape (N_dyad,2) est acceptée. **Scénario** : `import_dyadic_predictors(mat_NxN)` → `TypeError` dans `Neteffect.dyadic_effect(None)`.

**S3 — `SRM.py:17-18, 50, 69, 83, 108-110, 117-123, 135` — les erreurs sont `print`ées, jamais levées.** `[RÉSOLU 4a318ce5]` Toutes les méthodes d'import continuent après « Error: … » ; l'état se retrouve `None` et l'échec réel survient à `fit`. **Scénario** : `import_exposure(2D array)` → print, `self.exposure` reste `None`, échec opaque dans le modèle.

**S4 — `SRM.py:457-460` — `print` de valeurs échantillonnées dans le corps du modèle.** `[RÉSOLU 4a318ce5]` Sous le jit de numpyro, ces prints de tracers polluent la compilation et, en exécution non-jit, s'exécutent à chaque appel du modèle. Hygiène : à supprimer avant release.

### PCA.py

**P1 — `PCA.py:19` vs `PCA.py:100-106` + `Main/main.py:578` — contrat d'orientation de `X` scindé, avec mutation in-place de la donnée utilisateur.** `[RÉSOLU 4a318ce5]` La doc de classe annonce `X.shape = (num_datapoints, data_dim)` ; le corps du modèle exige `(data_dim, num_data_points)` (`z` shape `(latent_dim, N)` L100, `w @ z` L106). Ça ne fonctionne que parce que `fit` fait `self.data_on_model["X"] = self.data_on_model["X"].T` — **mutation du dict de l'utilisateur**. Second `fit` sur le même objet : X transposé deux fois → `data_dim`/`num_data_points` déjà présents non recalculés → incompatibilité de shapes (N≠D) ou **vraisemblance transposée silencieusement** (N=D). Et `get_attributes` (L327-340), lui, attend `(N,D)` (`X @ components` L304). **Scénario** : `m.fit(m.models.pca(type='classic')); m.fit(m.models.pca(type='classic'))` → crash ou résultats faux selon N vs D.

**P2 — `PCA.py:506` — `individual_correlation` par défaut `X = self.X.T` projette dans le mauvais sens.** `[RÉSOLU 4a318ce5]` `extract_results_for_plot` calcule `X @ components` (L350) : avec `X` en (D,N) → erreur de shapes, ou scores **silencieusement faux** si N=D. Incohérent avec `plot()` (L743-760) qui passe `X` sans transposer. **Scénario** : `p=pca(X=X_nd); p.get_attributes(); p.individual_correlation()` → crash (N≠D).

**P3 — `PCA.py:575` — `plt.cScalarMappable` : faute de frappe, `AttributeError` garanti.** `[RÉSOLU 4a318ce5]` La branche `color_var and size_var and shape_var` (L560-584) plante ; la branche jumelle L598 utilise correctement `matplotlib.cm.ScalarMappable`. **Scénario** : `p.individual_correlation(color_var=c, size_var=s, shape_var=sh)` → crash.

### surv.py

**V1 — `surv.py:549` vs `surv.py:693-705` — collision de nom : la méthode `observed` est écrasée par un attribut.** `[RÉSOLU 4a318ce5]` `import_time_even` fait `self.observed = jnp.ones((N,T))` (L549) ; toute instance de la classe définit la **méthode** `observed(observed)` (L693). Après `import_time_even`, `surv.observed(arr)` lève `TypeError: 'JaxArray' object is not callable`. De plus la matrice `observed` n'est jamais consommée par la vraisemblance (L817-883) — masque d'observation mort. **Scénario** : le workflow documenté « import_time_even puis observed(...) » est cassé.

**V2 — `surv.py:799-813` + `885-903` — modèle sans covariable : soit `UnboundLocalError`, soit modèle sans vraisemblance silencieux.** `[RÉSOLU 4a318ce5]` Sans `import_covF`, `cov_all` est `np.array([])` (ndim 1) : `priors()` n'entre dans aucune branche et retourne un `beta` non lié (`UnboundLocalError` L815) ; `model()` (L897-903) ne rentre dans aucune branche non plus et **retourne `None` sans site d'observation** selon le chemin d'appel → échantillonnage du prior. **Scénario** : `surv.import_time_even(t,e); m.fit(m.models.survival)` → échec obscur ou fit sans données.

**V3 — `surv.py:520-538` — aucune validation de `time`/`event` ; temps nul ou négatif → vraisemblance catastrophique silencieuse.** `[RÉSOLU 4a318ce5]` `last_period` est clipé à [0, T−1] (L527) mais `exposure[i, last] = time - bounds[last]` (L538) : `time=0, event=1` → `mu=0+tiny`, `Poisson(tiny | death=1)` → log-prob ≈ −87 (float32) : chaînes bloquées, divergences, sans message. `time<0` → `exposure<0` → `mu<0` → `Poisson` à taux négatif → **NaN silencieux** dans l'énergie potentielle. `event∉{0,1}` est accepté comme compteur. **Scénario** : une date saisie en négatif dans le CSV → trace de NaN.

**V4 — `surv.py:790-797` — géométrie de prior dépendante du backend.** `[RÉSOLU 4a318ce5]` Backend numpyro : `_LogGamma` (dont le Jacobien L43-48 est **correct** — vérifié) ; autres backends : `Gamma(0.01, 0.01)` brut — exactement la pathologie décrite au L18-25 (underflow → `-inf` → divergences). Le même nom de modèle a deux comportements NUTS selon le backend.

---

## 3. Fragilités mineures

### GMM.py
- **`GMM.py:34, 38`** — aucune validation : `data` 1D → `IndexError` sur `shape[1]` ; `K>N` → erreur sklearn lointaine ; `N=0` non traité. `[RÉSOLU b94ec2bc]`
- **`GMM.py:162`** — `distance_threshold=0.5` codé en dur, non exposé (DPMM expose `threshold`/`linkage_method`, L247) ; incohérence d'API entre modèles jumeaux. `[RÉSOLU b94ec2bc]`
- **`GMM.py:24-27`** — docstring : « initial_means … breaks the mixture's label symmetry » est faux (des moyennes de prior distinctes ne suppriment pas le label switching en MCMC) ; c'est le PSM qui le gère. `[RÉSOLU b94ec2bc]`
- **`GMM.py:212-213, 220`** — `plot` suppose D=2 (`data[:,0]`, `data[:,1]`) : crash si D=1 (`np.trace` sur scalaire), D>2 ignoré silencieusement ; `np.cov(data.T)` → NaN si N=1. `[RÉSOLU b94ec2bc]`
- **`GMM.py:6-10`** — imports dupliqués (deux fois `fcluster/linkage`, `pdist/squareform`). `[RÉSOLU b94ec2bc]`

### DPMM.py
- **`DPMM.py:21-24` et `497`** — commentaires « ten nested levels, 5%-95% » alors que `DEFAULT_CREDIBLE=(0.25,0.50,0.75,0.95)` (4 niveaux). `[RÉSOLU b94ec2bc]`
- **`DPMM.py:374`** — `print` du nombre de clusters dans `predict` (API bruyante ; BNNC.py:150 idem). `[RÉSOLU b94ec2bc]`
- **`DPMM.py:177` / `GMM.py:147`** — `jnp.log(w)` : `w` peut sous-flotter à 0 en float32 (stick-breaking extrême) → `-inf` ; le softmax absorbe sauf ligne entièrement nulle → NaN possible. `[RÉSOLU b94ec2bc]`
- **`DPMM.py:199-203`** — aplatissage des chaînes conditionné à `w.ndim==3` uniquement ; suppose `mu/sigma/Lcorr` cohérents avec `w`, sinon silencieusement faux. `[RÉSOLU b94ec2bc]`
- **`DPMM.py:534`** — `plot` suppose D=2 (comme GMM). `[RÉSOLU b94ec2bc]`

### BNNC.py
- **`BNNC.py:36`** — aucune validation (`data` 1D, `K>N`, `N=0`, classes vides non testées). `[RÉSOLU b94ec2bc]`
- **`BNNC.py:109`** — `log(theta + 1e-8)` : biais d'epsilon sur les poids quasi nuls (le modèle L67 utilise 1e-10, la plot L187 aussi — trois epsilons différents). `[RÉSOLU b94ec2bc]`
- **`BNNC.py:193-194`** — `plot` suppose D=2. `[RÉSOLU b94ec2bc]`

### SRM.py
- **`SRM.py:351-357`** — boucle Python sur les dyades avec `.at[].set()` et `int(block_predictors[i,0])` : trace O(N²) non compilable (`int()` d'un tracer interdit le jit) ; lent sur N modeste. `[RÉSOLU 54581898]`
- **`SRM.py:220`** — `jnp.zeros(..., dtype=jnp.float64)` avec x64 désactivé par défaut → warning + rétrogradation silencieuse. `[RÉSOLU 54581898]`
- **`SRM.py:462-465`** — branches `if/else export_network==1` strictement identiques : code mort. `[RÉSOLU 54581898]`
- **`SRM.py:12-13`** — `logit` sans clipping : sûr avec les constantes actuelles (0.01/0.1), fragile dès qu'une base rate est branchée sur une fréquence observée. `[RÉSOLU 54581898]`
- **`SRM.py:228`** — hypothèse labels contigus 0-based (`max` = nombre de groupes) ; labels 1-based sans `index1` → off-by-one (cf. C1). `[RÉSOLU 54581898]`
- **`SRM.py:156`** — détection « colonne sans groupe » par `sum()==0` : une colonne avec labels symétriques (−1,+1) est confondue avec un groupe unique. `[RÉSOLU 54581898]`
- Deux importeurs de blocs publics divergents (L125 vs L224) — surface d'API dupliquée (cf. C2). `[RÉSOLU 54581898]`

### PCA.py
- **`PCA.py:330`** — `return Warning(...)` au lieu de `raise`/`warnings.warn` ; **`PCA.py:508, 513`** — `Warning(...)` construit puis **jeté**, puis poursuite sur `X=None` → `AttributeError` opaque. `[RÉSOLU 571b34f2]`
- **`PCA.py:102-106`** — `epsilon` documenté « noise variance » mais passé comme **scale** (écart-type) à `normal` : confusion variance/écart-type dans le contrat du prior (`Exponential(rate=1)` sur σ). `[RÉSOLU 571b34f2]`
- **`PCA.py:562, 611, 642`** — normalisation `100*(s-min)/(max-min)+10` : `size_var` constant → division par zéro → tailles NaN. `[RÉSOLU 571b34f2]`
- **`PCA.py:363-367`** — `confidence_ellipse` : groupe à variance nulle → `pearson` NaN ; |pearson|>1 numérique → `sqrt` négatif → NaN. `[RÉSOLU 571b34f2]`
- **`PCA.py:482-483, 680-681`** — `ratio[1]`/`explained_variance_ratio[1]` : `IndexError` si `latent_dim=1`. `[RÉSOLU 571b34f2]`
- **`PCA.py:405`** — `if ax is None: plt.show()` mort (ax vient d'être créé en L399-400) : `components()` n'affiche jamais sa propre figure. `[RÉSOLU 571b34f2]`
- **`PCA.py:25` vs `59`** — `type` par défaut `"classic"` dans `__init__`, `"ARD"` dans `__call__`. `[RÉSOLU 571b34f2]`
- **`PCA.py:68-84`** — `get_model(type inconnu)` retourne `None` (pas d'`else`) → `TypeError` différé dans `fit`. `[RÉSOLU 571b34f2]`
- **`PCA.py:276-282`** — `set_deterministic_sign` : boucle Python avec branche sur valeur tracée → incompatible jit (ok en eager, piège si l'utilisateur jitte). `[RÉSOLU 571b34f2]`

### surv.py
- **`surv.py:988-995`** — `_posterior_draws` : la branche `else` (L994-995) duplique la première → `KeyError` sans message utile quand ni `beta` ni `Hazard_rate_*` n'existent. `[RÉSOLU 35409a83]`
- **`surv.py:355-356`** (legacy `hazards`) — les paramètres `lambda0`/`beta` (noms) sont ignorés, clés codées en dur. `[RÉSOLU 35409a83]`
- **`surv.py:176-181`** (legacy `plot_censoring`) — colonne `'time'` codée en dur au lieu du paramètre `time` → `KeyError` si la colonne s'appelle autrement. `[RÉSOLU 35409a83]`
- **`surv.py:71`** (legacy `survival_old.__init__`) — `self.df = None` passe par le **setter de propriété** (L77-79) → `self.parent.df = None` : effacement des données du parent à la construction ; de plus `self.dist = dist()` rappelle une instance (code mort, mais toxique s'il est réactivé). `[RÉSOLU 35409a83]`
- **`surv.py:545-547, 579-581, 624, 636-638`** — `print` dans l'API publique (dont `print(self.cov_v)` L624, debug évident). `[RÉSOLU 35409a83]`
- **`surv.py:460`** — le constructeur mute `self.parent.model_name = 'survival'` : effet de bord à l'instanciation. `[RÉSOLU 35409a83]`
- **`surv.py:526` / legacy `249`** — epsilons de frontière `1e-7` / `0.01` dépendants de l'échelle des temps (temps < 1e-7 mal bucketés ; legacy : temps < 0.01 → `last_period=-1` → wrap à la **dernière** colonne de `death`, L253-254, avant le clip introduit en V3). `[RÉSOLU 35409a83]`
- **`surv.py:572`** — `self.cov_names += names` : si `names` est une chaîne, elle est étendue caractère par caractère. `[RÉSOLU 35409a83]`
- **`surv.py:583`** — `n_cov` mis à jour uniquement dans `import_covF` ; jamais dans `import_covV` → attribut absent ou périmé. `[RÉSOLU 35409a83]`

### models.py
- **`models.py:18-20`** — `self.gmm.predict = predict_gmm` etc. **mute l'objet fonction du module** : effet de bord global partagé par toutes les instances ; fragile (deuxièmes imports, rechargements). `[RÉSOLU 35409a83]`
- **`models.py:13-17`** — `parent` reçu mais non stocké (seuls les sous-constructeurs l'utilisent) ; `pass` mort en L17. `[RÉSOLU 35409a83]`
- Surface hétérogène : `self.gmm` = fonction brute + attributs greffés, `self.dpmm/bnnc/survival` = instances, `self.pca` = classe, `self.nbda` = `NBDA.model` — quatre styles de contrat dans un même dict `available` (L27-34). `[RÉSOLU 35409a83 — documenté dans la docstring de models, API inchangée]`

---

## 4. Points corrects vérifiés (pour mémoire)

- `DPMM.py:62-65` `mix_weights` : stick-breaking correct (`w_k = β_k·∏_{j<k}(1−β_j)`, dernier = `∏(1−β)`), somme = 1, pas d'inversion de `cumprod` ni de borne mal incluse.
- `surv.py:43-48` `_LogGamma.log_prob` : Jacobien `+u` correctement intégré algébriquement (densité de `log X` exacte) ; l'échantillonnage L36-38 est cohérent.
- `surv.py:1028-1051` `sample_times` : inversion de hazard cumulé piecewise-exponential correcte (recherche d'intervalle, extrapolation au-delà de la dernière borne, taux nul → `inf`), PRNG local seedé (`default_rng(seed)`) — pas de PRNG global caché.
- Log-sum-exp utilisé correctement partout où c'est nécessaire (`GMM.py:148`, `DPMM.py:180`, `BNNC.py:88, 110`, softmax stable `BNNC.py:188-190`) ; la vraisemblance marginalisée de BNNC via `numpyro.factor` (`BNNC.py:85-89`) est correcte.
- PSM de consensus (`GMM.py:154`, `DPMM.py:361`, `BNNC.py:139`) : `S = E[r rᵀ]` correct, symétrisation + diagonale nulle + clip avant `squareform` — bonne pratique.
- `PCA.py:255-260` alignement de Procrustes par draw : correct ; `PCA.py:420-431` moments de mélange de `_cluster_moments` (avec ancrage intra-composante) : corrects, avec garde PSD (L432-434).
- `SRM.py:467/476/486` construction sender/receiver/dyad `S_i+R_j, S_j+R_i` : conforme au BSRM.
- Aucun usage de `np.random`/`jax.random` global non seedé dans les modèles ; `KMeans(random_state=0)` (`GMM.py:38`) est déterministe.

---

## (a) Résumé du module

`BayesForge/Models/` est un ensemble de modèles pré-construits de qualité très inégale. Le noyau probabiliste des mélanges (GMM/DPMM/BNNC) est **mathématiquement juste dans ses densités** (stick-breaking, logsumexp, PSM anti-label-switching) et les trois modèles partagent une même recette de consensus clustering globalement bien pensée. `surv.py` contient le meilleur morceau du module (`_LogGamma`, `sample_times`, tous deux vérifiés exacts) mais gâche son output par un `plot_surv` qui applique `λ0·β` au lieu de `λ0·exp(β)`. La famille des bugs dominants n'est donc pas dans les formules mais dans **les contrats** : orientation de `X` en PCA (résolue par une mutation in-place dans `Main`), priors par défaut dépendants de l'échelle des données (GMM/DPMM/BNNC), deux jeux de priors pour un même DPMM selon `method`, et surtout `SRM.py`, qui est en dessous du seuil de release : erreurs `print`ées au lieu d'être levées, design de blocs silencieusement faux avec `index1=True`, structure de blocs codée en dur à 2 groupes, et mode de résultat non validé menant à un échantillonnage de prior sans données. Aucun modèle ne valide ses entrées (1D vs 2D, N=0, K>N, temps négatifs, classes vides) ; les échecs arrivent tard, loin de la cause. Le module est explicitement marqué « en développement » (`warn_dev` y compris sur `predict`/`plot`), et l'écart entre la spec (Cox/Weibull/AFT) et le code (Poisson à temps discret) doit être résolu avant publication.

## (b) Fragilités architecturales

1. **Contrats de shapes implicites et non documentés** : l'orientation de `X` en PCA dépend d'une transposition mutatrice dans `Main/main.py:578` ; `get_attributes`/`plot` attendent l'orientation inverse des défauts (`PCA.py:506`). Le même type de dérive existe entre `get_basic_info` (clé `'cov'`, L136-139) et `surv_object` (clés par nom, L277-279) dans surv legacy.
2. **Duplication non factorisée** : ~100 lignes de consensus clustering (PSM → distance → linkage → fcluster) recopiées dans GMM/DPMM/BNNC avec des comportements divergents (cap de draws présent en DPMM/BNNC, absent en GMM ; seuil exposé en DPMM, codé en dur en GMM/BNNC). GMM dépend en retour des méthodes statiques privées de DPMM (`GMM.py:15, 208, 281, 284`) — couplage Models↔Models par le privé.
3. **État caché dans `parent`** : `dpmm.proportion_of_data_assigned_to_cluster` (L191-209), `survival._posterior_draws` (L979), `bnnc.model` (`m.bnn.layer_linear`, L52-61) et `ppc_km_by_cov` (L1122) lisent `parent.posteriors`/`parent.data_on_model`/`parent.diag`/`parent.bnn` ; le constructeur de `survival` mute `parent.model_name` (L460), celui de `survival_old` mute `parent.df` (L71). Les modèles ne sont pas testables isolément de `Main`.
4. **Prior par défaut non adaptatifs à l'échelle** (GMM 0.1·I / half-Cauchy(1) ; DPMM marginal 5.0·I ; BNNC N(0,5)+Exp(1)) : la convention « standardisez vous-même » n'est nulle part vérifiée ni documentée comme contrat.
5. **Robustesse d'API uniformément absente** : le module n'élève aucune `ValueError` sur entrées invalides (SRM `print`e, les autres laissent numpyro/sklearn planter) ; deux méthodes homonymes d'un même import (`import_block_predictors` / `import_block_predictors2`) et une collision méthode/attribut (`observed`) complètent un tableau de surface publique non stabilisée.
6. **Code legacy conservé dans les fichiers de production** : `survival_old` (dont le `__init__` est toxique s'il est instancié), `models.py` qui greffe des attributs sur des fonctions de module, branches `if/else` identiques (SRM L462-465), prints de debug dans les corps de modèles.

## (c) Décompte par sévérité

| Sévérité | Nombre |
|---|---|
| **Critique** | **5** (SRM ×4 : C1-C4 ; surv ×1 : C5) |
| **Majeure** | **19** (GMM 3, DPMM 2, BNNC 4, SRM 4, PCA 3, surv 3) |
| **Mineure** | **43** (GMM 5, DPMM 5, BNNC 3, SRM 7, PCA 9, surv 9, models 3, +2 hérités surv) |
| **Total** | **67** |

**Recommandation de release** : bloquer `SRM.py` (les 4 critiques + S1-S4 touchent le cœur de la log-vraiesemblance) ; corriger `surv.py:1160` (C5, une ligne) et la collision `observed` (V1) ; aligner GMM sur la politique « raw draws + cap de draws » de DPMM (G1-G2) ; trancher le contrat d'orientation PCA (P1-P2) ; le reste (mineurs) peut suivre en backlog post-release.

---

### Module Main

**Périmètre** : `BayesForge/Main/main.py` (2247 l., lu intégralement), `BayesForge/Main/__init__.py` (0 l., vide), `BayesForge/__init__.py` (41 l.). Vérifications croisées dans `Samplers/mcmc_numpyro.py`, `Samplers/mcmc_tfp.py`, `Samplers/SVI/svi_numpyro.py`, `Samplers/SBI/*`, `Utils/rng.py`, `Data/manip.py`, `Diagnostic/Diag2.py`, `Models/PCA.py`, `Models/models.py`, `NBDA/NBDA.py`, `Resources/datasets.py`, `SetDevice/set.py`, et test d'exécution de `numpyro.infer.util.initialize_model` (numpyro 0.22.0, jax 0.11.2).

---

## 1. Bugs constatés

### M-01 — Sévérité : **Critique** — `BayesForge/Main/main.py:1783` (`log_prob`) `[RÉSOLU 82860ed8]`
`initialize_model(rng_key, model, model_args=(kwargs))` : les parenthèses autour de `kwargs` ne forment **pas** un tuple — `model_args` reçoit le dict lui-même. NumPyro appelle `model(*model_args)`, ce qui déballe les **clés** du dict comme arguments positionnels. Vérifié par exécution : le modèle reçoit `y='y'`, `N='N'` (chaînes), jamais les données.
**Scénario** : `m.log_prob(model, y=np.array([1.,2.,3.]), N=3)` →
- si le modèle consomme `y` (`obs=y`) : `TypeError: unsupported operand type(s) for -: 'str' and 'LinearizeTracer'` (message sans rapport avec la cause) ;
- si le modèle ignore ses arguments : **aucune erreur**, log-prob calculé **sans conditionnement sur les données** — résultat silencieusement faux.
De plus, les données ne transitent jamais par `model_kwargs`. Correction : `model_kwargs=kwargs`. Corrections annexes : `constrain_fn(init_params.z)` (l. 1786) plante si le modèle n'a aucun site latent (`init_params=None`) ; la méthode imprime sur stdout au lieu de ne retourner que le tuple ; le docstring annonce « seed defaults to 0 » alors que le code tire une clé du stream (l. 1781).

### M-02 — Sévérité : **Critique** — `BayesForge/Main/main.py:578` (branche `pca` de `fit`) `[RÉSOLU 82860ed8]`
`self.data_on_model["X"] = self.data_on_model["X"].T` mute **le dict de l'utilisateur** : au l. 540, `self.data_on_model = obs` stocke le dict par référence (aucune copie, contrairement à `_sbi_setup` l. 856 qui fait `dict(data)`). La transposition est un ré-assignement dans le dict de l'utilisateur (vue `.T` sur son array).
**Scénario 1** : `d = {"X": X}; m.fit(model=pca_inst, obs=d); assert d["X"].shape == X.T.shape` — l'appelant voit ses données transposées.
**Scénario 2** : second `m.fit(model=pca_inst, obs=d)` sur le même dict → double transposition → `X` revient à l'orientation (N, D) alors que le modèle attend (D, N) (cf. `pca_classic`, `w @ z` de shape `(data_dim, num_data_points)`) ; `data_dim`/`num_data_points` étant déjà dans le dict, ils ne sont pas recalculés (l. 579-584). Si N ≠ D : crash de broadcasting ; **si N = D : posterior silencieusement faux**.
Même famille : l. 566-569, la branche `gmm` injecte `"initial_means"` dans le dict de l'utilisateur.

### M-03 — Sévérité : **Critique** — `BayesForge/Main/main.py:815-836` (`svi`) et `592/689` (`fit`) : backend inconnu ou `tfp` → **no-op silencieux** `[RÉSOLU 82860ed8]`
- `fit` : `if self.backend == "numpyro": … elif self.backend == "tfp": …` — **pas de `else`**. Le constructeur (l. 131, 251, 291) ne valide jamais `backend` : `bf(backend="numpyro ")` (espace, majuscule, faute de frappe) ou `backend="stan"` → `fit()` n'exécute aucun branchement, ne lève rien, pose `run_model_name` (l. 726) et **retourne None comme un succès**. Les `m.posteriors` restent ceux d'un fit antérieur ou sont absents.
- `svi` : `if self.backend == "numpyro":` sans `else` → avec le backend `tfp`, `svi()` prépare guide/optim/loss, ne fit **rien**, ne lève rien, et écrase `run_model_name`.
**Scénario** : `m = bf(backend="TFP"); m.fit(model, obs=d)` → aucun échantillonnage, aucun message d'erreur (le print d'avertissement l. 690 n'est atteint que pour `"tfp"` exact), puis `m.summary()` analyse les restes d'un fit précédent. Correction : valider `backend in {"numpyro","tfp"}` dans `__init__` et ajouter un `else: raise`.

### M-04 — Sévérité : **Majeure** — `BayesForge/Main/main.py:711-712` : contrat des postérieurs enfreint par le backend TFP `[RÉSOLU 19f93e5d]`
`self.posteriors = self.sampler.get_samples()` puis `self.posteriors_by_chain = self.posteriors`. Or `mcmc_tfp.mcmc.get_samples(group_by_chain=…)` **ignore son argument** (`mcmc_tfp.py:124-133`) et renvoie toujours les draws structurés par chaîne. Conséquences :
- `m.posteriors` côté TFP a shape `(n_chains, num_results, …)` alors que côté numpyro c'est le pool `(n_chains*num_samples, …)` — tout code aval (scripts, `m.plot`, export) qui suppose un pool à 1 dim de batch casse ou silencieusement mal dimensionné ;
- `m.sample()` (l. 1653) appelle `self.sampler.get_samples()` et passe le résultat à `Predictive` comme `posterior_samples` : avec TFP, l'axe 0 est l'axe chaîne → draws PPC à 2 dims de batch ;
- la sous-population `samples=n` (l. 1683) compte **les chaînes** : `m.sample(samples=500)` après un fit TFP à 4 chaînes → `ValueError: Cannot subsample 500 posterior draws: the posterior has only N = 4 draws`.

### M-05 — Sévérité : **Majeure** — `BayesForge/Main/main.py:832-836` (`svi`) : `posteriors_full` / `posteriors_by_chain_full` / `posteriors_by_chain` non rafraîchis → retour fantôme au fit précédent `[RÉSOLU antérieurement — vérifié dans le code courant]`
`svi` ne met à jour que `self.posteriors`. `posteriors_full`, `posteriors_by_chain_full` et `posteriors_by_chain` gardent les valeurs du **`fit` précédent**.
**Scénario** : `m.fit(A)` → `m.svi(B)` → `m.filter_posteriors(include="a")` (ou `m.summary(include="a")`, l. 1280-1281) → l. 1421 : `self.posteriors = _filter(self.posteriors_full)` → `m.posteriors` est **remplacé par les échantillons MCMC du modèle A**, silencieusement, alors que le dernier fit est le SVI de B. Idem `m.summary(backend="jax")` qui consomme `posteriors_by_chain` périmé (l. 1296-1298).

### M-06 — Sévérité : **Majeure** — `BayesForge/Main/main.py:835` et `Samplers/SVI/svi_numpyro.py:30-46` : diagnostics post-SVI avec `num_chains = num_samples` [RÉSOLU antérieurement — vérifié, test 365eb5fe]
`svi_numpyro.get_samples(group_by_chain=True)` n'ajoute l'axe chaîne que dans son **fallback** `Predictive` (l. 44-45) ; dans le chemin principal (`guide.sample_posterior`, l. 39) l'argument est ignoré et les draws restent poolés. `diag(sampler=self.sampler)` (Diag2, `Diag2.py:328-336`) déduit alors `num_chains = shape[0] = num_samples` (ex. 1000 « chaînes »), et le garde-fou `declared = getattr(sampler, "num_chains", None)` ne se déclenche pas (l'objet SVI n'a pas d'attribut `num_chains`). Les tracés de diagnostics post-SVI itèrent sur 1000 « chaînes » de 1 draw.

### M-07 — Sévérité : **Majeure** — `BayesForge/Main/main.py:297` : le backend TFP force `jax_enable_x64=False`, contredisant `float_precision=64` `[RÉSOLU 19f93e5d]`
Le constructeur appelle `setup_device(float_precision=…)` (l. 166-173) qui pose `jax.config.update("jax_enable_x64", float_precision == 'float64')` (`SetDevice/set.py:106`), puis, dans la même méthode, la branche tfp fait `jax.config.update("jax_enable_x64", False)` (l. 297).
**Scénario** : `m = bf(backend="tfp", float_precision=64)` → l'utilisateur a demandé explicitement le 64 bits et l'obtient… nié 130 lignes plus bas, **globalement pour le processus** : tout calcul JAX ultérieur (y compris d'autres instances `bf` numpyro du même notebook) repasse en float32, sans le moindre warning. Erreur de conception silencieuse sur la précision numérique.

### M-08 — Sévérité : **Majeure** — `BayesForge/Main/main.py:279-289` : singletons de module mutés → deux instances `bf` se marchent dessus `[RÉSOLU 19f93e5d]`
Pour le backend numpyro, le constructeur écrit dans les singletons `dist` des modules `Effects`, `Gaussian`, `model_effects`, `DPMM`, `GMM`, `surv` : `_mod.dist.seed = self.seed ; _mod.dist._rng = self._rng`. La dernière instance construite **gagne pour tout le processus**, y compris pour les instances précédentes.
**Scénario** : `m1 = bf(rand_seed=1); m2 = bf(rand_seed=2); m1.dist.normal(0,1,sample=True)` → le tirage « de m1 » utilise le RNG de m2 (le commentaire l. 260-271 assume le partage mais ne traite pas la multi-instances). Côté tfp, `self.dist = tfp_dists` (l. 295) est la **classe** : l'état est de toute façon global et partagé entre instances.

### M-09 — Sévérité : **Majeure** — `BayesForge/Main/main.py:560/562` + `Data/manip.py:270` : `pd_to_jax` retourne la **chaîne** `"Error, no data found"`, stockée comme data puis splattée `[RÉSOLU antérieurement — vérifié dans le code courant]`
Si `m.df` ne couvre pas les arguments du modèle, `pd_to_jax` retourne un `str`. `fit` le stocke dans `self.data_on_model` (l. 560) puis :
- l. 564 : `"initial_means" not in self.data_on_model` devient une **recherche de sous-chaîne** sur le message d'erreur (toujours `True`) ;
- l. 567 : `self.data_on_model["data"]` → `TypeError: string indices must be integers` ;
- ou l. 665 : `**run_data` → `TypeError: argument after ** must be a mapping, not str`.
Aucun de ces messages ne dit « colonnes manquantes dans m.df ». Même chemin dans `svi` (l. 774-775 puis 830).

### M-10 — Sévérité : **Majeure** — `BayesForge/Main/main.py:542-548` + `557-560` : `fit()` sans modèle → `print` au lieu d'lever, puis `AttributeError` ; chemin NBDA mort `[RÉSOLU 82860ed8]`
`model is None` et `nbdaModel` False → `print("Argument model can't be None")` (l. 544) **sans return**, puis la suite s'exécute et plante en l. 560 (`self.pd_to_jax(self.model)` → `AttributeError: 'bf' object has no attribute 'model'` — `manip.__init__` ne définit pas `model`). Le chemin `else` (l. 547) référence `self.nbda`, attribut **jamais assigné** sur `bf` (seuls `self.NBDA` l. 235 et `models.nbda` existent), et `self.nbdaModel` ne peut jamais valoir `True` sur `bf` (`NBDA.py:53` le pose sur l'objet NBDA, pas sur `bf`) : branche morte. Un appel fautif doit lever `TypeError` immédiatement, pas imprimer puis crasher 16 lignes plus loin.

### M-11 — Sévérité : **Majeure** — `BayesForge/Main/main.py:564/571-572` : dispatch `gmm`/`pca` sur le **nom** de l'objet fourni par l'utilisateur `[RÉSOLU 19f93e5d]`
`self.model_name` est dérivé de `model.__name__` (l. 552-555) puis compare à `"gmm"`/`"pca"`.
- Une fonction utilisateur nommée `pca` déclenche l. 572 `model.get_model(model.type)` → `AttributeError: 'function' object has no attribute 'get_model'` ;
- une fonction nommée `gmm` déclenche l. 566-569 → `KeyError: 'data'` ou `KeyError: 'K'` sur un dictionnaire qui n'a rien à voir ;
- un callable sans `__name__` (instance de classe wrapper, partiel de partiel l. 552) → `AttributeError` en l. 555.
Le dispatch de modèle ne doit pas reposer sur une chaîne contrôlée par l'appelant.

### M-12 — Sévérité : **Majeure** — `BayesForge/Main/main.py:698-709` : le backend TFP ignore silencieusement ~20 options de `fit` `[RÉSOLU 19f93e5d]`
`mcmc_tfp.run` ne reçoit que `seed`, `model`, `n_chains`, `num_results`, `num_burnin_steps` et les données. Sont **avalés sans avertissement** : `step_size`, `inverse_mass_matrix`, `adapt_step_size`, `adapt_mass_matrix`, `dense_mass`, `target_accept_prob`, `trajectory_length`, `max_tree_depth`, `init_strategy`, `find_heuristic_step_size`, `forward_mode_differentiation`, `regularize_mass_matrix`, `thinning`, `postprocess_fn`, `chain_method`, `progress_bar`, `jit_model_args`, `extra_fields`, `init_params`, `potential_fn`, `kinetic_fn`, et `shard` (le bloc l. 523-535 s'exécute, émet même le warning EXPERIMENTAL, puis le résultat `run_data` n'est jamais utilisé côté tfp).
**Scénario** : `m.fit(model, obs=d, target_accept_prob=0.95, max_tree_depth=12, init_params=x)` sur tfp → l'échantillonneur tourne avec les défauts TFP, l'utilisateur croit avoir configuré NUTS. Le simple `print` l. 690-692 ne couvre pas ces oublis.

### M-13 — Sévérité : **Majeure** — `BayesForge/Main/main.py:661-666` et `702-709` : collision entre les clés du dictionnaire de données et la signature de `run` `[RÉSOLU 19f93e5d]`
`self.sampler.run(key, extra_fields=…, init_params=…, **run_data)` (numpyro `MCMC.run(rng_key, *args, extra_fields=(), init_params=None, **kwargs)`) et `self.sampler.run(seed=…, model=…, n_chains=…, num_results=…, num_burnin_steps=…, **self.data_on_model)` (tfp) : un argument de modèle nommé `extra_fields`, `init_params` (numpyro) ou `model`, `obs`, `seed`, `init`, `name`, `n_chains`, `num_results`… (tfp) produit `TypeError: got multiple values for argument '…'` — message qui incrimine les données de l'utilisateur sans explication. Un filtrage/préfixage ou un `raise` explicite s'impose.

### M-14 — Sévérité : **Majeure** — `BayesForge/Main/main.py:308` : `_resolve_key` casse sur un seed non entier et tronque les flottants `[RÉSOLU 19f93e5d]`
`jax.random.PRNGKey(int(seed))` :
- `m.fit(model, seed=jax.random.PRNGKey(0))` (idiome NumPyro réflexe) → `TypeError: Only scalar arrays can be converted to Python scalars` (vérifié par exécution) ;
- `seed=42.9` → `int()` **tronque silencieusement** en 42 ;
- `seed=True` → clé fixe `PRNGKey(1)` alors que l'utilisateur attend de l'entropie (cohérent avec `rand_seed=True` → aléatoire, pas ici).
Le même `int(seed)` apparaît dans `sample` (l. 1583) et `svi` (l. 833).

### M-15 — Sévérité : **Majeure** — `BayesForge/Main/main.py:702-703` + `Samplers/mcmc_tfp.py` (`init_Model(..., seed=0)`) : la graine du fit n'atteint pas l'initialisation TFP `[RÉSOLU 19f93e5d]`
`m.fit(seed=42)` côté tfp transmet la clé au sampler (`seed=self._resolve_key(...)`) mais `mcmc_tfp.run` appelle `self.init_Model(model, init=init, bijectors=bijectors, seed=0, …)` avec **0 en dur** : l'état initial de la chaîne est identique quel que soit le seed demandé. La promesse docstring de `_resolve_key` (« honoured verbatim », l. 302-304) est rompue pour la phase d'init du backend tfp.

### M-16 — Sévérité : **Majeure** — `BayesForge/Main/main.py:1280-1281` + `1365-1376` : `summary()` mute `m.posteriors` et le chemin arviz ignore `group_by_chain`/`filter_regex` `[RÉSOLU 19f93e5d]`
Double fragilité : (a) une méthode de lecture (`summary`) appelle `filter_posteriors` et **réduit de façon persistante** `m.posteriors`/`m.posteriors_by_chain` (documenté, mais combiné à M-05 cela peut ressusciter les postérieurs d'un ancien fit) ; (b) `group_by_chain` et `filter_regex` ne sont honorés que par le backend `jax` (l. 1296-1305) — avec le backend arviz par défaut ils sont **silencieusement ignorés** (le tableau vient de `diag.trace`, toujours groupé).

### M-17 — Sévérité : **Mineure** — `BayesForge/Main/main.py:1874` : le nom de fichier par défaut de `save()` est du code mort [RÉSOLU ec5589f3]
`path = f"{self.model_name}_bf.pkl" if self.model_name else "bf_model.pkl"` — or `fit`/`svi`/`_sbi_finish` remettent `self.model_name = None` à la fin (l. 727, 839, 877) : après tout fit, `save()` écrit toujours `bf_model.pkl`. Le nom disponible est `run_model_name`. Deux sauvegardes successives (gmm puis dpmm) s'écrasent mutuellement.

### M-18 — Sévérité : **Mineure** — `BayesForge/Main/main.py:1224-1225` : `sampler_stats(per_chain=True)` plante si `num_steps` absent [RÉSOLU ec5589f3]
`range(len(steps))` avec `steps=None` (p. ex. `extra_fields=('diverging',)` passé à `fit`, ou fit `mixed_hmc` où `extra_fields=()` l. 657) → `TypeError: object of type 'NoneType' has no len()`. Le garde l. 1182 ne teste que la non-vacuité du dict.

### M-19 — Sévérité : **Mineure** — `BayesForge/Main/main.py:658` : `extra_fields="diverging"` (chaîne) éclaté en caractères [RÉSOLU ec5589f3]
`tuple(extra_fields)` transforme `"diverging"` en `('d','i','v',…)` ; l'erreur levée ensuite par NumPyro liste des champs d'une lettre, sans lien avec l'oubli du list/tuple.

### M-20 — Sévérité : **Mineure** — `BayesForge/Main/main.py:511-514` + `Utils/rng.py:50-58` : validation des compteurs incomplète [RÉSOLU ec5589f3]
`_as_count` n'accepte que `float` entier : `num_chains="4"` passe tel quel et casse profond dans NumPyro ; `num_samples=0`/négatif, `thinning=0` ne sont pas rejetés tôt ; `rand_seed=1000.0` (flottant venant de R, le cas d'usage revendiqué de `_as_count` l. 509-510) lève `ValueError: Invalid rand_seed type` — incohérence entre les deux politiques de tolérance aux flottants.

### M-21 — Sévérité : **Mineure** — `BayesForge/Main/main.py:1098` et `1112` : `npe_posterior`/`sbc_coverage` avant tout fit → `AttributeError` au lieu du `RuntimeError` promis [RÉSOLU ec5589f3]
`hasattr(self.sampler, …)` évalue `self.sampler`, attribut **absent** tant que `fit`/`svi`/`abc`/`npe` n'a pas tourné avec le backend numpyro (le constructeur ne le pose que pour tfp, l. 296) → `AttributeError: 'bf' object has no attribute 'sampler'` au lieu de « Run m.npe(...) first ».

### M-22 — Sévérité : **Mineure** — `BayesForge/Main/main.py:1593` : `sample()` sans données → `AttributeError: 'NoneType' object has no attribute 'copy'` [RÉSOLU ec5589f3]
`data = self.data_on_model.copy()` sans garde `None` (ni validation du type dict). `m.sample(posterior=False)` sur une instance fraîche (prior predictive, usage légitime) plante avec un message non parlant.

### M-23 — Sévérité : **Mineure** — `BayesForge/Main/main.py:576` : `return print("X is required")` — abandon de fit muet et état à moitié muté [RÉSOLU ec5589f3]
Branche pca sans clé `X` : `fit` a déjà remplacé `self.model` (l. 572) et `self.data_on_model` (l. 540/560), puis retourne `None` sans exception — l'appelant ne peut pas tester l'échec (`fit` retourne toujours `None`) et l'objet reste dans un état incohérent.

### M-24 — Sévérité : **Mineure** — `BayesForge/Main/main.py:1801-1807` : `history` stocke des références, pas un instantané [RÉSOLU ec5589f3]
`history["data"] is m.data_on_model` et `history["posteriors"] is m.posteriors` : une mutation ultérieure (double transposition M-02, `filter_posteriors`) **rétro-modifie le snapshot** du fit précédent. Un historique censé rejouer un fit doit copier (au moins les dicts).

### M-25 — Sévérité : **Mineure** — `BayesForge/Main/main.py:634-643` + `2093-2099` : `shard_report()` mensonger quand le garde-fou retombe sur replicated [RÉSOLU ec5589f3]
Quand `_shard_check` renvoie `False` (l. 640-643), `run_data` redevient `self.data_on_model` (répliqué) mais `_last_shard_report`, écrit par `_auto_shard_data`, continue d'annoncer les tableaux dans `"sharded"` — l'audit demandé par `m.shard_report()` est faux après fallback.

### M-26 — Sévérité : **Mineure** — `BayesForge/Main/main.py:2102-2121` : le contrôle de correction du sharding est **fail-open** [RÉSOLU ec5589f3]
`_shard_check` avale toute exception interne (`l. 2118-2121`) et retourne `True` : si `run_shard_check` échoue (import, OOM, trace), le fit part en sharding **potentiellement biaisant** (le warning l. 1953-1974 dit lui-même « SILENTLY bias the posterior ») avec un simple « shard-check skipped ». Pour un garde de correction numérique, fail-closed (retour replicated) serait plus sûr.

### M-27 — Sévérité : **Mineure** — `BayesForge/Main/main.py:1820` : `plot()` avant fit → `AttributeError` sur `self.sampler` [RÉSOLU ec5589f3]
`kwargs.pop("sampler", self.sampler)` : avec le backend numpyro, `self.sampler` n'existe qu'après un fit. Et pour tout `run_model_name` hors {gmm, dpmm, bnnc, pca}, `plot` retourne `None` sans un mot (l. 1822-1834) — le commentaire l. 723-725 se plaint ironiquement de ce no-op pour les autres modèles sans le corriger.

### M-28 — Sévérité : **Mineure** — `BayesForge/Main/main.py:1877-1884` : `save()` détruit la sauvegarde précédente en cas d'échec [RÉSOLU ec5589f3]
`open(path, "wb")` tronque le fichier existant **avant** le `cloudpickle.dump` ; si le sérialisable échoue (closure nu-pickleable, device dans l'état), le `except` supprime le fichier : l'utilisateur perd l'ancienne sauvegarde valide en plus de la nouvelle. Écrire vers `path + ".tmp"` puis `os.replace`.

### M-29 — Sévérité : **Mineure** — `BayesForge/Main/main.py:1652-1655` : `except Exception` large autour de `get_samples()` [RÉSOLU ec5589f3]
Une vraie erreur de `get_samples` (état corrompu, device perdu) est convertie en `posterior_samples = None` puis en « No posterior samples: call m.fit() first » — diagnostic mensonger qui oriente l'utilisateur vers la mauvaise cause.

### M-30 — Sévérité : **Mineure** — `BayesForge/Main/main.py:1763` : attribution site observé ↔ entrée de data par égalité de valeur [RÉSOLU ec5589f3]
`next((k for k, v in data.items() if same(v, site["value"])), None)` : si deux entrées du dict sont des tableaux égaux (`{"y": y, "y_copy": y}`), le **premier** match gagne ; `sample(remove_obs=True)` (l. 1636-1637) peut alors mettre `y_copy` à `None` et laisser `y` conditionnant — le PPC conditionne silencieusement sur les données. Coût annexe : `np.array_equal` sur chaque paire (site × entrée) à chaque appel.

### M-31 — Sévérité : **Mineure** — `BayesForge/Main/main.py:1704-1713` : `trace = {}` en dernier recours → `sample()` retourne `{}` sans erreur [RÉSOLU ec5589f3]
Si les deux tentatives de trace échouent (`remove_obs=False`), `return_sites` est vide et `sample()` retourne `{}` (l. 1720-1721) — un résultat vide silencieux là où une exception nommant la cause serait honnête.

### M-32 — Sévérité : **Mineure** — `BayesForge/Main/main.py:1673/1676/1682` : `samples` négatif ou nul non validé [RÉSOLU 8c5b23ba]
`samples=0` → `Predictive(num_samples=0)` ou `jax.random.choice(..., (0,))` selon le chemin ; `samples=-5` → crash d'indexing en profondeur. Un `_as_count` + contrôle `> 0` (comme pour `fit`) s'impose.

### M-33 — Sévérité : **Mineure** — `BayesForge/Main/main.py:1847-1858` : `__setstate__` peut désynchroniser `n_devices` du mesh reconstruit [RÉSOLU 8c5b23ba]
`setup_multi_device(platform)` est rappelé **sans** `cores` : sur une machine au nombre de devices différent, `self.n_devices` (désérialisé de l'ancien état) ne correspond plus au mesh reconstruit ; `_auto_shard_data` (l. 2070) calcule alors la divisibilité avec le mauvais diviseur. Seul `RuntimeError` est intercepté (l. 1855).

### M-34 — Sévérité : **Mineure** — `BayesForge/Main/main.py:1945-1952` : branche morte dans `_maybe_warn_shard_limits` [RÉSOLU 8c5b23ba]
Le test `if self._data_sharding is None` est inatteignable depuis `fit` (le garde l. 524-532 fait déjà retomber `shard=False` avant l'appel l. 534) ; le warning « only 1 device » ne peut jamais être émis.

### M-35 — Sévérité : **Mineure** — `BayesForge/Main/main.py:206-207` + `Utils/rng.py:98-107` : `bf()` re-seed les RNG **globaux** du processus utilisateur [RÉSOLU 8c5b23ba]
`self._rng.seed_globals()` appelle `np.random.seed(base)` et `random.seed(base)` à chaque construction. Un script qui a seedé `np.random` pour sa propre reproductibilité perd son état dès qu'il instancie un `bf()` — effet de bord destructeur non documenté dans le docstring du constructeur.

### M-36 — Sévérité : **Mineure** — `BayesForge/Main/main.py:537-538` : réutilisation silencieuse des données du fit précédent [RÉSOLU 8c5b23ba]
`fit(B)` sans `obs` après `fit(A, obs=dA)` réutilise `dA` (l. 538) même si les clés ne correspondent pas au modèle B — l'erreur remonte alors de NumPyro, obscure. Le docstring l. 475-476 promet par ailleurs « built automatically from m.df when None », ce qui n'arrive que si `data_on_model` n'a **jamais** été posé.

### M-37 — Sévérité : **Mineure** — `BayesForge/Main/main.py:1334-1339` : `summary()` avant tout fit → `AttributeError` brut [RÉSOLU 8c5b23ba]
`getattr(self.diag, "trace", None)` : `self.diag` n'existe qu'après un fit → `AttributeError: 'bf' object has no attribute 'diag'` au lieu d'un « Run fit() first » (comparer au bon message de `filter_posteriors`, l. 1401-1402).

### M-38 — Sévérité : **Mineure** — `BayesForge/Main/main.py:1683` : `next(iter(...))` sur dictionnaire de draws vide → `StopIteration` [RÉSOLU 8c5b23ba]
`posterior_samples` vide (sampler sans sites latents, cas dégénéré mais atteignable avec `posterior={}`) → `StopIteration` non attrapé au lieu d'une `ValueError` explicite.

### M-39 — Sévérité : **Mineure** — `BayesForge/Main/main.py:113-117` (`_ExpandObserved`) : découpe batch/event silencieuse en cas d'incohérence de rang [RÉSOLU 8c5b23ba]
`batch = shape[:len(shape) - n_event] if n_event else shape` : si le rang du data stocké est **inférieur** à `event_shape` (data modifiée entre fit et `sample()`), le slice négatif produit un batch tronqué sans erreur ; l'`except (ValueError, TypeError, NotImplementedError): pass` l. 118-119 avale en plus toute incompatibilité d'expand. Le PPC peut alors dessiner un nombre de valeurs différent des données fittées — exactement ce que la classe cherche à empêcher.

### M-40 — Sévérité : **Mineure** — `BayesForge/Main/main.py:148-149` : `self.load = load()` masque le staticmethod `bf.load` sur les instances [RÉSOLU 8c5b23ba]
L'attribut d'instance écrase la méthode de classe (l. 1888). Ça fonctionne parce que `load.__call__` (`Resources/datasets.py:65-68`) redélègue à `bf.load`, mais le nom `load` a deux sémantiques (chargeur de datasets au niveau paquet, chargeur de session sur l'objet) et `from BayesForge import load; load("x.pkl")` lève `TypeError` (le constructeur `load.__init__` ne prend pas d'argument).

### M-41 — Sévérité : **Mineure** — `BayesForge/Main/main.py:1910-1924` : `cloudpickle.load` sur un fichier = exécution de code arbitraire [RÉSOLU 8c5b23ba]
Chargeur non sécurisé, documenté sans la mise en garde d'usage (ne charger que des fichiers de confiance). À noter positivement : la validation `isinstance(obj, bf)` (l. 1916) et l'avertissement de version (l. 1918-1922) sont bien faits.

### M-42 — Sévérité : **Mineure** — `BayesForge/Main/main.py:581-582` : défaut `latent_dim = data_dim` pour PCA [RÉSOLU 8c5b23ba]
Quand `latent_dim` est absent, il prend `X.shape[0]` (= data_dim après la transposition l. 578) : une PCA probabiliste à dimension latente égale à la dimension des données est un modèle dégénéré (le modèle est aussi riche que les données). Un défaut explicite (2, ou une erreur demandant la valeur) serait plus sûr.

### M-43 — Sévérité : **Mineure** — `BayesForge/Main/main.py:551-555` : `functools.partial` de `functools.partial` → `AttributeError` [RÉSOLU 8c5b23ba]
`model.func.__name__` suppose que `.func` est une fonction ; un partiel imbriqué (fréquent avec `partial(partial(model, a=1), b=2)`) n'a pas de `__name__`.

### M-44 — Sévérité : **Mineure** — `BayesForge/Main/main.py:719-721` : post-traitement PCA avec les postérieurs bruts du backend [RÉSOLU 8c5b23ba]
`self.models.pca.posterior = self.posteriors` : après un fit TFP, ce sont les draws chain-structurés (M-04) que `get_attributes` consomme comme s'ils étaient poolés.

### M-45 — Sévérité : **Mineure** — `BayesForge/Main/main.py:1227-1230` : E-BFMI poolé d'abord calculé sur les chaînes concaténées puis écrasé [RÉSOLU 8c5b23ba]
`_row(slice(None))` calcule un `ebfmi` avec `np.diff` **à travers les frontières de chaînes** (l. 1200-1204), valeur incorrecte qui n'est sauvée que parce que l. 1228-1230 la remplace par le min par chaîne. Tout consommateur futur de `_row(slice(None))` hériterait du calcul faux ; le code ne le signale pas.

### Points vérifiés **sains** (extraits de la liste de l'énoncé et surroundings)
- **l. 661-662** : la clé est bien passée en premier argument positionnel de `MCMC.run(rng_key, …)` (signature numpyro 0.22 vérifiée) ; `seed=0` explicite est bien distingué de `None` (test `is None`, l. 306) — pas de bug de falsiness.
- **l. 703-710** : le dispatch tfp passe bien `seed=` en mot-clé et `mcmc_tfp.run` accepte une clé PRNG (l. 176-180 de mcmc_tfp.py) — le problème n'est pas le passage de la clé mais les options ignorées (M-12) et l'init à seed 0 (M-15).
- **l. 707-714** : construction de `m.posteriors` correcte côté numpyro (pool + by_chain + copies `_full` superficielles — les tableaux sont partagés, pas dupliqués : pas de surcoût mémoire anormal).
- **l. 251-295** : la branche `dist = np_dists` (instance via `DistProxy`) vs `dist = tfp_dists` (classe statique) est telle qu'annoncée ; le partage de RNG avec les singletons (l. 279-289) est une bonne idée mal isolée (M-08).
- **l. 494-508** : validation de `sampler` contre `SAMPLERS`, garde `hmc_gibbs` et refus des samplers Gibbs hors numpyro — corrects et bien placés.
- **l. 667-676** : conversion de l'`AssertionError` « all sites must have unique names » en `ValueError` pédagogique — bon UX.
- **l. 842-877** : `_sbi_setup`/`_sbi_finish` copient le dict (`dict(data)`, l. 856) et rafraîchissent les quatre attributs de postérieurs — le seul chemin sans faille de cycle de vie (contraste avec `svi`, M-05). Regret : `_sbi_finish` et `svi` n'appellent pas `bind_diag_to_model` (contrairement à `fit`, l. 685-686/715-716) — `m.diag` post-svi/abc/npe n'a pas les méthodes liées.
- **l. 1198** : `2 ** max_tree_depth - 1` — précédence correcte.
- **l. 1340-1344** : détection runtime `ci_prob`/`hdi_prob` selon la version d'arviz — robuste.
- **l. 1379-1423** : `filter_posteriors` repart toujours de `_full` (idempotent, non cumulatif) — sémantique propre, message d'erreur clair.
- **l. 1465-1516** : chirurgie de signature `build_model_with_Y_None` — gestion correcte des contraintes POSITIONAL_OR_KEYWORD/KEYWORD_ONLY.
- **l. 1591-1637** : `sample()` travaille sur une copie puis reconstruit un dict neuf — **pas de mutation** de `data_on_model` (le point noir de mutation est `fit`, M-02).
- **l. 1639-1646** : validation de `posterior_draws` (axe 0 commun) — bon message.
- **l. 1662-1670** : contournement documenté du bug de `Predictive(infer_discrete=True)` — juste et bien commenté.
- **l. 2026-2100** : `_auto_shard_data` ne mute pas l'entrée, respecte les placements explicites, avertit sur les dimensions non divisibles — le meilleur morceau du module côté hygiène.
- **`BayesForge/__init__.py`** : `__all__` cohérent avec les exports (l. 22-25, 40-41) ; `__version__` avec fallback (l. 30-33) ; alias `BayesForge = BF` (l. 28) ; `BF_QUIET` pour les workers (l. 35-39).

---

## 2. Fragilités architecturales (hors bugs individuels)

- **F1 — Chaîne d'import JAX contre le propre contrat du paquet.** `BayesForge/__init__.py:3` (patch pxla) et `main.py:7` importent JAX **avant** tout appel à `setup_device`, alors que sa docstring (`SetDevice/set.py:52-55`) exige d'être appelée « before any JAX imports or usage » pour poser `XLA_FLAGS`. Ça ne tient que grâce à l'initialisation paresseuse de JAX ; un usage de JAX entre `import BayesForge` et `bf()` fige le nombre de devices. Les deux monkey-patches de l'`__init__.py` (l. 2-7, 13-20) avalent leur `ImportError` en silence et peuvent masquer une incompatibilité de version réelle.
- **F2 — La configuration JAX est un état global négocié à la loyale.** Plateforme, cores, x64 : chaque `bf()` écrase la config du processus (`set.py:105-106`), et la branche tfp la contredit (M-07). Deux instances aux réglages différents sont une illusion d'isolation.
- **F3 — Cycle de vie d'attributs non déclaré.** `model`, `sampler`, `diag`, `posteriors`, `posteriors_by_chain`, `extra_fields*`, `_max_tree_depth`, `nbda` (inexistant, M-10) naissent au fil des appels ; le code se protège par `getattr`/`hasattr` épars (l. 352, 1181, 1198, 1297, 2156) au lieu d'un `__init__` exhaustif — source directe des `AttributeError` non verbeux (M-21, M-22, M-27, M-37).
- **F4 — Dispatch sur noms de chaînes** (`"gmm"`, `"pca"`, `"NBDA"`, `run_model_name` dans `plot`) au lieu de types ou de marqueurs — M-11, et `plot` muet pour les autres noms (M-27).
- **F5 — Deux backends, deux contrats non écrits** : shapes des postérieurs (M-04), options honorées (M-12), graine d'init (M-15), `sampler_name`/`extra_fields` absents côté tfp. Rien dans le module ne formalise l'interface commune (`get_samples`, `run`, attributs posés).
- **F6 — Le dictionnaire de données est un objet vivant partagé** : référence stockée (M-02), mutations partielles (gmm/pca), réutilisation inter-fits (M-36), splat dans des signatures d'appel (M-13). Un `dict(data)` + validation des clés contre la signature du modèle en entrée de `fit` éliminerait quatre bugs d'un coup.
- **F7 — `print` comme politique d'erreur** (l. 544, 576, 690, 1785-1788, 1885, 1923) là où `raise`/`warnings` sont requis — l'appelant ne peut pas programmatiquement détecter l'échec.
- **F8 — Imports wildcard** (`main.py:69-70`, GMM/DPMM sans `__all__`) : `plot_gmm`, `gmm`, `dpmm` arrivent dans le namespace de `main` par magie d'ordre d'import ; tout renommage dans ces modules casse `plot()` (l. 1823) sans que le typage ni un linter ne protège.
- **F9 — Imports redondants** (jax l. 178/2028, os l. 199/2110, numpyro l. 780, numpy l. 1179, jnp l. 1582/2029, warnings l. 525/1944) et code mort (l. 61, 547, 1945) — signe d'une dette d'entretien dans un fichier de 2247 lignes qui gagnerait à extraire sharding, SBI et persistance dans des modules dédiés.

---

## 3. Décompte par sévérité

| Sévérité | Nombre | IDs |
|---|---|---|
| **Critique** | 3 | M-01, M-02, M-03 |
| **Majeure** | 11 | M-04 → M-16 |
| **Mineure** | 29 | M-17 → M-45 |
| **Total** | **43** | |

## 4. Résumé du module

`BayesForge/Main/main.py` est la façade monolithique du paquet : la classe `bf(manip)` y concentre la construction (device/mesh/RNG/distributions), le dispatch d'inférence (NUTS et variantes Gibbs via NumPyro, MCMC TFP, SVI, ABC/ABC-RF/NPE), la gestion des données (`data_on_model`, `pd_to_jax`), les postérieurs et leur filtrage, les résumés (ArviZ ou JAX), les diagnostics de sampler, le sharding multi-device expérimental, la persistance cloudpickle et le PPC (`sample`). L'ossature est globalement soignée sur les chemins numpyro : clés reproductibles via `RNG` (label-fold_in vs stream), copies `_full` des postérieurs, garde-fous de sharding, messages d'erreur pédagogiques à certains endroits (noms de sites dupliqués, autoguides inconnus, version de fichier sauvegardé).

Les trois défauts critiques résument le risque de release : `log_prob` est **cassé silencieusement** (les données n'atteignent jamais le modèle — M-01) ; `fit` **mute les données de l'utilisateur** avec une double transposition PCA qui peut produire des postérieurs faux sans erreur (M-02) ; et un **backend inconnu ou le SVI sur tfp est un no-op silencieux** (M-03). Autour, le backend tfp viole le contrat commun des postérieurs (M-04), le SVI laisse un état périmé capable de ressusciter un ancien fit (M-05) et force la descente en float32 de tout le processus (M-07), tandis que les singletons de module partagés (M-08) rendent la multi-instances trompeuse. Aucun de ces points ne nécessite une réécriture : validation du `backend` et des entrées en tête de `fit`/`svi`, copie du dict `obs`, `model_kwargs=` dans `log_prob`, normalisation des sorties TFP et rafraîchissement complet des attributs dans `svi` en viennent à bout. **Avis : non prêt pour release** tant que M-01 à M-03 (et idéalement M-04/M-05/M-07) ne sont pas corrigés ; le reste relève d'un durcissement robustesse planifiable.

---

### Module Diagnostic

**Périmètre** — audit en lecture seule de `BayesForge/Diagnostic/` (projet `/home/sosa/work/BF`) :
`Diag.py` (619 l., classe `diag`, legacy), `Diag2.py` (1 821 l., classe `diagWIP`, classe active — `main.py:60` l'importe sous le nom `diag`), `jax_diagnostics.py` (1 407 l.), `jax_summary.py` (76 l., déprécié), `patch_diag.py` (1 105 l.), `ppc.py` (1 608 l.), `regression_plot.py` (261 l.), `sensitivity.py` (744 l.), `sharding_safeguards.py` (393 l.), `__init__.py` (vide).

**Méthode** — relecture intégrale ligne par ligne depuis le disque (`nl -ba`), vérification empirique de la parité ArviZ 1.3.0 (`az.rhat/ess/mcse` vs implémentations manuelles sur chaînes normales, autocorrélées, gamma, poisson, S impair), vérification des contrats avec `Samplers/` (TFP, SVI, SBI) et `Main/main.py`, et tests d'exécution de l'API de tracé ArviZ installée. Aucun code modifié. Les numéros de ligne sont ceux du disque à la date d'audit (mtime : Diag/Diag2/patch_diag/ppc/sensitivity = 2026-11-09 ; jax_diagnostics/jax_summary/regression_plot/sharding_safeguards = 2026-08-30).

**Limite** — quelques micro-tests de bout en bout (tracé mono-paramètre, `to_az` sur noisy-chorus, SVI complet) n'ont pas pu aboutir pour des raisons de script ; les points correspondants reposent sur la lecture du code. En revanche, l'incompatibilité de tracé avec ArviZ 1.3.0 (bugs 3 et 4) a été **vérifiée empiriquement** dans cette session.

---

#### Points positifs vérifiés

- **Parité ArviZ exacte** des statistiques manuelles de `diagWIP` : `_rhat_manual` (Diag2.py:1238), `_ess_bulk_manual` (Diag2.py:1254), `_ess_tail_manual` à prob 0.1 (Diag2.py:1283), `_mcse_mean`/`_mcse_sd` (Diag2.py:1296/1305) donnent **0 écart** avec `az.rhat`/`az.ess`/`az.mcse` sur tous les profils testés. Le R-hat est bien le rank-normalized *folded* split R-hat de Vehtari et al. 2021 (Diag2.py:1238-1244) et l'E-BFMI utilise le carré moyen des différences d'énergie, conforme à Stan/`az.bfmi` (Diag2.py:1386-1390).
- **Gardes numériques** dans `Diag2._ess_raw_manual` : `n_draw < 4` → NaN (Diag2.py:1327-1328), chaîne constante → ESS = taille (Diag2.py:1330-1331) ; `_rhat_core` renvoie NaN si W = 0 (Diag2.py:1232-1233).
- **Garde de contrat** dans `diagWIP.__init__` : divergence entre `sampler.num_chains` déclaré et la dimension principale des échantillons → `ValueError` explicite (Diag2.py:329-338).
- **Aucun monkey-patching d'arviz** dans le module (les patches jax/numpyro de `BayesForge/__init__.py` sont hors périmètre mais signalés en fragilités architecturales).
- **PRNG des PPC reproductible** : `np.random.default_rng(0)` (ppc.py:1529) pour les bandes PIT.
- `_waic` manuel (Diag2.py:259) justifié : `az.waic` a été supprimé dans ArviZ ≥ 1.x ; ddof=0 et SE ×|sign| conformes à l'ancienne implémentation ArviZ.
- `jax_summary.py` est déprécié mais son renvoi vers `jax_diagnostics` est cohérent.
- `Diag2.plot_ess` (Diag2.py:803) et `Diag2.rank` (Diag2.py:817) sont bien adaptés à l'API ArviZ 1.x (vérifié empiriquement : OK).

---

#### Volet A — Exactitude mathématique

**Bug 1 — CRITIQUE — Côté TFP, `log_accept_ratio` contient en réalité `leapfrogs_taken` et la vraie statistique est silencieusement jetée** `[RÉSOLU 21232cba]`
- Emplacement : `BayesForge/Diagnostic/Diag2.py:349-354` (liste par défaut à 4 noms) et `Diag2.py:385-388` (`zip` sans `strict`) ; même schéma dans `BayesForge/Diagnostic/Diag.py:53` et `Diag.py:68`. Cause racine : `BayesForge/Samplers/mcmc_tfp.py:43-50`, le `trace_fn` renvoie **5** statistiques dans l'ordre `(target_log_prob, leapfrogs_taken, has_divergence, energy, log_accept_ratio)`.
- Risque : le `zip` apparie positionnellement 4 noms à 5 tenseurs → `log_accept_ratio` ← nombre de leapfrogs (entiers 16…1024), et le vrai `log_accept_ratio` est **perdu sans erreur** (zip non strict). Tout ce qui consomme `sample_stats.log_accept_ratio` (histogrammes, plots de taux d'acceptation, analyses manuelles) affiche des valeurs sans rapport avec le taux d'acceptation, et l'information est irrécupérable a posteriori.
- Scénario : `m.fit(backend="tfp")` puis `m.diag.to_az(backend="tfp")` avec les noms par défaut → `az.plot_pair(diag.trace, var_names=["log_accept_ratio"])` montre un escalier de comptes de leapfrogs ; l'utilisateur diagnostique à tort une acceptation pathologique. Les divergences et l'énergie (donc E-BFMI) restent correctes, ce qui masque le problème.

**Bug 2 — CRITIQUE — Postérieurs SBI : les 4 « pseudo-chaînes » garantissent structurellement R-hat ≈ 1 et ESS ≈ n** `[RÉSOLU 21232cba]`
- Emplacement : `BayesForge/Samplers/SBI/simulate.py:324-329` (`num_chains` = 4 si n ≥ 16 et n % 4 == 0) et `simulate.py:331-335` (`reshape` des draws indépendants en (4, n/4)) ; accepté sans avertissement par `BayesForge/Diagnostic/Diag2.py:370-377` (construction de la trace via `az_from_dict`) — la garde de cohérence `Diag2.py:329-338` passe puisque 4 == 4.
- Risque : R-hat compare variance inter-chaînes et intra-chaînes ; ici les « chaînes » sont le simple reshape d'un même lot de draws i.i.d. de l'approximateur NPE/ABC. R-hat ≈ 1,00 est **garanti quel que soit le calage réel du posterior** (un NPE mal entraîné, effondré sur une mode, passera avec mention). ESS ≈ n pour la même raison. Le diagnostic de convergence principal du module devient une machine à rassurer.
- Scénario : entraînement NPE avec 2 000 draws acceptés → `diag(sbi_result).diagnose()` → « rhat 1.00, ess 2000 » alors que la network flow a manqué une mode. Le docstring de `simulate.py:326-327` assume ce comportement (« expected ~1 and ~n without warnings »), mais le module Diagnostic n'émet **aucun avertissement** et le rapport ne le signale pas.

**Bug 3 — MAJEUR — R-hat ≈ 0,998 avec une seule chaîne au lieu de NaN** `[RÉSOLU 0f3ba554]`
- Emplacement : `BayesForge/Diagnostic/Diag2.py:1227-1236` (`_rhat_core`, aucune garde `m < 2`) et `Diag2.py:1238` (`_rhat_manual`).
- Risque : avec C = 1, le split produit deux moitiés de la même chaîne ; le R-hat mesure la cohérence de deux moitiés d'une même chaîne et tend vers 1 (mesuré : **0,998** là où `az.rhat` renvoie NaN précisément pour dire « non calculable »). Le seuil 1,01 de `diagnose()` (Diag2.py:1404) est toujours satisfait : fausse assurance de convergence.
- Scénario : `m.fit(..., num_chains=1)` (chaîne unique, courant sur petits modèles ou en CI) → `diag.summ()` affiche rhat ≈ 0,998 pour tous les paramètres ; l'utilisateur conclut « convergé » alors que R-hat n'a aucun sens avec une chaîne.

**Bug 4 — MAJEUR — Chaînes de longueurs inégales : ESS = NaN là où ArviZ calcule** `[RÉSOLU 27fe8dd1 — prémisse corrigée : des chaînes ragged ne peuvent pas entrer (tableaux rectangulaires) ; l'écart réel était la garde n_draw<4 appliquée après le split, chaînes de 4 à 7 draws]`
- Emplacement : garde `BayesForge/Diagnostic/Diag2.py:1327-1328` (`n_draw < 4` après split → NaN pour toute la variable).
- Risque : avec des chaînes ragged (mesuré : longueurs 4 et 6), la chaîne la plus courte donne 2 draws après split et la garde invalide l'ESS de **toutes** les chaînes ; ArviZ, lui, calcule (mesuré : 33,1). Le résumé devient inutilisable précisément dans les cas courts/interrompus où l'utilisateur a le plus besoin de l'ESS.
- Scénario : arrêt anticipé d'une chaîne (préemption, thinning asymétrique, reprise de warmup) → `diag.summ()` → colonne ESS entièrement NaN, sans message expliquant pourquoi.

**Bug 5 — MAJEUR — `jax_diagnostics._ess_raw` cumule les deux implémentations sans les gardes de `Diag2`** `[RÉSOLU 0f3ba554]`
- Emplacement : `BayesForge/Diagnostic/jax_diagnostics.py:212-245` (aucune garde `n_draw < 4`, aucune garde chaîne constante, contrairement à `Diag2.py:1327-1331`).
- Risque : `n_draw = 1` → division par `n_draw - 1 = 0` (jax_diagnostics.py:230) → inf/NaN ; chaîne constante → `var_plus = 0` → NaN à `jax_diagnostics.py:238` au lieu de ESS = taille ; ESS négatif ou aberrant propagé silencieusement dans les résumés et les plots. Deux portages de la même formule coexistent avec des comportements numériques différents selon l'entrée (lequel est appelé dépend du chemin choisi par l'utilisateur).
- Scénario : paramètre pinné à son prior (posterior constant, ex. donnée non informative) → `ess(...)` renvoie NaN au lieu de 4×n_draw ; un test CI qui assert `ess > 0` échoue sans expliquer la cause.

**Bug 6 — MAJEUR — Interprétation des axes de la log-vraisemblance par heuristique `shape[0] <= 20`** `[RÉSOLU 0f3ba554]`
- Emplacement : `BayesForge/Diagnostic/jax_diagnostics.py:1175` (`compute_log_likelihood`) et `jax_diagnostics.py:1340` (variante `_manual`).
- Risque : le code décide « première dimension = chaînes si ≤ 20, sinon observations ». Un posterior **aplati** (sans dimension chaîne) d'un modèle à ≤ 20 observations est lu comme 12 « chaînes » → matrice (n_obs, n_draws) transposée/erronée → LOO, WAIC, p_loo et k-hat **silencieux et faux** (aucun avertissement). Inversement, un fit à > 20 chaînes est lu comme observations.
- Scénario : modèle à 12 observations, `compute_log_likelihood(model, m.sampler.get_samples(group_by_chain=False))` → ELPD absurde accepté sans erreur ; la comparaison de modèles conclut à l'envers.

**Bug 7 — MINEUR — Formule de tree depth correcte seulement pour NUTS** `[RÉSOLU 4fd19f24]`
- Emplacement : `BayesForge/Diagnostic/Diag2.py:1446` : `tree_depth = ceil(log2(num_steps + 1))`.
- Risque : exacte pour NUTS (numpyro renvoie `num_steps = binary_tree.num_proposals = 2^d − 1`, cf. `Samplers/numpyro/hmc.py:456` et le docstring l.70-73), mais pour un HMC simple à `num_steps` fixe la « profondeur » est inventée (num_steps = 100 → profondeur 7) et le compteur de saturation `per_chain_max_treedepth` (Diag2.py:1447) compte des violations inexistantes.
- Scénario : `m.fit(method="hmc", num_steps=100, max_treedepth=10)` OK, mais avec `max_treedepth=3` le rapport signale une saturation alors que HMC n'a pas d'arbre.

**Bug 8 — MINEUR — Percentiles de la tail-ESS codés en dur (5 %/95 %)** `[RÉSOLU 4fd19f24]`
- Emplacement : `BayesForge/Diagnostic/jax_diagnostics.py:278-282` (`_ess_tail_1d`, q05/q95 en dur).
- Risque : la valeur par défaut correspond à ArviZ (prob = 0,1), mais l'utilisateur ne peut pas calculer une tail-ESS aux quantiles 1 %/99 %, utile pour les posteriors à queues très lourdes ; de plus `az.ess(method="tail")` exige désormais `prob=` explicite en ArviZ 1.3 — toute future délégation à ArviZ devra adapter l'appel.
- Scénario : posterior de loi de Pareto à alpha ≈ 2, l'ESS 5/95 masque l'effondrement de l'ESS au quantile 1 %.

**Bug 9 — MINEUR — LOO et LOO-PIT supposent `reff = 1,0` par défaut** `[RÉSOLU 4fd19f24]`
- Emplacement : `BayesForge/Diagnostic/jax_diagnostics.py:793-794` (`loo`) et `BayesForge/Diagnostic/ppc.py:1516` (`ppc_loo_pit`, `reff=1.0 if reff is None`).
- Risque : `reff = 1` suppose des draws indépendants ; sur une chaîne autocorrélée (ESS/n = 0,05), les poids PSIS sont sur-informés et les bandes de calibration du PIT sont trop étroites → faux « le modèle ne cale pas » (ou fausse inversion). Le paramètre existe mais le défaut est le mauvais choix pour des sorties MCMC.
- Scénario : chaîne NUTS fortement autocorrélée sur un modèle hiérarchique → LOO-PIT sort un drapeau rouge alors que le modèle est correct.

---

#### Volet B — Dimensions, vectorisation, axes (chains, draws)

**Bug 10 — MAJEUR — `_plot_ess_evolution` normalise toutes les chaînes sur la longueur de la première** `[RÉSOLU 0f3ba554]`
- Emplacement : `BayesForge/Diagnostic/patch_diag.py:343` : `S_total = expanded[0][1].shape[1]`.
- Risque : la grille `ns = [max(4, int(f * S_total)) ...]` (patch_diag.py:345) est calculée sur la chaîne 0 ; si les chaînes ont des longueurs différentes, les tranches `[:, :n]` des autres chaînes dépassent leur borne → `IndexError` ou courbe ESS tronquée silencieusement selon le chemin de découpe. Le plot « ESS en fonction du nombre de draws » est faux dès que les chaînes ne sont pas identiques.
- Scénario : reprise d'échantillonnage sur une seule chaîne (1 000 draws d'un côté, 500 de l'autre) → `plot_ess_evolution()` plante ou affiche une courbe qui ignore la moitié des draws.

**Bug 11 — MAJEUR — Contrat `get_samples(group_by_chain=...)` violé côté TFP ; Main confond les deux vues** `[RÉSOLU 19f93e5d]`
- Emplacement : `BayesForge/Samplers/mcmc_tfp.py:120-132` (`get_samples` ignore complètement l'argument et renvoie toujours la forme pmap (C, S, …)) ; `BayesForge/Main/main.py:712` (`posteriors_by_chain = posteriors`, même objet pour les deux vues).
- Risque : l'appelant qui demande des draws aplatis (`group_by_chain=False`) reçoit (C, S, …) et itère sur les **chaînes** comme si c'étaient des draws (PPC, prédictions postérieures, moyennage) → résultats faux d'un facteur n_chains en effectif. Le docstring (mcmc_tfp.py:124) l'admet (« currently always returns with chain dimension ») mais rien n'empêche l'appel silencieux.
- Scénario : `ppc.get_yrep` ou un script utilisateur appelle `sampler.get_samples(group_by_chain=False)` sur un fit TFP à 4 chaînes → les 4 chaînes sont traitées comme 4 draws.

**Bug 12 — MINEUR — `zip` sans `strict` sur les paires noms/valeurs (motif général)** `[RÉSOLU 4fd19f24]`
- Emplacement : `BayesForge/Diagnostic/Diag2.py:387` et `Diag2.py:393` ; `BayesForge/Diagnostic/Diag.py:68` et `Diag.py:73`.
- Risque : toute désynchronisation de longueur entre les listes de noms et les tenseurs du sampler est tronquée silencieusement (le bug 1 en est la conséquence concrète) ; un futur champ ajouté au `trace_fn` sans mise à jour de `sample_stats_name` disparaît sans bruit.
- Scénario : ajout d'une 6ᵉ statistique dans `mcmc_tfp.trace_fn` → ignorée sans erreur ni warning.

---

#### Volet C — Architecture, reproductibilité, contrats avec Main et les Samplers

**Bug 13 — MAJEUR — Contrôle d'intégrité du sharding « fail-open »** `[RÉSOLU 0f3ba554]`
- Emplacement : `BayesForge/Main/main.py:2118-2123` (`_shard_check` renvoie `True` sur exception **et** sur rapport `None`) et `BayesForge/Diagnostic/sharding_safeguards.py:390-393` (`run_shard_check` ne lève jamais, renvoie `None` sur la moindre exception).
- Risque : la sauvegarde conçue pour détecter une log-densité shardée divergeant de la référence répliquée se **désactive précisément quand elle plante** (environnement XLA mal configuré, `initialize_model` qui échoue, changement d'API numpyro) : un simple warning « shard-check skipped » et le fit continue **shardé**, potentiellement faux. Un garde-fou de correction ne doit pas être fail-open par défaut.
- Scénario : déploiement sur une machine sans initialisation multi-processus JAX → exception avalée (sharding_safeguards.py:390-393) → `main.py:2122-2123` renvoie True → fit shardé avec log-densité possiblement erronée, sans que l'utilisateur ait choisi ce niveau de risque.

**Bug 14 — MAJEUR — SVI : les draws variationnels acceptés comme « chaîne unique », R-hat/ESS produits sans avertissement** `[RÉSOLU 0f3ba554]`
- Emplacement : `BayesForge/Diagnostic/Diag2.py:370-377` (`to_az` accepte tout objet portant `svi`/`az_from_dict`) ; `BayesForge/Samplers/SVI/svi_numpyro.py:11` (`num_chains = 1`) et `svi_numpyro.py:62-64` (`get_samples(group_by_chain=True)` → (1, N)).
- Risque : les draws d'une approximation variationnelle ne sont pas des chaînes MCMC ; la garde de cohérence passe (1 == 1) et `diagnose()` calcule R-hat (≈ 0,998 via le bug 3) et ESS comme s'il y avait une convergence à vérifier. Le rapport ne distingue nulle part « échantillons d'approximation » et « échantillons MCMC ».
- Scénario : `m.fit(method="svi")` → `m.diag.diagnose()` → « rhat 0,998, ess 4 900 » sur un posterior variationnel effondré en une mode ; l'utilisateur empile ce chiffre dans un papier.

**Bug 15 — MINEUR — Duplication `_acf`** `[RÉSOLU 4fd19f24]`
- Emplacement : `BayesForge/Diagnostic/Diag2.py:25` et `BayesForge/Diagnostic/patch_diag.py:20` (implémentations actuellement cohérentes).
- Risque : deux copies de l'autocorrélation ; la correction future d'un bord (normalisation, mean global vs par chaîne) ne sera appliquée que sur l'une → divergence silencieuse entre le résumé matplotlib et le résumé plotly.
- Scénario : fix de normalisation dans Diag2 uniquement → les plots plotly d'ESS local deviennent incohérents avec le tableau.

**Bug 16 — MINEUR — Duplication E-BFMI** `[RÉSOLU 4fd19f24]`
- Emplacement : `BayesForge/Diagnostic/Diag2.py:1386` (`_ebfmi_manual`) ≡ `BayesForge/Diagnostic/sensitivity.py:593` (dans `divergence_energy_plot`).
- Risque : même formule (correcte) codée deux fois ; le seuil et la convention peuvent diverger entre le rapport `diagnose()` et le plot de sensibilité.
- Scénario : changement de seuil dans `diagnose` (paramètre `ebfmi_threshold`, Diag2.py:1403) non répercuté dans sensitivity.py → deux verdicts contradictoires sur le même fit.

**Bug 17 — MINEUR — Graine fixe `random_state=42` pour le sous-échantillonnage des pair plots** `[RÉSOLU 4fd19f24]`
- Emplacement : `BayesForge/Diagnostic/Diag2.py:939` ; `BayesForge/Diagnostic/patch_diag.py:378`.
- Risque : reproductible (bien) mais **toujours le même sous-ensemble** : une multimodalité rare ou un outlier structurel peut être systématiquement absent de tous les plots, run après run, sans que l'utilisateur puisse changer la graine.
- Scénario : 50 000 draws, deux modes de poids 95/5 → le sous-échantillon de 2 000 points rate la mode mineure de façon identique à chaque inspection visuelle.

**Bug 18 — MINEUR — `__init__.py` vide : aucune API publique du module** `[RÉSOLU 4fd19f24]`
- Emplacement : `BayesForge/Diagnostic/__init__.py` (0 ligne).
- Risque : les appelants importent les fichiers par leur chemin (`from BayesForge.Diagnostic.Diag2 import diagWIP`) et la dualité `diag`/`diagWIP` fuit jusqu'à l'utilisateur final ; `main.py:60` fait lui-même `import diagWIP as diag`, confirmant que le nom public n'est nulle part fixé.
- Scénario : renommage/fusion des fichiers → tous les scripts utilisateurs cassent ; entre-temps, deux classes aux signatures différentes (`to_az`, plots) restent publiquement importables.

**Bug 19 — MINEUR — `diagnose()` retourne une chaîne formatée, sans `var_names`** `[RÉSOLU 4fd19f24]`
- Emplacement : `BayesForge/Diagnostic/Diag2.py:1400-1406` (signature `-> str`, pas de paramètre `var_names`).
- Risque : pas exploitable programmatiquement (pas de dict/DataFrame), impossible de filtrer sur un sous-ensemble de paramètres, difficile d'asserter en test ; le parsing de texte devient l'intégration par défaut.
- Scénario : un CI qui veut `assert report["mu"]["rhat"] < 1.01` doit parser une chaîne.

**Bug 20 — MINEUR — Effet de bord de `summary` (filtrage persistant des posteriors)** `[RÉSOLU 19f93e5d — M-16]`
- Emplacement : `BayesForge/Main/main.py:1274-1306` (le chemin `summary` applique `filter_posteriors`, qui retire les sites vides de l'état stocké).
- Risque : une méthode en apparence en lecture seule modifie l'objet ; le résultat d'un second appel dépend du premier (et des plots intercalés), ce qui casse l'idempotence attendue d'un résumé.
- Scénario : `m.summary()` puis changement de données puis `m.summary()` → le filtrage du premier appel contamine le second.

**Bug 21 — MINEUR — Dualité `diag` (legacy) / `diagWIP` (active) maintenue en parallèle** `[RÉSOLU 4fd19f24]`
- Emplacement : `BayesForge/Diagnostic/Diag.py` (classe `diag`, 619 l.) vs `BayesForge/Diagnostic/Diag2.py` (classe `diagWIP`) ; `main.py:60` n'utilise que `diagWIP`.
- Risque : la classe legacy reste importable avec des signatures différentes (`to_az`, plots, pas de cross-check num_chains) ; les bugs sont corrigés deux fois (zip non strict, plots) ou pas du tout (voir bug 22) ; la documentation et les exemples peuvent pointer la mauvaise classe. Le nom « diagWIP » signale de plus une classe officiellement « work in progress » comme implémentation de production.
- Scénario : un tutoriel `from BayesForge.Diagnostic.Diag import diag` reproduit des comportements différents de `m.diag` et l'utilisateur ne peut pas comparer les chiffres.

---

#### Volet D — Robustesse API et compatibilité

**Bug 22 — MAJEUR — Toute la surface de tracé de `Diag.py` est incompatible avec l'ArviZ épinglé (≥ 1,2)** `[RÉSOLU 0f3ba554]`
- Emplacement : `BayesForge/Diagnostic/Diag.py:132` et `Diag.py:298` (`az.plot_posterior` — **n'existe plus** en ArviZ 1.3.0), `Diag.py:260` (`az.plot_density` — supprimé), `Diag.py:159-160` et `Diag.py:304` (`az.plot_rank(..., ax=axes)` — l'API 1.x rejette `ax=`), `Diag.py:619` (`az.plot_compare` avec kwargs legacy), plus `Diag.py:117` (`plot_trace(kind=...)`), `Diag.py:182` (`plot_forest(ess=...)`), `Diag.py:237` (`plot_pair(kde_kwargs=...)`) dont les signatures 0.x ne passent plus. Épinglage : `pyproject.toml:44` (`arviz >= 1.2.0`). Vérifié empiriquement : `AttributeError` sur `plot_posterior`, `ValueError` sur `plot_rank(ax=)`.
- Risque : la classe legacy est **totalement morte** sur l'environnement supporté ; les exemples/doc qui l'utilisent plantent dès le premier plot. Fragilité latente associée : `plt.subplots(1, len(priors_name))` (Diag.py:131, 158, 297, 303) avec `squeeze` par défaut renvoie un `Axes` scalaire pour un modèle mono-paramètre — tout code aval qui itère sur `axes` casserait même sur un ArviZ 0.x.
- Scénario : `diag(trace).plot_posterior()` sur l'environnement installé → `AttributeError: module 'arviz' has no attribute 'plot_posterior'`.

**Bug 23 — MAJEUR — `diagWIP.plot_compare()` plante toujours sur ArviZ ≥ 1,2** `[RÉSOLU 0f3ba554]`
- Emplacement : `BayesForge/Diagnostic/Diag2.py:656` (méthode, docstring recopié de l'API arviz 0.x) et appel `Diag2.py:748-763` (`insample_dev`, `plot_standard_error`, `plot_ic_diff`, `order_by_rank`, `legend`, `figsize`, `textsize`, `labeller`, `plot_kwargs`, `ax`, `backend`…). Vérifié empiriquement sur arviz 1.3.0 : `ValueError: Keyword arguments ['insample_dev', ...] have been passed as **kwargs but have no active aesthetic mapped to them`.
- Risque : la méthode de comparaison de modèles *avec graphique* de la classe active est inutilisable sur la version épinglée ; la signature publique (celle d'arviz 0.x) est en outre un mensonge sur le comportement réel. Contrairement aux bugs silencieux, celui-ci est bruyant — mais il ne peut pas « marcher par accident ».
- Scénario : `m.compare([m1, m2]).plot_compare()` → ValueError systématique ; le tableau `compare()` (sans plot) fonctionne, ce qui rend la panne d'autant plus surprenante.

**Bug 24 — MAJEUR — `exclude` silencieusement ignoré par `_summary` (backend arviz)** `[RÉSOLU 0f3ba554]`
- Emplacement : `BayesForge/Diagnostic/patch_diag.py:766-767` : quand `exclude` est non vide, le code pose `kw["filter_vars"] = None` et ne propage jamais la liste d'exclusion à `var_names`.
- Risque : `summary(exclude=[...])` renvoie le tableau **avec** les variables à exclure ; l'utilisateur qui masque des dérivées (`divergence`, `energy`, paramètres de nuisance) les voit rester dans le rapport, et peut publier un tableau qu'il croyait filtré.
- Scénario : `m.summary(exclude=["~mu"])` → `mu` toujours dans la table, sans warning.

**Bug 25 — MINEUR — `diagWIP.__init__` exige un objet sampler, pas un dict d'échantillons** `[RÉSOLU 4fd19f24]`
- Emplacement : `BayesForge/Diagnostic/Diag2.py:329-338` (lit `sampler.num_chains` et `posterior_samples` issus de `get_samples`).
- Risque : impossible de construire un diagnostic sur des draws externes (provenance d'un autre logiciel, draws sauvegardés) sans fabriquer un faux objet duck-typé ; l'erreur remontée (AttributeError lointain) ne dit pas quoi faire. Le module fournit bien un cross-check exigeant (bien), mais aucune porte d'entrée « dict + num_chains ».
- Scénario : reprise de draws Stan sauvegardés en npz → l'utilisateur doit réimplémenter SBIPosterior-like lui-même.

**Bug 26 — MINEUR — Divergences d'API entre `diag` et `diagWIP` non documentées dans le module** `[RÉSOLU 4fd19f24]`
- Emplacement : `BayesForge/Diagnostic/Diag2.py` (pas de `plot_ppc`, pas de paramètre `num_chains=` sur les plots, pas d'attribut `warnings`) vs attentes héritées de `Diag.py`.
- Risque : les appelants migrés `diag` → `diagWIP` (comme `main.py:60` qui masque le changement par l'alias) obtiennent des AttributeError ou des comportements absents sans message de migration.
- Scénario : code utilisateur `m.diag.plot_ppc(...)` → AttributeError ; rien dans le module n'indique où aller (ppc.py est un namespace séparé).

---

#### (a) Résumé du module

`Diagnostic` est le cœur d'évaluation des fits BayesForge : statistiques de convergence manuelles (`diagWIP`, Diag2.py) portées fidèlement d'ArviZ et **vérifiées à zéro écart** contre `az.rhat/ess/mcse`, boîtes à outils jax/numpyro pures (jax_diagnostics.py : ESS, HDI, PSIS, LOO, WAIC, compare, PIT), PPC plotly (ppc.py), sensibilité aux priors et checks de multimodalité (sensitivity.py), instrumentation plotly additionnelle (patch_diag.py), tracé de régression (regression_plot.py) et garde-fous de sharding (sharding_safeguards.py). La justesse mathématique du noyau (R-hat folded, Geyer, E-BFMI, MCSE) est excellente et supérieure à ce qu'on attend d'un port maison. Les faiblesses sont ailleurs : (1) deux accidents d'intégration qui **corrompent silencieusement les diagnostics** — statistiques TFP décalées (bug 1) et pseudo-chaînes SBI (bug 2) ; (2) une migration ArviZ 1.x **inachevée** — `diagWIP.plot_compare` et toute la surface de plots de `Diag.py` sont cassés sur la version épinglée (bugs 22-23) ; (3) des cas limites (1 chaîne, chaînes inégales, posterior constant, ≤ 20 observations) où les chiffres produits sont faux ou NaN sans explication (bugs 3-6) ; (4) un garde-fou d'intégrité du sharding fail-open (bug 13).

#### (b) Fragilités architecturales

1. **Contrats sampler→diagnostic non fermés.** Le diagnostic fait confiance aux noms (`sample_stats_name` positionnel), à la forme (chaîne-major) et à la sémantique (« des draws i.i.d. sont des chaînes ») déclarés par les samplers. TFP décale les stats, SVI et SBI déclarent des « chaînes » qui n'en sont pas ; la seule garde réelle (Diag2.py:329-338) vérifie une arithmétique de dimensions, pas la sémantique. Il manque un type de contrat explicite (« MCMC multi-chaînes » vs « draws indépendants ») qui désactiverait R-hat/ESS pour les seconds.
2. **Trois portages parallèles des mêmes statistiques** (Diag2 manuel, jax_diagnostics, arviz) avec des gardes différentes (bug 5) et des implémentations dupliquées (`_acf`, E-BFMI, bugs 15-16) : le comportement d'un utilisateur dépend du chemin d'import qu'il choisit.
3. **Dualité diag/diagWIP** sans API publique (`__init__.py` vide) ni dépréciation effective : la classe legacy, cassée sur l'ArviZ épinglé, reste importable sous le nom historique.
4. **Migration ArviZ 1.x partielle** : le noyau numérique a été mis à jour (et la suppression de `az.waic` contournée proprement), mais la couche plots de Diag.py et `plot_compare` de Diag2 sont restées à l'API 0.x — signe qu'aucun test d'intégration n'exécute les méthodes de tracé sur la version épinglée.
5. **Garde-fous fail-open** (shard-check) et **effets de bord** (`summary` qui filtre l'état) : les invariants d'un run dépendent de l'ordre des appels et de l'environnement.
6. Hors module mais adjacent : les monkey-patches jax/numpyro de `BayesForge/__init__.py` s'appliquent à tout import du paquet et peuvent déplacer les valeurs de diagnostics entre environnements (à documenter avec le module).

#### (c) Décompte par sévérité

| Sévérité | Nombre | Bugs |
|---|---|---|
| Critique | 2 | 1 (stats TFP décalées), 2 (pseudo-chaînes SBI) |
| Majeure | 11 | 3 (R-hat 1 chaîne), 4 (ESS NaN chaînes inégales), 5 (`_ess_raw` sans gardes), 6 (heuristique ≤ 20 obs), 10 (S_total chaîne 0), 11 (contrat `group_by_chain` TFP), 13 (shard-check fail-open), 14 (SVI sans avertissement), 22 (plots Diag.py vs ArviZ 1.x), 23 (`plot_compare` Diag2), 24 (`exclude` ignoré) |
| Mineure | 13 | 7 (treedepth HMC), 8 (percentiles en dur), 9 (`reff=1.0`), 12 (zip sans strict), 15 (`_acf` dupliqué), 16 (E-BFMI dupliqué), 17 (graine 42 fixe), 18 (`__init__.py` vide), 19 (`diagnose()` → str), 20 (effet de bord `summary`), 21 (dualité diag/diagWIP), 25 (sampler requis), 26 (divergences d'API) |
| **Total** | **26** | |

---

### Module Utils

Audit de release — `BayesForge/Utils/` (racine `/home/sosa/work/BF`). Neuf fichiers lus intégralement ; les affirmations marquées ✅ ont été **vérifiées empiriquement** dans l'environnement du projet (JAX 0.11.2, `jax_enable_x64=False`, CPU).

---

## 1. BayesForge/Utils/Effects.py (680 l.)

### [Critique] `varying_slope_non_centered` — `NameError: name 'm' is not defined`, fonction publique 100 % cassée `[RÉSOLU 94e80673]`
- **Emplacement** : `BayesForge/Utils/Effects.py:458`, `:462`, `:467`
- **Risque** : trois appels utilisent `m.dist.lkj_cholesky`, `m.dist.exponential`, `m.dist.normal` alors que `m` n'existe pas dans le module (le reste du fichier utilise le singleton `dist`, l. 9). La fonction lève `NameError` à **chaque** appel, y compris quand l'appelant fournit `L_Rho` et `sigma` (le site `z`, l. 467, passe toujours par `m.dist`).
- **Scénario** ✅ : `effects.varying_slope_non_centered(cat_idx, grp_idx, 2, 2, sample=True)` → `NameError: name 'm' is not defined`. Aucun test dans `Test/` ne couvre cette fonction (grep : 0 occurrence), alors que `PostModel/to_latex.py:530` et `PostModel/model_graph.py:213` la reconnaissent comme API documentée — la doc promet donc une fonction qui ne s'exécute pas.

### [Critique] `varying_intercept_slope` — covariance construite avec le facteur de Cholesky à la place de la matrice de corrélation `[RÉSOLU 94e80673]`
- **Emplacement** : `BayesForge/Utils/Effects.py:250`
- **Risque** : `cov = jnp.outer(group_std, group_std) * L_corr` où `L_corr` vient de `dist.lkj_cholesky` (l. 246). Or `outer(σ,σ) * R` exige la **matrice de corrélation** `R = L Lᵀ`, pas son facteur de Cholesky `L`. La covariance passée à `multivariate_normal` (l. 253) est donc mathématiquement fausse (elle n'est même pas garantie définie positive : `L` triangulaire inférieure × `outer` n'est pas symétrique) → modèle erroné **silencieux**, posteriors biaisés, ou erreur Cholesky aléatoire selon les draws.
- **Scénario** : `m.effects.varying_intercept_slope(N_group=10, group=g, global_intercept=a, global_slope=b)` avec les défauts → `cov` asymétrique ; soit `multivariate_normal` produit des draws à structure de corrélation arbitraire, soit le sampler diverge. Comparer au centre correct de `varying_effects` (l. 193 : `diag(σ) @ corr @ diag(σ)` avec `dist.lkj`, qui est la bonne pratique) et à la branche non centrée correcte (l. 428 : `(σ[...,None] * L_Rho) @ z`).

### [Critique] `hdi` de SampledData — voir section SampledData (le module Diagnostics le consomme) `[RÉSOLU 94e80673]`

### [Majeure] `_parent_ids` — aucune validation du design emboîté ; parents faux en silence [RÉSOLU ab217dad]
- **Emplacement** : `BayesForge/Utils/Effects.py:22-32`, consommé par `nested_varying_effects` (`:639`, `:653`, `:669`)
- **Risque** : `jnp.zeros(n_group).at[child_obs_ids].set(parent_obs_ids)` : (a) si un id de niveau enfant apparaît sous **deux parents différents** (design croisé au lieu d'emboîté strict), dernier write gagne → hiérarchie silencieusement fausse ; (b) si une unité enfant n'apparaît dans **aucune** observation, son parent reste l'initialisation `0` → rattachement arbitraire au premier parent, sans erreur.
- **Scénario** : données où le groupe 3 du niveau 2 est observé une fois sous le parent 1 et une fois sous le parent 2 → `pid[3]` = valeur du dernier scatter, dépend de l'ordre des lignes ; l'utilisateur obtient un modèle multiniveau faux sans le moindre avertissement. Le docstring (l. 25-27) suppose l'emboîtement strict mais ne le vérifie pas (un `jnp.where` + `check` ou un `dev_warning` serait minimal).

### [Majeure] Singleton `dist` de module, muté par `Main/main.py:288-289` — reproductibilité croisée `[PARTIEL 19f93e5d — rebind au début de chaque appel d'inférence ; un tirage ad hoc hors fit utilise la dernière instance liée]`
- **Emplacement** : `BayesForge/Utils/Effects.py:6-9` (idem `Gaussian.py:4-5`) ; mutation dans `BayesForge/Main/main.py:282-289` (`_mod.dist.seed = self.seed ; _mod.dist._rng = self._rng`)
- **Risque** : le `UnifiedDist()` créé à l'import est **partagé par tout le processus** ; chaque constructeur `bf()` réécrit son `.seed`/`._rng`. Conséquences : (1) deux modèles `m1 = bf(rand_seed=1)`, `m2 = bf(rand_seed=2)` → tout tirage `sample=True` dans `Effects`/`Gaussian` (ex. `phylo_effect`, `gaussian_process`, l. 90 / `Gaussian.py:161-166`) utilise le RNG **du dernier modèle construit**, quel que soit le modèle en cours d'exécution ; (2) un `bf()` construit entre la définition d'un modèle et son `m.fit()` change le stream des draws ad-hoc intercalés. L'audit Main a épinglé `seed_globals` ; ici c'est le volet « état partagé » du même problème : le contrat de reproductibilité n'est pas porté par l'objet modèle mais par un global muté.
- **Scénario** : notebook avec `m1 = bf(rand_seed=7)` ; l'utilisateur dessine `m1.effects.phylo_effect(..., sample=True)` ; puis construit `m2 = bf(rand_seed=99)` ; rejoue le même dessin → valeurs différentes, sans que `m1` ait changé.

### [Mineure] Docstrings « centered »/« non-centered » inversées [RÉSOLU a9f66175]
- **Emplacement** : `Effects.py:111-112` (`varying_effects` : « uses a "centered" parameterization » alors que `centered=False` par défaut, l. 105, et que la branche par défaut est non centrée) ; `Effects.py:351` (`varying_intercept` : « non-centered parameterization » alors que l. 372-378 est la forme **centrée** `alpha ~ N(a_bar, sigma)`) ; `Effects.py:170` (paramètre `corr` documenté « correlation matrix » mais utilisé comme facteur de Cholesky `L_corr`).
- **Risque** : l'utilisateur choisit sa paramétrisation d'après la doc et obtient l'inverse — impact direct sur l'efficacité de NUTS (funnel) et l'interprétation des traces.

### [Mineure] Code mort et pièges divers [RÉSOLU a9f66175]
- `Effects.py:256-258` : `varying_intercept, varying_slope = group_params[:,0], group_params[:,1]` calculés puis écrasés deux lignes plus bas par l'indexation `[group]`.
- `Effects.py:53` : `diag_pre_multiply` matérialise `jnp.diag(v)` (O(N²) mémoire et FLOPs) alors que `v[:, None] * m` est exact et O(N²)→O(N·M) sans allocation ; c'est la fonction que Stan implémente sans diag.
- `Effects.py:672-679` : `nested_varying_effects` retourne un **array** si `N_vars == 1`, un **tuple** sinon → contrat d'arithmétique de retour dépendant d'une valeur d'argument ; `a, b = ...` plante avec `N_vars=1` avec une erreur d'unpacking obscure.
- `Effects.py:1` : `import jax as jax` (repli cosmétique).
- `Effects.py:616` : `mu[0]` de forme `(1,)` broadcaste silencieusement sur `(n_group, N_vars)` (l. 667) : un `mu=[jnp.zeros(1)]` avec `N_vars=2` ajoute le même offset à intercept et pentes sans erreur.

### Points sains (contrôle positif)
- `phylo_effect` (l. 90-92) : einsum `"...ij,...j->...i"` correct pour le stack `(T,N,N)`, broadcast `sd` correct via `expand_dims(-1)` ; testé (`Test/0.Core/test_effects.py`).
- Branche non centrée de `varying_effects` (l. 176) et de `varying_slope` (l. 339) : `(σ[...,None] * L) @ z` est la forme correcte.
- Branche centrée (l. 193, 647) : `diag(σ) @ corr @ diag(σ)` correcte (avec `dist.lkj`, matrice de corrélation).

---

## 2. BayesForge/Utils/Gaussian.py (336 l.)

### [Critique] `gaussian_process2` — `NameError: name 'Mgaussian' is not defined`, fonction publique 100 % cassée `[RÉSOLU 94e80673]`
- **Emplacement** : `BayesForge/Utils/Gaussian.py:323`, `:326`, `:332`
- **Risque** : la fonction référence `Mgaussian`, qui n'est défini nulle part dans le dépôt (grep : uniquement ces 3 lignes). Tout appel lève `NameError`, quel que soit `kernel_type`.
- **Scénario** ✅ : `gaussian.gaussian_process2(Dmat, kernel_type="periodic", sigma=1.0)` → `NameError: name 'Mgaussian' is not defined`. Zéro test. La correction évidente (`gaussian.kernel_sq_exp` etc.) n'a jamais été branchée sous test.

### [Majeure] `kernel_exp` — le passage en float64 est un no-op silencieux avec la config par défaut [RÉSOLU ab217dad]
- **Emplacement** : `BayesForge/Utils/Gaussian.py:101-108` (dupliqué `:240-247`)
- **Risque** ✅ : le code monte explicitement en `jnp.float64` « for Cholesky stability », mais avec `jax_enable_x64=False` (défaut JAX, et `Main/main.py` pousse même explicitement `jax.config.update("jax_enable_x64", False)` dans une branche backend), `astype(jnp.float64)` est **tronqué en float32** avec un simple `UserWarning` : les dtypes de sortie mesurés sont `float32/float32/float32`. La stabilité promise n'existe pas ; le jitter `delta=1e-3` reste la seule protection réelle.
- **Scénario** : grande matrice de distances avec points quasi confondus → le Cholesky « float64 » échoue exactement là où il devait réussir ; l'utilisateur voit un `UserWarning` noyé dans le log et des NaN.

### [Majeure] `kernel_sq_exp` — diagonale `set` + `delta=0` par défaut → Cholesky NaN silencieux sur points dupliqués [RÉSOLU ab217dad]
- **Emplacement** : `BayesForge/Utils/Gaussian.py:71-74` (dupliqué `:210-213`)
- **Risque** : `K.at[diag].set(sq_alpha + delta)` avec `delta=0` (défaut, l. 46) rend `K` singulière dès que deux points sont confondus (ou que `sq_rho` est grand, rendant les lignes quasi colinéaires). `jnp.linalg.cholesky` ne lève **pas** d'erreur en JAX : il retourne des NaN qui se propagent dans `k = L @ z` et donc dans toute la log-densité du modèle.
- **Scénario** ✅ : deux sites aux coordonnées identiques (fréquent : nids, quadrats répétés) avec `delta=0` → `diag(L) = [0.707, 1.7e-4]` dès N=2, NaN au-delà ; le sampler reçoit une log-prob NaN sans message pointant vers le kernel. Un jitter par défaut (comme le `delta=1e-3` de `kernel_exp`, l. 82) ou un `check_valid` s'impose.

### [Mineure] Corps de classe dupliqué — dont un bug de précédence dans le code mort [RÉSOLU a9f66175]
- **Emplacement** : `Gaussian.py:168-287` redéfinit intégralement `__init__`, `distance_matrix`, `kernel_sq_exp`, `kernel_exp`, `kernel_periodic`, `kernel_periodic_local` (première version l. 30-147). La seconde définition gagne ; la première est morte.
- **Risque** : la version morte `kernel_periodic_local` (l. 147) contient `jnp.exp(-(m**2/ 2*length_scale**2))` = `exp(−(m²/2)·ls²)` au lieu de `exp(−m²/(2·ls²))` — bug **corrigé** dans la version active (l. 285), mais le cadavre reste : toute reprise par copier-coller réintroduit l'erreur. La première version a aussi `sigma=1` (int, l. 113/132) contre `1.0` dans la seconde. 120 lignes à supprimer.

### [Mineure] `gaussian_process` — contrat flou et sites cachés [RÉSOLU a9f66175]
- **Emplacement** : `Gaussian.py:150-166`
- **Risque** : le docstring promet « Returns: the covariance matrix » mais la fonction retourne un **draw** `dist.multivariate_normal(0, SIGMA, name='kernel')` et crée au passage deux sites nommés `etasq`/`rhosq` sur le singleton `dist` muté (cf. Effects.py plus haut). `cov_GPL2` (l. 9-26) attend `x` = matrice de distances et applique `exp(-sq_rho·x²)` : correct en L2, mais le nom du paramètre (`sq_rho`) et le docstring de `gaussian_process` (l. 154-156 : « etasq: Length-scale parameter » — faux, `etasq` est l'amplitude) inversent amplitude et length-scale → l'utilisateur qui fixe ses priors d'après la doc échange deux hyperparamètres de rôle opposé.

### Points sains
- `distance_matrix` (l. 42) correct ; `kernel_periodic` (l. 266) et `kernel_periodic_local` actif (l. 285-287) exacts ; `cov_GPL2` utilise `.at[].add` correctement (l. 25).

---

## 3. BayesForge/Utils/link.py (254 l.)

### [Critique] `LKJcor` — densité LKJ mathématiquement fausse ; η n'a aucun effet sur la forme du prior `[RÉSOLU 94e80673]`
- **Emplacement** : `BayesForge/Utils/link.py:228-254`
- **Risque** : trois erreurs cumulées :
  1. `term1 = (2η−2)·Σ log(diag(C))` (l. 251) : la diagonale d'une matrice de **corrélation** vaut 1 → `term1 ≡ 0`, terme mort (il visait probablement la diagonale du facteur de Cholesky).
  2. `term3 = Σ log(det(C))` (l. 253) : l'exposant de `|C|` vaut **1 au lieu de (η−1)** → la forme du prior ne dépend pas de η. Vérifié ✅ : `LKJcor(R)/LKJcor(I) = det(R) = 0.75` pour η=0.5 comme pour η=10. Un LKJ correct avec η=1 doit être uniforme (exposant 0) ; ici η=1 donne `|C|¹`, et monter η ne concentre **pas** vers l'identité.
  3. `term2 = −(k(k−1)/2)·gammaln(η)` (l. 252) n'est pas la constante de normalisation LKJ (qui met en jeu `gammaln(η + (k−1−i)/2)`), et la fonction retourne la **densité** `exp(·)` (l. 254) au lieu du log — sous-flux garanti en grande dimension.
- **Scénario** : `m.dist` ou un utilisateur branche `link.LKJcor(C, eta)` comme log-prior → l'hyperparamètre η est décoratif, le prior effectif est figé à `|C|¹`, et la « normalisation » est fausse (comparaison de modèles faussée). Actuellement aucun appelant ni test (grep) — mais c'est une API publique documentée, placée de surcroît dans une classe de *fonctions de lien*.

### [Majeure] `inv_logit` — gradients NaN aux valeurs extrêmes ; utilisé dans des modèles NUTS [RÉSOLU antérieurement — vérifié]
- **Emplacement** : `BayesForge/Utils/link.py:50`
- **Risque** ✅ : `1/(1+exp(-x))` sature : `exp(-x)` déborde pour `x ≲ -88` et le **gradient** vaut `nan` (mesuré : `d/dx inv_logit(-100) = nan`, alors que `jax.nn.sigmoid` donne un gradient propre partout et que `logit`/`inv_logit` sont `@jit`, donc le NaN entre dans la trace sans warning).
- **Scénario** : `NBDA.py:597-662` appelle `link.inv_logit(R_asocial)` / `link.inv_logit(R_social)` **dans le modèle** ; avec des linéaires prédits étendus (hiérarchie non centrée mal initialisée, données mal standardisées), NUTS reçoit un gradient NaN → divergence en chaîne ou crash `NaN in log-density`, sans que le lien soit identifié comme coupable.

### [Majeure] `logit` — aucun clipping aux bornes 0/1 ; `probit`/`cloglog` idem [RÉSOLU ab217dad]
- **Emplacement** : `BayesForge/Utils/link.py:30` (`logit`), `:87` (`probit`), `:156` (`cloglog`)
- **Risque** ✅ : `logit(0) = -inf`, `logit(1) = +inf` (mesuré) ; `probit(0/1)` et `cloglog(0/1)` → ±inf de même. `log(x/(1-x))` de surcroît perd la précision et peut overflow près de 1 (`x/(1-x) → inf` dès `1-x < ~1e-38` en float32) là où `log(x) − log1p(−x)` resterait fini.
- **Scénario** : transformation d'une proportion observée contenant un 0 ou un 1 (survie totale dans un groupe, succès à l'essai 0) → `inf` entre dans le modèle ou les plots, puis NaN. Un `eps` optionnel (`clip(eps, 1-eps)`) est le standard (Stan, `qbeta` de R).

### [Mineure] `inv_logit_scale` — docstring faux sur le range, utilisé dans NBDA [RÉSOLU a9f66175]
- **Emplacement** : `BayesForge/Utils/link.py:54-69`
- **Risque** : la doc annonce « scaling between 0 and 1 » et « Returns … 1/(1+exp(-x)) » ; la fonction calcule `2·sigmoid(x) − 1` ∈ **(−1, 1)** ✅ (`inv_logit_scale(0) = 0.0`, mesuré). `NBDA.py:605/632/658` l'utilise comme `social_influence_weight` : quiconque suppose un poids ∈ [0,1] interprète mal le modèle (poids négatifs possibles par construction).

### [Mineure] Code mort et cohérence de classe [RÉSOLU a9f66175]
- `link.py:32` : `return (inv_logit(x) - 0.5) * 2` **après** le `return` de `logit` — inatteignable ; c'est l'indice d'un demi-refactor de `inv_logit_scale` abandonné.
- Le module ne fournit **aucun Jacobien** alors que son rôle (fonctions de lien) et l'attente de l'audit les suppose présents : toute transformation de variable avec correction jacobienne doit les réécrire ailleurs, source de divergences silencieuses lien/Jacobien.
- `LKJcor` est une densité, pas un lien : erreur de placement architectural dans la classe `link`.
- Incohérence `@jit` : `logit`/`inv_logit`/`inv_logit_scale` sont jittés (l. 15, 35, 53), `probit`, `cloglog`, `reciprocal` ne le sont pas — mêmes usages, comportements de recompilation différents. `reciprocal(0)` → `inf` sans mot (l. 191).
- `link.inv_logit` n'est pas l'inverse exact de `link.logit` en précision finie près des bornes (log/exp vs sigmoid) — acceptable, mais à documenter.

---

## 4. BayesForge/Utils/SampledData.py (884 l.)

### [Critique] `hdi` — silencieusement faux en 2D et avec NaN ; aucune validation d'entrée `[RÉSOLU 94e80673]`
- **Emplacement** : `BayesForge/Utils/SampledData.py:588-608`
- **Risque** :
  1. **2D** : le docstring (l. 593) dit « 1D array » mais rien ne l'impose. Sur un tableau `(chains, draws)` ou `(draws, vars)`, `jnp.sort` trie le long du dernier axe, `n = shape[0]` compte les **lignes**, et `lows[min_idx]`/`highs[min_idx]` (l. 607-608) retournent des **lignes entières** : mesuré ✅, `SampledData(jnp.arange(12.).reshape(4,3)).hdi()` retourne `[[0,1,2],[9,10,11]]` — un tableau 2×3 que l'appelant dépaquette comme `(lo, hi)` sans erreur, avec des valeurs dénuées de sens.
  2. **NaN** : `jnp.sort` pousse les NaN en fin de tableau, `argmin` sélectionne une fenêtre de largeur NaN → mesuré ✅ `hdi = (0.1, nan)`. Un seul draw NaN (divergence MCMC non filtrée) contamine tout l'intervalle.
  3. Cas dégénérés sans message : 0 draw → `argmin` sur tableau vide ; 1 draw → `(x, x)` (mesuré) ; `cred_mass·n < 1` → largeur 0 partout, intervalle arbitraire (l. 601-608).
- **Scénario** : `Test/15.Diagnostics/Parity_Tests/test_diag_parity.py:260` (« summary_uses_real_hdi_not_percentiles ») montre que le module Diagnostics s'appuie sur ce calcul ; un appel sur des draws à plusieurs chaînes ou une trace avec une divergence produit un « HDI 95 % » faux publié dans un résumé. Les tests existants (`Test/14.Jax extension (WIP)/test_sampleddata.py:1074-1097`) ne couvrent que le 1D propre.

### [Majeure] `corr_heatmap` — ne calcule **pas** la corrélation : affiche les draws bruts sous un titre « Correlation Matrix » [RÉSOLU ab217dad]
- **Emplacement** : `BayesForge/Utils/SampledData.py:153-210`, en particulier `:172` (`corr_matrix = self._data`)
- **Risque** : le docstring (l. 154-158) et le commentaire « FIX 1: Always calculate the correlation from the samples » (l. 168-171, avec son garde-fou « fewer than 2 samples ») promettent le calcul depuis les draws ; le code prend `self._data` **tel quel**. Un appel sur `(n_draws, n_vars)` dessine une heatmap 4000×5 de draws étiquetés « Correlation Matrix » (et le formatage d'annotations l. 176-179 tourne en Python O(n_draws·n_vars) → minutes de gel). L'inverse — passer une vraie matrice de corrélation — marche mais contredit la doc. Contrat doublement incohérent, et le rendu est **scientifiquement trompeur** (des draws dans [-1,1] passent pour des corrélations).
- **Scénario** : `SampledData(m.posteriors["beta"]).corr_heatmap()` → heatmap titrée « Correlation Matrix » montrant les draws bruts ; l'utilisateur la collecte pour un papier.

### [Majeure] Effets de bord à l'import : imports lourds en thread pool jamais fermé [RÉSOLU ab217dad]
- **Emplacement** : `BayesForge/Utils/SampledData.py:9-17` (+ `ImportManager.py:10`)
- **Risque** : importer `SampledData` (donc `BayesForge`) planifie 6 imports en arrière-plan (plotly×4, seaborn, matplotlib) sur un `ThreadPoolExecutor(4)` global **jamais shut down** ; les threads non-daemon sont rejoints à l'exit (un import bloqué = process qui ne rend pas la main). Les `ImportError` (plotly absent) ne surgissent que tard, au premier `get_module` (l. 30 `ImportManager.py`), dans un contexte d'appel sans rapport. L'import concurrent de modules Python croisés (matplotlib/seaborn partagent des dépendances) est une source classique de deadlocks d'import selon les versions.
- **Scénario** : environnement sans plotly : `import BayesForge` réussit, puis un `sd.hist()` échoue avec `ModuleNotFoundError: plotly` remontée du futur — diagnostic délocalisé.

### [Mineure] `hist` 3D — axes inversés et bins figés [-4, 4] [RÉSOLU 5d8270bd]
- **Emplacement** : `SampledData.py:110` vs `:118-120` ; bins `:112`
- **Risque** : le dépaquetage dit `(n_samples, n_groups, n_times)` (l. 110) mais l'indexation `self._data[:, t, g]` (l. 120) itère `t` sur `range(n_times)` en **axe 1** (celui des groupes) et `g` sur `range(n_groups)` en axe 2 : `IndexError` si `n_times > n_groups`, tranche silencieusement fausse sinon. `bin_edges = linspace(-4, 4)` (l. 112) exclut silencieusement toute donnée hors [-4,4] (`np.histogram` ignore les valeurs hors bins) → densités 3D fausses pour des draws non standardisés.

### [Mineure] `ridgeline` interactif — itère sur les draws au lieu des catégories ; 3D accepté mais cassé [RÉSOLU 5d8270bd]
- **Emplacement** : `SampledData.py:467-484`
- **Risque** : `zip(self._data, colors)` (l. 474) itère les **lignes** (draws) d'un `(n_samples, n_categories)` alors que `colors` est de longueur `n_categories` (l. 470) : le violon trace les `n_categories` **premiers draws**, pas les variables. `ndim==3` est accepté (l. 467) mais `go.Violin(x=<matrice 2D>)` plante. La branche non interactive (l. 494-512) est correcte, ce qui rend le contraste d'autant plus trompeur.

### [Mineure] `ppc_plot` — RNG global numpy et krach sur données constantes [RÉSOLU 5d8270bd]
- **Emplacement** : `SampledData.py:556` (`np.random.choice`), `:558`/`:568` (`gaussian_kde`)
- **Risque** : le sous-échantillonnage des draws utilise le RNG **global numpy**, pas le `RNG` de session : la reproductibilité du PPC dépend du hasard d'appels antérieurs… ou du re-seed global de `bf()` (`rng.py:98-108` appelé par `main.py:207`) — couplage implicite fragile. `stats.gaussian_kde` lève `LinAlgError` (matrice singulière) sur un draw ou des données constantes (binaire dégénéré, n_obs=1) sans message utile.

### [Mineure] `autocorr` — NaN sur séries courtes/constantes, incohérence np/jnp, O(lags) syncs [RÉSOLU 5d8270bd]
- **Emplacement** : `SampledData.py:388`, `:396`, `:407`, `:412`
- **Risque** : `np.corrcoef(series[:-k], series[k:])` avec `k ≥ len(series)` → tranches vides → NaN + RuntimeWarning ; série constante → NaN. La branche non-interactive 1D utilise `np.corrcoef` (l. 407) et sa jumelle 2D `jnp.corrcoef` (l. 412) — deux sémantiques NaN différentes selon le chemin. Chaque lag déclenche une conversion/sync device.

### [Mineure] `__getattr__` — récursion infinie si `_data` absent ; délégation trop large [RÉSOLU 5d8270bd]
- **Emplacement** : `SampledData.py:751-771`
- **Risque** : `__getattr__` lit `self._data` sans garde : sur un objet partiellement construit (exception dans `__init__` sur une entrée ragged, `object.__new__` hors pytree), tout accès d'attribut — y compris `_data` lui-même — repasse par `__getattr__` → `RecursionError` qui masque l'erreur d'origine. La délégation aveugle rend aussi les fautes de frappe (`sd.meann()`) obscures (`AttributeError` sur `jax.Array`).

### [Mineure] Contrats de types incohérents et code dupliqué [RÉSOLU 5d8270bd]
- `__array_ufunc__` (l. 725-728) : `np.add(sd, x)` retourne un **ndarray nu** (le résultat numpy ne passe pas le `isinstance(result, jnp.ndarray)` de `_wrap_result`, l. 37) là où `sd + x` retourne un `SampledData` ; `out=` et `where=` cassent silencieusement la sémantique.
- `__eq__`/`__ne__` définis sans `__hash__` (l. 686-689) → `SampledData` inhashable : `dict[sd] = ...` et `set` échouent, ce qu'aucun docstring ne signale.
- `_tree_flatten`/`_tree_unflatten` définis **deux fois** (l. 47-62 et l. 803-821) ; la seconde paire gagne (celle qui contourne `__init__`, correctement commentée) ; la première, qui appelait `jnp.asarray` dans unflatten, est du code mort — le commentaire l. 810-818 décrit d'ailleurs le bug qu'elle provoquait.
- `__contains__` (l. 705-706) : `x in sd` sur un array multi-éléments lève via `__bool__` (l. 707-708) — comportement numpy, mais à documenter pour un « conteneur ».

### [Mineure] Contrat public en deçà de la fiche module : pas de statistiques ni d'export [RÉSOLU 5d8270bd]
- Le module s'intitule conteneur « statistiques, résumé, export » ; en réalité il n'expose que `hdi`, `to_jax` et des plots. `sd.mean()`, `sd.std()`, `sd.quantile()` sont délégués à jnp : `ddof=0` par défaut (biais d'échantillon sur peu de draws), pas de `nanmean`, aucune sémantique `(chains, draws)` (l'axe 0 n'est jamais traité comme un axe-chaîne), pas d'export CSV/ArviZ. Les plots acceptent des données à NaN sans le signaler (sauf `density`, l. 451, le seul endroit du fichier qui filtre les non-finis). Point positif : l'immuabilité des tableaux JAX neutralise le risque « vues vs copies » sur le postérieur source (`__init__` l. 33, `__setitem__` refusé l. 735-743 avec un bon message).

### Points sains
- Le pytree `register_pytree_node` (l. 853-857) + `__jax_array__` (l. 795-800) + `__reduce__` (l. 791-792) sont corrects et bien commentés ; `_repr_pretty_`/`_repr_html_` propres ; `density` (l. 424-459) gère NaN et données constantes (`np.ptp(s) > 0`, l. 454) — c'est le modèle à imiter dans le reste du fichier.

---

## 5. BayesForge/Utils/rng.py (108 l.)

### [Majeure] `seed_globals` — effet de bord global sur `np.random` et `random` à chaque `bf()` `[RÉSOLU 19f93e5d — opt-in bf(seed_global_rngs=True)]`
- **Emplacement** : `BayesForge/Utils/rng.py:98-108`, appelé par `Main/main.py:207`
- **Risque (gravité réelle confirmée)** : chaque construction de modèle écrase l'état des RNG globaux avec le seed de base. (1) Le code utilisateur qui positionne `np.random.seed(...)` pour son propre splitting de données voit son état détruit par un `bf()` postérieur ; (2) deux modèles coexistants se volent l'état global (le dernier construit gagne) ; (3) avec `rand_seed=True`, le seed global devient `time.time_ns()`-dépendant : toute la chaîne numpy du processus devient non reproductible **sans que l'utilisateur ait demandé l'aléa** (il ne l'a demandé que pour BF). Le module Utils contient d'ailleurs un consommateur de ce canal : `SampledData.ppc_plot` (`:556`) — la reproductibilité du PPC repose sur ce side-effect, pas sur le `RNG` de session. Majeure, pas critique : le chemin JAX (inférence) reste pur et non affecté.
- **Scénario** ✅ (code lu) : `np.random.seed(0); split = train_test_split(X); m = bf(rand_seed=42)` → tout `np.random.*` ultérieur de l'utilisateur reprend la séquence de BF.

### [Mineure] `stream()` — reproductibilité dépendante de l'ordre ; garde Tracer sur API privée [RÉSOLU 5d8270bd]
- **Emplacement** : `rng.py:69-86`
- **Risque** : le split mutateur rend la séquence des draws `sample=True` dépendante du **nombre** de draws effectués avant (documenté l. 16-20, acceptable) ; mais la garde `isinstance(sub, jax.core.Tracer)` (l. 78) s'appuie sur un chemin privé en cours de déplacement dans JAX (`jax._src.core`) — à surveiller au prochain upgrade. La logique elle-même est correcte : pas de réutilisation de clé, `new_key` stocké, `sub` retourné.
- `key(label)` (l. 63-67) : `fold_in` sur `PRNGKey(base)` avec hash SHA-256 tronqué 32 bits — propre, stable, sans collision pratique ; pas de mutation. ✅

### [Mineure] `set_seed` incohérent avec le constructeur [RÉSOLU 5d8270bd]
- **Emplacement** : `rng.py:92-96`
- **Risque** : `set_seed` rembobine le stream JAX mais **ne rappelle pas** `seed_globals()`, contrairement au chemin `bf()` → après `m.set_seed(k)`, le chemin numpy n'est pas resynchronisé avec le chemin JAX : deux moitiés du même modèle divergent d'état.

---

## 6. BayesForge/Utils/infer_discrete.py (70 l.)

### [Mineure] Hypothèses d'entrée non vérifiées et API privée numpyro [RÉSOLU 5d8270bd]
- **Emplacement** : `infer_discrete.py:42`, `:45`, `:20`
- **Risque** : `tree_leaves(posterior_samples)[0]` (l. 42) lève `IndexError` sur dict vide et suppose que la **première** feuille porte le batch shape correct (ordre de dict) ; `x[0]` (l. 45) suppose ≥1 draw partout (`IndexError` obscur sur un site vide). `_sample_posterior` (l. 20) est un symbole **privé** de numpyro — cassure silencieuse au upgrade de numpyro, sans test de fumette dans `Test/` à ma connaissance. Le module lui-même est autrement solide : le contournement du `mask(mask=False)` de `Predictive` est correctement documenté (l. 1-12) et la logique FFFB (l. 51-70) reproduit fidèlement numpyro sans le bug de masquage ; `temperature=1` commenté (l. 57).

---

## 7. BayesForge/Utils/dev_warning.py (25 l.)

### [Mineure] `print` au lieu de `warnings.warn` [RÉSOLU 5d8270bd]
- **Emplacement** : `dev_warning.py:5-9`
- **Risque** : le warning de développement passe par `print` sur stdout : non filtrable/silenciable via `warnings`, non capturable par les logs, absent des captures stderr. L'anti-doublon `_WARNED` (l. 2) est un set global : dans une suite de tests, seul le premier test voit le message (il existe `reset_dev_warnings`, l. 12-14, mais rien ne l'accroche aux fixtures). `dev_warning()` (l. 18-25) dépend d'IPython mais n'est appelé que depuis la doc — OK.

---

## 8. BayesForge/Utils/ImportManager.py (35 l.)

### [Mineure] `get_module` sans délai ; erreurs différées ; executor jamais fermé [RÉSOLU 5d8270bd]
- **Emplacement** : `ImportManager.py:20-30`, `:32-34`
- **Risque** : `future.result()` (l. 30) bloque sans `timeout` — un import gelé (filesystem réseau, import circulaire sous thread) gèle l'appelant indéfiniment ; l'exception d'import n'est levée qu'à la consommation (cf. SampledData plus haut) ; `shutdown()` existe mais aucun appelant ne l'invoque (l'instance globale `SampledData.py:9` vit jusqu'à l'exit). `schedule_import` ignore les doublons par alias (l. 15) — correct.

---

## 9. BayesForge/Utils/__init__.py (1 l.)
Vide. Aucun export public, pas de `__all__` : les consommateurs importent les sous-modules en dur (`from BayesForge.Utils.link import link`), ce qui verrouille les chemins d'import. Neutre pour la release, à noter pour l'API.

---

## (a) Résumé du module

`Utils/` est le socle importé partout, et c'est là que le bât blesse : **deux de ses fonctions publiques sont 100 % cassées par des `NameError`** (`Effects.varying_slope_non_centered`, `Gaussian.gaussian_process2` — jamais testées), **deux calculs scientifiques sont silencieusement faux** (`link.LKJcor`, dont η ne pilote rien ; `SampledData.hdi`, qui retourne une matrice au lieu d'un intervalle dès que les draws sont 2D et un NaN dès qu'un draw est contaminé), et **une covariance hiérarchique est construite avec le facteur de Cholesky à la place de la matrice de corrélation** (`Effects.varying_intercept_slope:250`). À côté de cela, le conteneur `SampledData` est architecturalement soigné (pytree, immuabilité, `__jax_array__`) mais son API statistique se réduit à `hdi` + plots, avec des plots dont plusieurs (`corr_heatmap`, `ridgeline`, `hist` 3D) trahissent leur propre docstring. `rng.py` est le meilleur fichier du module (clés pures, contrat documenté), amputé d'un unique défaut réel : `seed_globals`, dont la gravité confirmée est **majeure** (destruction de l'état RNG global de l'utilisateur, et le PPC en dépend). La dette de copie (120 lignes dupliquées dans `Gaussian.py`, paires de méthodes dupliquées dans `SampledData.py`, code mort après `return` dans `link.py`) montre que ces fichiers n'ont pas eu de passe de nettoyage avant release.

## (b) Fragilités architecturales

1. **État global muté comme mécanisme de câblage** : le singleton `dist` de chaque module (Effects, Gaussian) est reconfiguré à la volée par `Main/main.py:288-289`, et `seed_globals` écrit dans les RNG de processus. La reproductibilité est un contrat inter-modules porté par des mutables sans propriétaire — deux modèles, deux seeds, ou un simple ordre d'appels différent suffisent à le briser.
2. **Aucune validation aux frontières publiques** : ni dimension (`hdi` 2D), ni domaine (`logit(0)`, `probit(1)`), ni finitude (NaN partout sauf dans `density`), ni design ( `_parent_ids`). Les erreurs surgissent trois étages plus loin, ou jamais (silence).
3. **Contrats docstring ≠ comportement** : « correlation » qui n'en calcule pas, « 0 and 1 » qui rend (−1,1), « centered » qui rend non centré, « returns the covariance matrix » qui retourne un draw. Pour un module consommé par `to_latex`/`model_graph`/NBDA, chaque écart est une mauvaise interprétation scientifique potentielle.
4. **Le float64 comme ornement** : `kernel_exp` « monte » en float64 que la config JAX par défaut annule — la stabilité numérique annoncée dépend d'un flag global que d'autres modules touchent (`main.py` le force à False).
5. **Absence de tests sur les chemins cassés** : les trois fonctions en `NameError` et `LKJcor` n'ont zéro test ; les tests `hdi` ne couvrent que le 1D propre. Un smoke-test « appeler chaque méthode publique avec ses défauts » aurait attrapé les quatre.
6. **API privée d'autrui** : `numpyro.contrib.funsor.discrete._sample_posterior`, `numpyro.infer.util._guess_max_plate_nesting`, `jax.core.Tracer` — trois dépendances privées qui rendent les upgrades amont cassants.

## (c) Décompte par sévérité

| Sévérité | Nombre |
|---|---|
| **Critique** | **5** — `Effects.py:458/462/467` (NameError `m`) ; `Gaussian.py:323/326/332` (NameError `Mgaussian`) ; `Effects.py:250` (Cholesky pris pour la corrélation) ; `link.py:228-254` (`LKJcor` fausse, η inopérant) ; `SampledData.py:588-608` (`hdi` faux en 2D/NaN, sans validation) |
| **Majeure** | **8** — `link.py:50` (gradient NaN de `inv_logit`, dans les modèles NBDA) ; `link.py:30/87/156` (±inf aux bornes, sans clipping) ; `rng.py:98-108` + `main.py:207` (`seed_globals`) ; `Gaussian.py:101-108` (float64 no-op sous x64 off) ; `Gaussian.py:71-74` (Cholesky NaN silencieux, `delta=0`) ; `Effects.py:6-9`/`Gaussian.py:4-5` (singleton `dist` muté, `main.py:288-289`) ; `Effects.py:22-32` (`_parent_ids` sans validation) ; `SampledData.py:9-17` (imports threadés à l'import, executor jamais fermé) |
| **Mineure** | **24** — `link.py:32` (code mort) ; `link.py:54-69` (range docstring faux) ; `link.py` (pas de Jacobians, `LKJcor` mal placée, `@jit` incohérent) ; `Gaussian.py:168-287` (classe dupliquée + bug de précédence mort l. 147) ; `Gaussian.py:150-166` (contrat/docstring `gaussian_process`) ; `Effects.py:111/351/170` (docstrings centered/non-centered) ; `Effects.py:256-258` (code mort) ; `Effects.py:53` (`diag` matérialisé) ; `Effects.py:672-679` (arité de retour variable) ; `Effects.py:616` (broadcast mu silencieux) ; `SampledData.py:110/120` (axes `hist` 3D inversés) ; `SampledData.py:112` (bins [-4,4] figés) ; `SampledData.py:467-484` (`ridgeline` axe draws + 3D cassé) ; `SampledData.py:556/558` (`ppc_plot` RNG global + `LinAlgError`) ; `SampledData.py:380-420` (`autocorr` NaN, np/jnp mêlés) ; `SampledData.py:751` (`__getattr__` récursif si `_data` absent) ; `SampledData.py:47-62` vs `:803-821` (pytree dupliqué) ; `SampledData.py:725-728` (`__array_ufunc__` non wrapé, `out=`) ; `SampledData.py:686-689` (inhashable) ; `SampledData.py` (pas d'API stats/export, ddof=0, pas de nanmean) ; `rng.py:69-86` (ordre-dépendant, `jax.core.Tracer` privé) ; `rng.py:92-96` (`set_seed` sans `seed_globals`) ; `dev_warning.py:5-9` (`print` vs `warnings`) ; `ImportManager.py:20-30` (pas de timeout, erreurs différées) ; `infer_discrete.py:42-45` (dict vide, API privée numpyro) |
| **Total** | **37 constats** (dont 11 vérifiés empiriquement ✅) |

**Recommandation de release** : bloquants = les 5 critiques (deux suppressions/corrections triviales de `NameError`, une correction de covariance `Effects.py:250`, réécriture de `LKJcor` vers `log_density` de numpyro, verrouillage 1D + filtre NaN dans `hdi`). Les 8 majeures doivent au minimum être documentées comme known issues ; `seed_globals` et le singleton `dist` méritent un correctif d'architecture avant toute promesse publique de reproductibilité multi-modèles.

---

### Module PostModel

**Audit de release — `BayesForge/PostModel/`** (racine `/home/sosa/work/BF`)
Fichiers lus intégralement : `to_latex.py` (939 l.), `model_graph.py` (702 l.), `model_trace.py` (344 l.), `__init__.py` (vide).
Méthode : lecture ligne à ligne + vérification croisée des tables de reconnaissance contre le code réel (`Utils/Effects.py`, `Utils/link.py`, `Utils/Gaussian.py`, `Network/model_effects.py`) + exécution de 6 démos de reproduction (AST, graphe, trace) sur numpyro 0.22 / JAX, CPU.

---

## 1. Exactitude mathématique (contenu scientifique affiché)

### [CRITIQUE] C1 — `to_latex.py:485-511` : hyperpriors fabriqués pour `varying_intercept_slope` / `varying_intercept_slope_non_centered` `[RÉSOLU 0a421299]`
La branche `if fn in ("varying_effects", "varying_intercept_slope", "varying_intercept_slope_non_centered")` (l. 485) rend systématiquement les lignes `ᾱ ~ Normal(0, 2), β̄ ~ Normal(0, 1)` (l. 497-498 et 509-510). Or `Effects.varying_intercept_slope` (Utils/Effects.py:343-380) et sa variante non centrée (Effects.py:443… signature `global_intercept, global_slope` / `global_intercept_mean, global_slope_mean`) **ne créent aucun site `global_intercept`/`global_beta`** : ces moyennes sont des **arguments fournis par l'utilisateur**. Le rendu invente donc deux lignes de priors absentes du modèle réel.
**Déclencheur (reproduit)** : `to_latex(model2)` avec `a, b = effects.varying_intercept_slope(N_group, group, gi, gs)` → sortie incluant `\bar{\alpha} \sim \text{Normal}(0, 2), \quad \bar{\beta} \sim \text{Normal}(0, 1)`. L'utilisateur qui publie ce `gather*` décrit un modèle qui n'est pas le sien. C'est du contenu scientifique faux, produit de façon déterministe à chaque appel.

### [MAJEUR] M1 — `to_latex.py:530` et `model_graph.py:213` : reconnaissance d'une API 100 % cassée (`varying_slope_non_centered`) [RÉSOLU antérieurement 94e80673 — vérifié, tests 739af172]
Les deux tables listent `"varying_slope_non_centered"` comme bloc de construction valide. La fonction Utils (`Utils/Effects.py:443-479`) appelle `m.dist.lkj_cholesky` / `m.dist.exponential` / `m.dist.normal` (l. 458, 462, 467) alors que `Effects.py` n'importe que `dist` (l. 6) et qu'aucun `m` n'est injecté (vérifié : `hasattr(Effects, 'm') == False`, aucune injection dans `ImportManager.py`). **Confirmé à l'exécution** : `effects.varying_slope_non_centered(0, 0, 2, 3)` → `NameError: name 'm' is not defined` (inconditionnel, la ligne 467 est hors branche conditionnelle).
**Conséquence PostModel** : `to_latex` (chemin AST, sans exécution) rend un jeu d'équations propre et crédible (`(diag(σ)·L·Z)^T_[g,c]`, `Z ~ Normal(0,1)`, `L ~ LKJCholesky(2)`, `σ ~ Exponential(1)`) et `model_to_graph` fabrique les nœuds latents `L_Rho_group`, `sigma_group`, `z_group` (reproduit, démo 6) pour un modèle qui **crashe au moindre fit**. Les tables ne réagissent pas à la cassure : rien n'indique à l'utilisateur que l'API documentée par le rendu est morte. Le jumeau fonctionnel `varying_slope` (Effects.py:262, qui utilise `dist.*`) est lui bien valide et produit les mêmes noms de sites — la table est donc exacte sur le fond, périmée sur la forme.

### [MAJEUR] M5 — `to_latex.py:878-894` (`_latex_from_trace`) : les plates sont ignorées, l'observation non distinguée [RÉSOLU 739af172]
`_latex_from_trace` itère sur `tr.sites` (l. 885) sans jamais lire `site.plates` (pourtant capturé dans `model_trace.py:266,276`) ni distinguer `OBSERVED` de `LATENT`. C'est **le seul chemin de rendu pour les modèles R/Julia** (docstring l. 879-880) : la structure d'indépendance conditionnelle disparaît entièrement, contredisant directement l'exigence « un plate doit multiplier les indices, pas les ignorer ».
**Déclencheur (reproduit)** : modèle avec `with numpyro.plate("i", N): z ~ Normal(0,1); y ~ Normal(z,1), obs=...` rendu sans source → `y \sim \text{Normal}(z, 1)` / `z \sim \text{Normal}(0, 1)` — un modèle scalaire au lieu de N observations i.i.d. De plus `Independent(Normal, 1)` est déshabillé par `_base_dist` (`model_trace.py:103-106`) en `Normal` sans que la réduction soit visible.

### [MINEUR] m2 — `to_latex.py:568, 576, 584` : support tronqué omis sur `TruncatedNormal` [RÉSOLU b45052b6]
Le rendu `σ_sr ~ TruncatedNormal(0, 2.5)` omet la contrainte réelle `low = 0` (`Network/model_effects.py:56, 217` : `dist.truncated_normal(mu, sd, low=0, ...)`). La densité affichée est celle d'une normale tronquée dont on tait le support — ambiguïté sur un paramètre d'échelle qui doit rester positif.

### [MINEUR] m3 — `to_latex.py:616` : `σ²` présenté comme paramètre libre du noyau GP [RÉSOLU b45052b6]
`K_SE(D, η², ρ², σ²)` suggère trois hyperparamètres ; or `Gaussian.gaussian_process` (Utils/Gaussian.py:150) a `sigmaq=0.01` **fixe** (nugget), seuls `etasq ~ Exponential(2)` et `rhosq ~ Exponential(0.5)` sont échantillonnés (Gaussian.py:161-163) — ces deux priors rendus (l. 617) sont corrects.

### [MINEUR] m4/m5 — rendu non centré imposé quand le code est centré [RÉSOLU b45052b6]
- `to_latex.py:592-598` (`edge_weights`) : rend toujours la décomposition non centrée `μ + σ·z` ; avec `partial_pooling=False` (Network/model_effects.py:343-366) le site réel est un simple `Normal(0, 2.5)`.
- `to_latex.py:542-550` (`nested_varying_effects`) : rend toujours `θ_ℓ = θ_{ℓ-1} + (diag(σ)L Z)^T` ; avec `centered=True` (Effects.py:585-608) les sites réels sont `corr_{name} ~ LKJ` et `{name}_effects ~ MVN`.

### [MINEUR] m1 — `to_latex.py:646` : noms de classes LaTeX incorrects [RÉSOLU b45052b6]
`"".join(part.capitalize() ...)` transforme `lkj_cholesky` → `LkjCholesky` et `lkj` → `Lkj` au lieu de `LKJCholesky`/`LKJ` affiché ailleurs (l. 500, 507). Incohérence typographique du contenu rendu.

### [MINEUR] m17 — traductions d'expressions mathématiquement fausses (groupe) [RÉSOLU b45052b6]
- `to_latex.py:388` : `slogdet(A)` → `\det(A)` (c'est `sign, log|det|`).
- `to_latex.py:407-410` : `ppf`/`cdf` génériques mappés sur `Φ⁻¹`/`Φ` — faux pour toute loi non normale (`t.ppf`, etc.).
- `to_latex.py:239` : opérateur de comparaison inconnu (dont `is`) rendu `=` → `a is None` devient `a = None`.
- `to_latex.py:219` : `a // b` → `a \lfloor / \rfloor b` (syntaxe nonsens).
- `to_latex.py:305` : appel avec mots-clés seuls, `sqrt(x=a)` → `\sqrt{x=a}`.
- `to_latex.py:322` : `repeat(x, k)` perd la répétition (choix de notation, mais perte d'information).
- `to_latex.py:97` : `_escape_text` n'échappe que `_` ; `%`, `&`, `#` dans un nom de site cassent KaTeX/MathJax.
- `to_latex.py:63` : `convert_to_greek` minuscule en silence les majuscules hors liste (`Epsilon` → `\epsilon`).
Bons points vérifiés : `inv_logit_scale` rendu `2·logit⁻¹(x) − 1` = formule réelle (link.py:54-76) ; `diag(σ)·R·diag(σ)` = `jnp.diag(sigma) @ corr @ jnp.diag(sigma)` (Effects.py) ; priors par défaut de `varying_effects` (`Normal(0,2)`, `Normal(0,1)`, `Exponential(1)`, `LKJ(2)`) conformes à Effects.py ; pas de terme jacobien revendiqué à tort (les liens sont rendus comme fonctions, pas comme densités).

---

## 2. Vectorisation, dimensions, parsing

### [MAJEUR] M4 — `to_latex.py:733` : inversion aveugle du corps d'une plate, les en-têtes `for`/`with` imbriqués sont séparés de leur corps [RÉSOLU 739af172]
`return headers + [f"\\quad {bl}" for bl in reversed(body_lines) if bl]` inverse la **liste aplatie** des lignes du corps. Un `for` imbriqué produit `[header, body…]` : après inversion, le header du `for` se retrouve **après** son corps.
**Déclencheur (reproduit)** :
```python
with numpyro.plate("i", N):
    z = 0.0
    for t in range(3): z = z + 1.0
    y = numpyro.sample("y", dist.normal(z, 1.0))
```
→ rendu : `for i ∈ 1..N : / y ~ Normal(z,1) / z = z + 1.0 / for t ∈ 0..2 : / z = 0.0` — équation structurellement illisible (processus AR dans une plaque d'observations : motif courant).

### Point d'architecture positif (grille, item 2)
PostModel ne touche **jamais** les draws du posterior (`parent.posteriors` n'est référencé nulle part dans le module) : pas de risque de confusion `(chains, draws)` vs `(draws,)`, pas de matérialisation inutile de gros draws. Les « traces » manipulées sont des traces d'exécution numpyro ; `model_trace.py:264` enregistre `np.shape(s["value"])` sans distinguer batch/event shape (consigné ci-dessous avec m19).

### [MINEUR] m6 — `model_graph.py:233-246` : `nested_varying_effects`, le paramètre `names` passé positionnellement est ignoré [RÉSOLU f4ecf90b]
La table ne scanne que `call.keywords` (l. 238-239) alors que la signature réelle est `nested_varying_effects(N_vars, names, N_groups, group_ids, …)` (Effects.py:518-528) et que l'exemple de la doc passe `names` en positionnel.
**Déclencheur (reproduit)** : `effects.nested_varying_effects(2, ["region","group"], [5,20], [r_id, g_id])` → graphe = `{region_id, group_id, global_intercept, global_beta, a, b}` ; **tous** les sites par niveau (`sigma_region`, `L_corr_region`, `z_region`, …) manquent, sans avertissement. La table ne gère en plus que la variante non centrée (cf. m5).

### [MINEUR] m7 — `model_graph.py:181-229` : sites fantômes quand les hyperpriors sont fournis en arguments [RÉSOLU f4ecf90b]
Les tables statiques ajoutent inconditionnellement `corr_{gname}`, `{gname}_sd_intercept`, `L_Rho_{gname}`, etc., même quand l'utilisateur passe `corr=`, `sd_intercept=`, `L_Rho=` (les sites ne sont alors pas créés, cf. Effects.py:140-160, 457-464). Graphes AST-only sur-peuplés ; le chemin trace (avec `data`) corrige — d'où deux graphes différents pour un même modèle avant/après fit.

### [MINEUR] m19 — `model_trace.py:123-151` : paramètres des transformations non tracés ; shapes sans distinction batch/event [RÉSOLU f4ecf90b]
`_param_leaves` ne lit que l'`arg_constraints` de la distribution **de base** : pour une `TransformedDistribution`, les paramètres des `transforms` (ex. `LowersTransform(lower_bound=f(σ))`) ne sont ni nommés ni ajoutés aux leaves → dépendances perdues silencieusement. Les leaves anonymes du fallback (`arg0…`, l. 148-150) peuvent aussi remonter comme placeholders `loc`/`scale` dans le LaTeX (cf. m15).

### [MINEUR] m23 — `to_latex.py:842, 927` : chemins de repli ligne-par-ligne dégénérés [RÉSOLU b45052b6]
`_body_lines` (l. 842) ampute `[1:]` même si la signature `def` s'étale sur plusieurs lignes → fragments de signature rendus comme équations ; `to_latex` l. 927 inverse **ligne par ligne** (pas par groupe) → en-têtes `for`/`with` séparés de leur corps, comme M4 mais sans possibilité de correction.

---

## 3. Architecture, état & reproductibilité

### [MAJEUR] M2 — `model_graph.py:483-488` : échec silencieux du traçage d'exécution, le graphe « exact » promis devient un graphe deviné [RÉSOLU 739af172]
```python
if data:
    try:
        tr = trace_model(model, data)
    except Exception:
        tr = None
```
Aucun warning, aucune trace : la docstring (l. 11-18) promet des arêtes *exactes* dès que `data` est fourni ; en réalité n'importe quelle exception (modèle qui crashe, données incompatibles, `range(traced_N)`, control-flow sur valeurs) fait retomber sans bruit sur le graphe AST approximatif.
**Déclencheur (reproduit, démo 6)** : `model_to_graph(model5, data=…)` avec `model5` appelant le `varying_slope_non_centered` cassé → `trace_model` lève `NameError`, `model_to_graph` retourne **le même graphe fantôme que sans data**, sans un mot. L'utilisateur croit disposer de la reconstruction par exécution. Contraste avec `model_trace.py:335-338` qui, lui, avertit (`warnings.warn`) quand la détection d'arêtes échoue.

### [MAJEUR] M3 — `model_graph.py:670-694` : la suppression d'« arêtes de contournement » efface de vraies dépendances directes [RÉSOLU 739af172]
`_reaches_via_det(u, v)` supprime l'arête directe `u→v` dès qu'un chemin `u → (nœud déterministe) → v` existe. Mais dans un modèle graphique, l'arête directe est une affirmation d'indépendance conditionnelle : la supprimer change la sémantique du DAG.
**Déclencheur (reproduit)** :
```python
sigma = numpyro.sample("sigma", dist.Exponential(1.0))
mu = sigma * 2.0
numpyro.deterministic("mu_site", mu)
numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
```
→ arêtes finales `[mu→y, sigma→mu, mu→mu_site]` : l'arête `sigma→y` a été **supprimée** alors que `sigma` est l'échelle de `y` (`parents["y"] == {"sigma"}` dans la trace). Le graphe affirmé : `y ⫫ sigma | mu` — faux.

### [MINEUR] m11 — `model_graph.py:504-511` : décalage de lignes faux pour les fonctions décorées [RÉSOLU f4ecf90b]
`line_offset = start_line - func.lineno` : `inspect.getsourcelines` inclut les décorateurs (première ligne = premier décorateur) alors que `func.lineno` pointe le `def` → l'offset est décalé du nombre de décorateurs, et le rattachement `site.line in line_to_lhs` (l. 652) se fait sur des lignes fausses (mauvais LHS ou rattachement manqué).

### [MINEUR] m12 — `model_graph.py:311-418` : figures matplotlib jamais fermées ; `plates`/`shapes` capturés mais jamais rendus [RÉSOLU f4ecf90b]
`draw()` crée une figure par appel sans `plt.close` (fuite mémoire dans les notebooks / boucles `for name in models: m.graph()`), et `plt.show()` bloque en script. Les attributs `self.plates` et `self.shapes` (l. 316-317), alimentés par la trace (l. 647-648), ne servent **nulle part** dans le rendu : le graphe dessiné ne montre aucune plaque, contrairement à la promesse « plate nesting » de l'architecture. (Note : pas de graphviz dans ce module — rendu matplotlib ; aucun fichier temporaire écrit.)

### [MINEUR] m27 — `model_graph.py:40-48` : avertissement « expérimental » global one-shot (`_WARNED`) [RÉSOLU f4ecf90b]
État module mutable, non thread-safe, émis une seule fois par processus — `to_latex` n'émet aucun avertissement équivalent alors que ses tables sont tout aussi expérimentales : incohérence UX.

### Points positifs vérifiés (grille, item 3)
- **Handlers numpyro corrects** : la substitution des sites `deterministic` par `handlers.substitute` est bien supportée en numpyro 0.22 (`process_message` couvre `sample/param/mutable/plate/deterministic`) ; le messager `_RawDeterministic` imbriqué **dans** `substitute` (`model_trace.py:296-305`) capture la valeur brute avant substitution — l'ordre inner-first de `apply_stack` est correctement exploité. `reversed(cond_indep_stack)` (l. 265, 276) donne bien l'ordre outermost-first (les frames sont appendues inner-first).
- **Aucune API privée** : imports limités à `numpyro.handlers` et `numpyro.primitives.Messenger` (publics) ; rien en `numpyro._src`.
- **Reproductibilité** : `trace_model(seed=0)` (l. 231, 246, 302) — graine fixe, double passage cohérent (même sous-échantillon de plaque), propagation jaxpr conservative pour scan/while/cond (l. 189-204).
- **Découplage Main** : PostModel n'importe pas Main ; `Main.main.latex()/graph()` (main.py:348-381) passent `self.model`/`self.data_on_model` via `getattr` — pas de dépendance à `parent.posteriors`.

---

## 4. Robustesse de l'API publique

### [MINEUR] m26 — `to_latex.py:907-932` : chaîne de `except Exception` nus, aucune dégradation signalée [RÉSOLU b45052b6]
Chemin AST → chemin ligne-par-ligne → chemin trace : chaque bascule avale l'exception d'origine sans log ni warning. Un bug dans `_stmt_to_latex_lines` (p. ex. nœud AST inattendu sur une seule ligne) fait silencieusement perdre l'expansion des blocs BF pour tout le modèle. `data=None` + échec des deux chemins source → `raise` (l. 932) : l'erreur remonte enfin, mais sans mentionner les causes intermédiaires.

### [MINEUR] m20 — `model_trace.py:240-247, 154-170` : données mal formées = crash non guidé [RÉSOLU f4ecf90b]
- Clé de `data` absente de la signature du modèle → `TypeError` de `get_trace(**data)` (l. 247) propagée telle quelle dans `_latex_from_trace` (to_latex.py:883) ; message illisible pour l'utilisateur final.
- `data` non-mapping (liste) → `AttributeError` sur `.items()` (l. 157).
- Scalaires numpy (`np.int64`) placés dans `arrays` (l. 158-169 ne teste que les scalaires *python*) puis tracés par `make_jaxpr` : un modèle avec `for i in range(N)` et `N` numpy-int échoue au second passage → warning « no edges » (l. 335-338), graphe sans arêtes.

### [MINEUR] m21 — sites `numpyro.param` (SVI) : invisibles dans la trace, traités comme distributions dans l'AST [RÉSOLU f4ecf90b]
`model_trace.py:251-279` n'ignore que `fn is None` ; les messages de type `param` ne sont ni `sample` ni `deterministic` → les paramètres SVI disparaissent du graphe tracé. À l'inverse, `_NUMPYRO_SITE_FNS` de `model_graph.py:80-81` inclut `param` → nœud « latent » avec pour étiquette de distribution `param` (`_dist_name`, l. 118). `to_latex.py:37-40` ne liste pas `param` : les trois fichiers divergent.

### [MINEUR] m8 — `model_graph.py:626` : site latent sans nom étiqueté `obs_<dist>` et collisions [RÉSOLU f4ecf90b]
`node = name_kw or obs_name or f"obs_{_dist_name(call)}"` : un `m.dist.normal(mu, 1)` nu (latent, sans `name` ni `obs`) devient un nœud `obs_normal` — préfixe mensonger ; deux appels nus de même loi fusionnent en un seul nœud.

### [MINEUR] m9/m10 — nœuds manquants ou injustement élagués [RÉSOLU f4ecf90b]
- `model_graph.py:567` : les affectations multi-cibles (`a = b = expr`) sont ignorées → nœuds absents.
- `model_graph.py:696-700` : un argument d'entrée utilisé uniquement comme taille de plaque (`with numpyro.plate("i", N)`) ne génère aucune arête (le `context_expr` n'est jamais scanné comme dépendance) → `N` est élagué, le graphe masque une entrée du modèle.

### [MINEUR] m15/m16 — rendus « creux » ou trompeurs au lieu d'erreurs explicites (grille, item 4) [RÉSOLU f4ecf90b — partiel : le 3e sous-point (distributions inconnues, chemin AST du graphe) reste en l’état]
- `to_latex.py:868-875` : paramètre vectoriel sans parent tracé → l'étiquette interne (`loc`, `scale`, `arg0`) est imprimée à la place d'une valeur : `x ~ Normal(loc, scale)`.
- `to_latex.py:885-893` : site observé rendu comme latent (cf. M5).
- Distribution inconnue : `_process_dist_call` (l. 642-665) ne consulte **aucune** table — `m.dist.foo(...)` rend `Foo(args)` avec le nom capitalisé à l'aveugle ; un bloc `m.effects.inconnu(...)` retombe sur `\text{Inconnu}(...)` (l. 440-443). Jamais d'erreur, jamais de mention « non reconnu » : rendu silencieusement plausible.
- Trace vide (modèle sans sites) : `_latex_from_trace` renvoie `[]` → `gather*` vide, sans message (to_latex.py:934-937).
- Modèle non fitté sans data : comportement **correct** par conception (chemin AST sans exécution).
- Écriture de fichiers : le module n'écrit aucun fichier (pas de dossier, pas de permissions en jeu) ; `to_latex` retourne la chaîne et `display()` (effet de bord, m13/m14 ci-dessous).

### [MINEUR] m13/m14/m24 — cohérence d'API [RÉSOLU f4ecf90b]
- Docstring `to_latex` (l. 898) annonce « LaTeX ``align*`` environment » mais émet `gather*` (l. 934).
- `display(Latex(...))` (l. 938) : effet de bord d'affichage dans une fonction qui retourne déjà la chaîne ; hors IPython, imprime le LaTeX brut (fallback l. 10-12).
- `to_latex.py:748-749` : `mu = numpyro.deterministic("mean", e)` rendu sous le nom Python `mu`, alors que `model_graph` (l. 576-583) utilise le nom de site `mean` — les deux vues d'un même modèle divergent.
- `model_graph.py:114-118` : `numpyro.sample("x", ma_loi)` (distribution en variable) reçoit l'étiquette de distribution `sample`.
- `to_latex.py:692-697` : `numpyro.plate("i", N, subsample_size=n)` rendu `for i ∈ 1..N` — le sous-échantillonnage est nié ; `dim=` ignoré ; indexing 1-based affiché là où numpyro indexe 0-based.

---

## (a) Résumé du module

PostModel est un module d'introspection à trois étages : `to_latex.py` (source Python → AST → `gather*` LaTeX, avec tables d'expansion des blocs BF `m.effects`/`m.net`/`m.gaussian` et des liens), `model_trace.py` (le modèle est **exécuté** sous `handlers.seed/trace/substitute` + `jax.make_jaxpr` pour obtenir sites, plaques et dépendances exactes), `model_graph.py` (fusion AST+trace en `ModelGraph` avec rendu matplotlib). L'ossature est bonne : découplage total de Main, aucune API privée numpyro, handlers utilisés correctement (vérifié contre numpyro 0.22), graine fixe, propagation jaxpr prudente, et le module ne touche jamais les draws du posterior. La table des blocs `effects` est **matériellement juste** sur les noms de sites et les priors par défaut (vérifiée contre Effects.py, Gaussian.py, model_effects.py — y compris `inv_logit_scale`, `diag(σ)·R·diag(σ)`, `LKJCholesky(2, 2.5)`).

Le problème central est épistémique : **le module produit du contenu scientifique (équations, DAG) sans jamais signaler l'incertitude ni l'échec**. Les tables de reconnaissance ignorent que `varying_slope_non_centered` est un NameError ambulant (M1) ; le chemin AST invente des priors pour `varying_intercept_slope` (C1) ; les échecs de traçage et de rendu sont avalés par des `except Exception` nus (M2, m26) ; des heuristiques de mise en forme suppriment de vraies arêtes (M3) ou désorganisent les blocs imbriqués (M4) ; le chemin réservé aux modèles R/Julia perd toute la structure de plaques (M5). Un utilisateur ne peut pas distinguer un rendu exact d'un rendu fabricqué.

## (b) Fragilités architecturales

1. **Duplication des tables de reconnaissance** entre `to_latex.py:471-620` et `model_graph.py:164-305` (deux codages indépendants des mêmes blocs BF, déjà désynchronisés : `nested names` positionnel, `centered`, `partial_pooling`, `_NUMPYRO_SITE_FNS` divergents). Toute évolution de Utils/Network exige deux mises à jour manuelles — le décalage avec l'API réelle est structurel, pas accidentel.
2. **Absence de canal d'erreur/diagnostic** : le module ne distingue jamais « rendu exact (trace) », « rendu approché (AST) », « rendu dégradé (fallback ligne) », « non reconnu ». Un champ `confidence`/`warnings` sur la sortie, ou au minimum des `warnings.warn` systématiques (comme le fait déjà model_trace.py:335), supprimerait la classe entière de bugs « silencieusement faux ».
3. **Heuristiques de présentation qui altèrent la sémantique** (inversion des corps de plaques, suppression d'arêtes de contournement) : la mise en forme devrait être réversible et ne jamais supprimer d'information probabiliste.
4. **Traitement inégal des trois modes** (AST seul / AST+trace / trace seule) : les fonctionnalités (plaques, noms de sites, priors) se comportent différemment selon la présence de `data` ou la disponibilité du source — deux « vrais » graphes pour un même modèle.
5. **Rendu matplotlib terminal** : figures non fermées, plaques/shapes capturés mais non dessinés, `plt.show()` bloquant — le `ModelGraph` est une bonne IR (nodes/edges/kinds/plates) sous-exploitée par son renderer.

## (c) Décompte par sévérité

| Sévérité | Nombre | Références |
|---|---|---|
| **Critique** | **1** | C1 (priors fabriqués `varying_intercept_slope`) |
| **Majeure** | **5** | M1 (table ↔ API cassée), M2 (échec de trace silencieux), M3 (arêtes directes supprimées), M4 (corps de plaque inversé), M5 (plaques ignorées au rendu trace) |
| **Mineure** | **20** | m1–m27 consolidés (noms `Lkj*`, supports omis, `σ²` GP, `centered`/`partial_pooling` ignorés, `names` positionnel, sites fantômes, `obs_` mensonger, multi-cibles, élagage d'entrées, offset décorateurs, figures/plates non rendus, docstring `align*`, `display`, placeholders `loc/scale`, comparateurs, traductions d'expressions, sous-échantillonnage nié, transforms non tracées, données mal formées, `param` divergents, fallbacks nus, `_WARNED` global) |
| **Total** | **26** | |

Verdict de release : **bloqueur** tant que C1, M1 et M2 ne sont pas traités — le module affiche aujourd'hui, sans le moindre avertissement, des modèles qui ne sont pas ceux de l'utilisateur et un graphe « exact » pour un modèle qui ne s'exécute pas.

---

### Module Network

**Audit de release — `BayesForge/Network/`** (racine `/home/sosa/work/BF`), 9 fichiers lus intégralement (4 286 lignes). Toutes les affirmations marquées « mesuré » ont été reproduites sur l'environnement du projet (Python 3.14, JAX float32, bf v0.0.60) ; aucun code source n'a été modifié.

---

## 1. Bugs et fragilités

### CRITIQUE

**C1. Betweenness pondérée fausse sur graphe déconnecté (silencieux)** — `BayesForge/Network/metrics.py:551` + `metrics.py:582-596` `[RÉSOLU 042c9601]`
Dans `dijkstra` (le helper de betweenness pondérée, L508-552), les sommets atteignables sont renvoyés dans `S_reversed` avec le sentinel `-1` pour les non-atteignables (L548-551). `_optimized_accumulate_basic` (L582-596) boucle sur **les n_nodes entrées**, y compris les `-1` : `w = S[i] = -1` → `P[w, :]` et `delta[w]` (L588-589) **s'enroulent sur le dernier sommet** (sémantique indice négatif, vérifiée : `jnp.zeros(4).at[-1].add(5)` → `[0,0,0,5]`), et `betweenness.at[w].add(delta[w])` (L592-593) **ajoute delta[n-1] au sommet n-1 une fois par sommet non atteignable**. Le chemin non pondéré (`bfs`, L554-578) n'est pas touché car son `S` est une permutation sans `-1` — d'où un résultat correct… jusqu'à ce qu'on passe `weight_matrix`.
*Scénario mesuré* : étoile centrée sur le sommet 4 (n=5) + sommet 3 isolé, `met.betweenness(A, 5, weight_matrix=A)` → **1.25** pour le centre au lieu de **0.5** (×2.5) ; la version non pondérée du même graphe donne bien 0.5. Toute normalisation/centrality consommée en aval est silencieusement fausse.

**C2. `betweenness(endpoints=True)` plante systématiquement** — `BayesForge/Network/metrics.py:598-600`, `616`, `655` `[RÉSOLU 042c9601]`
`_optimized_accumulate_endpoints` déclare `s_len` dans `static_argnames`, mais `s_len` est un **traced value** calculée sous `vmap` (L653-655). N'importe quel appel avec `endpoints=True` lève `ValueError: Non-hashable static arguments are not supported … BatchTracer` (mesuré). Le paramètre documenté (L685) est inutilisable sur tout graphe.

**C3. `sender_receiver` sans `sender_predictors` → crash AttributeError** — `BayesForge/Network/model_effects.py:161-163` `[RÉSOLU 042c9601]`
Dans la branche `receiver_predictors is not None`, la ligne 163 lit `N_id = sender_predictors.shape[0]` alors que `sender_predictors` est `None` → `AttributeError: 'NoneType' object has no attribute 'shape'` (mesuré). La doc (L135-136) présente les deux arguments comme indépendants. Le jumeau `net2` gère correctement le cas (`model_effects2.py:157-159`) — la dérive entre les deux versions est déjà matérialisée. Consommateur direct : `BayesForge/Models/SRM.py:305`.

**C4. `single_source_dijkstra` silencieusement faux sur matrice d'adjacence brute** — `BayesForge/Network/metrics.py:440-466` `[RÉSOLU 042c9601]`
La relaxation L460 (`jnp.minimum(dist, dist[u] + m[u])`) ne consulte **aucun masque d'adjacence** : une absence d'arête (valeur 0) est relaxée comme une arête de coût 0. La conversion `0 → inf` n'est faite que dans `geodesic_distance` (L480) et n'est documentée nulle part sur la méthode publique.
*Scénario mesuré* : `met.single_source_dijkstra(B, 0)` sur le chemin 0-1-2-3 (B binaire) → distances `[0, 1, 0, 0]` (les sommets 2 et 3 à distance **0** !), alors que `met.dijkstra_jax(B, 0)` renvoie `[0, 1, 1, 2]`. Deux méthodes publiques adjacentes, sémantiques d'entrée contradictoires, aucun avertissement.

### MAJEUR

**M1. Deux méthodes `dijkstra` dans la même classe : l'API publique est écrasée** — `BayesForge/Network/metrics.py:271-273` vs `metrics.py:508-515` `[RÉSOLU 4cac9f71]`
Le wrapper public `dijkstra(m, source)` (L272) est redéfini plus bas par le helper de betweenness `dijkstra(adjacency_matrix, weight_matrix, source, n_nodes)` (L510). `met.dijkstra(adj, 0)` lève `TypeError: missing 2 required positional arguments` (mesuré). L'API documentée aux L271-273 n'existe plus ; aucun test ne le détecte.

**M2. `eigenvector` : contrat NetworkX non tenu, paramètre ignoré** — `BayesForge/Network/metrics.py:136-208` `[RÉSOLU 4cac9f71]`
- `component_aware` (L136) est déclaré statique et documenté (L147-150) mais **jamais utilisé** dans le corps : `component_aware=True` et `False` renvoient le même vecteur (mesuré `same=True`).
- Graphes **dirigés** : la power iteration utilise `A` (L185-192) alors que NetworkX utilise `A.T` pour un `DiGraph` (centralité du *receveur*). Mesuré sur le chemin orienté 0→1→2 : `[0.9998, 0.020, 0.0002]` (émetteur dominant) — l'inverse du comportement revendiqué L138 (« matching NetworkX behavior »).
- `tol` est ignoré (L146, assumé) : 100 itérations fixes, sans test de convergence → résultat dépendant de `max_iter` sur les graphes à spectre serré.
- `jnp.float64` (L178, 185, 188) est **rétrogradé en float32 avec UserWarning** quand `jax_enable_x64` est désactivé (mesuré) — la précision demandée est silencieusement perdue.

**M3. Clustering : self-loops et graphes dirigés non gérés, bornes [0,1] violées** — `BayesForge/Network/metrics.py:36-68` `[RÉSOLU 4cac9f71]`
`triangles_and_degree` (L36-51) ne zero pas la diagonale : une self-loop sur un sommet de degré 1 donne `cc = 1.5` (mesuré sur `A=[[1,1],[1,0]]` → `[1.5, 0]`), violant la promesse « in [0, 1] » de `cc` (L113). Le sens `>` (L42) traite les poids négatifs comme absents. Matrice asymétrique : la formule est appliquée telle quelle (degré = somme de ligne, triangles /2) sans avertissement — résultat dépourvu de signification pour un graphe dirigé.

**M4. `density` : NaN en N≤1, >1 avec self-loops, doc trompeuse** — `BayesForge/Network/metrics.py:420-436` `[RÉSOLU 4cac9f71]`
`n_possible_edges = n(n-1)/2` → division par zéro : `met.density(zeros((1,1)))` → **NaN**, idem N=0 (mesuré). Self-loop comptée comme arête : `[[1,1],[1,0]]` → **1.5** (mesuré). La docstring annonce « from the weighted adjacency matrix » mais `count_nonzero` (L432) calcule une densité **binaire**. (Formule correcte par coïncidence pour le cas dirigé : nnz/(n(n-1)).)

**M5. `degree`/`strength` : le défaut `sym=False` double les degrés des graphes non dirigés** — `BayesForge/Network/metrics.py:296-311`, `359-381`, `18-29` `[RÉSOLU 4cac9f71]`
Sur matrice symétrique, `met.degree(A)` (défaut) renvoie in+out = **2× le degré** : mesuré `[4,2,2]` au lieu de `[2,1,1]` ; idem `strength`. C'est documenté, mais le piège par défaut est patent — `vizfeatures.py:111-112` doit passer `sym=not directed` explicitement. De plus `normalize` (L29) divise par N−1 même le **degré total** dirigé (max 2(N−1)) → valeurs jusqu'à 2 ; et **N=1 → NaN** (mesuré `[nan]`).

**M6. `diameter` renvoie `inf` en silence sur graphe déconnecté** — `BayesForge/Network/metrics.py:491-502` `[RÉSOLU 4cac9f71]`
`jnp.max(geodesic_distance(m))` sur composantes déconnectées → `inf` (mesuré). NetworkX lève une exception ; ici l'`inf` se propage dans les stats/JSON (`_js_val` le convertit en `NaN`, `explorer.py:1445-1446`).

**M7. `logit` sans clipping + `block_build_mu_ij` → NaN silencieux dans les priors de bloc** — `BayesForge/Network/model_effects.py:18-32`, `427-450`, `332-338` `[RÉSOLU 4cac9f71]`
`logit` = `jnp.log(x/(1-x))` sans clamp : mesuré `logit(0) = -inf`, `logit(1) = +inf`, `logit(1.2) = NaN`. Dans `block_model` (L334), `mu_ij = logit(block_build_mu_ij(N_by_group, N_group))` : un groupe **vide** (`N_by_group` contient 0) donne `ss = 0` → `0.1/0 = inf` (mesuré matrice avec `inf`) → `logit(inf) = NaN` → `dist.normal(NaN, …)` : **prior mean NaN, NaN dans le log-density, sans aucune erreur**. C'est le motif « logit sans clipping » déjà rencontré ailleurs dans le projet, ici sur le chemin des tables LaTeX (PostModel) et de SRM.

**M8. `mat_to_edgl` (contrat SRM) : matrice non carrée tronquée en silence** — `BayesForge/Network/util.py:163-180` `[RÉSOLU 4cac9f71]`
`N = mat.shape[0]` puis triu sur `N` : une matrice 2×3 renvoie **1 dyade** sans erreur (mesuré, shape `(1,2)`) ; une entrée 1-D lève un `IndexError` obscur (mesuré) ; la diagonale non nulle est **silencieusement écartée** (L175 `k=1`). Contrat consommé par `SRM.py:23`, `:93`, `:280` — une matrice mal découpée en shard produit un edgelist tronqué sans signal. (Le aller-retour `mat_to_edgl → edgl_to_mat` sur matrice dirigée est correct, mesuré ; `edgl_to_mat` valide bien sa longueur, `util.py:194-198`.)

**M9. `vis_net(metrics=True)` : matérialisation (N,N,N) du betweenness → OOM sur grands graphes** — `BayesForge/Network/explorer.py:1301-1312` + `metrics.py:659` `[RÉSOLU 4cac9f71]`
`_vmapped_betweenness_computation` vmape la logique source-par-source dont le `P_matrix` (N,N) → JAX matérialise (N,N,N) : ≈ 2 Go à N=800, ≈ 13 Go à N=1500 (seuil canvas, `explorer.py:1371`), aucune garde N dans `centralities` (`vizfeatures.py:119`). Le `try/except` de `_try` (`vizfeatures.py:103-109`) peut rattraper l'exception… si le processus survit au thrash/OOM. S'y ajoutent, pour N ≤ 800, **tous** les layouts PINNED calculs à chaque appel (`explorer.py:1328-1344`, spectral=eigh O(N³), layered=N BFS complets, `layouts.py:246`) et `chord_matrix` dès qu'un `layout_col` est fourni (`explorer.py:1351-1353`).

### MINEUR

**m1. `betweenness` : asymétrie acceptée en mode non dirigé, float32, n_nodes séparé** [RÉSOLU a7cae02f] — `metrics.py:664-717`, `517-521`, `559`
`directed=False` ne vérifie pas la symétrie : matrice asymétrique → chemins calculés dans un seul sens puis **divisés par 2** (L711-712), résultat faux en silence. `sigma`/`delta` en **float32** (L517-518, 559) : le comptage de chemons déborde (→ inf/NaN) sur les graphes denses (~n! dépasse 3.4e38 dès N≈30-40 en pire cas). `n_nodes` est un argument indépendant de la matrice : incohérence → erreur de broadcast (mesuré, explicite mais tardive).

**m2. `onnela_clustering_coefficient` : diagonale non masquée malgré la doc, poids négatifs → 0** [RÉSOLU a7cae02f] — `metrics.py:72-100`
La docstring (L87-88) dit « Assumes a zero diagonal » mais rien ne la zero : avec une self-loop de poids 5 sur un triangle complet, cc du hub mesuré **0.46** au lieu de 1.0 (le max_w et le einsum L96 intègrent la diagonale). Poids négatifs : `cbrt` négatif → cc mesuré `[0,0,0]` sans avertissement. `deg` = somme de ligne (L98) : faux pour matrice dirigée.

**m3. Échauffement JIT dans `met.__init__`** [RÉSOLU a7cae02f] — `metrics.py:13-16`
Chaque instanciation de `met`/`net`/`net2` exécute `met.betweenness(zeros((4,4)), 4)` pour pré-compiler ; couplé au `_LazyViz` de `Net.py:8-30`, la construction de `m.net` déclenche compilation JAX + (au premier `.viz`) `FileNotFoundError` possible si les assets d3 manquent (documenté, mais l'échauffement, lui, n'apporte rien au premier appel réel qui re-compile pour le vrai N).

**m4. Méthodes sans `@staticmethod` : appels sur instance cassés** [RÉSOLU a7cae02f] — `metrics.py:18-19`, `53-54`
`normalize` et `clustering_coefficient` sont décorées seulement de `@jit` : l'appel via classe fonctionne (PjitFunction n'est pas descripteur), mais `m.net.normalize(x, A)` lève `TypeError` (le `self` est tracé comme tableau, mesuré). Incohérent avec le reste de la classe.

**m5. Trois Dijkstra aux sémantiques incompatibles** [RÉSOLU a7cae02f] — `metrics.py:211-269`, `440-466`, `510-552`
`dijkstra_jax` (masque `>0`, zéro = absente), `single_source_dijkstra` (relaxe tout, exige un pré-masquage inf non documenté — voir C4), `dijkstra` de betweenness (masque `>0`, tolérance absolue 1e-9 L537). Poids négatifs acceptés par `geodesic_distance` (L480 ne convertit que les 0) alors que Dijkstra est invalide → distances fausses en silence.

**m6. Décorateur orphelin `@staticmethod` : double décoration de `block_model`** [RÉSOLU 6f014b67] — `model_effects.py:307-311`
Reste d'une fonction supprimée : `@staticmethod` (L307) suivi de `@staticmethod` (L310) sur `block_model`. Fonctionne en Python ≥ 3.10 (staticmethod callable, mesuré `callable=True`) mais **TypeError à l'import en ≤ 3.9** ; de plus `Neteffect.block_model` renvoie un `staticmethod` enveloppé — accès non conventionnel.

**m7. `dyadic_effect` renvoie une chaîne d'erreur au lieu de lever** [RÉSOLU 6f014b67] — `model_effects.py:277-279`
Sans `dyadic_predictors` ni `shape` : `print` + `return 'Argument shape must be…'` — un `str` se propage dans l'addition de logits aval avec une erreur de type obscure. `prepare_dyadic_effect` (L194-197) : pas de `else` → **retourne None** pour tout ndim ∉ {2,3} ; le `vmap(in_axes=2)` (L197) impose (N,N,K) sans le documenter.

**m8. Dérive de contrat `dyadic_effect` net vs net2 sur l'entrée 2-D** [RÉSOLU 6f014b67] — `model_effects.py:248-256` vs `model_effects2.py:291-313`
Même nom, même classe-mère, sémantiques **différentes** : `net.dyadic_effect(X)` avec X 2-D traite X comme un edgelist (E,K) avec **un scalaire partagé** sur les K colonnes (L249-250, `shape=(1,)`) ; `net2.dyadic_effect(X)` avec X 2-D traite X comme une **matrice (N,N)** (L304-309, mesuré (4,4)→(4,4)). Passer de `m.net` à `m.net2` (la migration shard-compatible annoncée `Net.py:51-57`) change silencieusement la sémantique des predicteurs. `dyadic_effect(shape=5)` (non triangulaire) : N arrondi à 4 (`model_effects2.py:312`) → erreur de broadcast obscure (mesurée) au lieu d'un message explicite.

**m9. `nodal_random_effects` : index de groupe 1-based, enroulement silencieux** [RÉSOLU 6f014b67] — `model_effects.py:411`
`group_idx_0 = random_group_index - 1` : un vecteur 0-based (erreur fréquente) produit `-1` → **dernier groupe** (gather JAX), affectation silencieusement fausse, aucun contrôle de bornes.

**m10. `dyad_vec_to_mat` dupliqué + erreurs obscures** [RÉSOLU 6f014b67] — `util.py:280-318` et `util.py:321-358`
Deux implémentations identiques (méthode + fonction module) — dérive future garantie. Longueur non triangulaire : N déduit par `round` puis erreur de broadcast incompréhensible (mesuré : « Incompatible types for broadcasting: input type=float32[2] and requested type=float32[3] ») au lieu de « length 5 is not N(N−1)/2 ».

**m11. Division par zéro dans les normalisations d'explorer (colonnes constantes)** [RÉSOLU 2fee13b0] — `explorer.py:957`, `explorer.py:1259`
`v = (v - v.min()) / (v.max() - v.min())` : colonne d'opacité constante → **NaN** (mesuré `[0.001, nan, nan]`) ; même motif pour `link_opacity` sur poids constants. Le chemin `col_size` (L976) protège pourtant `hi == lo` — incohérence interne.

**m12. `vis_net` ne vérifie pas `len(df) == N`** [RÉSOLU 2fee13b0] — `explorer.py:1201-1243`
Si le tableau de nœuds n'a pas la taille de la matrice : `IndexError` profond dans `mat_to_edgl` (`ids[i]`, L810-811) ou, si df > N, décalage nœuds/métriques dans `_nodes_json` (`v[i]`, L1545). Une validation en tête de `vis_net` s'impose.

**m13. Layouts : N=0/N=1 non gardés, Fiedler arbitraire sur graphe déconnecté** [RÉSOLU 2fee13b0] — `layouts.py:147-158`, `223-249`
`spectral` N=1 → `IndexError: index 1 is out of bounds` (mesuré, L158) ; `layered` N=0 → `ValueError: attempt to get argmax of an empty sequence` (mesuré, L245). `spectral` prend `vecs[:, 1:3]` (L156) : avec k composantes, les k−1 premiers vecteurs sont des zéros-eigenvalue → la doc « smallest non-zero eigenvalue » est fausse sur graphe déconnecté. `layered` (L246) calcule `_bfs_all_pairs` (N BFS, matrice N×N) pour ne lire **qu'une ligne** — un BFS mono-source suffirait.

**m14. `chord_matrix` : double boucle Python O(N²), calculée inconditionnellement** [RÉSOLU 2fee13b0] — `layouts.py:269-272` + `explorer.py:1351-1353`
Agrégation groupe×groupe en Python pur sur toutes les paires (i,j) ; déclenchée dès que `layout_col` est fourni, **même sans `layout="chord"`**. Vectorisable en une ligne (one-hot @ A).

**m15. Stats : densité de repli incohérente pour graphes dirigés ; self-loops gonflent tout** [RÉSOLU 2fee13b0] — `vizfeatures.py:52-53`, `74`
Le repli L74 (`edges / (n(n-1)/2)`) utilise la formule non dirigée même quand `directed=True`, alors que `met.density` (chemin normal) donne nnz/(n(n−1)) — deux valeurs possibles pour le même graphe selon que l'exception a sauté. Self-loop : `network_stats([[1,1],[1,0]])` → densité **1.5**, clustering 0.75 (mesuré) ; les self-loops entrent dans `edges`, `deg`, `abin`.

**m16. Hygiène et état mort** [RÉSOLU 6f014b67] — `model_effects.py:14` (`self._cache` jamais lu/écrit), `util.py:1-22` (imports dupliqués, `import jax as jax`, `lk` inutilisé), `util.py:210-230` (bloc commenté), `util.py:138` (docstring `get_tri` inverse : le code empile **upper** en première colonne, L153-155), `metrics.py:110` (commentaire mort après `return`), `model_effects.py:221-229` (`print` de diagnostic dans le corps du modèle — trace à chaque recompilation sous jit).

**m17. Injection/échappement HTML-JS depuis les données utilisateur** [RÉSOLU 2fee13b0] — `explorer.py:1450`, `1457-1465`
`_js_val` n'échappe que les apostrophes (L1450) ; un id ou nom de colonne contenant un saut de ligne, `</script>` ou une balise casse le HTML généré ou injecte du JS dans le fichier produit. `_tooltip_js` interpole les noms de colonnes bruts dans du JS.

**m18. Masquage du diagonal par multiplication dans l'exemple net2** [RÉSOLU 6f014b67] — `model_effects2.py:47` (docstring d'usage)
`bernoulli(logits=logits * (1 - jnp.eye(N)))` met le diagonal à logit 0 → **p = 0.5** (contribution constante `−N·log 2` à la log-likelihood), pas une exclusion ; si `Y_mat` a un 1 au diagonal, le modèle est contredit sans avertissement. Un `where` avec `-inf` ou un masquage de l'`obs` serait le contrat correct.

---

## 2. Résumé du module

`Network/` est un module à trois étages de maturité très inégale. **`metrics.py`** est le noyau à risque : les métriques simples (degré, force, clustering binaire, densité sur cas nominal, betweenness **non pondéré** — validé numériquement sur cas connecté et déconnecté) sont correctes, mais le diable est dans les bords : le chemin pondéré du betweenness est **silencieusement faux sur graphe déconnecté** (C1), `endpoints=True` est **cassé** (C2), `eigenvector` ne tient pas son contrat NetworkX sur graphes dirigés et ignore son paramètre `component_aware` (M2), et trois implémentations de Dijkstra aux conventions incompatibles coexistent (C4, M5 de la liste mineure m5). **`model_effects.py` / `model_effects2.py`** (contrats SRM/PostModel : `mat_to_edgl`, `block_prior_to_edglelist`, `dyadic_effect`, `sender_receiver`) sont structurellement sains — parametrisation non-centrée correcte, échantillonnage LKJ/Cholesky juste, aller-retour edgelist↔matrice validé numériquement — mais portent un chemin de crash (C3), un chemin de NaN silencieux dans les priors (M7) et une dérive sémantique nette entre `net` et `net2` sur les entrées 2-D (m8). **`explorer.py`/`layouts.py`/`vizfeatures.py`** sont défensifs (try/except partout, PRNG pur et seedé — `default_rng(seed)` avec seed=0 partout, reproductibilité OK, pas de mutation de la matrice de l'utilisateur — `_sym_binary` alloue, `fill_diagonal` travaille sur la copie) mais avec des pièges de performance/memory (M9, m14) et de validation d'entrée (m11, m12). Les conventions matrice↔edgelist (triangle supérieur, colonnes i→j / j→i, diagonale exclue) sont **cohérentes** entre `util.mat_to_edgl`, `vec_node_to_edgle`, `edgl_to_mat`, `dyad_index_maps` et SRM — c'est le point fort du module.

## 3. Fragilités architecturales

1. **Duplication `model_effects` / `model_effects2`** : `sender_receiver`, `block_model`, `dyadic_effect` sont *réimplémentés* (pas dérivés) avec les mêmes noms de sites numpyro (bien) mais des sémantiques d'entrée divergentes (2-D edgelist vs 2-D matrice) et des chemins d'erreur différents (C3 vs net2 qui tient). Toute correction future (ex. C3) devra être faite deux fois — c'est le schéma de dérive classique. Un noyau commun « nodal → (N,N) ou (E,2) » avec un seul adaptateur de format éliminerait la classe.
2. **Trois Dijkstra + un BFS à sémantiques incompatibles** dans `metrics.py` (0 = absente / 0 = arête gratuite / pré-masquage inf implicite) : la bonne pièce est `dijkstra_jax` ; les deux autres devraient être privées ou unifiées derrière un unique `geodesic` validé.
3. **Classes-espaces de noms** : `met`/`array_manip` sont des fourre-tout de staticmethods composés par MRO (`net(met, Neteffect, array_manip)`) avec un `__init__` qui compile du JIT en effet de bord (m3) et un `_cache` mort ; aucune validation d'entrée aux frontières du module (carré, N, len(df), symétrie, diagonal) — les erreurs émergent tard et loin de leur cause (M8, m12, m10).
4. **Génération HTML par substitution de motifs** (`explorer.py:1715-1859`) : les ancres `p2.replace(old_link, …, 1)` sont des no-ops silencieux si le patron évolue ; l'échappement des données dans le JS est minimal (m17). Le budget de rendu (tous les layouts PINNED + chord + centralités à chaque appel, M9/m14) devrait être piloté par la taille du graphe, pas par un seuil fixe N≤800.
5. **Aucune garde sur la complexité mémoire du betweenness vmap** ((N,N,N) materialisé) — le seul mécanisme de protection est un `try/except` qui suppose que l'OOM est rattrapable (M9).

## 4. Décompte par sévérité

| Sévérité | Nombre | Références |
|---|---|---|
| **Critique** | **4** | C1 betweenness pondéré faux (déconnecté), C2 `endpoints=True` cassé, C3 `sender_receiver` receiver-only crash, C4 `single_source_dijkstra` faux en silence |
| **Majeure** | **9** | M1 shadowing `dijkstra`, M2 `eigenvector` (dirigé + `component_aware` + float64), M3 clustering/self-loops (>1), M4 `density` NaN/>1, M5 défaut `sym=False` (×2) + normalize N=1, M6 `diameter` inf silencieux, M7 logit sans clipping → NaN priors, M8 `mat_to_edgl` non carrée tronquée, M9 OOM betweenness dans `vis_net` |
| **Mineure** | **18** | m1–m18 (asymétrie non dirigée, float32 sigma, onnela diagonale/négatifs, échauffement JIT, staticmethods manquants, triple Dijkstra, décorateur orphelin, chaîne d'erreur retournée, dérive net/net2 2-D, index 1-based, `dyad_vec_to_mat` dupliqué, divisions par zéro explorer, len(df) non vérifié, layouts N≤1, chord O(N²), stats dirigé/self-loops, hygiène, échappement JS, masquage diagonal par multiplication) |
| **Total** | **31** | |

**Recommandation de release** : bloquants = C1–C4 + M7 (chemins SRM/priors) ; à corriger avant publication = M1–M6, M8–M9. Les cas de régression minimaux sont dans ce rapport (étoile+isolé pondéré, `endpoints=True`, receiver-only, `single_source_dijkstra` brut, `logit`/groupe vide, cc avec self-loop, `density` N=1, `mat_to_edgl` 2×3).

---

### Module Parallel

**Audit de release — `BayesForge/Parallel/`** (racine `/home/sosa/work/BF`)
Fichiers lus intégralement : `simulate.py` (938 lignes), `__init__.py` (5 lignes).

**Mécanisme réel (vérifié, pas celui qu'on suppose)** : ni threads, ni `pmap`.
`_parallel` (L829–938) utilise `multiprocessing.get_context("spawn")` (L853) +
`concurrent.futures.ProcessPoolExecutor` (L892–897), avec `cloudpickle` pour
`fn`/`fn_kwargs`/payloads (L846–848), un `ctx.Manager().Queue()` (L887–889)
pour les événements de progression, un épinglage CPU par
`os.sched_setaffinity` (L215–261) et un budget JAX posé par variables
d'environnement (`XLA_FLAGS --xla_force_host_platform_device_count`,
`BF_MAX_CORES`, pools BLAS — L172–192). Le chemin `serial` (L815–826) partage
`_run_one` (L320–350) avec les workers — bonne idée. Pas de math embarquée
(densités, poids) : le volet A « math » est largement N/A ici ; l'audit porte
donc sur le cycle de vie des ressources, la reproductibilité et la robustesse
API.

---

## Bugs et fragilités

### CRITIQUE

**C1. Une erreur hors du corps de `fn` dans le worker avorte toute la run et jette tous les résultats déjà collectés, même avec `on_error="record"`** `[RÉSOLU 4184752b]`
`BayesForge/Parallel/simulate.py:289-292, 306, 912`
La promesse documentaire (L485–486 : « `'record'` keeps going and puts the
traceback in the row's `error` column ») ne couvre que les exceptions levées
*par* `fn` (capture L331–337). Tout ce qui entoure l'appel dans
`_worker_entry` est hors de ce try :
- dé sérialisation différée de `fn_kwargs` à la première tâche (L290–292) ;
- `cloudpickle.dumps((rows, error))` du résultat (L306) ;
- un `BaseException` de `fn` (`KeyboardInterrupt`, `SystemExit` — non capturé
  par `except Exception`, L335).

Ces échecs remontent en exception du future ; `future.result()` (L912) les
relance dans le parent, hors de toute gestion `on_error`, et la liste `rows`
(toutes les simulations déjà terminées) est perdue avec la stack.
*Scénario* : étude de 200 fits de 20 min, `on_error="record"` (défaut). La
fonction renvoie `{"rho_hat": ..., "diag": <objet non picklable : verrou,
figure matplotlib ouverte, bf niché dans un dict>}`. Le détection amicale
`_looks_like_bf` (L118–135) ne voit qu'un `bf` **retourné directement**, pas
nichés. Le `dumps` L306 échoue après 200 min de calcul → exception propagée
L912 → **0 ligne de DataFrame**. Même sort si un `fn_kwargs` picklable côté
parent mais dont le `__setstate__` échoue côté worker (L292) : la première
tâche de chaque worker lève, run entière perdue. Le contrat « record » est
cassé pour tout ce qui n'est pas le corps de `fn`.

### MAJEURES

**M1. Incohérence `os.cpu_count()` vs `sched_getaffinity` : pas d'avertissement de sur-souscription et blocs d'affinité qui se chevauchent sous `taskset`/cgroup** [RÉSOLU 42455665]
`simulate.py:402, 407-415` vs `simulate.py:417-419, 858-860, 253-254`
Le plafond de workers (L402, L407) est calculé sur `os.cpu_count()` — le nombre
**physique** — alors que le `block_size` (L419, L858) et le pinning (L249)
utilisent `sched_getaffinity`, qui respecte `taskset`/cgroups. Le module se
vante d'ailleurs en commentaire (L248) de respecter ces limites.
*Scénario* : hôte 24 cœurs, `taskset -c 0-7 python study.py`, 30 scénarios,
`cores_per_worker=1`. `_plan` : `n_cpu=24`, `workers=min(30,24)=24`, contrôle
L407 `24*1 ≤ 24` → **aucun avertissement, aucun plafond**. `_parallel` :
`n_cpu=8`, `block_size=max(1, 8//24)=1`, et L253 `start=(block*1)%8` → les 24
workers se répartissent sur 8 CPU (3 par CPU), 16 threads oisifs par ailleurs ;
le facteur de sur-souscription ×3 passe silencieusement, exactement le cas que
l'avertissement L409–414 existe pour attraper.

**M2. `workers == 1` bascule silencieusement en in-process : plus d'isolation d'environnement, `quiet` ignoré, et le mode « chaînes parallèles » dégénère** [RÉSOLU 42455665]
`simulate.py:627` (+ `_serial` L815–826, `_env_override` L195–212 utilisé uniquement dans `_parallel` L881)
`if backend == "serial" or workers == 1:` exécute `fn` dans le processus
parent : `_env_override` n'est jamais appliqué (pas de `XLA_FLAGS`,
`BF_MAX_CORES`, `OMP_NUM_THREADS` dédiés), le backend JAX du parent (device
count figé à l'import, cf. docstring L11–14) s'applique, et `quiet=True` ne
supprime ni le banner BayesForge ni les prints de `fn` (le devnull L273 est
réservé aux workers).
*Scénario* : portable 8 cœurs, `run_simulations(fn, scenarios, num_chains=4)`
→ plafond L407–415 ramène `workers` à 2, puis un jeu de 8 scénarios… non :
pire, sur 2 cœurs, `capped = max(1, 2//4) = 1` → chemin serial → `fn` reçoit
`num_chains=4` (injecté L398–399) et lance `chain_method='parallel'` sur le
device count du parent (1) → chaînes **séquentielles**, exactement le failure
mode documenté L580–585, sans que l'avertissement L391–397 ne se déclenche
(ici `cores_per_worker == num_chains`). Le backend demandé (`'process'`) est
silencieusement remplacé.

**M3. Aucune validation d'unicité de `sim` : graines dupliquées et répliques « indépendantes » qui tirent les mêmes données** [RÉSOLU 42455665]
`simulate.py:98-100, 378, 104-111, 637`
La graine est dérivée de `(base, sim)` (L378, L110). `_normalise_scenarios` ne
vérifie pas que les `sim` sont distincts.
*Scénario* : `scenarios = grid(N=[50,100], reps=5) + grid(N=[200], reps=5)` —
composition naturelle de deux grilles, chacune numérotée depuis 0 (L81). Les 5
premières simulations de chaque moitié partagent `sim` 0–4 → graines identiques
→ le DGP piloté par `seed` produit **les mêmes données synthétiques** pour des
cellules censées être indépendantes ; les colonnes `seed`/`sim` du DataFrame
sont identiques, donc rien ne le signale. Tri L637 par `sim` stable → ordre
d'entrelacement des deux grilles = ordre d'achèvement → **non déterministe**
d'un run à l'autre.

**M4. `BrokenProcessPool` (OOM, worker tué) : perte de 100 % des résultats, sans reprise partielle** [RÉSOLU 42455665]
`simulate.py:929-934` (+ L912)
Un worker tué par l'OS casse tout le pool (sémantique ≥ 3.9) ; le message est
excellent, mais `rows` — potentiellement des heures de fits terminés avec
`on_error="record"` — n'est ni retourné ni écrit. Un seul scénario gourmand
(`N=100000`) dans une étude de 500 ruine les 499 autres. Aucun mécanisme de
checkpointing/streaming des lignes au fil de l'eau.

**M5. `KeyboardInterrupt` / arrêt : aucun chemin de terminaison, sortie du `with` bloquante, lignes perdues** [RÉSOLU 42455665]
`simulate.py:892-897, 904-928`
Le pool n'est fermé que par le `with` ; en cas d'interruption (Ctrl+C pendant
`wait` L907, stop d'IDE, `kill -INT <parent>`), `__exit__` rappelle
`shutdown(wait=True)` et **join les workers en cours**. Un worker occupé à une
compilation XLA ou un échantillonnage long (signaux différés en C++) continue ;
le parent semble pendu, sans message ni option `terminate()`. Les mêmes
trajets `raise` (L913–921) et exception inattendue (L912) jettent `rows`
collecté. Le chemin `on_error="raise"` lui-même attend les fits en vol avant
d'afficher l'erreur (commentaire L914–915 assumé, mais aucun délai maximal).

**M6. Script sans garde `__main__` + `progress=True` (défaut) : le Manager grille l'avertissement amical et produit un échec obscur** [RÉSOLU 42455665]
`simulate.py:887-889` vs `simulate.py:38-46, 838-842, 929-934`
Le garde anti-récursion (L838) et le message `_GUARD_MSG` (L38–46) ne
s'exécutent que dans les **workers du pool**, remontés via
`BrokenProcessPool` (L929–934). Mais avec `progress=True` (défaut), le
`ctx.Manager()` (L888) est créé **avant** le pool : son processus serveur,
spawné, ré-importe le script de l'utilisateur ; sans garde, l'appel de niveau
module relance `run_simulations` → L838 lève dans le manager → le serveur meurt
pendant son démarrage → le parent attend et finit par lever une erreur de
démarrage de manager (timeout/pipe), **sans jamais voir `_GUARD_MSG`**.
L'erreur la plus fréquente des débutants (oubli du garde, documentée L588–590)
donne un diagnostic opaque par défaut, et propre seulement avec
`progress=False`.

**M7. GPU : `CUDA_VISIBLE_DEVICES` n'est pas partitionné ; tous les workers initialisent le(s) même(s) GPU** [RÉSOLU 42455665]
`simulate.py:184-192` (rien sur les GPU dans `_worker_env`)
Le budget est pensé CPU (`--xla_force_host_platform_device_count`). Sur une
machine à GPU, chaque worker spawné initialise CUDA et JAX préalloue ~75 % de
la mémoire par processus.
*Scénario* : 1 GPU, `run_simulations(fn, grid(...), workers=8)` → le premier
worker prend le GPU, les suivants échouent à l'init → workers tués →
`BrokenProcessPool` → message L929–934 « killed by the OS (out of memory) —
lower workers » qui n'est qu'à moitié vrai : même `workers=2` est mort, et
`workers=1` retombe en serial (M2). Aucun `CUDA_VISIBLE_DEVICES=""` ni
répartition par device n'est proposé.

### MINEURES

**m1. Chevauchement de blocs d'affinité si les workers ne prennent pas leur première tâche en ordre de soumission** [RÉSOLU b20468a9]
`simulate.py:224-229, 898, 241`
Le bloc est `i % workers` (L898) et le pinning est figé à la première tâche
(L241). L'hypothèse « workers pick up their first tasks in submission order »
n'est pas une garantie de `ProcessPoolExecutor` : un worker au démarrage lent
(peu d'imports) peut laisser un worker rapide enchaîner deux tâches de même
classe `i % workers` avant son premier pull → deux workers sur le même bloc,
un bloc orphelin. Performance seulement, mais contredit « disjoint block »
(L544).

**m2. `_env_override` mute l'environnement du parent pendant tout le run ; non thread-safe ; écrase le `XLA_FLAGS` existant de l'utilisateur dans les workers** [RÉSOLU b20468a9]
`simulate.py:195-212, 186, 203-204`
Deux `run_simulations` concurrents dans deux threads se marchent sur le
save/restore. `XLA_FLAGS` (L186) remplace sans fusion toute valeur héritée (ex.
`--xla_cpu_multi_thread_eigen=false` posé par l'utilisateur) ; le parent garde
les variables pendant la durée du run (fenêtre visible par tout sous-processus
lancé en parallèle).

**m3. `sys.stdout = open(os.devnull, "w")` non restauré, jamais fermé, sans échappatoire** [RÉSOLU b20468a9]
`simulate.py:273`
Documenté (« Suppress worker stdout » L487), mais aucun `stderr` distinct,
aucun moyen pour `fn` d'écrire un log utile depuis un worker ; le fichier
devnull n'est pas refermé (mineur, le worker meurt avec le pool).

**m4. `_normalise_scenarios` : conversion avant validation → le `TypeError` promis ne se déclenche jamais pour un non-mapping** [RÉSOLU b20468a9]
`simulate.py:94-96`
`dict(s)` lève avant le `isinstance` : `run_simulations(fn, ["abc"])` →
`ValueError`/`TypeError` confus de `dict()`, jamais le message clair L96.

**m5. `grid` n'accepte que `list`/`tuple` : ndarray, `range`, `set` traités comme scalaires** [RÉSOLU b20468a9]
`simulate.py:75`
`grid(N=range(10, 51, 10))` produit **une** scénario dont la valeur `N` est
l'objet `range` (picklé dans chaque payload L848), au lieu de 5 scénarios.
`np.array([50, 100])` idem : un seul scénario avec un array comme paramètre.
Pas d'erreur, un résultat silencieusement faux.

**m6. Les clés retournées par `fn` écrasent les méta-colonnes `sim`, `seed`, `error`, `worker`** [RÉSOLU b20468a9]
`simulate.py:343-349, 642, 637`
`merged.update(row)` donne le dernier mot à `fn` (choix assumé L343–344). Un
`fn` qui renvoie `{"error": "convergence divergentes"}` (chaîne, pas une
exception) crée une ligne « en erreur » pour le comptage L642 alors que la
notification `done` (L304) n'a rien compté — compteur du panneau ≠ compteur du
DataFrame. Un `fn` qui renvoie son propre `sim` casse le tri L637 et
l'association graine↔ligne.

**m7. Validations API tardives ou silencieuses** [RÉSOLU b20468a9]
`simulate.py:405, 401, 110`
`workers=0` ou `workers=-3` → `max(1, …)` = 1, sans avertissement (et donc
bascule serial, M2). `cores_per_worker=2.7` → tronqué à 2 (L401). `seed=None`
→ `int(None)` (L110) lève un `TypeError` obscur en plein `_plan`. `reps=2.5` →
`range(2.5)` lève loin de `grid`.

**m8. Logique de budget dupliquée `_plan`/`_parallel`, avec des sources CPU différentes ; `plan["env"]` et `plan["block_size"]` inutilisés côté Python** [RÉSOLU b20468a9]
`simulate.py:417-421` vs `simulate.py:858-860, 850`
`_plan` (partagé avec le pont R, L361–365) calcule `env` et `block_size` que
`run_simulations` ignore : `_parallel` recalcule les deux. Toute divergence
future entre les deux calculs (ex. correctif M1 appliqué à un seul endroit)
fêlera silencieusement la parité Python/R.

**m9. Exception inattendue (hors `BrokenProcessPool`) : `display.close()` non appelé → frame ANSI orpheline dans le terminal** [RÉSOLU b20468a9]
`simulate.py:910-912, 928`
Sur la trajectoire L912 (C1), le `close()` L928 est sauté ; le panneau reste
incomplet sur le terminal. Le `finally` L935–937 ne s'occupe que du manager.

**m10. La progression (cosmétique) est sur le chemin critique : un Manager qui n'arrive pas à lier une socket plante tout le run, sans repli** [RÉSOLU b20468a9]
`simulate.py:887-889`
Aucun `try`/repli `progress=False` ; dans un environnement sans localhost
(sandbox stricte), une étude de 10 h refuse de démarrer pour une barre de
progression.

**m11. Ordre des colonnes du DataFrame dépend de l'ordre d'achèvement** [RÉSOLU b20468a9]
`simulate.py:635-637`
`pd.DataFrame(rows)` est construit **avant** le tri (L635 vs L637) : avec des
schémas hétérogènes entre scénarios, l'ordre d'insertion des colonnes suit
l'ordre d'achèvement → non déterministe entre deux runs (cosmétique, mais
casse les diffs).

**m12. `_normalise_result` : colonnes dupliquées ou MultiIndex d'un DataFrame silencieusement écrasées** [RÉSOLU b20468a9]
`simulate.py:144-145`
`to_dict("records")` fusionne les colonnes de même nom (dernier gagne) et
aplatit/abîme un MultiIndex → perte de données sans avertissement.

---

## Points sains (pour l'équité de révue)

- Dérivation des graines sha256 `(base, sim)` (L104–111) : correcte entre
  processus (pas de sel `hash()`), indépendante de l'ordre d'achèvement et du
  nombre de workers (promesse L482–484 tenue).
- Désérialisation différée de `fn`/`fn_kwargs` **après** le pinning
  (L275–279, L289–292) : l'ordre backend-XLA/affinité est le bon, et le
  commentaire explique pourquoi.
- `finally` sur `_env_override` (L205–212) et sur `manager.shutdown()`
  (L935–937), fermé **après** la sortie du pool — l'ordre de teardown est
  correct ; `_notify` best-effort (L309–317) ne peut pas casser une sim.
- Détection précoce d'un `bf` retourné, avec message pédagogique (L128–135) ;
  `cloudpickle.dumps(fn)` côté parent (L846) échoue tôt pour une closure
  non transportable.
- Tri stable par `sim` (L637) : assemblage déterministe entre runs (sous
  réserve de M3/m6) ; multi-lignes par sim conservées dans l'ordre d'`fn`.
- Pas de mémoire partagée : chaque worker reconstitue ses arrays par
  dé-pickle (L292) → pas de mutation croisée d'arrays utilisateur.
- `payloads` pré-sérialisés côté parent (L848), `fn_kwargs` sérialisé une fois
  par worker et non par tâche (L278–279, 289–292) : le bon choix.

---

## (a) Résumé du module

`Parallel/simulate.py` est un distributeur de simulations indépendantes :
`grid` construit un produit cartésien de scénarios (L53–84), `_plan` fixe
graines, workers, budget de cœurs et environnement JAX (L357–430), `_parallel`
déroule les tâches dans un `ProcessPoolExecutor` spawn (L829–938) avec
pinning CPU, environnement XLA par worker, panneau de progression live via
Manager-queue, et un chemin `serial` qui partage `_run_one` avec les workers.
La conception d'ensemble est soignée et documentée avec des mesures ; la
reproductibilité des graines est correcte. Le point faible structurel est le
**positionnement de la frontière d'erreur** : tout ce qui touche à la
sérialisation dans le worker (L289–292, L306) et la mort d'un worker (L929)
court-circuite `on_error="record"` et détruit des heures de calculs. S'y
ajoutent deux trous de cohérence système (cpu_count vs affinity, M1 ; GPU
ignoré, M7) et deux pièges d'ergonomie par défaut (serial silencieux en
`workers=1`, M2 ; manager qui enterre le message du garde `__main__`, M6).
Le module est utilisable en l'état pour le cas nominal (script gardé, CPU,
`workers>1`, retours plats picklables), mais **pas encore release-quality**
pour un outil dont la promesse est « des heures de MCMC sans surveillance ».

## (b) Fragilités architecturales

1. **Frontière d'erreur trop étroite** : le contrat `on_error="record"` est
   implémenté dans `_run_one` (L331–337) alors que les modes de défaillance
   réels (sérialisation du retour, dé-sérialisation des kwargs, BaseException)
   vivent dans `_worker_entry`/le transport. La capture devrait envelopper
   toute la tâche, avec repli « ligne d'erreur » sérialisable par construction.
2. **Aucune persistance incrémentale** : les lignes vivent dans une liste en
   mémoire du parent (L879, 922) ; toute interruption (raise, Ctrl+C, OOM,
   pool cassé) est un saccage total. Un callback/append JSONL au fil de l'eau
   supprimerait la moitié des sévérités ci-dessus.
3. **Budget calculé deux fois, avec deux sources de vérité CPU** (`_plan`
   L402/L417 vs `_parallel` L858) et un pont R qui consomme `_plan` : risque
   de dérive Python/R à chaque correctif partiel (m8, M1).
4. **Pinning fondé sur un comportement non garanti de l'ordonnanceur**
   (`i % workers` + « first task wins », m1) : fragile aux changements de
   CPython ; un compteur partagé sûr (ou `initializer` avec index attribué par
   le parent) serait plus robuste.
5. **Pas de recyclage de workers ni de timeout de tâche** : pas
   d'équivalent `max_tasks_per_child` (cache JIT et RSS croissants sur une
   étude longue), aucun délai maximal par simulation, aucune annulation des
   tâches en vol (M5).
6. **Progression cosmétique sur le chemin critique** (manager + socket,
   M6/m10) : la dégradation gracieuse existe pour le *rendu* (TTY vs log) mais
   pas pour l'*infrastructure*.
7. **Périmètre CPU-only assumé mais non gardé** : rien n'avertit ni
   n'isole sur machine GPU (M7) alors que BayesForge supporte les GPU.

## (c) Décompte par sévérité

| Sévérité | Nombre |
|---|---|
| Critique | 1 (C1) |
| Majeure | 7 (M1–M7) |
| Mineure | 12 (m1–m12) |
| **Total** | **20** |

---

### Module NBDA

**Audit de release — `BayesForge/NBDA/`** (racine `/home/sosa/work/BF`)
Périmètre lu intégralement : `BayesForge/NBDA/NBDA.py` (735 lignes), `BayesForge/NBDA/__init__.py` (vide, 0 ligne).
Dépendances vérifiées pour le croisement : `BayesForge/Utils/link.py` (l. 50 `inv_logit`, l. 54–69 `inv_logit_scale`), `BayesForge/Main/main.py` (l. 214, 235, 543–548, 557), `BayesForge/Models/models.py` (l. 24, 31), `BayesForge/Distributions/np_dists.py` (l. 5655 `normal`, l. 6837 `uniform`, branche `sample=False` l. 5724–5741), `Test/5.NBDA.ipynb` (usage réel).

---

## 1. Bugs critiques

### C1 — Appels « non liés » de `UnifiedDist` : toutes les priors sont silencieusement fausses `[RÉSOLU 2e8f60cc]`
- **Sévérité** : Critique
- **Emplacement** : `BayesForge/NBDA/NBDA.py:5` (import) + `NBDA.py:492, 495, 498, 503, 506, 510, 515` et `NBDA.py:526, 529, 532, 537, 540, 544, 549`
- **Risque** : la ligne 5 importe `from BayesForge.Distributions.np_dists import UnifiedDist as dist`, c'est-à-dire **la classe**, pas une instance. Or `normal` (np_dists.py:5655) et `uniform` (np_dists.py:6837) sont des **méthodes d'instance** `def normal(self, loc=0.0, scale=1.0, ...)`. L'appel positionnel `dist.normal(0, 2, shape=(1,), sample=False, name=...)` se lie donc `self=0, loc=2, scale=1.0` (défaut). Rien ne plante, car dans la branche `sample=False` (np_dists.py:5724–5741) le corps n'utilise jamais `self` avant `numpyro.sample` : le site est créé avec la **mauvaise distribution**, silencieusement.
- **Scénario concret** : `nbda = NBDA(net, status); nbda.get_cov(); m.run(nbda.model())` →
  - `dist.normal(0, 2, ...)` → prior réelle **Normal(loc=2, scale=1)** au lieu de Normal(0, 2) : tous les coefficients (`social_learning`, `asocial_learning`, tous les `Beta_*`) sont priorisés centrés sur **+2** ;
  - `dist.uniform(0.0001, 0.9999, name='ces_alpha')` (l. 492/526) → `self=0.0001, low=0.9999, high=1.0` (défaut) → **Uniform(0.9999, 1.0)** : `ces_alpha ≈ 1`, donc `ces()` (l. 485) donne `Q ≈ K` — le modèle CES dégénère en **social-only**, l'apprentissage asocial est éliminé du mélange.
  Toute inférence produite par le module est statistiquement invalide, sans le moindre message d'erreur.

### C2 — `covNF_dims` : axes i/j inversés → le covariate du « learner » est celui du démonstrateur, et l'effet asocial devient une constante `[NON REPRODUIT, test de non-régression 2e8f60cc]`
- **Sévérité** : Critique
- **Emplacement** : `BayesForge/NBDA/NBDA.py:212–218` (transpositions), consommé par `NBDA.py:438–439` et `NBDA.py:602/608`
- **Risque** : ligne 212, `arrays[v,i,j,t] = df[i,v]` (la valeur varie le long de l'axe 1 = i). Ligne 215, `result_array_j = transpose(arrays,(1,2,3,0))` → `covNF_j[a,b,t,v] = df[a,v]` : la valeur du **learner** dans le tableau censé porter le démonstrateur. Ligne 216, `result_array_i = transpose(arrays,(2,1,3,0))` → `covNF_i[a,b,t,v] = df[b,v]` : la valeur varie le long de l'axe 1 = **démonstrateur**. Dans le modèle, l'axe 0 est l'individu qui apprend (`attention[i,j]`, l. 604–605) : le terme « covariate du learner » en diffusion sociale utilise donc les valeurs du **démonstrateur**. Pire, le tronc asocial `objects[k][:,0,:,i]` (l. 439) fige l'axe 1 à 0 → `df[0,v]`, **scalaire constant** (la valeur du nœud 0) répété sur tous les individus : le covariate asocial n'a plus aucune variance inter-individus.
- **Scénario concret** : `nbda.import_covNF(pd.DataFrame({'age': ages}))` puis `get_cov()` → le coefficient `Beta_social_...age` mesure l'effet de l'âge du **démonstrateur** (déjà capturé ailleurs), et `Beta_asocial_...age` régresse sur une colonne constante → coefficient non identifiable, posterior ≈ prior (Normal(2,1) à cause de C1), conclusions inverses sur l'effet de l'âge.

### C3 — `convert_covNV` : transpositions fausses → le chemin covNV time-varying est inutilisable (crash) et sémantiquement perdu `[RÉSOLU 2e8f60cc]`
- **Sévérité** : Critique
- **Emplacement** : `BayesForge/NBDA/NBDA.py:294–296`
- **Risque** : pour `covV` de forme `(num_var, n, t)` (docstring l. 289, et c'est ce que passe le notebook : `reshape((-1,N,times))`), `arrays` a la forme `(t, n, n, num_var)` (l. 294). Les transpositions `(2,3,1,0)` et `(3,2,1,0)` (l. 295–296) produisent `(n, num_var, n, t)` et `(num_var, n, n, t)` — **pas** le `(n, n, t, num_var)` promis. Les valeurs dépendent alors des mauvais axes (la distinction i/j est perdue : les deux tableaux portent la valeur indexée sur le même axe).
- **Scénario concret** : `nbda.import_covNV(covV)` puis `nbda.get_cov()` → dans la boucle l. 436, `objects['covNV_i'].shape[3]` vaut **t** (le nombre de pas de temps) au lieu du nombre de variables → `self.names['covNV'][i]` (l. 444) lève `IndexError` dès `i ≥ len(names)` ; si le nombre de noms coïncide, `jnp.concatenate` (l. 477) reçoit des tranches 3D `(1,n,n)` à côté de l'intercept 4D `(n,n,t,1)` → `ValueError`. Le covariate time-varying est soit inutilisable, soit silencieusement faux.

### C4 — `inv_logit_scale ∈ (−1,1)` utilisé comme probabilité → poids d'influence négatif → `lk < 0` → densité NaN `[RÉSOLU 2e8f60cc]`
- **Sévérité** : Critique
- **Emplacement** : `BayesForge/NBDA/NBDA.py:605, 632, 658` (consommation) ; cause racine `BayesForge/Utils/link.py:69` (doc l. 56 qui promet « scaling between 0 and 1 » alors que `(sigmoid(x)−0.5)*2 ∈ (−1,1)`)
- **Risque** : `social_influence_weight = link.inv_logit_scale(Σ_j φ·A_ij·z_j)` est sensé être une probabilité d'adoption par influence (intervalle [0,1] d'après la doc du lien, supplément aax9070). En réalité il peut être **négatif**. La mise à jour l. 612/639/665, `lk = θ + (1−θ)·w`, devient négative dès que `w < −θ/(1−θ)`. Le `mask = ~jnp.isnan(lk)` (l. 587) ne détecte **pas** les probabilités négatives : `Binomial(probs=lk)` produit un `log_prob` NaN sur les entrées **non masquées** → énergie NaN, NUTS échoue (`Initial energy is NaN`) ou, pire selon le sampler, échantillonnage corrompu.
- **Scénario concret** : réseau signé (valence, `A_ij = −1` pour les dyades antagonistes, usage NBDA classique) avec `status[:,t-1] ≠ 0` → somme attentionnelle négative → `w ≈ −0.9` ; avec `θ = inv_logit(R_asocial) ≈ 0.1` → `lk ≈ 0.1 + 0.9·(−0.9) = −0.71` → NaN. Même sans réseau signé, un `R_social` très négatif (cf. C1/C5) donne `φ ≈ 0` mais la somme peut rester négative si `A` contient des poids négatifs.

### C5 — Gradients NaN de `inv_logit`/`inv_logit_scale` aux extrêmes dans le cœur du modèle `[RÉSOLU 2e8f60cc]`
- **Sévérité** : Critique
- **Emplacement** : `BayesForge/NBDA/NBDA.py:597, 603, 609, 623, 630, 635, 650, 656, 662` (via `link.inv_logit`) et `605, 632, 658` (via `link.inv_logit_scale`) ; cause `BayesForge/Utils/link.py:50` et `link.py:69`
- **Risque** : `1/(1+jnp.exp(−x))` (link.py:50) a un forward stable mais un gradient autodiff `exp(−x)/(1+exp(−x))²` qui vaut `inf/inf = NaN` dès `|x| ≳ 88` en float32 (le sigmoid factorisé `s·(1−s)` donnerait 0). Dans NBDA, deux voies grossissent `x` : (i) `R_social/R_asocial = Σ β·cov` avec les β priorisés **Normal(2,1)** (C1) et des covariates non standardisés (`scale=False`, l. 248/300) ; (ii) la somme `Σ_j φ·A_ij·z_j` (l. 605/632/658) sur un réseau dense : n = 1000, φ ≈ 1, A = 1 → argument ≈ 10³, très au-delà du seuil.
- **Scénario concret** : réseau dense 500 nœuds, une covariate non scalée d'échelle 100 → `R_social ≈ 200` → gradient NaN → `Cannot find valid initial parameters` au warmup, ou divergences massives en cours de chaîne, sans diagnostic pointant le lien.

### C6 — `print_model` plante systématiquement (attribut inexistant + `display` non importé) `[RÉSOLU 2e8f60cc]`
- **Sévérité** : Critique
- **Emplacement** : `BayesForge/NBDA/NBDA.py:668–730` (l. 680, 688/690/693, 698/700/703, 730)
- **Risque** : la condition l. 680 `if len(self.names) > 2` est **toujours vraie** (le dict `names` est initialisé avec 8 clés, l. 43–52) ; la boucle rencontre donc toujours `k == 'covNF'` et lit `self.covNF_location` (l. 688) — attribut **jamais assigné nulle part** dans le codebase (grep : uniquement lu en 688–705 ; les imports stockent `self.locations['covNF_i']`, l. 281). → `AttributeError` garanti. Et même en passant, l. 730 `display(Markdown(r))` : `display` n'est jamais importé (seul `Markdown`, l. 9–12) → `NameError`.
- **Scénario concret** : `nbda.print_model()` sur n'importe quel objet NBDA correctement construit → `AttributeError: 'NBDA' object has no attribute 'covNF_location'`. La seule fonction de documentation du modèle est morte.

### C7 — Contrat avec `Main` rompu : NBDA est inexécutable via `bf.run` (branche morte + attribut fantôme) `[RÉSOLU 2e8f60cc]`
- **Sévérité** : Critique
- **Emplacement** : `BayesForge/NBDA/NBDA.py:53` (côté module) ; `BayesForge/Main/main.py:214, 543–548, 557` et `BayesForge/Models/models.py:24, 31` (côté Main)
- **Risque** : triple rupture :
  1. `NBDA.py:53` pose `self.nbdaModel = True` **sur l'instance NBDA**, alors que `main.py:543/557` teste `self.nbdaModel` **sur bf** — jamais posé à True sur bf (`main.py:214` le fige à `False`). La branche NBDA de `run()` est donc **morte**.
  2. Si elle était atteinte, `main.py:547` fait `self.model = self.nbda.model` : `self.nbda` **n'existe jamais sur bf** (grep : seul `models.py:24` pose `self.nbda = NBDA.model` sur l'objet helper `models`, et c'est une fonction non liée) → `AttributeError`.
  3. `models.py:24/31` : `bf.models.nbda` est la **fonction non liée** `NBDA.model` ; l'appeler sans `self` → `TypeError`. Et `main.py:557` (`data_on_model = pd_to_jax(self.model)`) ne traite jamais le cas NBDA (pas de DataFrame).
- **Scénario concret** : `m = bf(); m.run(model=None)` avec `nbdaModel` forcé à True → `AttributeError: 'bf' object has no attribute 'nbda'` ; avec la valeur par défaut → `print("Argument model can't be None")` (l. 544) puis `run()` poursuit sans `self.model` → `AttributeError` plus loin. Le module n'a **aucun chemin d'intégration vivant** dans le framework ; le notebook `Test/5.NBDA.ipynb` ne passe que par l'objet NBDA autonome.

---

## 2. Bugs majeurs

### M1 — `import_covNF(..., where='asocial')` plante sur objet neuf (`covNF_j` est `None`) [RÉSOLU 0331f099]
- **Sévérité** : Majeure
- **Emplacement** : `BayesForge/NBDA/NBDA.py:280` (et `NBDA.py:329` pour covNV)
- **Risque** : l. 261–262, `where='asocial'` force `get_cov_j=False`, donc `self.covNF_j` reste `None` (l. 269–273) ; mais l. 280, `self.covNF_get_j.extend([get_cov_j]*self.covNF_j.shape[3])` déréférence `None.shape` **inconditionnellement**.
- **Scénario concret** : `nbda = NBDA(net, status); nbda.import_covNF(df, where='asocial')` → `AttributeError: 'NoneType' object has no attribute 'shape'`. Le mode « covariate asocial-only » est cassé au premier import.

### M2 — `extend(str(...))` sur les noms auto-générés : un string est itéré caractère par caractère [RÉSOLU 0331f099]
- **Sévérité** : Majeure
- **Emplacement** : `BayesForge/NBDA/NBDA.py:276, 325, 361, 389`
- **Risque** : `self.names['covNF'].extend(str(len(self.names['covNF'])))` ajoute les **caractères** du string. Tant que le compteur < 10, un seul caractère → effet de bord masqué ; au 10ᵉ covariate, `str(10) = "10"` ajoute **deux** noms `'1'` et `'0'` → décalage noms/colonnes. Les noms auto sont de toute façon `'0','1',…` : collisions avec les noms fournis sur d'autres imports, et noms illisibles dans les sites numpyro (`Beta_social_3`).
- **Scénario concret** : 12 imports `import_covNF` successifs sans `names` → `len(names['covNF']) = 13` pour 12 colonnes → `priors()` (l. 498) nomme les sites avec les mauvais noms, et `get_cov` l. 463–469 décale tout ; `IndexError` possible l. 441/472.

### M3 — `self.locations` : un scalaire partagé écrase la localisation de tous les covariates [RÉSOLU 0331f099]
- **Sévérité** : Majeure
- **Emplacement** : `BayesForge/NBDA/NBDA.py:281–282, 330–331` (consommé l. 437)
- **Risque** : `self.locations['covNF_i'] = where` stocke une **chaîne unique** pour toutes les colonnes covNF, écrasée à chaque import. `get_cov` (l. 437) applique la dernière valeur à **toutes** les colonnes.
- **Scénario concret** : `import_covNF(a, where='social'); import_covNF(b, where='asocial')` → `locations['covNF_i'] = 'asocial'` → la covariate `a`, déclarée sociale, est injectée **uniquement dans le canal asocial** (et `b` aussi) ; le design social perd une colonne, le design asocial en gagne une à tort.

### M4 — Paramètre `scale` ignoré dans `import_covDF` et `import_covDV` [RÉSOLU 0331f099]
- **Sévérité** : Majeure
- **Emplacement** : `BayesForge/NBDA/NBDA.py:333–352` (signature l. 333, scaling inconditionnel l. 344 et 350) ; `BayesForge/NBDA/NBDA.py:365–381` (3D scalé l. 377, **4D jamais scalé** l. 381, `scale` jamais consulté)
- **Risque** : `scale=False` est silencieusement ignoré pour covDF (toujours standardisé) ; pour covDV le comportement est l'inverse selon la dimension d'entrée (3D toujours scalé, 4D jamais) — deux contrats incompatibles avec le même mot-clé.
- **Scénario concret** : `import_covDF(kinship, scale=False)` sur une matrice 0/1 de parenté (cas réel du notebook `5.NBDA.ipynb`) → la matrice est standardisée (moyenne/variance mélangées sur la diagonale) → le coefficient dyadique est exprimé dans une unité que l'utilisateur n'a pas choisie, sans aucun avertissement.

### M5 — `scale_data` sans epsilon : colonne constante ou NaN → NaN silencieux jusqu'à la vraisemblance [RÉSOLU 0331f099]
- **Sévérité** : Majeure
- **Emplacement** : `BayesForge/NBDA/NBDA.py:132–135` (consommé l. 235, 244, 302, 344, 350, 377)
- **Risque** : `(x − mean)/std` sans garde-fou : colonne constante (catégorie unique, réseau vide, statut binaire constant) → `std = 0` → `0/0` → colonne **entièrement NaN** sans erreur. Un NaN unique dans `df` contamine mean/std → colonne entière NaN. En aval, `mask = ~jnp.isnan(lk)` (l. 587) **supprime silencieusement** les observations NaN de la vraisemblance au lieu de signaler l'erreur : l'utilisateur fit un sous-échantillon fantôme.
- **Scénario concret** : `import_covNF(pd.DataFrame({'sex': np.ones(n)}))` (groupe mono-sexué) → design NaN → lk NaN → mask faux partout → vraisemblance ≈ 0 contribution, chaînes « convergentes » sur les seules priors.

### M6 — `scale_along_time` : vmap sur le mauvais axe → standardisation qui mélange variables (ou nœuds) [RÉSOLU 0331f099]
- **Sévérité** : Majeure
- **Emplacement** : `BayesForge/NBDA/NBDA.py:137–139` (utilisé l. 302 pour covNV `(var,n,t)` et l. 377 pour covDV `(n,n,t)`)
- **Risque** : `vmap(scale_data, in_axes=1)` sur `(var,n,t)` scalé chaque tranche `(var,t)` en **mélangeant les variables** dans le mean/std communs ; sur covDV `(n,n,t)` (l. 377) il scalé chaque `(n,t)` en mélangeant les **nœuds**. L'intention (« scale along time », éviter la colinéarité avec l'intercept, cf. warning l. 304–305) exige de standardiser chaque série temporelle par (variable, nœud).
- **Scénario concret** : covNV avec une variable d'exposition d'amplitude 1000 et une d'âge d'amplitude 50 → la standardisation de l'âge est dictée par la variance de l'exposition → coefficients mal échoués, warning l. 304 mensonger (le scaling appliqué ne protège pas de ce qu'il promet).

### M7 — Mode multiprocess (`N_process`) : soit crash, soit processus 1..P−1 silencieusement ignorés [RÉSOLU b87fe728]
- **Sévérité** : Majeure
- **Emplacement** : `BayesForge/NBDA/NBDA.py:68–72, 77` (préparation), `NBDA.py:188–190` (crash), `NBDA.py:589` (ignorance), `NBDA.py:478` (crash concatenate)
- **Risque** : `status` 3D `(n,t,P)` active `multiprocess=True` (l. 69) et `intercept_asocial (n,t,P)` (l. 77), mais `convert_status` (l. 188–190) suppose `status` 2D : `jnp.concatenate([zeros((n,1)), status[:,:-1]], axis=1)` → mismatch de rangs → crash immédiat. Si l'utilisateur bricole un `(n,t,1)`, `model()` l. 589 ne prend `obs=status[:,:,0]` : **seul le processus 0 est fit**, les autres sont ignorés sans message. Et pour P>1, `jnp.concatenate(D_asocial)` (l. 478) échoue (intercept `(n,t,P)` vs colonnes `(n,t,1)`).
- **Scénario concret** : deux comportements simultanés `(n,t,2)` → `TypeError`/`ValueError` dans le constructeur ; ou fit monoculaire silencieux que l'utilisateur croit multivarié (`self.multiprocess == True` lui confirme le contraire de la réalité).

### M8 — Mode multi-network : réseaux 2..K silencieusement ignorés ; code multinetwork mort [RÉSOLU b87fe728]
- **Sévérité** : Majeure
- **Emplacement** : `BayesForge/NBDA/NBDA.py:94–97` (drapeau posé), `NBDA.py:604, 631, 657` (`network[:,:,t,0]`), `NBDA.py:575–585` (n'appelle jamais les variantes multinetwork), `NBDA.py:522–554` et `NBDA.py:616–641` (jamais appelées)
- **Risque** : `model()` ne route **que** vers `priors` + `compute_probs[_ces]` ; `priors_multinetwork` et `compute_probs_multinetwork` ne sont référencées nulle part (et sont incohérentes entre elles : `soc (N_networks,1)` vs `asoc (1,)`, l. 529/540). Avec un réseau 4D `(n,n,t,K)`, `network[:,:,t,0]` ne garde que le réseau 0.
- **Scénario concret** : `NBDA(net4d_with_K=2, status)` → `multi_network=True`, le second réseau (p.ex. réseau de tolérance vs réseau de préférence, cas d'usage NBDA standard) n'entre **nulle part** dans la vraisemblance ; l'utilisateur n'a aucun signal.

### M9 — Réseau 3D `(n,n,t)` : accepté au constructeur, fait planifier le modèle (`IndexError`) [RÉSOLU b87fe728]
- **Sévérité** : Majeure
- **Emplacement** : `BayesForge/NBDA/NBDA.py:91–93` (acceptation), consommé `NBDA.py:604, 631, 657` et `NBDA.py:477`
- **Risque** : la branche 3D stocke `self.network` en 4 indices sans ajouter l'axe réseau : `network[:,:,t,0]` (l. 604) indexe un tableau 3D avec 4 indices → `IndexError: too many indices for array` ; de plus `self.intercept = jnp.ones(network.shape)` (l. 107) est 3D et se heurte à `jnp.concatenate` avec les covariates 4D (l. 477).
- **Scénario concret** : `NBDA(net_nt, status)` avec `net_nt.shape == (n,n,t)` — cas explicitement documenté l. 91 — → crash dans `model()`, loin du point d'erreur réel.

### M10 — t = 0 traité comme un événement de diffusion : les adopteurs initiaux sont sur-pondérés dans la vraisemblance [RÉSOLU b87fe728]
- **Sévérité** : Majeure
- **Emplacement** : `BayesForge/NBDA/NBDA.py:594–598, 622–624, 648–651` (et `612/639/665` pour t ≥ 1)
- **Risque** : `lk[:,0] = θ` pour **tous** les individus, sans masquage (le masquage par `status[:,t-1]` ne démarre qu'à t = 1, l. 612). L'NBDA standard (Allen et al.) **conditionne** sur l'état initial : `status[:,0]` n'est pas un événement. Ici chaque adopteur initial ajoute un pseudo-événement Bernoulli(θ) à la vraisemblance, et chaque non-adopteur initial aussi — N pseudo-événements parasites qui biaisent `asocial_learning` (tiré vers la prévalence initiale).
- **Scénario concret** : 30 % d'adopteurs à t = 0 sur n = 100 → 100 observations binomiales supplémentaires centrées sur 0.3 → le taux asocial estimé mélange prévalence initiale et taux d'acquisition ; comparaisons entre groupes avec prévalences initiales différentes invalides.

### M11 — `get_cov` : noms ajoutés aux mauvaises listes (l. 459, 452) → décalage noms/colonnes [RÉSOLU 0331f099]
- **Sévérité** : Majeure
- **Emplacement** : `BayesForge/NBDA/NBDA.py:459` et `NBDA.py:452`
- **Risque** : dans la branche `where='asocial'` pour covNV, la colonne part dans `D_asocial` (l. 455) mais son nom est ajouté à **`D_social_names`** (l. 459) ; dans la branche `'social'`, le nom est ajouté sans préfixe `k` (l. 452, incohérent avec l. 444). Les listes `T_social_names`/`T_asocial_names` (l. 479–480) ne correspondent plus aux colonnes de `T_social`/`T_asocial`.
- **Scénario concret** : `import_covNV(covV, names=['exp'], where='asocial')` → `T_asocial` a 2 colonnes mais `T_asocial_names` 1 seul → `priors()` l. 510 `self.T_asocial_names[a+1]` → `IndexError`, ou nom de site numpyro attribué au mauvais coefficient.

### M12 — `where` non validé : une valeur invalide fait disparaître la covariate de la vraisemblance [RÉSOLU 0331f099]
- **Sévérité** : Majeure
- **Emplacement** : `BayesForge/NBDA/NBDA.py:437–459` (pas de `else`), aucune validation à l'import (`NBDA.py:248, 300`)
- **Risque** : `get_cov` teste `'both'/'social'/'asocial'` sans branche finale : une faute de frappe (`'Asocial'`, `'both '`) fait **silencieusement** sauter la colonne du design, alors que son nom reste dans `names` → décalage noms/colonnes (comme M11) et covariate fantôme.
- **Scénario concret** : `import_covNF(df, where='social ')` (espace) → le modèle est fit **sans** cette covariate ; l'utilisateur croit l'avoir contrôlée.

### M13 — Préconditions et validations absentes : erreurs obscures au lieu d'erreurs explicites [RÉSOLU 0331f099 + b87fe728]
- **Sévérité** : Majeure
- **Emplacement** : `BayesForge/NBDA/NBDA.py:489–490` (`get_cov` non appelé → `AttributeError: 'NoneType' object has no attribute 'shape'`), `NBDA.py:20–130` (aucune validation : réseau carré, `n` réseau = `n` status, `t` réseau = `t` status, statut binaire 0/1, pas de NaN dans status), `NBDA.py:258` (`df.shape[1]` sur un array 1-D `(n,)` → `IndexError`), `NBDA.py:612` (`status[:,t-1] == 1` : toute valeur ≠ 1, p.ex. 2 ou 0.999, est traitée comme « naïf »)
- **Risque** : chaque erreur de données se matérialise des dizaines d'appels plus tard, dans un traceback JAX illisible, ou silencieusement (statut non binaire).
- **Scénario concret** : `status` avec un 2 (état « mort ») → l'individu est traité comme perpétuellement naïf → il peut « ré-adopter » ; `network (m,m,...)` avec `m ≠ n` → erreur de broadcast dans `compute_probs`, pas dans le constructeur.

---

## 3. Fragilités mineures

### m1 — `is` avec un littéral [RÉSOLU a17c80e0]
- **Sévérité** : Mineure — `BayesForge/NBDA/NBDA.py:142` : `if names is  None:` → `SyntaxWarning` (≥ 3.8), comportement futur CPython : erreur. Fonctionne aujourd'hui par chance (None est un singleton).

### m2 — `give_name` mort et incohérent [RÉSOLU a17c80e0]
- **Sévérité** : Mineure — `BayesForge/NBDA/NBDA.py:141–154` : jamais appelé (commentés l. 81, 100, 323) ; mélange `append` (l. 145) et affectation écrasante (l. 147/149/154) du dict `names`.

### m3 — Message d'erreur faux sur la dimension du réseau [RÉSOLU a17c80e0]
- **Sévérité** : Mineure — `BayesForge/NBDA/NBDA.py:99` : « must be a 2 (n, n) or 4 dimensional » alors que 3D est accepté l. 91–93 (et cassé, cf. M9).

### m4 — Constructeur `network=None, status=None` : objet vide [RÉSOLU a17c80e0]
- **Sévérité** : Mineure — `BayesForge/NBDA/NBDA.py:39–40` : aucun attribut posé (pas de `self.names`, `self.n`) → tout appel ultérieur plante en `AttributeError` au lieu d'un message du type « NBDA requires network and status ».

### m5 — `self.status` écrasé en 3D par `convert_status` [RÉSOLU a17c80e0]
- **Sévérité** : Mineure — `BayesForge/NBDA/NBDA.py:74` puis `194` : l'attribut `status` passe de `(n,t)` (donnée utilisateur) à `(n,t,1)` interne ; le contrat public de l'attribut change sans que la doc (l. 27) le dise.

### m6 — `status_i`/`status_j` calculés, stockés, jamais utilisés [RÉSOLU a17c80e0]
- **Sévérité** : Mineure — `BayesForge/NBDA/NBDA.py:80, 188–195` : deux tableaux `(n,n,t,1)` (mémoire O(n²t)) que `compute_probs*` n'utilisent pas (elles utilisent `status[:,t-1]`). Commentaire l. 735 confirme que leur usage est un chantier inachevé.

### m7 — Densification mémoire O(num_var · n² · t) [RÉSOLU a17c80e0]
- **Sévérité** : Mineure — `BayesForge/NBDA/NBDA.py:212, 294, 381` : boucles Python construisant des copies `(n,n,t)` pleines par variable, ×2 (i et j). n = 1000, t = 50, 5 covariates ≈ 1 Go par tableau, sans exploitation de la parcimonie du réseau.

### m8 — `jax.jit(static_argnums=(0,))` sur `self` [RÉSOLU a17c80e0]
- **Sévérité** : Mineure — `BayesForge/NBDA/NBDA.py:483–486` : `ces` ne référence pas `self` ; le rendre statique lie le cache de compilation à l'identité de l'instance (recompilation par objet, rétention mémoire) pour aucun bénéfice.

### m9 — `print()` au lieu de `warnings.warn` [RÉSOLU a17c80e0]
- **Sévérité** : Mineure — `BayesForge/NBDA/NBDA.py:304–305` : avertissement non captable/filtrable ; coquilles (« coeffcients », « affet »).

### m10 — `print_model` : LaTeX incohérent et remplacements no-op [RÉSOLU a17c80e0]
- **Sévérité** : Mineure — `BayesForge/NBDA/NBDA.py:674–675` : affiche `Normal(0,4)` (variance) là où le code (s'il était correct, cf. C1) utilise scale = 2 ; `NBDA.py:722` : le motif `replace` (avec `\sum A_{ij}`) ne correspond pas à la chaîne du modèle (l. 673, `\sum_{j = 1}^{N}`) → remplacement silencieusement inopérant. (Écran de fumée sur C6, qui fait planter la fonction avant.)

### m11 — Test mort dans `get_cov` [RÉSOLU a17c80e0]
- **Sévérité** : Mineure — `BayesForge/NBDA/NBDA.py:434` : `if k is not None:` teste la **clé** de boucle (toujours une chaîne non-None) ; les valeurs None sont déjà filtrées l. 415.

### m12 — `import_covDV` 4D : doc et code en désaccord sur l'axe des variables [RÉSOLU a17c80e0]
- **Sévérité** : Mineure — `BayesForge/NBDA/NBDA.py:370` (doc : `(n,n,t,num)`) vs `NBDA.py:381` (code : itère `covDV[i,:,:]` sur `shape[0]`, donc `(num,n,n,t)`). Combiné à M4 (jamais scalé en 4D).

### m13 — « CES » est un mélange linéaire (élasticité constante = ∞/substituts parfaits), pas une CES [RÉSOLU a17c80e0]
- **Sévérité** : Mineure — `BayesForge/NBDA/NBDA.py:485` : `Q = αK + (1−α)L` ; la formulation CES du supplément cité (aax9070, cf. `link.py:57`) comporte un paramètre de courbure ρ. Le nom et la docstring (l. 561–562) surpromettent le modèle.

### m14 — Le NaN comme sentinelle de masquage avale les NaN réels [RÉSOLU a17c80e0]
- **Sévérité** : Mineure — `BayesForge/NBDA/NBDA.py:587, 612` : `mask = ~jnp.isnan(lk)` ne distingue pas « individu déjà informé » (voulu) de « θ ou w calculé NaN » (accident de données/numerique, cf. C5/M5) : les entrées NaN accidentelles sont **retirées de la vraisemblance en silence**.

### m15 — Contournement de l'abstraction backend et nom de site générique [RÉSOLU a17c80e0]
- **Sévérité** : Mineure — `BayesForge/NBDA/NBDA.py:7, 588–589` : `numpyro.handlers.mask` et `numpyro.distributions.Binomial` utilisés en direct (le reste de BF passe par `bf.dist`/le proxy backend) → le module ne peut pas fonctionner sous un autre backend que numpyro ; site d'observation nommé `"y"` (générique, collision possible dans les modèles composites).

---

## (a) Résumé du module

`NBDA.py` implémente une **Network-Based Diffusion Analysis** bayésienne : préparation de données (réseau 2D/3D/4D, statut `(n,t)`, quatre familles de covariates nodales/dyadiques fixes ou temporelles expandues en tenseurs `(n,n,t,k)`), spécification de priors (taux social/asocial + β par covariate, option « CES »), et un modèle `model()` (l. 556–589) où la probabilité d'adoption de i au temps t est `lk = θ + (1−θ)·w` avec `θ = inv_logit(λ_a·x_i)` (asocial) et `w = inv_logit_scale(Σ_j φ_ij A_ij z_j)` (sociale), les individus déjà informés étant exclus par un `numpyro.handlers.mask` piloté par des NaN. L'ingénierie des données est soignée sur le papier (i/j, time-varying, multi-réseaux, multi-processus) mais **presque aucun de ces modes n'arrive vivant au modèle** : covNV et réseau 3D crashent, multi-réseaux et multi-processus sont ignorés, les covariates NF sont montées i/j à l'envers, et les priors sont silencieusement décalées par l'appel non lié de la classe `UnifiedDist`. Le module n'a aucun chemin d'exécution via `bf.run` (branche morte `nbdaModel`/`self.nbda`) et son unique fonction de documentation plante. En l'état, seule la configuration minimale (réseau 2D, pas de covariate, mono-processus, fit numpyro autonome du notebook) produit un ajustement — avec des priors fausses (C1) et un t = 0 sur-pondéré (M10).

## (b) Fragilités architecturales

1. **Aucune frontière de contrat avec Main** : le drapeau `nbdaModel` vit sur le mauvais objet, `self.nbda` est un fantôme, `models.py` expose une fonction non liée. Le module est un notebook déguisé en classe, pas un modèle BF intégrable (C7).
2. **Import de classe au lieu d'instance/proxy** (`NBDA.py:5`) : le module bypasse `bf.dist`/`DistProxy` et le RNG seedé de la session ; d'autres modules de `Models/` qui copieraient ce motif hériteraient du même empoisonnement silencieux des priors (C1, m15).
3. **Représentation dense `(n,n,t,k)` de tout** : les covariates nodales sont dupliquées sur l'axe j, les statuts laggués stockés en n²·t ; aucune structure (broadcast, sparse) n'est exploitée — limites mémoire et coût de trace (m7, m6).
4. **Le NaN comme mécanisme de contrôle** (masquage des informés, l. 587/612) : le même canal transporte les accidents numériques ; aucune validation en amont ne distingue les deux (m14, M5, M13).
5. **Contrats implicites et non vérifiés** : ordre d'appel `import_* → get_cov → model`, cohérence des noms avec les colonnes, `where` comme chaîne libre, `scale` honoré ou non selon la méthode — tout est de la convention non testée ; `__init__.py` vide, zéro test unitaire dans `Test/` hors notebook (M2/M3/M11/M12/M13).
6. **Code mort structurel** : `priors_multinetwork`, `compute_probs_multinetwork`, `give_name`, `status_i/j`, mode multiprocess — quatre chantiers inachevés laissés apparents, qui font croire à des fonctionnalités existantes (M7, M8, m2, m6).

## (c) Décompte par sévérité

| Sévérité | Nombre | Repères |
|---|---|---|
| **Critique** | **7** | C1 priors fausses (unbound), C2 i/j covNF inversés, C3 covNV crash, C4 probs négatives (inv_logit_scale), C5 gradients NaN, C6 print_model mort, C7 contrat Main rompu |
| **Majeure** | **13** | M1 crash where='asocial', M2 extend(str), M3 locations écrasé, M4 scale ignoré, M5 std=0 → NaN, M6 vmap mauvais axe, M7 multiprocess, M8 multi_network ignoré, M9 réseau 3D crash, M10 pseudo-événements t=0, M11 noms décalés, M12 where non validé, M13 validations absentes |
| **Mineure** | **15** | m1–m15 |
| **Total** | **35** | |

**Verdict de release : NON PUBliable.** Le module produit des inférences statistiquement fausses sans erreur (C1, C2, C4, M10) dans sa configuration minimale, et plante dans la plupart des modes avancés (C3, M1, M7, M9) ; son intégration à `bf.run` est morte (C7).

---

### Module BNN

**Audit de release — `BayesForge/BNN/`** (racine `/home/sosa/work/BF`)
Fichiers lus intégralement : `activations.py` (249 l.), `bnn.py` (352 l.), `__init__.py` (1 l., vide).
Contexte vérifié : `Models/BNNC.py:52-61` (appels `m.bnn.layer_linear`), `Main/main.py:245` (`self.bnn = bnn()`) et `main.py:251-254` (`self.bnn.dist = _unified`), `Samplers/SBI/npe.py:37` (`_BNN = bnn()` au niveau module) et `npe.py:56-59`, `Distributions/np_dists.py:175-224` (`UnifiedDist`, `normalize_seed`), `Test/10.BNN/`.

---

## 1. Exactitude mathématique & stabilité numérique

### [CRIT-1] `bnn.py:289-296` — `__call__` référence des attributs et méthodes inexistants `[RÉSOLU 5a927884]`
`__call__` (l.289-296) appelle `self.layer(X, self.q_proj)` : **ni `self.layer` ni `self.q_proj`/`k_proj`/`v_proj` n'existent nulle part** dans la classe (les méthodes s'appellent `layer_linear`, les poids de l'attention sont `emb_q`, `emb_kv`, `q_k_v_proj`, `out`, définis uniquement dans `layer_attention`, l.153-165).
**Scénario** : `net = bnn(); net(X)` → `AttributeError: 'bnn' object has no attribute 'layer'`. Pire : après un appel à `layer_attention`, `self.q_proj` n'est toujours pas défini → crash identique. C'est du code public mort (aucun appelant trouvé dans `BayesForge/` ni `Test/`), qui contredit de plus `layer_attention:176` (échelle différente, cf. MAJ-3).
**Risque** : API publique cassée ; un utilisateur qui suit la doc de la classe (« forward pass through the Bayesian attention mechanism ») obtient un crash immédiat.

### [MAJ-1] `activations.py:212` — `softplus` non stable : overflow `exp` → `inf` puis gradients `NaN` [RÉSOLU 657b6a3a]
`softplus(x) = jnp.log(1.0 + jnp.exp(x))` : en float32, `exp(x)` déborde dès `x ≳ 88,7` → `inf` → `softplus = inf` au lieu de ≈ `x`. Le commentaire l.213 admet lui-même « *A more numerically stable version is: jax.nn.softplus(x)* » — et ne l'applique pas. Le gradient autodiff `exp(x)/(1+exp(x))` donne `inf/inf = NaN`.
**Scénario** : `bnn.layer_linear(X, W, activation="softplus")` avec une pré-activation à 100 (possible avec des priors `N(0,1)` non scalés, cf. MAJ-5) → sortie `inf`, log-densité `NaN`, HMC/NUTS meurt sur `NaN`. Or `softplus` est enregistré dans `_activation_map` (`bnn.py:34`) **et** utilisé comme activation de sortie recommandée pour les paramètres d'échelle (docstring l.200-204). Contraste : `layer_cholesky:99` et les couches de covariance (`bnn.py:207, 215, 243, 257, 278`) utilisent, eux, le `jax.nn.softplus` stable — l'incohérence est le bug.

### [MAJ-2] `activations.py:237-249` — activation `exp` exposée : overflow silencieux [RÉSOLU 657b6a3a]
`exp` est une activation enregistrée (`bnn.py:35`). `exp(x) = inf` dès `x > 88,7` (float32), sans clamp ni avertissement. Couche cachée avec `activation="exp"` et pré-activations ~ N(0, fan_in) → `inf` au premier forward. À réserver, documenter comme sortie-only, ou borner (`jnp.exp(jnp.clip(x, -30, 30))`).

### [MAJ-3] `bnn.py:176` — `layer_attention` : mauvaise échelle du scaled dot-product [RÉSOLU 657b6a3a]
```python
scores = jnp.matmul(Q, K.T) / jnp.sqrt(self.d_model ** 0.5)
```
`jnp.sqrt(d_model ** 0.5) = d_model**0.25` (≈ 2,38 pour `d_model=32`) au lieu du correct `sqrt(d_k) = sqrt(d_model)` (≈ 5,66). Les scores sont donc **2,38× trop grands** → softmax saturé (attention quasi one-hot), gradients quasi nuls. La bonne formule existe deux méthodes plus bas, `scaled_dot_product_attention:285` (`/ jnp.sqrt(d_k)`) — incohérence interne qui confirme l'intention.
**Scénario** : `bnn.layer_attention(b_kv=5, b_q=4)` avec embeddings ~ N(0,1) : scores d'ordre `sqrt(32)·1 ≈ 5,7` non divisés correctement → `attn` ≈ one-hot dès l'init, apprentissage de l'attention bloqué silencieusement.

### [MAJ-4] `bnn.py:224 vs 255` — `layer_compound_symmetry` : `sigma` au lieu de `sigma**2` [RÉSOLU 657b6a3a]
La docstring (l.224) documente `C = sigma^2 * [(1 - rho) I + rho 11^T]` ; le code calcule `comp_sym = sigma * ((1.0 - rho) * I + rho * ones)` (l.255) avec `sigma = softplus(log_sigma) + 1e-6`. L'échelle de la covariance rendue est la racine carrée de ce que la docstring promet. Un utilisateur qui dimensionne son prior sur `log_sigma` d'après la formule documentée obtient des blocs de covariance d'un ordre de grandeur faux (et incohérents avec `layer_toeplitz`, où `sigma` multiplie bien une structure de *covariance*, l.211).

### [MAJ-5] `bnn.py:311-314`, `BNNC.py:52-61`, `bnn.py:155-165` — priors de poids fixes `N(0,1)`, aucune échelle par fan-in (ni He/Glorot) [PARTIEL 657b6a3a — prior_scale='fan_in' opt-in, défaut N(0,1) inchangé]
Tous les poids sont tirés `N(0,1)` indépendamment du fan-in : `W1 ~ N(0,1)` de shape `(N, hidden_dim)` (l.311), `W2 ~ N(0,1)` `(hidden_dim, 2)` (l.314) ; dans BNNC, `w1 ~ N(0,1)` `(D_X, D_H1)` suivi de `tanh` ; dans `layer_attention`, `out ~ N(0,1)` de fan-in `d_model·b_q = 32·b_q` (l.165).
**Scénario** : BNNC avec `D_X = 20` : variance de la pré-activation ≈ 20 → écart-type ≈ 4,5 → `tanh` sature (|z|>2 sur ~65 % des neurones) → gradients exponentiellement petits, mixage HMC dégradé, funnel vers les régions saturées. Avec `out` de l'attention, fan-in 128 → pré-activations d'écart-type ≈ 11,3. Rien dans le module ne divise par `sqrt(fan_in)` ni n'expose d'échelle de prior par couche.

### [MAJ-6] `bnn.py:339-345` — `get_rho` : divisions par zéro → `NaN`/`Inf` [RÉSOLU 657b6a3a]
- l.339 : `cov_sample = ... / (N - 1)` avec `N = a_b.shape[0]` ; si le posterior a une seule ligne (ou `posterior` de shape `(draws, 2)` → après `mean(axis=0)`, `a_b` de shape `(2,)`, `N=2` mais `centered_data` 1-D → `dot` scalaire, `cov_sample` scalaire, `jnp.diagonal` échoue) → division par zéro ou comportement silencieux faux.
- l.342-345 : `sigma = sqrt(diag(cov_sample))` puis `rho = cov_sample / (sigma[:, None] * sigma[None, :])` **sans epsilon** : une colonne constante (p. ex. `a` passé comme scalaire dans `cov`, donc `a + delta[:,0]` constant si `delta[:,0]` ≈ 0, ou tout bloc dégénéré) → `sigma = 0` → `rho = 0/0 = NaN` propagé aux plots/rapports PPC.
**Scénario** : `m.bnn.get_rho(m2.posteriors['rf'])` avec un posterior où `a` est quasi non contraint (prior large) → `rho` contient `NaN` sans erreur.

### [MIN-1] `activations.py:141` — `elu` : branche non sélectionnée qui déborde [RÉSOLU 3b99a538]
`jnp.where(x >= 0, x, alpha*(jnp.exp(x)-1))` évalue `exp(x)` partout : pour `x = +1000` la branche fausse vaut `inf` (sélectionnée hors → valeur OK sous JAX, pas de NaN de gradient car `select` ne multiplie pas par 0), mais la forme stable canonique `alpha*jnp.expm1(jnp.minimum(x, 0))` évite l'`inf` maternel et gagne en précision près de 0. À aligner sur la politique « stable » appliquée ailleurs (`jax.nn.softplus`, `jax.nn.sigmoid`).

### [MIN-2] `bnn.py:348-352` — `make_pd_and_cholesky_jax` : jitter fixe, sans repli [RÉSOLU 3b99a538]
`jitter = 1e-6` (l.350) est insuffisant pour une matrice mal conditionnée issue d'un posterior corrélé ; `jnp.linalg.cholesky` lève alors `LinAlgError` sans retry (jitter croissant) ni repli sur une décomposition propre (eigh + plancher). La symétrisation l.349 est correcte ; le `A +=` l.351 est un rébind JAX fonctionnel, pas une mutation — OK.

### [MIN-3] `bnn.py:199-207, 232-243, 273-278` — commentaires « log space » faux : double transformation de positivité [RÉSOLU 3b99a538]
`log_sigma` est tiré de `log_normal(1, 1)` (déjà strictement positif, `bnn.py:200, 233`) puis repassé dans `softplus` (l.207, 243) ; `log_variances` vient de `normal(0,1)` (l.274) puis `softplus` (l.278). La paramétrisation est licite mais les commentaires (« *stored in log space* », l.199, 232, 273) sont faux, et le `log_` des noms est trompeur : personne ne peut reproduire l'échelle effective du prior sans lire le code.

---

## 2. Vectorisation, dimensions & mémoire

### [CRIT-2] `bnn.py:322-324` — `cov` : `jnp.stack([a, b])` de shape `(2, N)` additionné à `delta (N, 2)` → broadcasting faux `[RÉSOLU 5a927884]`
```python
delta = jnp.dot(hidden, W2)             # (N, 2)
return deterministic('rf', jnp.stack([a, b]) + delta)
```
Si `a` et `b` sont des **tableaux de longueur N** — ce que la docstring documente (« *The first set of offsets* », au pluriel, l.306-307) — `jnp.stack([a, b])` a shape `(2, N)` :
- `N ≠ 2` (ex. `m.bnn.cov(10, 20, jnp.zeros(20), jnp.ones(20))`) → `ValueError: operands could not be broadcast` (axes 20 vs 2 et 2 vs 20 incompatibles) ;
- `N = 2` → `(2,2) + (2,2)` s'additionne **sans erreur mais transposé** : l'offset `a` du café 1 est ajouté au delta du café 0 → résultats faux silencieusement.
Le seul test existant (`Test/10.BNN/code/BNN covariance.py:95`) passe `a = 3.5`, `b = -1` scalaires → `stack` donne `(2,)` qui diffuse par chance. Le chemin documenté (tableaux) est donc cassé, et le cas `N=2` est silencieusement faux. La correction attendue est `jnp.stack([a, b], axis=-1)` (ou `jnp.column_stack`).

### [MIN-4] `bnn.py:46-60` — `layer_linear` : aucune validation de rang/orientation des poids [RÉSOLU 3b99a538]
`jnp.matmul(X, dist)` suppose `W (in, out)` (convention respectée par BNNC l.52-61 et npe l.59). Un poids `(out, in)` à la PyTorch :
- couches non carrées → `ValueError` obscur de matmul ;
- **couches carrées (`D_H1 = D_X`, très courant en auto-encodeur) → produit calculé sur la transposée, résultat faux sans erreur**.
Pas de contrôle `X.ndim >= 2`, `W.ndim == 2`, `X.shape[-1] == W.shape[0]`.

### [MIN-5] `bnn.py:317-319` — `cov` : matérialisation inutile de `jnp.eye(N)` [RÉSOLU 3b99a538]
`X = jnp.eye(N)` puis `jnp.dot(X, W1)` ≡ `W1` directement (`tanh(W1)`). Dense `N×N` pour rien ; avec `N = 5000` points, 100 Mo gaspillés par appel et un matmul `O(N²·hidden)` au lieu de 0.

### [MIN-6] `bnn.py:185` — `layer_attention` : `context.reshape(1, -1)` force le batch à 1 [RÉSOLU 3b99a538]
`layer_linear(context.reshape(1, -1), self.out).reshape(self.b_q, self.b_kv)` ne marche que pour un seul bloc à la fois ; tout batch `B > 1` ou rappel vectorisé (`vmap`) casse silencieusement le reshape final. Combiné à l'état d'instance (ARCH-2), la méthode n'est pas réentrante.

### Point positif
`layer_cholesky` (l.92-99) et `layer_mixture_normal` (l.133-147) sont propres : validation de l'axe dernier (`l.93-95`, `l.135-137`), shapes batch quelconques, pas de matrice dense de mixture matérialisée, et `Test/10.BNN/test_mixture_layers.py` vérifie log-prob contre `numpyro.distributions` de référence.

---

## 3. Architecture, état & reproductibilité

### [MAJ-7] `bnn.py:13-16` + `np_dists.py:214-218` + `Samplers/SBI/npe.py:37` — instance `_BNN` de niveau module avec seed par défaut **temporelle** : draws `sample=True` non reproductibles [RÉSOLU 657b6a3a]
`bnn.__init__(rand_seed=True)` crée son propre `UnifiedDist(True)` ; `normalize_seed(True)` retourne `int(time.time_ns())` (`np_dists.py:216-217`). Dans `Main`, `main.py:251-254` recâble `self.bnn.dist = _unified` (partagé, avec `_rng` central) — mais **`npe.py:37` instancie `_BNN = bnn()` au niveau module, hors de tout `BF`** : ce câble n'est jamais posé. Tout tirage `sample=True` sans `seed` explicite via `_BNN` (p. ex. `layer_mixture_normal(..., sample=True)`, `bnn.py:146-147`) repart d'une clé `PRNGKey(time_ns())`.
**Scénario** : deux exécutions NPE identiques avec `rand_seed=42` sur `m` → les échantillons tirés par `_BNN` diffèrent à chaque run ; le test de non-régression de reproductibilité de l'audit Samplers échoue sur ce chemin. Le paramètre `rand_seed` du constructeur (l.13) est de plus trompeur pour `m.bnn` : il est écrasé par `main.py:254`.

### [ARCH-1] `bnn.py:9` — `class bnn(activation)` : héritage IS-A abusif, et `_activation_map` décorative `[PARTIEL 657b6a3a — lookup via _activation_map (MAJ-8) ; héritage bnn(activation) conservé pour compatibilité d'API]`
`bnn` hérite de `activation` uniquement pour que `layer_linear:67` fasse `getattr(self, activation)`. Conséquence directe : **la lookup ignore complètement `_activation_map`** (construit l.18-36, utilisé seulement par `available_activations()` l.44).
- [MAJ-8] **Alias `"swish" cassé** : enregistré l.31, listé par `available_activations()`, mais `getattr(self, "swish")` → `AttributeError` → `ValueError: Unknown activation function: 'swish'` (l.68-69). L'API promet ce qu'elle refuse. (npe se protège avec son `_check_activation` l.~52 qui exige map **et** getattr — l'incohérence est là, pas côté utilisateur direct.) [RÉSOLU 657b6a3a]
- [MAJ-8 bis] **Tout nom d'attribut est accepté** : `activation="cov"` passe le `getattr` (méthode liée, l.298) et plante au fond de `cov` avec une erreur sans rapport ; `activation="layer_linear"`, `"get_rho"`, `"make_pd_and_cholesky_jax"` idem. La lookup devrait être `self._activation_map.get(activation)` — la map existe, il faut s'en servir. [RÉSOLU 657b6a3a]
- [MIN-7] `softmax` et `exp` sont enregistrés comme activations génériques (`bnn.py:23, 35`) alors que leurs docstrings les restreignent à la sortie ; rien n'empêche `activation="softmax"` en couche cachée (normalisation par ligne, perte d'information silencieuse). `[DOCUMENTÉ — docstring de layer_linear ; aucun garde, la couche ne sait pas si elle est cachée]`

### [ARCH-2] `bnn.py:153, 165, 198, 230, 272` — état mutable stocké sur l'instance par les méthodes de couche `[DOCUMENTÉ 27d7ba49 — état « dernière couche construite » documenté (W13), jamais relu entre appels ; refonte non faite]`
`layer_attention` écrit `self.b_q/b_kv/d_model/emb_q/emb_kv/q_k_v_proj/out` ; `layer_toeplitz`, `layer_compound_symmetry`, `layer_diagonal` écrivent `self.b`, `self.log_sigma`, `self.raw_alpha`, `self.log_diag`, `self.log_variances`. Une instance = un seul état de couche :
- deux blocs (`layer_toeplitz(name='A')` puis `name='B'`) → les attributs du bloc A sont écrasés, tout code qui les relit (debug, `__call__`) voit B ;
- `m.bnn` est **partagé par tout le modèle** (`main.py:245`) et `_BNN` par tout le module npe → réexécution sous `scan`/`vmap` ou deux modèles dans le même process = état croisé. Ces valeurs ne sont de toute façon pas dans le posterior (ce sont des tirages/tampons), l'attribut d'instance n'apporte rien.

### [MIN-8] `bnn.py:155-165, 200-204, 233-239, 274, 311-314, 324` — noms de sites codés en dur ou à suffixe vide [RÉSOLU 3b99a538]
`name=f'attention_q_{name}'` avec `name=''` par défaut (l.149, 187, 219, 262) produit `'attention_q_'` ; `cov` déclare `'W1'`, `'W2'`, `'rf'` (l.311, 314, 324). Deux appels dans le même modèle numpyro → erreur de site dupliqué (`layer_attention()` deux fois, `cov()` deux fois) au lieu d'un nommage paramétrable systématique.

### [MIN-9] `bnn.py:149, 219, 262` — défauts `sample=True` dangereux [RÉSOLU 3b99a538]
`layer_attention`, `layer_compound_symmetry`, `layer_diagonal` ont `sample=True` par défaut (alors que `layer_toeplitz:187` a `False` — incohérence interne). Dans un contexte de modèle, le chemin attendu est `sample=False` (déclaration de site appris) ; le défaut tire du prior et renvoie une valeur figée : les paramètres ne sont jamais appris, sans erreur. Les defaults sûrs devraient être alignés sur `layer_toeplitz`.

### PRNG — point positif
Aucun `np.random` / `random` global dans les trois fichiers (grep vérifié) ; tout l'échantillonnage passe par `self.dist` (JAX). La pureté est respectée ; le problème est la *résolution de la clé* (MAJ-7), pas la source.

---

## 4. Robustesse de l'API publique

### [MIN-10] `bnn.py:57-58` — `bias=None` plante, `bias=0` passe [RÉSOLU 3b99a538]
`if bias is not False:` — `None is not False` est vrai → `jnp.matmul(...) + None` → `TypeError` obscur. Un utilisateur qui passe `bias=None` (valeur « naturelle » pour « pas de biais », cf. `activation=None` l.63) obtient une erreur de broadcasting, pas un message d'API.

### [MIN-11] `bnn.py:46` — paramètre de poids nommé `dist` [RÉSOLU 3b99a538]
`layer_linear(X, dist, ...)` : le deuxième argument est une **matrice de poids**, pas une distribution ; le nom masque l'import `dist` (l.1) et invite à passer un objet-distribution (`create_obj=True`), qui échoue alors dans `jnp.matmul` avec une erreur incompréhensible. La docstring l.51 parle d'ailleurs de `prior_dist (bf.dist)` — signature, docstring et sémantique (un tenseur) ne coïncident nulle part.

### [MIN-12] `bnn.py:144-147` — `layer_mixture_normal` : `name` silencieusement ignoré si `sample=True` [RÉSOLU 3b99a538]
`if name is not None and not sample:` déclare le site ; sinon (y compris `name` fourni **et** `sample=True`) on retombe sur le tirage anonyme, `obs` ignoré. L'utilisateur qui demande à la fois un site nommé et un échantillon ne reçoit ni l'un ni l'autre, sans avertissement. Un `raise` ou un warning explicite s'impose.

### [MIN-13] `bnn.py:187-197` — `layer_toeplitz` : docstring placée après la signature [RÉSOLU 3b99a538]
Le bloc l.189-197 est une expression statement morte : `help(bnn.layer_toeplitz)` n'affiche rien. Les autres méthodes de couche ont leur docstring en première position.

### [MIN-14] `bnn.py:326-346` — `get_rho` : docstring vide et concept discutable [RÉSOLU 3b99a538]
`a_b = jnp.mean(posterior, axis=0)` (l.331) moyenne **d'abord** sur les draws, puis calcule la covariance inter-lignes : le `rho` obtenu est la corrélation des *moyennes postérieures*, qui sous-estime la variabilité réelle (la moyenne écrase l'incertitude) et n'est pas la moyenne des `rho` par draw. Sans docstring, l'appelant ne peut pas savoir lequel des deux il obtient. (Le NaN de MAJ-6 s'ajoute au problème.)

### [MIN-15] `BayesForge/BNN/__init__.py` — fichier vide [RÉSOLU 3b99a538]
Aucun ré-export (`from BayesForge.BNN import bnn` échoue) ; chaque consommateur écrit `from BayesForge.BNN.bnn import bnn` (`main.py:72`, `npe.py:31`). Mineur mais incohérent avec le reste du paquet.

### [MIN-16] `bnn.py:272` — `layer_diagonal` écrit `self.b` sans s'en servir [RÉSOLU 3b99a538]
`block_size` est utilisé directement (l.274, 280) ; `self.b` est un résidu de copie de `layer_toeplitz`/`layer_compound_symmetry` (qui, eux, s'en servent). Confirme ARCH-2.

---

## (a) Résumé du module

`BayesForge/BNN` fournit (1) une bibliothèque d'activations JAX statiques (`activations.py`, correcte dans l'ensemble, deux pièges numériques : `softplus` naïf et `exp`), (2) la classe `bnn` : couche linéaire à poids échantillonnés hors méthode (`layer_linear`, convention `W (in, out)`, consommée par `Models/BNNC.py:52-61` et `Samplers/SBI/npe.py:59`), une famille de couches de covariance structurée (`layer_toeplitz`, `layer_compound_symmetry`, `layer_diagonal`), un bloc attention (`layer_attention`), et surtout le couple **`layer_cholesky` / `layer_mixture_normal`** — la partie saine du module : validée par shape-check, diagonale Cholesky rendue positive par `jax.nn.softplus` stable, testée contre `numpyro.distributions` dans `Test/10.BNN/test_mixture_layers.py`. La périphérie en revanche (`__call__`, `cov`, `get_rho`, `layer_attention`, la résolution des activations) est du code peu ou pas testé, dont deux chemins sont objectivement faux.

## (b) Fragilités architecturales

1. **Double registre d'activations** : `_activation_map` (l.18-36) et `getattr` (l.67) divergent — alias `swish` mort, noms de méthodes acceptés. La map devrait être la source unique.
2. **Héritage `bnn(activation)`** : choix fait pour alimenter le `getattr`, qui pollue l'espace de noms de l'instance (cause directe du 1).
3. **État d'instance comme mémoire de couche** (`self.b`, `self.emb_q`, `self.log_sigma`, …) : non réentrant, inadapté à un objet partagé par tout le modèle (`main.py:245`) ou par un module (`npe.py:37`).
4. **Câblage RNG implicite et partiel** : `Main` recâble `m.bnn.dist` après construction (`main.py:254`) ; toute instance hors `Main` (npe) garde la seed temporelle `time_ns()` de `normalize_seed(True)` — la reproductibilité dépend de *qui a instancié la classe*, pas d'une garantie du module.
5. **Aucune politique d'échelle de prior par fan-in** : le module laisse l'appelant fournir le prior (BNNC passe `N(0,1)` brut) sans helper He/Glorot ni avertissement — la bonne abstraction (échelle adaptée au shape de la couche) n'existe nulle part.
6. **Contrat `layer_linear(X, dist=…)`** : le poids est échantillonné *avant* l'appel par le distributeur, la couche ne valide rien — toute erreur de shape/orientation atterrit dans `jnp.matmul`.

## (c) Décompte par sévérité

| Sévérité | Nombre | Issues |
|---|---|---|
| **Critique** | 2 | CRIT-1 (`__call__` cassé), CRIT-2 (`cov` broadcasting `(2,N)`/`(N,2)`) |
| **Majeure** | 8 | MAJ-1 (softplus overflow), MAJ-2 (exp overflow), MAJ-3 (échelle attention `d_model^0.25`), MAJ-4 (`sigma` vs `sigma²`), MAJ-5 (priors sans fan-in), MAJ-6 (`get_rho` div/0 → NaN), MAJ-7 (`_BNN` npe, seed `time_ns`), MAJ-8 (lookup `getattr` : swish mort + noms arbitraires acceptés) |
| **Mineure** | 16 | MIN-1…MIN-16 (elu branche morte, jitter fixe, commentaires « log space », orientation carrée silencieuse, `eye(N)` inutile, reshape batch-1, softmax/exp en caché, noms de sites codés en dur, `sample=True` par défaut, `bias=None`, param `dist`, `name` ignoré si `sample=True`, docstring `layer_toeplitz`, `get_rho` concept, `__init__.py` vide, `self.b` résiduel) |
| **Total** | **26** | |

---

### Module Build

**Périmètre** : `BayesForge/Build/Build_NP_light.py` (221 l.), `BayesForge/Build/Build_TFP_light.py` (120 l.), `BayesForge/Build/Build_test_file.py` (84 l.). Audit statique uniquement — aucun générateur exécuté. Vérifications croisées sur les fichiers produits en production (`BayesForge/Distributions/np_dists.py`, `tfp_dists.py`) et le cache `BayesForge/Build/Doc_python.json`, à l'appui des constats.

**Constat directeur** : ces trois scripts ne sont pas des générateurs au sens industriel, mais des notebooks exportés (`# %%`) dont la sortie brute **ne compile pas** et n'a **jamais** pu être utilisée telle quelle. Le « pipeline » réel est : génération → correction manuelle globale par Ctrl+H dans VSCode (documentée dans l'en-tête de `tfp_dists.py` L1-15 et la docstring de `Build_NP_light.py` L5-11) → renommage manuel. Les bugs critiques C1/C2 relevés dans l'audit de `tfp_dists.py` prennent tous leur source ici, et **le générateur les régénérerait à l'identique**.

---

## 1. Bugs et fragilités — Build_TFP_light.py (générateur de tfp_dists.py)

### T-1 · CRITIQUE — `Build_TFP_light.py:19` : écrasement direct du fichier de production, sans garde `[RÉSOLU 1bc8055b]`
`with open("tfp_dists.py", "w")` écrit **sous le nom exact du fichier de production**, en chemin relatif (CWD). Exécuter le script depuis `BayesForge/Distributions/` détruit `tfp_dists.py` (235 Ko, corrigé à la main depuis mai) et le remplace par la version brute non corrigée. Aucune sauvegarde, aucun diff, aucun `py_compile`, aucune question. Pire : le script n'a pas de garde `if __name__ == "__main__"` — **un simple `import Build_TFP_light` déclenche la génération et l'écrasement**. Scénario : un développeur « essaie » le générateur pour comprendre un bug → le fichier de production est perdu avec tous les correctifs manuels.

### T-2 · CRITIQUE — `Build_TFP_light.py:59`, `:54-56`, `:93-94`+`:97` : le fichier émis ne compile pas (3 familles d'erreurs de syntaxe) — racine des bugs C1/C2 `[RÉSOLU 1bc8055b]`
- **L59** `", ".join([str(p) for p in parameters.values() ...])` : `str(inspect.Parameter)` rend le défaut par `repr`. Pour `Categorical(dtype=jnp.int32)` la ligne `def` générée contient `dtype=<class 'jax.numpy.int32'>` — syntaxe invalide. C'est exactement ce que corrige l'en-tête de `tfp_dists.py` L6-7 (« Replace `<class 'jax.numpy.int32'>` … »).
- **L54-56 + L64** : les arguments custom (`shape`, `sample`, `seed`, `obs`, `wrap`) sont **apposés après** les paramètres d'origine. Quand la signature TFP se termine par `**kwargs` (p. ex. `JointDistributionSequential`), on obtient `def x(..., **kwargs, shape=(), sample=False, ...)` — SyntaxError. C'est le correctif manuel n°5 de l'en-tête (L8-9).
- **L93-94** : `arg_names = [p.name for p in parameters.values() ...]` — pour un paramètre `VAR_KEYWORD`/`VAR_POSITIONAL`, `p.name` vaut `kwargs`/`args`, donc **L97** émet `tfd.X(args=args, kwargs=kwargs)` au lieu de `tfd.X(*args, **kwargs)` : le dictionnaire est passé comme *un mot-clé nommé* `kwargs`, les kwargs réels sont perdus. C'est le correctif n°1 de l'en-tête (L4) — **le bug C1**. Preuve qu'il est toujours vivant : `tfp_dists.py:4584` contient encore `kwargs=kwargs` (`ZeroInflatedNegativeBinomial`) — le Ctrl+H global a été appliqué **incomplètement**, et le générateur n'a jamais été corrigé à la source.

Scénario : régénérer → les trois familles de bugs reviennent intégralement, plus les omissions du Ctrl+H.

### T-3 · CRITIQUE — `Build_TFP_light.py:101-104` : sémantique de `obs` cassée à la source, divergence structurelle avec NP `[RÉSOLU 1bc8055b]`
```python
if obs is not None:
     final_dist = tfd.Independent(dist, reinterpreted_batch_ndims=shape)
else:
     final_dist = tfd.Sample(dist, sample_shape=shape) if shape else dist
```
`obs` n'est **jamais** transmis à la distribution : il ne fait que déclencher un `Independent` dont `reinterpreted_batch_ndims` reçoit `shape` (un **tuple**, alors qu'un `int` est requis) → `TypeError` à l'exécution dès qu'un utilisateur passe `obs`. Côté NP (`Build_NP_light.py:214`), `obs` est conditionné correctement via `numpyro.sample(name, d, obs=obs, ...)`. Le wrapper TFP ne conditionne donc **aucune observation** — divergence inter-backends structurelle, pas une régression locale. La production a été corrigée à la main en `reinterpreted_batch_ndims=len(jnp.shape(obs))` (visible dans `tfp_dists.py`, p. ex. corps de `log_normal` L2435+) — correction **absente du générateur**, donc perdue à la régénération. Note : les lignes 102/104 sont indentées de 13 espaces (12 ailleurs) — valide mais signe d'écriture à la main sans relecture.

### T-4 · MAJEURE — `Build_TFP_light.py:12-16` : filtre sans `issubclass(..., Distribution)` → les wrappers morts de tfp_dists viennent d'ici [RÉSOLU 5d666ffd]
Le filtre retient **toute classe** exposée par `tfp.distributions`, y compris les classes d'infrastructure. Ce sont les auteurs directs des wrappers morts relevés dans l'audit de `tfp_dists.py` : `AutoCompositeTensorDistribution` → `tfp_dists.py:30`, `Distribution` → `:882`, `RegisterKL` → `register_k_l` `:3576`, `ReparameterizationType` → `reparameterization_type` `:3680`. Les appeler construit un objet dénué de sens (`tfd.Distribution(dtype=...)`). L'homologue NP (`Build_NP_light.py:25-28`) fait le contrôle `issubclass(obj, Distribution)` — le TFP ne le fait pas.

### T-5 · MAJEURE — `Build_TFP_light.py:69` : collisions de noms non détectées → `delta` défini deux fois [RÉSOLU 5d666ffd]
`re.sub(r'(?<!^)(?=[A-Z])', '_', key).lower()` convertit `Deterministic` **et** `Delta` en `delta` : `tfp_dists.py:643` (`def delta(loc, …, name='Deterministic', …)`) puis `:767` (`def delta(v, …, name='Delta')`) — la seconde définition écrase silencieusement la première à l'exécution du `class` ; `Deterministic` est inatteignable. Même mécanique pour `LKJ*` → `k_k_j`/`l_k_j` (correctif manuel n°2 de l'en-tête L5). Aucune détection de collision ni journal des doublons.

### T-6 · MAJEURE — `Build_TFP_light.py:47`+`:99` : `seed=0` par défaut, `PRNGKey(seed)` recréé à chaque appel [RÉSOLU 5d666ffd]
Tous les appels sans `seed` explicite partagent la clé PRNG 0 : deux distributions différentes échantillonnées avec les mêmes `shape` renvoient **les mêmes nombres** (transformations d'une même suite de base). Bug statistique silencieux, commun aux deux générateurs (NP `:158`/`:192`).

### T-7 · MINEURE — `Build_TFP_light.py:74-85` : docstring « Default = <repr> » non nettoyée [RÉSOLU 7379f074]
`f"{param.name}: Default = {param.default}"` recolle le `repr` des défauts (`<class 'jax.numpy.int32'>`) dans la docstring — cosmétique, mais c'est une 2ᵉ source du correctif manuel n°3/4 de l'en-tête.

---

## 2. Bugs et fragilités — Build_NP_light.py (générateur de np_dists.py)

### N-1 · CRITIQUE — `Build_NP_light.py:158-159` : signature du wrapper concaténée à l'aveugle → `dists.py` ne compile pas `[RÉSOLU 1bc8055b]`
`full_signature = f"{param_str}, {wrapper_args}"` avec `wrapper_args = "name='x', obs=None, mask=None, sample=False, seed=0, shape=(), event=0, create_obj=False"` :
- classes dont la signature contient `**kwargs` → `def x(**kwargs, name='x', …)` = SyntaxError (docstring du générateur L7, correctif manuel n°2) ;
- classes ayant déjà un paramètre `mask` (p. ex. `MaskedDistribution`, incluse car non listée aux exclusions L27) → **paramètre `mask` deux fois** = SyntaxError (docstring L6, correctif n°1).
Le fichier généré est donc **cassé par construction** ; la docstring L5-11 l'assume (« edited by hand », « tracked through the VSCode Problems list ») : le compilateur Python tient lieu de suite de tests.

### N-2 · CRITIQUE — `Build_NP_light.py:115` + `:182-184` : la validité syntaxique du fichier dépend du formatage de la sortie LLM `[RÉSOLU 1bc8055b]`
`generate_doc_string` retourne `response['message']['content']` **brut**, et L184 l'insère **sans écrire les délimiteurs `"""`** — les guillemets de docstring viennent… du LLM lui-même. Vérification dans le cache committé `Doc_python.json` (92 entrées) : la sortie de `gemma3:12b` commence par `` ```python `` puis `"""HalfNormal` **non fermé**. Le fichier généré contient donc des fences Markdown et des docstrings syntaxiquement invalides pour (presque) chaque méthode. Toute la normalisation des docstrings de `np_dists.py` (`r"""### HalfNormal` etc.) est du travail manuel ultérieur.

### N-3 · CRITIQUE — exactitude documentaire : formules LLM jamais vérifiées (HalfNormal fausse) `[RÉSOLU 1bc8055b]`
Le prompt L49-101 exige une formule LaTeX, sans aucune vérification. Dans le cache `Doc_python.json`, `HalfNormal` porte la densité normale **sans le facteur 2** (déjà faux) ; dans la production `np_dists.py:3443`, la formule est devenue `\frac{2}{\pi,\sigma}\frac{1}{1+(x/\sigma)^2}` — **une demi-Cauchy avec faute de frappe**, pas une demi-normale. Le générateur est l'usine à docstrings fausses ; rien ici ne le détecte. (La contamination Bernoulli/BernoulliLogits signalée dans l'audit de `np_dists` est du même lot LLM ; la version production de `bernoulli_logits` `:509` a été réécrite à la main.)

### N-4 · CRITIQUE — reproductibilité : le générateur est **périmé** par rapport à sa production ; régénérer = régresser `[RÉSOLU 1bc8055b]`
`np_dists.py` (22 oct.) contient `to_jax=True`, `enum=False`, `infer=None`, `seed=None`, l'auto-nommage par inspection de bytecode, l'import `SampledData` — **aucun de ces éléments n'existe dans le générateur** (`grep to_jax` : 0 occurrence ; `wrapper_args` L158 émet `seed=0`). Le générateur n'a plus rien à voir avec le fichier qu'il est censé produire : la régénération écraserait des semaines de travail. Le build est un **cliquet unidirectionnel** : production ≠ f(générateur), et rien (ni version stamp, ni diff, ni test) ne signale la dérive.

### N-5 · MAJEURE — `Build_NP_light.py:17`, `:103-114`, `:119`, `:131-133` : LLM `ollama`/`gemma3:12b` dans le chemin de build, dépendance non déclarée et non rejouable [RÉSOLU 5d666ffd]
- `import ollama` absent de `pyproject.toml`/`requirements` (vérifié : 0 occurrence) → le build échoue sur un environnement propre (`ModuleNotFoundError`), ou au premier appel si le serveur ollama local ne tourne pas (`ollama.chat` L103 n'a **aucune** gestion d'erreur — la boucle L121-127, « up to 2h to run » L119, plante au premier refus de connexion, après N heures de travail perdu).
- `temperature: 0` ≠ déterminisme (version du modèle, non-déterminisme GPU) : les docstrings **ne sont pas reproductibles**.
- Pas de reprise sur panne : `Doc = {}` (L116) repart de zéro à chaque exécution ; `Doc_python.json` est écrit (L131-133) **mais jamais relu** — le cache n'est pas un cache. Le test L122 (`if name not in Doc.keys()`) est toujours vrai.

### N-6 · MAJEURE — `Build_NP_light.py:186-188` : même bug `args=args, kwargs=kwargs` que TFP, jamais corrigé à la source [RÉSOLU 5d666ffd]
`arg_str` construit depuis `param.name` → pour `*args`/`**kwargs` émet `d = module.Cls(args=args, kwargs=kwargs)`. La docstring L5-11 ne mentionne même pas ce correctif côté NP (contrairement à TFP) : les deux générateurs partagent le défaut, un seul des deux fichiers production porte la trace du correctif.

### N-7 · MAJEURE — `Build_NP_light.py:150-151` + `:123` : distributions silencieusement absentes [RÉSOLU 5d666ffd]
La docstring est générée pour tout sauf `SineBivariateVonMises` (exclusion codée en dur L123, workaround d'un plantage du LLM), et le wrapper n'est émis que `if name in Doc` (L151) → `sine_bivariate_von_mises` **disparaît du fichier final sans un mot**. Même logique d'échec silencieux L217-218 : seules `ValueError`/`TypeError` d'inspection sont capturées, en `print` ; toute autre exception plante la boucle **pendant l'écriture** du fichier (L136, `with` ouvert) → `dists.py` tronqué à blanc.

### N-8 · MAJEURE — reproductibilité à la version installée, sans empreinte [RÉSOLU 5d666ffd]
`dir(numpyro.distributions)` (L23, L33) au moment du build détermine **la liste des distributions** ; aucune épingle de version numpyro/jax, aucun horodatage/version stamp dans le fichier émis. Deux builds sur deux machines différentes produisent deux API publiques différentes, sans moyen de le savoir. (Le générateur TFP, L11, idem.)

### N-9 · MINEURE — `Build_NP_light.py:132`, `:136` : écritures sans `encoding=` [RÉSOLU 7379f074]
`open(..., "w")` sans encodage → dépendance à la locale ; les docstrings LaTeX/Unicode (`σ`, `→`) lèvent `UnicodeEncodeError` sur une machine en cp1252 (Windows) et produisent des octets différents selon la plateforme. Idem TFP L19 et test L61.

### N-10 · MINEURE — détails de style/hygiène révélant l'origine notebook non entretenue [RÉSOLU 7379f074]
Marqueurs `# %%` partout ; espaces parasites dans le code (`full_docstring.split    ('\n')` L183, `str   (dist_class.__name__)` L188, `if    mask` L213) — valides mais non formatés (aucun linter n'est passé sur ce répertoire) ; `dist_classes['NegativeBinomial'] = numpyro.distributions.conjugate.NegativeBinomial` (L40) accède à un sous-module privé pour compenser une absence de l'API publique ; instruction d'URL auto-contradictoire L92-93 (« `function_name_lowercase` … en PascalCase ») ; `Doc_R.json`/`Doc_python.json` (364 Ko de cache LLM) committés dans le dépôt — et le commit `e96c361b` (« LLM script removed ») montre que la branche R a déjà été nettoyée, pas la branche Python.

---

## 3. Bugs et fragilités — Build_test_file.py (générateur de tests par LLM)

### B-1 · CRITIQUE — `Build_test_file.py:84` : le fichier test généré ne compile pas — la « suite de tests » n'a jamais tourné `[RÉSOLU 5d666ffd]`
Dernière ligne émise : `file.write(f"print(erros):\n")` → `print(erros):` avec un deux-points final = **SyntaxError** dans le `test.py` généré. Le fichier produit par ce pipeline n'a donc **jamais pu être exécuté** : ce générateur de tests n'a jamais testé quoi que ce soit. Preuve d'abandon, corroborée par :
- **B-2 · CRITIQUE — `Build_test_file.py:77-78`** `[RÉSOLU 5d666ffd]` : `file.write(f"    {test[name]}\n")` n'indente **que la première ligne** de l'exemple LLM ; tout exemple multi-lignes (le cas courant : import + appel + print) produit un `IndentationError`. Deux raisons indépendantes et fatales.

### B-3 · MAJEURE — `Build_test_file.py:61` + absence d'intégration : `test.py` écrit dans le CWD, hors de toute suite [RÉSOLU 5d666ffd]
Écriture relative sans garde (écrase un `test.py` voisin), jamais lancée par la CI (vérifié : `.github/workflows/tests.yml` ne référence ni `Build_test_file` ni `test.py` ; les tests de documentation vivent dans `Test/3.Documentation/`). Le script est un **orphelin** : ni `if __name__`, ni pytest, ni code de sortie — même corrigé, il ne ferait **échouer aucune CI** (les erreurs sont seulement `print`ées dans `erros`, L72/L80-81/L84).

### B-4 · MAJEURE — circularité du test : exemples LLM extraits de docstrings LLM [PARTIEL 5d666ffd — tests générés = smoke tests documentés ; validation numérique par loi hors générateur]
La chaîne entière est `gemma3:12b` → docstrings → `gemma3:12b` → tests. Un exemple d'appel halluciné (mauvais nom d'argument, valeur par défaut impossible) est avalé par le `try/except` L79-81 et affiché comme erreur d'exécution, jamais comme échec. Le test ne peut pas invalider le générateur qui a produit sa propre hypothèse.

### B-5 · MINEURE — `Build_test_file.py:12`, `:66-68` : logique de chemin fausse, dupliquée dans le fichier généré [RÉSOLU 7379f074]
`os.path.dirname(os.path.abspath(""))` = **parent du CWD** (abspath de la chaîne vide = CWD) ; dans le fichier généré (L66), le double `dirname` donne l'**arrière-grand-parent**. Le `sys.path.append` est faux à l'endroit même où il est censé être le plus fiable — et il est recopié dans chaque `test.py` produit.

### B-6 · MINEURE — `Build_test_file.py:18-24`, `:56-58` : filtre sur `dir(m.dist)` trop large, docstrings `None` non traitées [RÉSOLU 7379f074]
Le filtre garde tout attribut non underscore « truthy » → `mask` et `plate` (méthodes utilitaires de `UnifiedDist`) passent et reçoivent des exemples inventés par le LLM ; `dist_classes[name].__doc__` (L58) peut être `None` → le prompt L28-36 reçoit la chaîne `"None"` et renvoie du code inventé. Import + instanciation `bf(platform='cpu')` au niveau module (L15-16) : importer ce script lance JAX.

---

## 4. Divergences structurelles NP ↔ TFP (racines des bugs inter-backends)

| Aspect | `Build_NP_light.py` | `Build_TFP_light.py` | Conséquence |
|---|---|---|---|
| `obs` | L214 : `numpyro.sample(name, d, obs=obs)` — conditionnement réel | L101-104 : `obs` ignoré, déclenche un `Independent` au `ndims` invalide | Un modèle « identique » aux deux backend n'a pas la même fonction de vraisemblance |
| `name` / site d'échantillonnage | L158 : `name='x'` | absent (TFP n'a pas de sample site) | API publique différente selon le backend, non documentée comme telle |
| `mask` | L158/L213 : via `infer={'obs_mask': …}` | inexistant | Masquage NP-only |
| `event` vs `shape` | L202-203 : `to_event(event)` distinct de `expand(shape)` | L102-104 : `shape` fait office des deux | Sémantique de forme divergente |
| `create_obj` / `wrap` | L208-209 : `create_obj` | L105-106 : `wrap` → `root(...)` | Deux conventions de sortie incompatibles |
| filtre de classes | `issubclass(Distribution)` (L26) | aucune vérification (L15) | Wrappers morts côté TFP uniquement |
| docstrings | LLM (gemma3:12b), non vérifiées | statiques | Qualité documentaire asymétrique |

---

## (a) Résumé du module

`BayesForge/Build/` n'est pas un système de build mais **trois notebooks exportés, à usage unique**, qui produisaient une première maquette des wrappers de distributions. Le flux réel est `génération LLM/inspect → fichier non compilable → corrections manuelles Ctrl+H → renommage`, documenté comme tel dans les en-têtes (`tfp_dists.py` L1-15, `Build_NP_light.py` L5-11). Depuis, la production a **quitté définitivement le générateur** (`np_dists.py` : `to_jax`, `enum`, auto-nommage bytecode absents du script ; `tfp_dists.py` : correctifs `obs` et `create_obj` absents du script). Les bugs critiques de `tfp_dists.py` (C1 `kwargs=kwargs`, C2 sémantique `obs`) sont **produits à la source** par `Build_TFP_light.py:93-94`/`:97` et `:101-104`, et le correctif manuel C1 est d'ailleurs resté incomplet (`tfp_dists.py:4584`). `Build_test_file.py` est un générateur de tests dont le produit ne compile pas (`:84`) et qui n'est branché à aucune CI. **Recommandation de release : classer les trois scripts comme non livrables** (`deprecated/` ou suppression), ou les réécrire avec : filtre `issubclass`, signature construite paramètre par paramètre (kind-aware), `obs`/`mask`/`event` alignés entre backends, docstrings vérifiées, `py_compile` + diff + tests post-génération, écriture hors chemin de production avec garde d'écrasement, `encoding="utf-8"`, et empreinte version (numpyro/tfp/modèle LLM) dans l'en-tête généré.

## (b) Fragilités architecturales

1. **Générateur ≠ source de vérité** : la production est éditée à la main en aval ; aucune boucle de retour vers le script. Le build est un cliquet : régénérer est toujours une régression (N-4, T-1).
2. **Le compilateur comme banc d'essai** : aucun `py_compile`, aucun import test, aucun diff post-génération ; la « suite de vérification » officielle est la liste Problems de VSCode (docstring NP L11, en-tête TFP L10).
3. **Ctrl+H comme mécanisme de correctif** : remplacements globaux non vérifiés → applications partielles (`tfp_dists.py:4584`), doublons (`log_normal` construit deux fois dans `tfp_dists.py`), collisions non détectées (`delta` ×2, `tfp_dists.py:643`/`:767`).
4. **LLM dans le chemin de build sans contrat** : dépendance undeclared, modèle et hyperparamètres codés en dur, aucune gestion d'erreur, aucun cache relu, aucune vérification de contenu (N-2, N-3, N-5, B-4).
5. **Échecs silencieux** : skips par `print` (NP L151/L217, TFP L112), exceptions non capturées en milieu d'écriture, tests qui `print`ent au lieu d'échouer — aucune voie d'alarme.
6. **Environnement non figé** : contenu du fichier = f(version installée de numpyro/TFP + version du modèle LLM + locale d'encodage + CWD d'exécution) ; rien de tout cela n'est enregistré.
7. **Absence de hygiène d'exécution** : pas de garde `__main__` — l'import des scripts est une opération destructrice (T-1).

## (c) Décompte par sévérité

| Sévérité | Nombre | Repères |
|---|---|---|
| **Critique** | **7** | T-1 (écrasement production), T-2 (fichier non compilable = C1/C2), T-3 (`obs` cassé = C2), N-1 (signature invalide), N-2 (délimiteurs docstring émis par le LLM), N-3 (formules fausses, HalfNormal), N-4 (générateur périmé vs production) — B-1/B-2 comptés ensemble : pipeline de test mort (`Build_test_file.py:84`, `:77-78`) |
| **Majeure** | **9** | T-4 (wrappers morts), T-5 (collisions `delta`/`k_k_j`), T-6 (`seed=0`), N-5 (ollama non rejouable), N-6 (`args=args` côté NP), N-7 (skips silencieux), N-8 (dérive de version), B-3 (test orphelin, hors CI), B-4 (circularité LLM→test) |
| **Mineure** | **7** | T-7 (repr dans docstrings), N-9 (encodage), N-10 (hygiène notebook/cache committé), B-5 (chemins faux), B-6 (filtre `dir(m.dist)`, `__doc__` None), typo `erros`/`warp`, URL contradictoire L92-93 |
| **Total** | **23** (dont 8 entrées critiques distinctes si B-1/B-2 séparés) | |

---

### Module Causal

**Périmètre** : `BayesForge/Causal/dag.py` (379 lignes) et `BayesForge/Causal/__init__.py` (3 lignes), racine `/home/sosa/work/BF`. Lecture intégrale ligne à ligne ; les 13 scénarios déclencheurs cités ci-dessous ont été **vérifiés par exécution** sur le code réel (les résultats annotés « vérifié » sortent d'un run effectif, pas d'une lecture seule).

**Note de cadrage sur la grille** : le module est un outillage graphique combinatoire pur Python (aucun numpy, aucune matrice d'adjacence, aucune densité). Les risques numériques classiques (NaN/Inf, divisions par zéro, log-densités, broadcasting `(N,)` vs `(N,1)`, symétrisation `A+A.T`, puissances de matrices) sont **structurellement absents** — c'est un point positif, pas une lacune d'audit. Les vrais risques sont la validation des entrées, la consistance des conventions internes et la complexité.

---

#### A.1 — Exactitude mathématique & stabilité numérique

**Points corrects vérifiés à la lecture :**
- La logique de d-séparation `_is_open` (dag.py:L165-L181) est manuellement conforme à Pearl : détection correcte du collisionneur `a → b ← c` (L172), ouverture par conditionnement sur `b` **ou descendant de `b`** (L175), blocage des chaînes/bifurcations par `b ∈ given` (L179).
- `backdoor_paths` (dag.py:L189-L192) : le critère `_edge_dir(p[0], p[1]) == "<-"` identifie bien les chemins commençant par une flèche **entrant** dans x.
- `ancestors`/`descendants` (dag.py:L119-L141) : DFS itératif strict (le nœud de départ est exclu), cohérent avec l'usage `descendants(x)` dans le critère back-door (L197, L213).
- Minimalité de `adjustment_sets` (dag.py:L221) : les ensembles étant collectés par taille croissante (L217), `any(v <= s for v in valid)` exclut bien tout surensemble redondant.
- Cycles : `add_edge` (dag.py:L77-L87) insère puis teste, avec rollback (L84-L85) ; les auto-boucles sont correctement détectées (`visit` voit `n` gris) et refusées. Pas d'infini-boucle possible via l'API publique : l'invariant acyclique est maintenu à la construction.

**Bugs et fragilités :**

**[CRITIQUE] C1 — M parsage silencieux des chunks multi-flèches et des arêtes bidirectionnes** — `BayesForge/Causal/dag.py:L43-L49` `[RÉSOLU 53c56830]`
La branche `<-` est testée **avant** `->` (L43) et `split` est appelé sans `maxsplit`. Tout chunk contenant deux opérateurs produit un nom de nœud corrompu **sans la moindre erreur**.
Scénario vérifié : `DAG("x -> m <- w")` (collisionneur écrit en un seul chunk, erreur d'écriture très plausible) → arête `('w', 'x -> m')`, nœud littéralement nommé `"x -> m"`. Pire : `DAG("x <-> y")` (arête bidirectionnelle, notation standard du confondement latent / diagrammes M-bias) → arête `('> y', 'x')`. Aucun message ; tous les calculs downstream (`d_separated`, `adjustment_sets`, `check`) tournent sur un graphe silencieusement faux. (`"a -> b -> c"` en un chunk lève bien un `ValueError: too many values to unpack` — obscur mais explicite ; le drame est le cas à deux opérateurs mixtes, qui ne lève rien.)

**[CRITIQUE] C2 — Nœuds inconnus acceptés silencieusement comme extrémités ou dans `given`** — `BayesForge/Causal/dag.py:L147-L159`, `L183-L186`, `L191-L199`, `L241-L274` `[RÉSOLU 53c56830]`
`_all_paths` explore depuis `x` ; si `y` n'est pas dans le graphe, la liste de chemins reste vide et **rien ne le signale**. Le graphe n'est validé nulle part en entrée des API publiques.
Scénarios vérifiés sur `g = DAG("z -> x, z -> y, x -> y")` :
- `g.d_separated("x", "w")` → `True` : le module **affirme l'indépendance conditionnelle avec un nœud inexistant** ;
- `g.adjustment_sets("x", "typo")` → `[set()]` : « aucun ajustement nécessaire » pour un résultat inexistant ;
- `g.check("x", "y", given={"z_typo"})` → `[]` : le diagnostic de sécurité déclare « tout va bien » alors que le nom de l'ensemble d'ajustement est fautif.
Incohérence aggravante : un `x` inconnu lève un `KeyError` sec (via l'accès direct `self._parents[node]`, L145) mais un `y` inconnu ne lève jamais rien — la faute de frappe, erreur utilisateur n°1, produit le résultat silencieusement faux que la grille interdit explicitement.

**[MAJEUR] M5 — `RecursionError` sur les longues chaînes : un DAG légitime est rejeté par un crash** — `BayesForge/Causal/dag.py:L92-L100` [RÉSOLU bb39454c]
`_creates_cycle` utilise un DFS **récursif** dont la profondeur égale le plus long chemin. Vérifié : construire une chaîne de 1500 nœuds par `add_edge` lève `RecursionError` vers le 995ᵉ arête (limite Python par défaut 1000). Un graphe valide est donc impossible à construire, avec un plantage inexpliqué au lieu d'une limite documentée ou d'un DFS itératif.

**[MINEUR] `d_separated(x, x)` renvoie `True`** — `BayesForge/Causal/dag.py:L147-L159`, `L186` [RÉSOLU c477610e]
Aucun chemin simple de `x` vers `x` n'est énuméré (`nb in path`, L153, exclut `x`), donc la disjonction vide donne « d-separated ». Une variable n'est jamais indépendante d'elle-même ; devrait renvoyer `False` ou lever.

**[MINEUR] Déviation non documentée du critère back-door : `y` est interdit dans `z`** — `BayesForge/Causal/dag.py:L197` [RÉSOLU c477610e]
Le critère de Pearl interdit `z ∩ (x ∪ desc(x))` ; le code interdit aussi `y` (`z & ({x, y} | self.descendants(x))`). Choix statistiquement défendable (conditionner sur le résultat ouvre des collisionneurs en `y`), mais le docstring L194-L195 revendique le critère standard — écart silencieux, par exemple pour un effet `x → y` où l'utilisateur voudrait tester `z = {y}`.

**[MINEUR] Noms de nœuds vides acceptés** — `BayesForge/Causal/dag.py:L43-L51` [RÉSOLU c477610e]
`DAG("a ->")` crée vérifié l'arête `('a', '')` ; `"a <-"` symétriquement. Un `strip()` qui aboutit à `""` devrait être rejeté immédiatement.

---

#### A.2 — Vectorisation, dimensions & mémoire

Aucun tableau numpy, aucune matrice d'adjacence, aucun DataFrame : les familles de bugs de broadcasting, d'axes de sommes, de symétrisation `A + A.T` et d'ordre de multiplication de matrices sont **inapplicables** ici. La reachability passe par DFS (`ancestors`/`descendants`, L119-L141) et non par puissances de matrices — pas de problème d'inclusion de soi-même sur la diagonale, les ensembles sont stricts et cette strictesse est utilisée de façon cohérente (L197, L213, L258).

**Hygiène des entrées utilisateur : bonne, vérifiée ligne à ligne.** `parents()`/`children()` renvoient des copies (L113, L116) ; `_is_open` (L167), `is_valid_adjustment_set` (L196), `check` (L248), `plot` (L311) recopient leurs ensembles d'entrée ; `nodes` (L106) et `edges` (L110) construisent des conteneurs neufs ; `do` (L231-L238) bâtit un graphe neuf. Aucune mutation in-place d'une donnée de l'appelant.

Points restants :

**[MINEUR] Coûts quadratiques/exponentiels cumulés hors chemin critique** — `BayesForge/Causal/dag.py:L83`, `L175`, `L266-L269`, `L291` [RÉSOLU c477610e]
- `_creates_cycle` est rejoué intégralement à chaque `add_edge` (L83) : coût de construction en O(E·(V+E)) ;
- `_is_open` recalcule `self.descendants(b)` pour chaque collisionneur de chaque chemin (L175) ;
- `check` réénumère **tous** les chemins `x↔y` pour chaque nœud de `given` (L268) ;
- `_topo_order` fait `queue.pop(0)` sur une liste, O(V²) (L291) — `collections.deque` s'impose.
Nuisible au-delà de ~20 nœuds ; anodin pour l'usage pédagogique visé.

**[MINEUR] Copies de listes dans l'énumération de chemins** — `BayesForge/Causal/dag.py:L156-L158` [RÉSOLU c477610e]
`path + [nb]` recopie le préfixe à chaque extension ; mémoire en O(chemins × longueur). Ce n'est gênant que du fait de M2 (voir A.4) : le nombre de chemins est déjà exponentiel.

---

#### A.3 — Architecture, état & reproductibilité

Aucun PRNG (ni `np.random` global, ni `random`) : le module est déterministe, point positif pour la reproductibilité. Aucun import de `numpy`, `networkx`, `dist` ou `Main` : découplage propre ; `__init__.py:L1` utilise l'import absolu `BayesForge.Causal.dag`, cohérent avec le paquet.

**[MAJEUR] M3 — Convention incohérente « chaîne-nœud » vs « itérable-de-nœuds » entre fonctions du même fichier** — `BayesForge/Causal/dag.py:L185`, `L248`, `L311` contre `L120-L121`, `L132-L133`, `L229-L230` [RÉSOLU bb39454c]
`ancestors`, `descendants`, `do` normalisent `str → {str}` ; `d_separated`, `check`, `plot` font `set(given or [])`, ce qui **découpe une chaîne en caractères**.
Scénario vérifié sur la chaîne `a → age → b` : `g.d_separated("a", "b", given="age")` → `False` (déclare a et b dépendants !) alors que `given={"age"}` → `True`. `check("a","b",given="age")` → `[]` (aucune alerte médiateur). Le bug est masqué par les tests utilisant des noms à un caractère (`given="z"` → `{"z"}`, fonctionne par accident) et explose dès qu'un nœud s'appelle `age`, `income`, etc.

**[MAJEUR] M4 — `DAG(iterable)` sans validation des éléments** — `BayesForge/Causal/dag.py:L68-L70` [RÉSOLU bb39454c]
`list(spec)` puis unpacking implicite. Vérifié : `DAG(["ab"])` produit silencieusement l'arête `('a', 'b')` (chaîne de 2 caractères dépaquetée caractère par caractère) ; `DAG(["z -> x"])` lève un `ValueError` de unpacking sans rapport avec l'intention de l'utilisateur. Le chemin « chaîne » passe par `_parse_spec`, le chemin « liste » par aucune validation : deux niveaux de robustesse pour un même constructeur.

**[MINEUR] Alerte d'expérimental par `print` + état global** — `BayesForge/Causal/dag.py:L21-L29` [RÉSOLU c477610e]
`_warn_experimental` écrit sur stdout (pollue notebooks, captures de sortie, CLI ; doit être `warnings.warn`, filtrable), et le drapeau global `_WARNED` n'est pas thread-safe (double affichage possible, bénin). Aucune possibilité de le supprimer.

**[MINEUR] Ordres non déterministes (sets Python)** — `BayesForge/Causal/dag.py:L110`, `L255`, `L262`, `L368` [RÉSOLU c477610e]
`edges` et `__repr__` itèrent des `set` de chaînes : l'ordre varie entre processus (hash randomization). Vérifié : `DAG("a -> b, a -> c, a -> d, a -> e")` → `DAG(a -> d, a -> e, a -> b, a -> c)`. Les messages d'alerte L255/L262 embarquent le `repr` de sets. Casse les doctests, les fichiers golden et la reproductibilité des rapports. Un `sorted()` à l'affichage suffirait.

**[MINEUR] Exemple du docstring du module faux** — `BayesForge/Causal/dag.py:L15-L16` [RÉSOLU c477610e]
Le graphe d'exemple (L10) ne contient **aucun nœud `m`** ; `g.check("x", "y", given={"m"})` renvoie vérifié `[]`, pas l'avertissement médiateur affiché. En doctest, l'exemple échoue.

**Contrat BF** : la façade `causal` (dag.py:L372-L379) expose `DAG` en attribut de classe et une fabrique `dag()` ; sans état ni `__init__(main)`, l'injection `m.causal = causal()` est triviale mais rien ne l'encode (pas de validation, pas d'accès aux données du modèle). Rien de cassé, rien de contracté non plus.

---

#### A.4 — Robustesse de l'API publique

**[MAJEUR] M1 — `plot()` plante sur le graphe vide (N=0)** — `BayesForge/Causal/dag.py:L355-L358` [RÉSOLU bb39454c]
`min(xs)` sur séquence vide. Vérifié : `DAG().plot()` → `ValueError: min() iterable argument is empty`. Aucun garde-fou N=0 ; N=1 fonctionne.

**[MAJEUR] M2 — Explosion combinatoire sans aucune garde** — `BayesForge/Causal/dag.py:L147-L159`, `L201-L224` [RÉSOLU bb39454c]
`_all_paths` énumère tous les chemins simples (exponentiel en l'occurrence) ; `adjustment_sets` itère **tous les sous-ensembles** des candidats (2^n, `max_size=None` par défaut, L215) en réénumérant les chemins à chaque sous-ensemble (via `is_valid_adjustment_set` → `backdoor_paths`). Sur un graphe de 20-25 nœuds, `adjustment_sets` suspend l'interpréteur pour des heures. Pas de borne par défaut, pas de timeout, pas de message. La doc promise (« Find adjustment sets ») n'est réaliste que pour de petits graphes — doit être documenté ou borné.

**[MINEUR] `sorted(candidates)` sur types hétérogènes** — `BayesForge/Causal/dag.py:L214` [RÉSOLU c477610e]
Des nœuds mixtes `int`/`str` (autorisés par les sets) → `TypeError: '<' not supported between instances of 'str' and 'int'`, tardif et obscur.

**[MINEUR] Nœuds orphelins laissés par une arête refusée** — `BayesForge/Causal/dag.py:L79-L86` [RÉSOLU c477610e]
Sur cycle détecté, l'arête est bien retirée (L84-L85) mais `_ensure` (L79-L80) a déjà inséré les deux nœuds : un `add_edge` **échoué** mute quand même le graphe (les nœuds apparaissent dans `nodes` et `repr`). Effet de bord d'une opération en erreur.

**[MINEUR] `plot` force `plt.show()`** — `BayesForge/Causal/dag.py:L364` [RÉSOLU c477610e]
Bloquant en script, effet de bord indésirable en CI/notebook ; devrait retourner l'axe sans afficher (paramètre `show=False`).

**[MINEUR] Code mort dans la logique de courbure des arêtes** — `BayesForge/Causal/dag.py:L334-L335` [RÉSOLU c477610e]
`span = abs(layer[c] - layer[p])` est toujours ≥ 1 pour une arête (la couche d'un enfant vaut ≥ couche parent + 1, L283), donc `same_row and span == 0` est **inatteignable** et la branche `else 0.4` (L335) est morte : la courbure des arêtes « same row » annoncée par le commentaire (L329-L330) n'est jamais appliquée.

**Ce qui est bien tenu** : graphe cyclique fourni → refusé à la construction (`add_edge` L83-L86, y compris auto-boucles) ; matrice non carrée / dtype objet → inapplicables (pas de matrices). `_topo_order` (L286-L297) ne détecte pas les cycles (retourne silencieusement un ordre partiel) — inatteignable via l'API grâce à l'invariant, mais hypothèse fragile si un futur `remove_edge` ou une construction « bulk » la contourne.

---

#### (a) Résumé du module

`BayesForge/Causal/` est un mini-module pédagogique de graphes causaux (379 lignes, zéro dépendance) : parsing de spécifications textuelles, DAG avec invariant acyclique maintenu à l'insertion, d-séparation par énumération de chemins, critère back-door, recherche d'ensembles d'ajustement minimaux, mutilation `do()`, diagnostics (`check`) et tracé matplotlib. **Le cœur mathématique (d-séparation, back-door, minimalité) est correct** et l'hygiène mémoire (copies systématiques, aucune mutation des entrées) est bonne. Le module pèche sur ses bords : deux voies d'entrée silencieusement corrompues (parsing multi-flèches, nœuds inconnus), une convention chaîne-vs-ensemble incohérente entre fonctions sœurs, une complexité exponentielle non bornée, et des détails de reproductibilité (ordres de sets, warning par `print`, exemple de docstring faux). Verdict de release : **prêt en tant qu'outil pédagogique expérimental sur de petits graphes à noms courts et validés par l'utilisateur ; non libérable comme API publique** tant que C1/C2/M3 ne sont pas corrigés — les trois produisent des réponses fausses sans erreur, ce qui est le mode de défaillance le plus dangereux pour un outil d'inférence causale.

#### (b) Fragilités architecturales

1. **Absence de couche de validation en amont** : chaque fonction applique sa propre politique (KeyError strict L145, `.get()` silencieux L125/L137, normalisation `str` L120/L229 ou découpage en caractères L185/L248/L311). Cause racine commune de C2, M3, M4. Une méthode privée `_check_nodes(*names)` appelée en tête de chaque API publique éliminerait les trois classes de bugs.
2. **Aucun plafond sur l'énumération** (chemins, sous-ensembles) : la surface API promet plus que la complexité n'autorise ; il faut une borne par défaut ou une documentation de limite explicite (M2).
3. **Pas d'interface matricielle** (export/import matrice d'adjacence) : le module est une île, sans intégration possible avec les composants numériques de BF (SEM, matrices de plan) ; la reachability n'existe que par DFS.
4. **Pas d'ADMG / arêtes bidirectionnes** (confondement latent), et le parseur **corrompt** cette entrée au lieu de la refuser (C1) — le trou est double : fonctionnalité manquante et surface d'erreur silencieuse.
5. **API de manipulation incomplète** : pas de `remove_edge`/`remove_node`/`copy`/`__eq__` ; `do()` est la seule mutilation ; pas de front-door ni de do-calculus. `_topo_order` fait l'hypothèse d'acyclicité sans l'asserter.
6. **Signalétique hors standards Python** : `print` + global mutable (L21-L29) au lieu de `warnings` ; ordres de sets non triés à l'affichage — tout ce qui consomme le module en pipeline reproductible (doctests, golden files) sera instable.

#### (c) Décompte par sévérité

| Sévérité | Nombre | Repères |
|---|---|---|
| Critique | 2 | C1 parsing silencieux (L43-L49) ; C2 nœuds inconnus silencieux (L147-L159, L183-L186, L241-L274) |
| Majeure | 5 | M1 plot vide (L355-L358) ; M2 explosion sans garde (L201-L224) ; M3 chaîne-vs-ensemble (L185/L248/L311) ; M4 iterable non validé (L68-L70) ; M5 récursion (L92-L100) |
| Mineure | 12 | d_sep(x,x) ; `y` interdit non documenté (L197) ; noms vides ; coûts cumulés (L83/L175/L266-L269/L291) ; copies L156-L158 ; print+global (L21-L29) ; ordres non déterministes (L110/L368) ; docstring d'exemple faux (L15-L16) ; sorted hétérogène (L214) ; orphelins sur arête refusée (L79-L86) ; `plt.show()` forcé (L364) ; code mort plot (L334-L335) |
| **Total** | **19** | dont 13 scénarios confirmés par exécution |

---

### Module Resources

**Périmètre** : `BayesForge/Resources/datasets.py` (667 lignes, lu intégralement) et `BayesForge/Resources/__init__.py` (0 octet, vide). Vérifications empiriques : les 31 chargeurs publics ont été exécutés avec `frame=True` et `frame=False` ; les fichiers de données embarqués ont été inspectés (en-têtes, séparateurs, MD5, non-ASCII) ; `pyproject.toml` (package_data), `BayesForge/__init__.py` et `BayesForge/Main/main.py` vérifiés pour la frontière de contrat.

**Faits de contrat vérifiés** (contexte d'audit confirmé, lignes exactes corrigées) :
- `BayesForge/Main/main.py:53` : `from BayesForge.Resources.datasets import load` ; l'instanciation `self.load = load()` est au **Main/main.py:249** (et non 148‑149).
- `bf.load` est un `@staticmethod` défini **Main/main.py:1889** (`def load(path=None)`, `path=None` → `"bf_model.pkl"` relatif au CWD, désérialisation `cloudpickle.load`). L'attribut d'instance `self.load` masque donc le staticmethod sur les instances ; `load.__call__` (datasets.py:65‑68) redélègue bien à `bf.load(path=path)`, ce qui rend le masquage comportementalement neutre à l'appel mais pas à l'introspection.
- `from BayesForge import load; load("x.pkl")` → **TypeError vérifié** : `load.__init__() takes 1 positional argument but 2 were given` (constructeur datasets.py:61 sans argument). Le bon usage est `load()("x.pkl")` — piège d'API réel, `load` est exporté dans `__all__` de `BayesForge/__init__.py`.
- `BayesForge/Resources/__init__.py` est vide : `from BayesForge.Resources import load` échoue ; il faut `BayesForge.Resources.datasets`. Incohérent avec l'exemple du docstring module (datasets.py:9 : `from BayesForge.resources.data import datasets` — **chemin inexistant**, minuscules + sous-module `data` qui n'existe pas).
- Package_data (pyproject.toml:69‑71) : `"BayesForge.Resources" = ["*.csv", "*.rda", "*.json"]` — les données sont bien embarquées à l'install pip. **Mais** `Trolley.csv` référencé par le code n'existe nulle part (voir C‑1).
- Aucun téléchargement réseau, aucun cache, aucun PRNG dans le module (positif : pas d'URLs/checksums/fichiers partiels à auditer ; les jeux « sim » sont des CSV statiques, `sim_nested_effects` documente seed=42, datasets.py:642).

---

#### Bugs et fragilités

**C‑1 — Critique — `BayesForge/Resources/datasets.py:241` — `trolley()` plante : fichier de données inexistant.** `[RÉSOLU 3b98e6c1]`
`_load_csv("Trolley.csv", …)` alors qu'aucun `Trolley.csv` n'existe dans le paquet (vérifié `ls` + `find` dans tout le dépôt) ni dans package_data. Scénario : `bf.load_data = load(); load().trolley()` ou `load().trolley(only_path=True)` → `FileNotFoundError: …/Resources/Trolley.csv` (vérifié à l'exécution, y compris `only_path=True` qui ne fait que construire le chemin). Une méthode publique documentée (docstring 219‑240, « Samples: Large dataset ») est du code mort qui ment dans l'API. Soit livrer le fichier, soit supprimer la méthode avant release.

**C‑2 — Critique — `BayesForge/Resources/datasets.py:514` — `elephants()` (défaut `frame=True`) retourne un DataFrame **vide** (41, 0).** `[RÉSOLU 3b98e6c1]`
`pd.read_csv(self.data_dir / "elephants.csv", index_col=0)` utilise le séparateur **virgule par défaut** alors que `elephants.csv` est **point-virgule** (`"AGE";"MATINGS"`). Sans `sep=None` (contrairement à `_load_csv:97`), chaque ligne entière devient une clé d'index : vérifié → `DataFrame shape=(41, 0)`, `index.name = 'AGE;"MATINGS"'`, zéro colonne. Pire, `index_col=0` est un non-sens ici : AGE est une variable, pas un index (le fichier n'a pas de colonne d'index, contrairement au pattern supposé). Le mode `frame=False` (ligne 516, via `_load_csv`) fonctionne — l'inversion défaut/correct est un piège massif. Scénario : `d = load().elephants(); d["AGE"]` → `KeyError` ; tout script d'exemple du docstring « Dataset with elephant data » renvoie un cadre vide **sans la moindre erreur**.

**C‑3 — Critique — `BayesForge/Resources/datasets.py:490` — `WaffleDivorce()` (défaut) retourne un DataFrame vide (50, 0), mécanisme identique à C‑2.** `[RÉSOLU 3b98e6c1]`
`WaffleDivorce.csv` est en point-virgule ; `pd.read_csv(…, index_col=0)` sans `sep=None` → `shape=(50, 0)`, index = ligne d'en-tête entière (vérifié). `frame=False` (ligne 492) marche et, lui, place `Location` en **colonne** de données (feature_names vérifié : `['Location','Loc','Population',…]`) alors que `frame=True` prétendait en faire un index : les deux modes ont des schémas incompatibles. Scénario : `w = load().WaffleDivorce(); w["Divorce"]` → KeyError sur un jeu de données phare de Statistical Rethinking.

**M‑1 — Majeure — `BayesForge/Resources/datasets.py:266` + `Resources/UCBadmit.csv` — fichier malformé (5 noms d'en-tête, 6 champs par ligne) → perte silencieuse d'une colonne.** [RÉSOLU 2906f8dd]
En-tête : `"dept";"applicant.gender";"admit";"reject";"applications"` (5) ; lignes : `"1";"A";"male";512;313;825` (6, vérifié via `csv.reader`). Avec `engine='python'`, pandas **rétrograde silencieusement le premier champ (n° de département) en index sans nom** : vérifié → `ucbadmit()` donne (12, 5) avec index 1,2,… et colonnes décalées d'un rang mais valeurs alignées. Conséquences : (a) `ucbadmit(frame=False).data` **perd le numéro de département** (seules les 5 colonnes nommées entrent dans `data`, l'index est hors Bunch hors `frame`) ; (b) le comportement est spécifique au moteur python — le moteur C lèverait `ParserError: Expected 5 fields, saw 6` ; un changement de moteur/paramètre (`sep=','` explicite, refactor) transforme ce chargement en crash ; (c) l'index unnamed casse toute jointure ultérieure sur `dept`. Scénario concret : `b = load().ucbadmit(frame=False); np.unique(b.data[:,0])` → `['A' 'B' …]` sans le dept numérique documenté par l'en-tête.

**M‑2 — Majeure — `BayesForge/Resources/datasets.py:449‑468` — `sim_multivariate_normal` ignore le paramètre `frame` : retourne un Bunch même avec le défaut `frame=True`.** [RÉSOLU 2906f8dd]
Contrairement aux 30 autres chargeurs (défaut → DataFrame), la méthode sort directement en `return Bunch(...)` (461) sans tester `frame`. Vérifié : `type(load().sim_multivariate_normal()).__name__ == 'Bunch'`. Scénario : code générique `dfs = [getattr(load(), n)() for n in names]; dfs[0].columns` → `AttributeError` sur ce seul jeu ; ou `df = load().sim_multivariate_normal(); df.mean()` → statistiques sur un dict. De plus la ligne 449 réaffecte inutilement `self.data_dir` (déjà posé en 63) — trace de copier-coller.

**M‑3 — Majeure — `BayesForge/Resources/datasets.py:381, 578, 590, 602, 614, 626` — rétroconversion silencieuse en float32 des matrices phylogénétiques via `jnp.array`.** [RÉSOLU 2906f8dd]
`Bunch(data=jnp.array(df.values), …)` : JAX en x64 **désactivé par défaut** (et `setup_device` de BF documente `float_precision=64` par défaut, Main/main.py ~146) → le CSV float64 (`0.99999999998205`, phylo_L_simple.csv) est tronqué en float32. Vérifié : `phylo_L_simple(frame=False).data.dtype == float32`, idem `islands_dist_matrix(frame=False)` (les 4 autres `_load_csv` retournent du numpy float64). Scénario : un modèle phylogénétique en float64 consomme `L = load().phylo_L_simple(frame=False).data` → facteur de Cholesky dégradé à ~1e‑7 relatif, biais dans la covariance des effets phylogénétiques et incohérence dtype avec le reste du pipeline, sans aucun avertissement.

**M‑4 — Majeure — `BayesForge/Resources/datasets.py:578, 590, 602, 614, 626` — schéma Bunch incohérent pour les chargeurs `phylo_L_*` : clé `columns` au lieu de `feature_names`, ni `DESCR`, ni `frame`, ni `filename`.** [RÉSOLU 2906f8dd]
Vérifié : `sorted(load().phylo_L_simple(frame=False).keys()) == ['columns','data']` contre `['DESCR','data','feature_names','filename','frame']` partout ailleurs. Tout consommateur générique de Bunch (boucle sur `feature_names`, affichage `DESCR`) casse sur ces cinq chargeurs. `islands_dist_matrix` (380‑386) est un troisième schéma : `feature_names` + `row_names` mais **pas de clé `frame`**. Trois contrats de retour coexistent dans le même module.

**M‑5 — Majeure — `BayesForge/Resources/datasets.py:106` — `data = df.values` produit des tableaux `dtype=object` empoisonnés de chaînes pour 12 des 25 chargeurs `frame=False`.** [RÉSOLU 2906f8dd]
Vérifié dtype object pour `iris`, `milk`, `kline`, `kline2`, `mastectomy`, `panama_ewa`, `phylo_simple/poisson/repeated/meta/slopes`, `reedfrogs`, `tulips`, `ucbadmit` (colonnes textuelles : species, clade, mono, dept…). Le docstring de classe (55) promet « data: numpy array of features ». Scénario : `jnp.asarray(load().iris(frame=False).data)` → `TypeError: float() argument must be…` ou, pire, conversion réussie mais silencieuse en chaîne ailleurs ; `data @ beta` → `TypeError: can't multiply sequence`. Aucun encodage catégoriel n'est fourni ni documenté (aucune mention dans les docstrings de milk/iris/ucbadmit que `data` n'est pas numérique).

**M‑6 — Majeure — `BayesForge/Resources/datasets.py:49‑68` — classe `load` : double rôle + masque du staticmethod `bf.load` + TypeError à l'usage racine.** [RÉSOLU 2906f8dd]
La classe s'appelle `load` (minuscule, non‑PEP8), charge des datasets **et** fait alias `bf.load` via `__call__` (65‑68, import différé de `BayesForge.Main.main`). Conséquences vérifiées : (a) `from BayesForge import load; load("x.pkl")` → `TypeError` (le constructeur 61 ne prend pas d'argument ; il faut `load()("x.pkl")`) alors que le nom exporté dans `__all__` suggère une fonction ; (b) `self.load = load()` (Main/main.py:249) masque `bf.load` (staticmethod, Main/main.py:1889) sur l'instance : `m.load` n'est plus la fonction de classe (plus de `__func__`, `inspect.signature` différent, `BF.load` ≠ `m.load`), et `m.load()` **sans argument** retombe sur `bf.load(None)` → charge silencieusement `./bf_model.pkl` **relatif au CWD** (Main/main.py:1889+) — lecture d'un fichier arbitre selon le répertoire courant ; (c) l'import différé 67 crée une dépendance circulaire package ↔ Main (neutralisée à l'exécution par l'ordre d'import de `__init__.py:22‑24`, mais `import BayesForge.Resources.datasets` isolé + appel de `__call__` déclenche tout le paquet). Le docstring de classe (50‑59) ne documente d'ailleurs pas du tout ce rôle `bf.load`.

**M‑7 — Majeure — `BayesForge/Resources/datasets.py:65‑68` — frontière de désérialisation non sécurisée exposée côté Resources.** [RÉSOLU 2906f8dd]
`load.__call__` délègue à `bf.load` qui fait `cloudpickle.load(f)` : exécution de code arbitraire pour tout `.pkl` non fiable ; le contrôle `isinstance(obj, bf)` intervient **après** désérialisation (Main/main.py:1889+), donc trop tard. Déjà signalé dans l'audit Main ; côté Resources, le point à consigner est que `load()(chemin)` **est** le point d'entrée public de cette frontière pour les utilisateurs des datasets, sans aucun avertissement dans le docstring (66) ni garde (chemin, extension, origine).

**m‑1 — Mineure — `datasets.py:93` vs `452`, `488`, `512` — type de retour de `only_path=True` incohérent : `str` (93, 376, 576…) vs `PosixPath` (452, 488, 512).** Vérifié à l'exécution. Scénario : `p = load().WaffleDivorce(only_path=True); p.startswith("/home")` → `AttributeError` (Path n'a pas `startswith`), alors que ça marche pour `howell1(only_path=True)`. [RÉSOLU 1a285825]

**m‑2 — Mineure — `datasets.py:374‑376` — `islands_dist_matrix` lit et parse le CSV **avant** de tester `only_path` : travail et allocation gaspillés pour une simple construction de chemin (et plantage inutile si le fichier est corrompu alors qu'on ne voulait que le chemin). [RÉSOLU 1a285825]

**m‑3 — Mineure — `datasets.py:53‑58` vs `99‑100` — le docstring de classe promet « Each method returns a Bunch object » ; en réalité le défaut `frame=True` retourne un simple DataFrame et **jette `description`** (DESCR inaccessible en mode défaut). Le contrat documenté est faux pour 28 des 31 chargeurs. [RÉSOLU 1a285825]

**m‑4 — Mineure — `datasets.py:9‑12` — exemple du docstring module : `from BayesForge.resources.data import datasets` — module inexistant (casse + `resources.data`). Un utilisateur qui copie-colle obtient `ModuleNotFoundError`. [RÉSOLU 1a285825]

**m‑5 — Mineure — `datasets.py:503‑510`, `479‑486` — métadonnées DESCRIPTIVES fausses : `elephants` décrit « experiments on elephant aggression … effect … on survival » alors que le fichier est AGE/MATINGS (données d'accouplement) ; `WaffleDivorce` « Data from experiments on waffle divorce » (observational, taux de divorce US par État, pas une expérimentation). Ces DESCR sont livrés dans les Bunch et propagés dans les notebooks. [RÉSOLU 1a285825]

**m‑6 — Mineure — `datasets.py:144‑173`, `268‑288` — NaN non documentés : `milk.csv` contient `NA` dans `neocortex.perc` (et chaînes dans clade/species), `chimpanzees.csv` a `recipient` entièrement `NA`. Aucun docstring ne mentionne les manquants → `milk(frame=False).data` contient `nan` silencieux ; une régression sur `neocortex.perc` sans `dropna` renvoie des coefficients biaisés. [RÉSOLU 1a285825]

**m‑7 — Mineure — paquet : fichiers morts ou dupliqués embarqués par package_data.** `Howell2.csv` (847 lignes, **aucun chargeur**, zéro référence dans le paquet) ; `KosterLeckie copy.rda` (fichier parasite avec « copy » dans le nom, dupliquant `KosterLeckie.rda`, tous deux référencés par zéro ligne de Python) ; `NBDA_data.json` (24,5 Mo, zéro référence `.py`) ; `phylo_L_meta/poisson/repeated/simple.csv` **byte-identiques** (MD5 `408a2b7a…` × 4, 266 Ko chacun) alors que leurs docstrings (573‑626) prétendent livrer « the Cholesky factor … for the Poisson model », « … for Meta-analysis », etc. — soit les modèles partagent la même phylogénie et les DESCR mentent, soit quatre matrices distinctes étaient attendues et trois sont fausses. Au minimum ~1 Mo + 24 Mo de poids de roue livrés. [RÉSOLU 1a285825]

**m‑8 — Mineure — `datasets.py:15‑16` — imports `os` et `json` jamais utilisés (vérifié par grep : aucune occurrence `os.`/`json.`). `import jax.numpy as jnp` (20) alourdit un module de données pour 6 usages (voir M‑3). [RÉSOLU 1a285825]

**m‑9 — Mineure — nomenclature API : classe `load` (minuscule), méthodes `WaffleDivorce`, `NBDA_events`, `NBDA_network` (PascalCase/accents d'acronyme) contre snake_case ailleurs (`howell1`, `ucbadmit`) — autocomplete et scripts génériques (getattr par nom) imprévisibles. Nom de fichier « Sim dat Gamma poisson.csv » (typo « dat », 407). [RÉSOLU 1a285825]

**m‑10 — Mineure — `datasets.py:97` — `pd.read_csv(filepath, sep=None, engine='python')` : (a) inférence du séparateur sur la **première ligne uniquement** — un CSV dont l'en-tête est ambigu (une seule colonne, ou `;` dans les données mais `,` en en-tête) est deviné de travers silencieusement ; (b) moteur python ~2‑5× plus lent que le moteur C sur `panama_ewa.csv` (1441×72) et `sim_gamma_poisson.csv` (3400 lignes) — anecdotique ici, mais le motif s'étend aux gros jeux utilisateurs ; (c) pas de paramètre `encoding` (UTF‑8 supposé — vérifié : tous les CSV actuels sont ASCII, donc pas de bug aujourd'hui, fragilité pour tout futur jeu à noms accentués) ni de `na_values` explicite (`NA` est géré par défaut, OK). [RÉSOLU 1a285825]

**m‑11 — Mineure — `datasets.py:384` — whitespace de fin sur `DESCR=description,            ` ; `datasets.py:449` — réaffectation de `self.data_dir` (voir M‑2). Cosmétique. [RÉSOLU 1a285825]

**Points sains vérifiés** (à consigner pour la release) : pas d'état global ni de mémoïsation — deux appels à `howell1()` retournent des objets distincts (vérifié `a1 is a2 == False`), donc pas de mutation partagée du DataFrame retourné ; `iris.csv` est bien standardisé (moyennes ≈ 0, écart-type populationnel = 1, le 1,0034 observé est l'effet ddof=1 de pandas — conforme au docstring 522‑525) ; `Howell1.csv` conforme au docstring (544×4, height/weight/age/male) ; `NBDA_event_data.csv`/`NBDA_edge_list.csv` conformes aux colonnes documentées (548, 560) ; `sim_nested_effects.csv` conforme (colonnes et DGP documentés 636‑649) ; package_data couvre bien `*.csv/*.rda/*.json` (pyproject.toml:69‑71) ; aucun réseau, aucun PRNG, aucun chemin absolu codé en dur ; `pathlib` utilisé partout (portable Windows).

---

#### (a) Résumé du module

`Resources/datasets.py` est un catalogue de 31 chargeurs de datasets de référence (Statistical Rethinking, phylogénie, NBDA, EWA) sur le modèle sklearn `Bunch`, avec un helper central `_load_csv` (détection automatique de séparateur, mode DataFrame ou Bunch, mode `only_path`). Le module est simple, sans état global, sans réseau, sans PRNG, et les données sont correctement embarquées par package_data. Mais il n'a visiblement **jamais été testé dans son mode par défaut** : les trois bugs critiques (trolley inexistant ; elephants et WaffleDivorce renvoyant un DataFrame **vide** à l'appel par défaut parce que leur lecture `index_col=0` maison ignore le séparateur `;` du fichier) sont tous détectables par un appel unique par chargeur. S'y ajoutent un fichier de données malformé (UCBadmit, 5 en-têtes pour 6 champs, parse « silencieux » dépendant du moteur python), un chargeur qui ignore son paramètre `frame` (sim_multivariate_normal), une rétroconversion float64→float32 silencieuse des facteurs de Cholesky phylogénétiques (`jnp.array` sans dtype, x64 JAX off), trois schémas de Bunch incompatibles et des tableaux `object` empoisonnés de chaînes en mode `frame=False`. La classe `load` cumule deux identités (catalogue de datasets + alias de `bf.load`) et son export racine (`from BayesForge import load`) est un piège à TypeError ; elle expose la désérialisation cloudpickle non sécurisée de Main. Verdict release : **bloquant** — C‑1/C‑2/C‑3 et M‑1/M‑3 doivent être corrigés avant publication.

#### (b) Fragilités architecturales

1. **Contrat de retour triple et non écrit** : DataFrame nu (défaut, DESCR jeté), Bunch à 6 clés, Bunch à 2 clés (`columns`), Bunch à 5 clés sans `frame` — plus un chargeur qui ignore `frame`. Aucun type de retour commun, pas de `TypedDict`/dataclass, pas de test de schéma : chaque chargeur est une petite île. Une fabrique unique `_load(name)` avec schéma figé éliminerait C‑2, C‑3, M‑2, M‑4, m‑1, m‑3 d'un coup.
2. **Doublon de la classe `load` / fonction `bf.load`** : le nom `load` désigne une classe de datasets exportée dans `__all__`, un staticmethod de `bf`, et un attribut d'instance qui masque ce dernier (Main:249). La frontière devrait être : `Resources` ne connaît que les datasets ; le rechargement `.pkl` reste dans `bf.load`, éventuellement exposé par une fonction module `load_session(path)`.
3. **Les données sont du code non testé** : en-têtes malformés (UCBadmit), fichiers manquants (Trolley), quadruplés (phylo_L ×4), orphelins (Howell2, KosterLeckie copy.rda, NBDA_data.json 24 Mo) — rien dans la CI ne valide « chaque chargeur public × {frame=True, frame=False} retourne le schéma documenté avec les colonnes documentées ». C'est LE test manquant (il aurait attrapé les 3 critiques).
4. **`_load_csv` comme unique chemin vs lectures à la main** : elephants/WaffleDivorce/islands/sim_mvnormal/phylo_L réimplémentent la lecture hors du helper (index_col, sep, types divergents). La règle « tout passe par `_load_csv` » (avec un paramètre `index_col`) aurait empêché C‑2/C‑3.
5. **JAX au cœur d'un module de données** (import module-level, DeviceArray float32 dans les Bunch) : couplage d'abord, perte de précision ensuite ; `Resources` ne devrait retourner que du numpy/float64 et laisser le conversion de dtype au pipeline.
6. **Sécurité héritée** : `load.__call__` = façade publique vers `cloudpickle.load` sans garde ni mention du risque ; à documenter (et à terme restreindre) comme déjà porté dans l'audit Main.

#### (c) Décompte par sévérité

| Sévérité | Nombre | Repères |
|---|---|---|
| **Critique** | **3** | C‑1 trolley() FileNotFoundError (datasets.py:241) ; C‑2 elephants() → DataFrame vide (514) ; C‑3 WaffleDivorce() → DataFrame vide (490) |
| **Majeure** | **7** | M‑1 UCBadmit.csv malformé, colonne perdue (266) ; M‑2 sim_multivariate_normal ignore `frame` (449‑468) ; M‑3 float32 via jnp (381, 578, 590, 602, 614, 626) ; M‑4 schéma Bunch `columns`/sans DESCR (578 ss.) ; M‑5 data dtype=object (106) ; M‑6 classe load : TypeError racine + masque bf.load (49‑68, Main:249/1889) ; M‑7 frontière cloudpickle non sécurisée (65‑68) |
| **Mineure** | **11** | m‑1 only_path str/Path ; m‑2 islands lit avant only_path ; m‑3 docstring « always Bunch » faux ; m‑4 exemple module chemin inexistant ; m‑5 DESCR faux elephants/WaffleDivorce ; m‑6 NaN non documentés ; m‑7 fichiers morts/dupliqués embarqués ; m‑8 imports morts os/json ; m‑9 nomenclature + typo « Sim dat » ; m‑10 sep inféré sur 1ʳᵉ ligne / moteur python / pas d'encoding ; m‑11 cosmétique (384, 449) |
| **Total** | **21** | |

---

### Module ML

**Audit de release — `BayesForge/ML/`** (racine `/home/sosa/work/BF`)
Fichiers audités ligne par ligne : `KMeans.py` (253 l.), `ml.py` (105 l.), `__init__.py` (vide).
Contrat Main vérifié : `Main/main.py:71` (import), `:244` (`self.ml = ml()`), `:564-569` (auto-init `initial_means` du GMM via `self.ml.KMEANS(data, n_clusters=K)` puis lecture `self.ml.results["centroids"]`). Autre consommateur : `Test/11.MGM and DPMM/cases/data/make_data.py:11-12`.

---

## 1. Contrat public et flux réel

- `ml.KMEANS(X, n_clusters, n_iterations=100, random_state=None)` (`ml.py:81`) construit un `JAXKMeans` (`ml.py:103`) dont **le constructeur lance déjà `fit`** (`KMeans.py:52`), puis copie la référence du dictionnaire `results` du modèle (`ml.py:104`).
- Le chemin utilisé par Main (`main.py:566-569`) et par le test `make_data.py` est donc : `KMEANS(...)` → lecture `ml.results["centroids"]`. Ce chemin **fonctionne** et est **déterministe** (voir C1 pour pourquoi c'est accidentel).
- Le chemin documenté `handler.KMEANS(X, n_clusters=3).fit()` (`ml.py:18`) est **cassé** (voir C2).
- Clés de `results` : `"centroids"`, `"labels"`, `"inertia"` (`KMeans.py:126-130`) — cohérent avec la lecture Main `results["centroids"]` (`main.py:569`). ✓

---

## 2. Bugs et fragilités

### CRITIQUE

**C1 — `random_state` totalement ignoré : les deux branches créent la même clé** — `BayesForge/ML/KMeans.py:44-47` `[RÉSOLU b4718662]`
```python
if random_state is not None:
    self.key = random.PRNGKey(0)
else:
    self.key = jax.random.PRNGKey(0)
```
`random` est `jax.random` (import L3) : les deux branches sont **strictement identiques** (`PRNGKey(0)`). La valeur de `random_state` n'est jamais utilisée (stockée L36, relue nulle part).
- Risque 1 (scientific) : une étude de sensibilité à la graine (`random_state=1,2,…,50`) produit **50 runs bit-à-bit identiques** ; l'utilisateur croit balayer l'aléa, il ne mesure rien. C'est le pire mode de défaillance : silencieux et faux.
- Risque 2 (API) : la docstring (`KMeans.py:28-30`) promet « Use an int to make the randomness deterministic » — en réalité tout est toujours déterministe avec la graine 0, et aucun seed n'est jamais honoré. Le fix évident (brancher `PRNGKey(random_state)` dans le `if`) n'a jamais été écrit.
- Note : le projet a le pattern inverse connu (`seed=True → time.time_ns()`) ; ici c'est l'excès contraire — graine figée. Pour le contrat GMM (`main.py:566`, `random_state` non passé) cela rend `initial_means` reproductible **par accident**, pas par conception.
- Déclencheur : `JAXKMeans(X, 3, random_state=1)` vs `JAXKMeans(X, 3, random_state=2)` → centroïdes identiques.

**C2 — `ml.fit()` lève toujours un `TypeError` : API fluent documentée morte** — `BayesForge/ML/ml.py:49` + `BayesForge/ML/KMeans.py:54` `[RÉSOLU b4718662]`
`ml.fit` appelle `self.model.fit(self.X)` (L49), mais `JAXKMeans.fit(self, X, n_clusters, n_iterations=100)` (`KMeans.py:54`) exige `n_clusters` en positionnel **sans défaut**.
- Déclencheur : l'exemple même de la docstring du handler (`ml.py:18`) `handler.KMEANS(X, n_clusters=3).fit()` → `TypeError: fit() missing 1 required positional argument: 'n_clusters'`.
- Conséquence release : une méthode publique documentée est inutilisable en l'état ; seul le chemin détourné (fit dans le constructeur, C4/M4) fonctionne. Pire, `ml.fit` ne sert à rien puisque `KMEANS` a déjà fit — code mort cassé livré.

### MAJEUR

**M1 — Initialisation aléatoire uniforme, pas de k-means++, pas de `n_init`** — `BayesForge/ML/KMeans.py:74` [RÉSOLU f80f5f12]
`random.choice(self.key, X_jnp, shape=(n_clusters,), replace=False)` tire k points uniformément (initialisation « random » à l'ancienne de sklearn, abandonnée par sklearn depuis la v0.20 au profit de k-means++ par défaut), et un **seul run** (sklearn : `n_init=10`).
- Scénario concret : le contrat Main (`main.py:564-569`) s'en sert pour initialiser les `initial_means` d'un GMM. Un tirage aléatoire peut mettre deux centroïdes dans le même blob → GMM initialisé dans un mauvais bassin → chaînes MCMC vers un optimum local.
- **Mesuré** (géométrie de production du test `make_data.py` : 500 points, 8 centres, std 0.8, box (−10,10), seed 101) : avec la clé 0 figée, BF trouve l'optimum (inertia 677.580 = k-means++ `n_init=10` de sklearn) — mais un `KMeans(n_init=1)` à init aléatoire de sklearn, balayées sur 50 graines, donne min 677.6 / médiane 677.6 / **max 1342.5** : le pire tirage est **2× pire**. La chance de la clé 0 sur ce cas précis ne généralise pas.
- Aggravé par C1 : graine figée à 0, **tout le monde reçoit le même init**, et il est impossible de s'en échapper par `random_state` pour retenter.

**M2 — Matérialisation (N, K, D) à chaque itération** — `BayesForge/ML/KMeans.py:88` (aussi `:117`, `:150`, `:229`) [RÉSOLU f80f5f12]
`jnp.sum((X_jnp[:, None, :] - centroids[None, :, :])**2, axis=2)` alloue un temporaire de N·K·D flottants par itération. Numériquement c'est la forme **sûre** (pas de terme croisé `-2X@C.T` qui peut devenir négatif — bon point), mais mémoire O(N·K·D) : N=100 000, K=50, D=100 → ~2 Go de temporaire par itération de la boucle `fori_loop` (L114). L'identité `‖x‖² − 2x·c + ‖c‖²` (avec clip à 0) donne O(N·K). Axes corrects au demeurant : `argmin(axis=1)` sur (N,K) = par échantillon ✓, `one_hot` → (N,K) ✓, `one_hot.T @ X` → (K,D) ✓, `counts` somme axis=0 ✓.

**M3 — Aucune validation d'entrée : NaN silencieux, 1D/k=0/k>n/X vide en erreurs obscures** — `BayesForge/ML/KMeans.py:70-71`, `:74`, `:88` [RÉSOLU f80f5f12]
- **NaN dans X (mesuré)** : `jnp.asarray` (L70) passe sans contrôle (aucun équivalent du `check_finite=True` de sklearn), un **seul** `NaN` dans X suffit à produire `inertia_ = NaN` et des centroïdes contaminés **silencieusement** (mesuré : les 3 centroïdes NaN sur un cas 3 blobs 900×2 ; contamination partielle sur un cas 2 blobs). Mécanisme aggravant mesuré : `jnp.argmin` de JAX arbitre NaN comme **minimum** (`jnp.argmin([nan,1,2]) → 0`), donc les points sont aspirés vers tout centroïde contaminé. Via Main (`main.py:569`), un GMM reçoit alors `initial_means` = NaN → trace MCMC NaN, sans aucune erreur émise.
- **X 1D (N,) (mesuré)** : `n_samples, n_features = X_jnp.shape` (L71) → `ValueError: not enough values to unpack (expected 2, got 1)`. Un GMM univarié passé par `main.py:566` plante avec ce message, sans que l'utilisateur fasse le lien avec k-means.
- **k=0 (mesuré)** : `IndexError: Too many indices: array is 1-dimensional, but 2 were indexed` — crash obscène, aucun message du domaine.
- **k>n (mesuré)** : `ValueError: Cannot take a larger sample (size 5) than population (size 3) when 'replace=False'` — explicite côté JAX mais non traduit depuis `bf.fit`.
- **X vide (0,D) (mesuré)** : `ValueError: a must be greater than 0 unless no samples are taken`.
- Aucune de ces erreurs n'est émise **tôt** avec un message du domaine (« n_clusters doit être dans [1, n_samples] », « X doit être 2D », « X contient des NaN »).

**M4 — `fit` appelé dans le constructeur + `fit` ignore l'état de l'instance** — `BayesForge/ML/KMeans.py:52`, `:54`, `:74`, `:114` ; `ml.py:104` [RÉSOLU f80f5f12]
- Le constructeur (L52) force un fit immédiat : impossible de construire un estimateur non ajusté (contraire du contrat sklearn), et `ml.KMEANS` + `ml.fit` font un double-fit conceptuel (le second étant cassé, C2).
- `fit` utilise ses **arguments** (`n_clusters` L74, `n_iterations` L114) et non `self.n_clusters`/`self.n_iterations` (stockés L34-35) : après `JAXKMeans(X,3).fit(X2,5)`, `self.n_clusters` vaut 3 alors que `centroids_` a 5 lignes — attributs d'instance menteurs.
- `self.results` du modèle est **aliasé** (pas copié) dans `ml.results` (`ml.py:104`, et `ml.py:50`) : tout re-fit du modèle mute rétroactivement le `results` du handler.
- `self.key` n'est jamais `split` : deux `fit` successifs réutilisent la même clé → mêmes centroïdes initiaux (lié à C1).

### MINEUR

**m1 — Pas de critère de convergence ni de `tol`** [RÉSOLU 0f68f118] — `KMeans.py:114` : `fori_loop(0, n_iterations, ...)` déroule toujours 100 itérations, sans test de stabilité des affectations/centroïdes. Pas faux (une fois convergé, l'itération est idempotente, et les centroïdes vides sont figés L106-110), mais gaspille ~10× le calcul nécessaire et rend le résultat dépendant d'un `n_iterations` arbitraire sans avertissement de non-convergence.

**m2 — Code mort et instabilité de types dans `predict`** [RÉSOLU 0f68f118] — `KMeans.py:148` : `centroids_jnp = jnp.asarray(self.centroids_)` calculé puis **jamais utilisé** (L150 utilise `self.centroids_` directement). `self.predictions` initialisé `{}` (dictionnaire, L50) puis réassigné en `Array` (L152) : le même attribut change de type selon le cycle de vie.

**m3 — Robustesse de `plot`** [RÉSOLU 0f68f118] — `KMeans.py:176` : `X.shape[1]` sur un `ArrayLike` liste → `AttributeError` (pas de `jnp.asarray` contrairement à `fit`/`predict`). `:183` : `plt.style.use('seaborn-v0_8-whitegrid')` exige matplotlib ≥ 3.6 ; sur mpl ancien lève `OSError`/`KeyError` **non capturé** (seul `ImportError` l'est L184). `:222` : pas de grille figé à 0.02 → sur données de grande étendue (ex. features en milliers), la grille `arange` explose en mémoire. `:229-241` : le contour tracé est le champ de **distance minimale**, pas les frontières de décision (titre L245 trompeur). Aucun contrôle que le `X` passé est celui du fit → couleurs `labels_` potentiellement décalées (L197-206).

**m4 — Rétrogradation float32 silencieuse** [RÉSOLU 0f68f118] — `KMeans.py:70` : `jnp.asarray` sur une entrée float64 ampute en float32 (`jax_enable_x64=False` par défaut, cohérent avec le reste du projet, cf. `main.py:562` `bit="32"`) sans avertissement ; `inertia_` (L121) et les `counts` float32 (L102) perdent la précision entière au-delà de 2^24 points ; les `initial_means` injectés au GMM sont float32.

**m5 — Docstrings fausses** [RÉSOLU 0f68f118] — `KMeans.py:62-63` : documente un paramètre `y` inexistant ; `:66-68` : promet « Returns self » alors que `fit` ne retourne rien (`None`). `ml.py:17` : l'exemple utilise `ModelHandler()`, classe qui n'existe pas (la classe s'appelle `ml`, `ml.py:9`) ; `ml.py:42-43` : `fit` annoncé `-> Array` retourne un dict.

**m6 — État mort / incohérent du handler** [RÉSOLU 0f68f118] — `ml.py:26` `self.y` et `:29` `self.model_params` ne sont jamais écrits ni lus ; `ml.py:28` `self.predictions` reste `None` même après `predict` (`ml.py:70` ne le remplit pas, seul le modèle interne le fait, `KMeans.py:152`).

**m7 — Hygiène** [RÉSOLU 0f68f118] — imports inutilisés : `Union` (`KMeans.py:12`, `ml.py:4`), `Array`/`Optional` partiellement (`ml.py:3-4` : `Optional` sert L81, `Array` sert en hints) ; `matplotlib.pyplot` importé au niveau module (`KMeans.py:6`, `ml.py:6`) — coût d'import pour tout utilisateur du package, y compris headless.

**m8 — Nommage** [RÉSOLU 0f68f118] — la classe `ml` (`ml.py:9`) masque le nom du module `BayesForge.ML.ml` (`from BayesForge.ML.ml import ml`, `main.py:71` fonctionne mais est déroutant) ; classes en minuscules (`ml`) vs PEP8 ; `JAXKMeans.__init__` prend `X` en premier argument (non-sklearn).

### Points sains (à créditer, validés empiriquement §3 #10-12)
- Gestion des clusters vides **correcte et validée** (`KMeans.py:106-110`) : le `0/0 → NaN` de L108 est bien masqué par `jnp.where` (qui sélectionne élément par élément, le NaN non sélectionné ne se propage pas) ; le centroïde d'un cluster vide reste en place, pas de NaN dans `centroids_` (mesuré #10).
- Axes de toutes les réductions corrects (voir M2) ; `inertia_` = somme des distances au carré minimales, définition sklearn-conforme (L121, mesurée à 2·10⁻⁵ près #12) ; labels finaux recalculés depuis les centroïdes finaux (L117-118) ✓.
- Aucun `np.random` global, pas de seed temporelle (l'inverse du pattern `seed=True → time.time_ns()` du projet) — reproductibilité acquise, mais uniquement grâce au bug C1.
- Données constantes (tous points identiques) : pas de NaN, inertia 0, clusters vides gérés (mesuré #11) ✓. k = n : chaque point son centroïde ✓.

---

## 3. Vérifications empiriques

Exécutées dans l'environnement du projet (`/home/sosa/work/BF`, JAX 0.11.2, CPU, `jax_enable_x64=False`, bf v0.0.60) :

| # | Test | Résultat mesuré | Verdict |
|---|---|---|---|
| 1 | `JAXKMeans(X,2,random_state=1)` vs `=2` vs `=None` | centroïdes **bit-à-bit identiques** | **C1 confirmé** |
| 2 | `ml().KMEANS(X,2).fit(X)` (exemple docstring `ml.py:18`) | `TypeError: JAXKMeans.fit() missing 1 required positional argument: 'n_clusters'` | **C2 confirmé** |
| 3 | X float64 en entrée | `centroids_.dtype = float32` | rétrogradation silencieuse (m4) confirmée |
| 4 | Type de `self.predictions` avant/après `predict` | `dict` → `ArrayImpl` | m2 confirmé |
| 5 | X 1D (50,) | `ValueError: not enough values to unpack (expected 2, got 1)` | M3 confirmé |
| 6 | Un seul NaN dans X | `inertia = nan`, centroïdes contaminés (3/3 NaN sur cas 3 blobs) | M3 confirmé |
| 7 | k=0 | `IndexError: Too many indices: array is 1-dimensional, but 2 were indexed` | M3 confirmé |
| 8 | k=5 > n=3 | `ValueError: Cannot take a larger sample (size 5) than population (size 3)...` | M3 confirmé |
| 9 | X vide (0,2) | `ValueError: a must be greater than 0 unless no samples are taken` | M3 confirmé |
| 10 | Cluster vide (1 point isolé, k=3) | centroïdes tous finis, 3 labels présents — **pas de fuite NaN** | gestion L106-110 **validée** |
| 11 | Données constantes (20×2 à 1.0), k=3 | `inertia=0.0`, centroïdes finis | comportement sain validé |
| 12 | 2 blobs séparés : inertia BF vs sklearn | 199.64011 vs 199.64013 | noyau algo **juste** |
| 13 | Géométrie de production (`make_data.py` : 500 pts, 8 centres) | BF (clé 0) 677.580 = sklearn kpp `n_init=10` 677.580 ; mais `n_init=1` aléatoire sur 50 graines : min 677.6 / médiane 677.6 / **max 1342.5** | M1 : juste sur ce cas, non protégé en général |
| 14 | `jnp.argmin([nan,1,2])` | `0` (NaN arbitré comme minimum) | mécanisme de la contamination M3 |

---

## 4. Résumé du module

`BayesForge/ML/` est un mini-module de clustering à deux étages : `JAXKMeans` (k-means JAX maison, 253 l., avec un `plot` sophistiqué) et `ml` (handler état « fluent » de 105 l. qui n'enveloppe que KMeans — **pas** de réduction de dimension ni de classification : la réduction PCA est ailleurs, gérée dans `main.py:571+` via `models.pca`, hors de ce module). Son unique raison d'être fonctionnelle dans le produit est d'initialiser les `initial_means` des GMM (`main.py:564-569`) et d'exposer `ml.results["centroids"]`. Le noyau algorithmique est **juste** (affectation, mise à jour, clusters vides, inertia : tout est correct et bien vectorisé sur les bons axes), mais l'enveloppe API est **peu fiable** : graine aléatoire factice, méthode `fit` du handler cassée, aucune validation d'entrée, initialisation de qualité médiocre pour le rôle critique qu'on lui demande (amorçage de GMM bayésien).

## 5. Fragilités architecturales

1. **Contrat sklearn rompu sur trois points** (constructeur qui ajuste, `fit` qui exige `n_clusters`, `fit` qui ne retourne pas `self`) : le module ne peut pas être substitué par sklearn ni l'inverse, et sa docstring promet le contrat qu'elle viole.
2. **Double responsabilité fit** (constructeur + méthode) sans autorité : `self.n_clusters`/`self.n_iterations` ne pilotent rien, l'état d'instance est décoratif — source directe de C2 et M4.
3. **Aliénation de l'état partagé** : `ml.results` alias de `model.results`, `predictions` changeant de type, handler mono-modèle sans cycle de vie clair — deux appels `KMEANS` successifs sur un même `bf` (`self.ml` unique, `main.py:244`) remplacent silencieusement le modèle.
4. **Le paramètre de reproductibilité est structurellement cassé** (C1) : dans un outil scientifique où la graine est une donnée de publication, c'est une faille d'intégrité, pas un détail d'API.
5. **Aucune frontière de validation** entre l'utilisateur et JAX : toutes les erreurs arrivent tard, en messages JAX internes (unpack, choice, argmin), y compris via le chemin grand public `bf.fit(gmm)`.

## 6. Décompte par sévérité

| Sévérité | Nombre | Repères |
|---|---|---|
| Critique | 2 | C1 (KMeans.py:44-47), C2 (ml.py:49 + KMeans.py:54) |
| Majeure | 4 | M1 (KMeans.py:74), M2 (KMeans.py:88), M3 (KMeans.py:70-71), M4 (KMeans.py:52,54) |
| Mineure | 8 | m1–m8 |
| **Total** | **14** | |

**Verdict de release : bloquant.** C1 et C2 doivent être corrigés avant publication ; M1 (k-means++ ou `n_init`) est fortement recommandé étant donné le rôle d'amorçage GMM.

---

### Module Data

**Audit de release — `BayesForge/Data/manip.py` (299 lignes)**
Environnement de vérification : pandas 3.0.6, numpy 2.3.5, jax 0.11.2 (`jax_enable_x64=False`). Toutes les affirmations marquées ✅ ont été reproduites par exécution en lecture seule.

---

## 0. Contrat réel du module

`manip` n'est **pas du code mort** : c'est la base d'héritage de la classe publique `bf`
(`BayesForge/Main/main.py:122` — `class bf(manip)`, initialisation via `manip.__init__(self)` à `main.py:163`) et il est aussi importé par `BayesForge/Samplers/mcmc_numpyro.py:5`. Le dictionnaire `m.data_on_model` produit par `pd_to_jax`/`data_to_model` est consommé directement par `fit()`/`svi()` (`main.py:558-562`, `774-775`, `857-858`, puis `**self.data_on_model` à `main.py:708` et `830`).

Qualification à la lecture : module de **pré-traitement tabulaire** — chargement (`data` L33), encodages catégoriels (`OHE` L64, `index` L90), standardisation (`scale` L142, `scale_var` L131, `z_score` L170), conversions de dtype (`to_float` L194, `to_int` L218) et pont pandas→JAX (`pd_to_jax` L240, `data_to_model` L285). Pas de PRNG, pas d'imputation, pas de binning.

**Consommateurs documentés** (Documentation/*.qmd) : `m.scale([...])` (liste — fonctionne), `m.index([...])`, `m.z_score`, mais aussi `m.scale(data='all')` (`reference/data.qmd:113`), `m.scale_var(x)` (`reference/data.qmd:129-132`), `m.to_float(cols='all')` (`:148`), `m.to_int(cols='all')` (`:168`) — **les quatre derniers sont cassés**, voir C1–C5. La référence publiée promet donc des API qui lèvent une exception.

---

## 1. Bugs et fragilités

### CRITIQUE

**C1 — `scale()` avec la valeur par défaut documentée `'all'` lève `TypeError` ; la branche `'all'` est du code mort** ✅ `[RÉSOLU 84f84c87]`
`BayesForge/Data/manip.py:152-153` : `if type(data) == str: return self.scale_var(data)`. Le défaut `data='all'` (L142) est une chaîne → il part dans `scale_var('all')` au lieu de la branche `data == 'all'` de L155, **définitivement inatteignable**. `m.scale()` ou `m.scale('all')` (publié dans `reference/data.qmd:113`) plante avec `TypeError: Error interpreting argument … as an abstract array` (voir C2). Et même si L155 était atteignable, L158 appelle `Series.sd()` qui **n'existe pas** dans pandas (`hasattr(pd.Series([1.]), 'sd') == False` ✅) → `AttributeError`. Triple défaut : branche morte, méthode inexistante, chemin par défaut cassé.

**C2 — `scale_var` est inutilisable avec n'importe quelle entrée : `@jax.jit` sur une méthode trace `self`** ✅ `[RÉSOLU 84f84c87]`
`BayesForge/Data/manip.py:130-131` : le décorateur est appliqué à la fonction non liée, donc à l'appel `m.scale_var(x)`, JAX reçoit `(self, x)` et échoue sur `self` : `TypeError: … problematic value is of type <class 'BayesForge.Data.manip.manip'> … passed at path self`. `m.scale_var(np.array([1.,2.,3.]))` lève systématiquement. La méthode est documentée dans la référence (`data.qmd:129`) et appelée par `scale` (L153) : **aucun appelant ne peut l'utiliser**. De plus, jitter une normalisation pandas→JAX ponctuelle n'apporte rien (recompilation par forme, L130).

**C3 — `OHE()` duplique toutes les colonnes non encodées dans `m.df` : corruption structurelle de la table** ✅ `[RÉSOLU 84f84c87]`
`BayesForge/Data/manip.py:76/80` : `pd.get_dummies(self.df, columns=colCat)` renvoie **un cadre complet** (colonnes numériques conservées, colonnes catégorielles remplacées par les dummies). Puis L86 `self.df = pd.concat([self.df, OHE], axis=1)` recolle ce cadre complet à côté de l'original → toute colonne numérique apparaît **deux fois**. Reproduit : `df = {y, x, sex}` → `m.df.columns == ['y','x','sex','y','x','sex_f','sex_m']`. Conséquences en cascade ✅ :
- `m.scale(['y'])` après `OHE` → `ValueError: Setting with non-unique columns is not allowed` (L163) ;
- `m.pd_to_jax` sur un modèle dont un argument est une colonne dupliquée → `AttributeError: 'DataFrame' object has no attribute 'dtype'` (L263, `self.df[arg]` renvoie un DataFrame) ;
- `m.index()` après `OHE` → `self.df[col]` (L113) renvoie un DataFrame → `AttributeError` sur `.cat`.
Le flux documenté « OHE puis scale » (`reference/data.qmd`, tutoriel 4) est donc empoisonné dès l'appel à `OHE`. L88 retourne de surcroît le cadre complet, pas « les nouvelles colonnes » comme l'annonce la docstring (L71).

**C4 — `to_int()` est cassé dans ses deux branches : indexation positionnelle avec des labels** ✅ `[RÉSOLU 84f84c87]`
`BayesForge/Data/manip.py:231` : `self.df.iloc[:, cols] = self.df.iloc[:, col].astype(type)` — dans la branche `'all'`, `cols` vaut la chaîne `'all'` et `col` est un **label**, passés à `iloc` → `ValueError: Location based indexing can only have [integer…]`. `BayesForge/Data/manip.py:235` (branche liste) : `self.df.iloc[:, cols[a]]` — même erreur avec le label de colonne. Reproduit pour `m.to_int()` et `m.to_int(['x'])` : les deux lèvent. La méthode est publiée (`data.qmd:168`). Aucun chemin d'appel ne fonctionne.

**C5 — `to_float()` : branche `'all'` qui lève, branche liste qui ne fait **rien** silencieusement (pandas ≥ 3)** ✅ `[RÉSOLU 84f84c87]`
`BayesForge/Data/manip.py:207` : RHS `self.df.iloc[:, col]` avec `col` label → `ValueError` identique à C4 pour `m.to_float()`. Branche liste L211 : `self.df.loc[:, cols[a]] = self.df.loc[:, cols[a]].astype(type)` — sous pandas 3.0, un setitem `.loc` sur une colonne existante **refond le dtype de la valeur dans le dtype de la colonne** : `m.to_float(['x'], 'float32')` sur une colonne int64 laisse `dtype('int64')`, et float64→float32 laisse `float64` ✅. Pire, L214 consigne `data_modification['float'] = cols` comme si la conversion avait eu lieu : **succès silencieux mensonger**, aucun avertissement.

**C6 — `pd_to_jax` retourne une chaîne d'erreur au lieu de lever ; `fit()` l'affecte à `data_on_model` et le plantage surgit très loin de sa cause** ✅ `[RÉSOLU 84f84c87]`
`BayesForge/Data/manip.py:270` : `return "Error, no data found"` si une colonne requise du modèle manque dans `m.df`. `main.py:560` fait `self.data_on_model = self.pd_to_jax(self.model)` sans contrôle, puis `main.py:708` `**self.data_on_model` → `TypeError: argument after ** must be a mapping, not str` dans le sampler, sans lien avec le vrai problème (nom de colonne absent). Un `raise KeyError(col)` à L259 s'impose.

**C7 — `pd_to_jax` plante sur tout argument par défaut array-like** ✅ `[RÉSOLU 84f84c87]`
`BayesForge/Data/manip.py:254` : `if param.default == inspect.Parameter.empty` — si un argument a une valeur par défaut `np.ndarray`, la comparaison renvoie un tableau et `if` lève `ValueError: The truth value of an array with more than one element is ambiguous`. Reproduit avec `def model(x, w=np.ones(3))`. Le test correct est `param.default is inspect.Parameter.empty`. De même, un défaut `None` fait planter L281 (`jnp.array(None)` → `ValueError: None is not a valid value for jnp.array` ✅) et un défaut `str` lèverait aussi ; les défauts non scalaires ne devraient pas être convertis en tenseurs.

**C8 — `index()` encode les NaN en code `-1` : indexation silencieusement fausse dans les modèles à groupes** ✅ `[RÉSOLU 84f84c87]`
`BayesForge/Data/manip.py:111` et `118` : `.astype("category").cat.codes` utilise **`-1` pour les valeurs manquantes**. Reproduit : `['a','b',None,'a']` → `index_g == [0, 1, -1, 0]` avec `index_map = {0:'a', 1:'b'}` (le `-1` n'est dans aucun map). En aval, un modèle numpyro/JAX qui fait `mu[group_idx]` résout `-1` comme **dernier groupe** (sémantique Python/JAX) : l'observation au NaN est affectée silencieusement à la mauvaise catégorie — postérieur faux, aucun message. `index()` devrait lever sur NaN ou créer une catégorie explicite.

### MAJEUR

**M1 — `scale` (branche liste) produit des NaN silencieux sur colonne constante ou N=1** ✅ `[RÉSOLU fe300cef]`
`BayesForge/Data/manip.py:163` : division par `.std()` sans garde. Colonne `[5.,5.,5.]` → `[nan, nan, nan]` dans `m.df`, consignée comme mise à l'échelle réussie (L166). En JAX, `z_score` (L189) protège correctement avec `jnp.where(std==0, 1.0, std)` — incohérence interne : le même calcul est protégé d'un côté du module et pas de l'autre. N=1 : `std` ddof=1 → NaN aussi.

**M2 — Incohérence mathématique `ddof` entre le chemin pandas et le chemin JAX** ✅ `[RÉSOLU fe300cef]`
`scale` (L158/L163) utilise `Series.std()` → **écart-type d'échantillon, ddof=1** ; `z_score` (L185) et `scale_var` (L140) utilisent `jnp.std` → **écart-type population, ddof=0**. Reproduit : `scale(['y'])` sur `[1,2,3,4]` → `[-1.162, -0.387, 0.387, 1.162]` (ddof=1) alors que `z_score` donnerait `[-1.342, -0.447, 0.447, 1.342]`. La doc du tutoriel 20 (`Measuring error.qmd:30-32`) mélange `m.scale` et `m.z_score` dans un même script : deux standardisations différentes coexistent silencieusement (facteur √(n/(n−1)) — négligeable pour n grand, visible pour n petit, et jamais documenté).

**M3 — `pd_to_jax` promet float64/int64 mais tronque silencieusement en 32 bits** ✅ `[RÉSOLU fe300cef]`
`BayesForge/Data/manip.py:19/265` : la map associe `'float64' → jnp.float64`, mais JAX tourne par défaut sans x64 : `jnp.array(col, dtype=jnp.float64)` est **tronqué en float32** avec un simple `UserWarning` (observé à l'exécution). Une colonne poids/prix en float64 perd sa précision sans que l'utilisateur ait demandé `bit='32'`. La map (L15-22) est un mensonge d'API tant que x64 n'est pas activé ; il faudrait mapper directement vers float32/int32 ou activer x64.

**M4 — `OHE` ignore le dtype `category` : les facteurs R (reticulate) ne sont pas encodés, silencieusement** ✅ `[RÉSOLU fe300cef]`
`BayesForge/Data/manip.py:75` : `select_dtypes(['object'])` — alors qu'un facteur R arrive comme pandas `Categorical` (dtype `category`), que `index()` traite explicitement (L109, avec commentaire L107-108). Reproduit : `OHE()` sur un `Categorical` ne crée **aucune dummy** et, à cause de C3, duplique toutes les colonnes (`['f','y','f','y']`). Le contrat « R users can pass their data directly » (docstring L39-41) est rompu pour OHE.

**M5 — `index()` renomme toutes les colonnes de `m.df` ; les appels suivants avec les noms originaux font `KeyError`** ✅ `[RÉSOLU fe300cef]`
`BayesForge/Data/manip.py:123-124` remplace `.` et espaces dans **tous** les noms de `m.df` (pas seulement les colonnes créées). Reproduit : colonne `my.col` → `my_col`, puis `m.scale(['my.col'])` → `KeyError: 'my.col'`. Incohérent avec `OHE` qui ne renomme que le cadre retourné (L82-83), laissant `m.df` avec un mélange de noms renommés/originaux après le concat L86. Effet aggravant : `data_modification['index']` (L126) garde les anciens noms, le journal de provenance ne correspond plus à la table.

**M6 — `pd_to_jax` : dtypes hors map → `None` implicite, comportement non spécifié** `[RÉSOLU fe300cef]`
`BayesForge/Data/manip.py:265/267` : `.get(varType)` retourne `None` pour `bool`, `int8`, `uint*`, `string`, `datetime64[ns]`, `category` (non listés L15-22) → `jnp.array(..., dtype=None)` infère. `bool`/`int8` s'en sortent ✅, mais `category` ou `datetime64` produiront une erreur obscure ou un objet invalide au lieu d'un message nommant la colonne et le dtype. Un `.get(varType)` à `None` devrait être une erreur explicite (ou un mapping complet).

### MINEUR

**m1 — `index_map` est écrasé à chaque appel** [RÉSOLU 0c36e593] — `manip.py:105` : `self.index_map = {}` en tête de `index()`. Deux appels successifs sur des colonnes différentes perdent le map du premier ✅ (`index(['g'])` puis `index(['h'])` → `index_map` ne contient plus que `'h'`). Le dictionnaire de décodage des catégories est un artefact utilisateur ; il devrait être initialisé dans `__init__` et mis à jour, pas réinitialisé.

**m2 — `data_modification` consigne `'all'` au lieu des colonnes résolues** [RÉSOLU 0c36e593] — `manip.py:87` et `126` : dans la branche `'all'`, la variable `cols` vaut encore `'all'` (les listes `colCat` calculées L75/L109 ne sont jamais stockées). Le journal de traçabilité censé rejouer/provenancer les transformations est inutilisable sur le chemin `'all'`.

**m3 — `to_int` retourne `None`, `to_float` retourne `self.df`** [RÉSOLU 0c36e593] — `manip.py:216` vs `226-238` : deux méthodes sœurs, deux contrats de retour. Chaînage `m.to_int(['x']).head()` impossible.

**m4 — `str.replace` sans `regex=False` explicite** [RÉSOLU 0c36e593] — `manip.py:82-83, 123-124` : correct sous pandas ≥ 2.0 (défaut littéral, vérifié ✅), mais sous pandas < 2.0 le défaut `regex=True` faisait de `'.'` un joker remplaçant **chaque caractère** des noms de colonnes. Portabilité à verrouiller par `regex=False` explicite.

**m5 — Paramètre `bit` : docstring fausse et concaténation naïve** [RÉSOLU 0c36e593] — `manip.py:240-245` : la doc annonce « default: 32 », le code a `bit=None`. L279 construit `type_name + bit` : `bit='64'` produirait `'float6464'` → `.get` → `None` → inférence silencieuse au lieu de 64 bits.

**m6 — `data_to_model` sans mapping de dtype ni garde de forme** [RÉSOLU 0c36e593] — `manip.py:297` : `jnp.array(self.df.loc[:, col].values)` ignore les maps de dtype ; int64 → int32 tronqué (warning seulement) ; après un `OHE` (C3), `loc[:, col]` sur colonne dupliquée renvoie un DataFrame → **tableau 2D silencieux** passé au modèle.

**m7 — NaN propagés sans un mot dans `z_score`/`scale_var`** [RÉSOLU 0c36e593] — `manip.py:140, 184-185` : un NaN dans la donnée contamine mean/std → sortie entièrement NaN, sans avertissement (contrairement à C8, au moins le NaN reste NaN et ne devient pas un index valide).

**m8 — `OHE` avec NaN : lignes tout-zéro non documentées** [RÉSOLU 0c36e593] — `manip.py:76` : `dummy_na` par défaut `False` ✅ ; une ligne à catégorie manquante a toutes ses dummies à 0 (et la colonne originale objet reste dans `m.df`, L86). Choix défendable mais jamais annoncé.

**m9 — `type` comme nom de paramètre** [RÉSOLU 0c36e593] — `manip.py:194, 218` : masque le builtin dans le scope des méthodes ; style.

**m10 — `Data/` sans `__init__.py`** [RÉSOLU 0c36e593] — 17 des 18 sous-packages de `BayesForge/` en ont un ; `Data/` (comme `Build/`, `R/`) fonctionne par package namespace implicite. Fonctionnel en py3, mais fragile au packaging (`find_packages()` classique l'omet) et incohérent avec le reste du dépôt.

**m11 — `@jax.jit` sans bénéfice** [RÉSOLU 0c36e593] — `manip.py:130` : sur un pré-traitement ponctuel, le jit ajoute recompilation par forme et tracing de `self` (C2) sans gain mesurable.

**m12 — Conversion `str(c)` des noms de colonnes en entrée** [RÉSOLU 0c36e593] — `manip.py:79, 116` : `[str(c) for c in cols]` casse les labels de colonnes non-strings (entiers, tuples) : `m.index([0])` sur colonne entière cherche `'0'` → `KeyError`.

---

## 2. Récapitulatif par défaut

| # | Ligne(s) | Défaut | Reproduit |
|---|----------|--------|-----------|
| C1 | 152-153, 155-158 | `scale('all')`/défaut → TypeError ; branche `'all'` morte ; `Series.sd()` inexistant | ✅ |
| C2 | 130 | `scale_var` : jit trace `self` → inutilisable | ✅ |
| C3 | 76, 80, 86 | `OHE` duplique toutes les colonnes non encodées dans `m.df` | ✅ |
| C4 | 231, 235 | `to_int` : `iloc` avec labels → cassé dans les 2 branches | ✅ |
| C5 | 207, 211 | `to_float` : `'all'` lève ; liste = no-op silencieux (pandas 3) | ✅ |
| C6 | 270 | `pd_to_jax` retourne une chaîne d'erreur → crash lointain dans `fit` | ✅ |
| C7 | 254, 281 | `pd_to_jax` : `==` ambigu sur défauts array ; défaut `None` → ValueError | ✅ |
| C8 | 111, 118 | `index` : NaN → code `-1` → indexation de groupe silencieusement fausse | ✅ |
| M1 | 163 | `scale` : std=0 ou N=1 → NaN silencieux | ✅ |
| M2 | 158/163 vs 140/185 | ddof=1 (pandas) vs ddof=0 (jnp) : deux standardisations incohérentes | ✅ |
| M3 | 19, 265 | float64→float32 tronqué silencieusement (x64 off) malgré la map | ✅ |
| M4 | 75 | `OHE` ignore dtype `category` (facteurs R) | ✅ |
| M5 | 123-124 | `index` renomme toute la table → KeyError sur les noms d'origine | ✅ |
| M6 | 265, 267 | dtypes hors map → `dtype=None` implicite | ✅ (bool/int8) |
| m1-m12 | voir ci-dessus | traçabilité, API, portabilité, packaging | partiellement ✅ |

---

## (a) Résumé du module

`manip` est le socle de pré-traitement de `bf` : chargement CSV/DataFrame/dict avec copie protectrice du données de l'appelant (L60, vérifié ✅ — le DataFrame de l'appelant n'est jamais modifié), encodage one-hot ou par indices 0-based, standardisation, conversions de dtype et conversion finale pandas→JAX pilotée par la signature du modèle (`inspect.signature`, L250). L'idée architecturale est saine et le chemin nominal documenté dans les tutoriels (`m.data(...)` → `m.scale([...])` → `m.index([...])` → `fit`) fonctionne pour des données numériques propres. **Mais le module est livrable ni robuste ni honnête** : sur les 9 méthodes publiques, 4 sont inutilisables en l'état (`scale_var`, `to_int`, et les branches `'all'` de `scale`/`to_float`), 1 corrompt la table de l'utilisateur (`OHE`), 1 encode silencieusement les NaN en index valides (`index`), et la plus critique pour le pipeline (`pd_to_jax`, appelée automatiquement par `fit`) retourne une chaîne d'erreur et plante sur des signatures de modèle parfaitement licites. Aucun PRNG n'est présent (reproductibilité : rien à signaler de ce côté).

## (b) Fragilités architecturales

1. **Boucle de validation absente entre `OHE`/`index` et le reste** : la concaténation L86 et le renommage L123 créent un état de `m.df` (colonnes dupliquées, noms modifiés) qui casse en cascade `scale`, `index`, `pd_to_jax`, `data_to_model`. Il manque un invariant « `m.df` a des colonnes uniques » vérifié à l'entrée de chaque méthode.
2. **Deux implémentations de la même mathématique** (pandas dans `scale`, JAX dans `scale_var`/`z_score`) avec des conventions différentes (ddof, garde std=0) : la source de vérité de la standardisation n'est pas définie.
3. **Erreurs comme valeurs de retour** (`pd_to_jax` L270 ; `to_float` no-op + `data_modification` mensonger L214) : le module préfère un résultat silencieusement faux à une exception précoce, alors qu'il est à 2 appels de distance d'un `**data_on_model` dans le sampler.
4. **Contrat de dtype non tenu** : les maps L15-30 promettent float64/int64 que JAX (x64 off) ne peut pas matérialiser ; `data_to_model` les ignore carrément.
5. **`@jax.jit` posé sur une méthode d'instance** sans conscience que `self` devient un argument tracé — signe que le chemin `scale_var` n'a jamais été testé, ce que corrobore la branche `'all'` morte de `scale` (C1) : **aucun test d'aucune de ces méthodes n'a pu passer** ; le module semble livré sans suite de tests.
6. **Provenance (`data_modification`) non fiable** : consigne `'all'` au lieu des colonnes résolues, consigne des conversions qui n'ont pas eu lieu (C5), invalidated par les renommages (M5).
7. `Data/` sans `__init__.py` là où 17/18 frères en ont un — cohérence de packaging.

## (c) Décompte par sévérité

| Sévérité | Nombre |
|----------|--------|
| **Critique** | **8** (C1–C8) |
| **Majeure** | **6** (M1–M6) |
| **Mineure** | **12** (m1–m12) |
| **Total** | **26** |

**Verdict de release : NON PRÊT.** C3 (corruption de `m.df` par `OHE`), C8 (NaN → index `-1` silencieusement faux dans les modèles à groupes) et C6 (chaîne d'erreur injectée dans `data_on_model`) sont des bugs de justitude scientifique, pas seulement des plantages : ils peuvent produire des postérieurs faux sans le dire. C1–C5 et C7 sont des plantages garantis sur des API publiées dans la référence. Minimum bloquant avant release : corriger C1–C8, retirer `scale_var`/`to_int`/branches `'all'` de la documentation tant qu'ils ne sont pas testés, et ajouter une suite de tests par méthode (le taux de succès actuel des API publiées est d'environ 4/9).

---

### Module SetDevice

**Périmètre** : `BayesForge/SetDevice/set.py` (199 l., lu intégralement), `BayesForge/SetDevice/__init__.py` (vide).
**Environnement de vérification** : JAX 0.11.2 / jaxlib CPU (machine avec GPU NVIDIA détecté mais sans jaxlib CUDA → fallback CPU), numpyro installé, tensorflow_probability **absent**, Python 3.14. Chaque test empirique exécuté dans un processus neuf, en lecture seule (aucune modification du code source).

---

## 1. Résumé du module

Le module expose trois fonctions :

- **`set_deallocate()`** (l. 5–7) : pose deux variables d'environnement XLA (`XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_ALLOCATOR=platform`).
- **`setup_device()`** (l. 9–109) : cœur du module. Valide `float_precision` (l. 65–70), pose `CUDA_VISIBLE_DEVICES` si `platform='gpu'` et `gpu_index` fourni (l. 72–74), résout `cores` via `BF_MAX_CORES`/`os.cpu_count()` avec cap et warning (l. 79–91), réécrit `XLA_FLAGS` avec `--xla_force_host_platform_device_count` (l. 97–99), **importe JAX** (l. 102), puis `jax.config.update("jax_platform_name", …)` et `jax_enable_x64` (l. 105–106), et finally force l'initialisation du backend via `jax.local_device_count(backend=None)` si `print_devices_found=True` (l. 109).
- **`setup_multi_device()`** (l. 112–195, alias `setup_multi_gpu` l. 199) : construit un Mesh 1-D `('device',)` sur `jax.devices(platform)`, tronqué à `cores`, renvoie `(mesh, data_sharding, rep_sharding)` ou `(None, None, None)` si < 2 devices.

**Chaîne d'appels réelle** (vérifiée) :
`BayesForge/__init__.py:22` → `Main/main.py` qui **importe `jax` et `jax.numpy` au niveau module** (main.py:7–8) et numpyro (main.py:11–13) → `BayesForge/__init__.py:23` exporte `setup_device`. Le constructeur `bf.__init__` appelle `setup_device(...)` (main.py:166–173) **à chaque instanciation**, puis `jax.devices(...)` (main.py:180–181) qui fige le backend, puis `setup_multi_device` si > 1 device (main.py:189–191). Pour le backend tfp, main.py:297 force `jax_enable_x64=False` **après** `setup_device`. `__setstate__` (main.py:1847–1858) rappelle `setup_multi_device` au chargement, avec `except RuntimeError` propre. `Parallel/simulate.py` documente (l. 12–14) et exploite (l. 851–853, spawn) la contrainte « XLA_FLAGS écrit une fois par processus », et pose `BF_MAX_CORES` par worker.

**Vérification empirique clé** (JAX 0.11.2) :

| Test | Résultat |
|---|---|
| `import BayesForge` → backend initialisé ? | **Non** (x64=False, platform='') → le **premier** `bf()` configure correctement |
| `jax.config.update("jax_enable_x64", …)` après init du backend | **Accepté ET effectif** (dtype float32→float64 vérifié) — pas de RuntimeError en 0.11.2 |
| `jax.config.update("jax_platform_name", "gpu")` après init | Accepté, `config.read` renvoie bien `'gpu'`, mais **`jax.devices()` reste CPU — no-op silencieux** |
| `jax_platform_name='bogus'` ou `'CPU'` avant init | `RuntimeError: Unknown backend CPU. Available backends are ['cpu']` — à l'**init**, pas à l'appel |
| `platform='gpu'` sans jaxlib CUDA / sans GPU | `RuntimeError: Unknown backend: 'gpu' requested…` à l'init (l. 109) |
| `jax.devices('gpu')` sans GPU | **RuntimeError**, pas une liste vide |
| `--xla_force_host_platform_device_count=0` | `RuntimeError: Unable to initialize backend 'cpu': INVALID_ARGUMENT: Could not determine process_index…` (message mensonger) |
| `cores=-1` | `RuntimeError … vector::reserve` (erreur C++ brute) |
| `cores=2.5` | XLA rejette le flag et **déverse son aide-mémoire complet** (centaines de lignes) |
| `BF_MAX_CORES='auto'` | `ValueError: invalid literal for int() with base 10: 'auto'` (ne nomme pas BF_MAX_CORES) |
| `bf(cores=2)` puis `bf(cores=4, float_precision=32)` | `n_devices` reste **2** alors que `XLA_FLAGS=…=4`, **aucun warning** ; float32, lui, est bien appliqué |
| `bf(platform='gpu')` après init CPU | affiche « jax.local_device_count 2 » (les devices **CPU** !) puis `RuntimeError` sur `jax.devices('gpu')` (main.py:180) |
| `setup_device(print_devices_found=False)` seul | backend **non initialisé**, config appliquée — seul usage propre du module |
| `jax.config.update('option_inconnue', …)` | `AttributeError: Unrecognized config option` → `jax_platform_name` est bien un flag **reconnu mais déprécié** (xla_bridge.py:73–77, « Deprecated, please use --jax_platforms instead ») |

---

## 2. Bugs et fragilités

### CRITIQUE

**C1 — Aucun garde-fou contre un backend déjà initialisé : toute 2ᵉ exécution de `setup_device` est un no-op silencieux partiel.** `[RÉSOLU 2928d611]`
`BayesForge/SetDevice/set.py:96-109` (et `main.py:166` qui l'appelle à chaque `bf()`).
Les quatre leviers du module sont lus **une seule fois par processus, à l'init du backend** : `XLA_FLAGS` (l. 99), `CUDA_VISIBLE_DEVICES` (l. 73), les flags `XLA_PYTHON_CLIENT_*` (l. 6–7) et `jax_platform_name` (l. 105, backend par défaut déjà mis en cache). Or `setup_device` ne teste jamais si le backend est déjà up (il existe `jax._src.xla_bridge.backends_are_initialized()`) et n'émet **aucun avertissement**. Le seul signal, l. 109, est le compteur de devices — qui affiche l'ancienne valeur.
*Scénario vérifié* : `m1 = bf(cores=2)` ; `m2 = bf(cores=4, float_precision=32)` → `XLA_FLAGS` vaut 4, `m2.n_devices` vaut **2**, rien n'est signalé ; l'utilisateur croit obtenir 4 devices et n'en a que 2 (chaînes MCMC et sharding deux fois plus lents, budgets `BF_MAX_CORES` faux). Pire : `m3 = bf(platform='gpu')` après un init CPU affiche « jax.local_device_count 2 » (devices CPU) puis plante dans `main.py:180` avec une erreur déconnectée de sa cause. La **reproductibilité dépend de l'ordre des appels `bf()` dans le processus** : le même script exécuté dans un ordre différent ou avec un `jnp.zeros(3)` ajouté en amont produit une configuration matérielle différente, silencieusement. Note : `float_precision`, lui, **est** effectif après init (vérifié) — le module est donc incohérent : certains paramètres changent à chaud, d'autres non, aucun n'est vérifié ni annoncé.

### MAJEURES

**M1 — `platform='gpu'` sans GPU : pas de fallback ni de validation précoce, message de succès mensonger avant le crash.** [RÉSOLU b4178df8]
`BayesForge/SetDevice/set.py:72-74, 105, 109`.
Aucune validation de `platform` (chaîne arbitraire, casse sensible : `'CPU'` → `RuntimeError: Unknown backend CPU`) ni sondage de disponibilité. Sur machine sans jaxlib CUDA, `bf(platform='gpu', gpu_index=0)` affiche « Setting CUDA_VISIBLE_DEVICES to: 0 » (l. 74, qui ne servira à rien), puis lève `RuntimeError: Unknown backend: 'gpu' requested…` en l. 109 — ou, si `print_devices_found=False`, plus loin dans `main.py:180`, c.-à-d. **l'endroit du crash dépend d'un paramètre d'affichage**. La valeur invalide reste dans la config (`jax_platform_name='gpu'` persiste après l'exception). Aucun repli CPU proposé, aucun message du type « GPU demandé, indisponible — relancez avec platform='cpu' ».

**M2 — `os.cpu_count()` ignore les limites d'affinité/cgroups : surallocation sur machines restreintes.** [RÉSOLU b4178df8]
`BayesForge/SetDevice/set.py:83`.
Sous SLURM, `taskset`, cpuset ou conteneur, `os.cpu_count()` renvoie les cœurs **physiques de l'hôte** ; `--xla_force_host_platform_device_count` est alors bien supérieur aux CPU alloués → threads XLA en surallocation, ralentissements et devices virtuels sans CPU porteur. Le projet sait déjà faire : `Parallel/simulate.py:858` utilise `os.sched_getaffinity(0)`. Incohérence interne : `run_simulations` compense via `BF_MAX_CORES`, mais un `bf()` nu sur nœud restreint (cas d'usage cluster typique de ce projet) ne bénéficie d'aucune compensation.

**M3 — `cores` et `BF_MAX_CORES` non validés : crashs XLA obscènes au lieu d'erreurs précoces.** [RÉSOLU b4178df8]
`BayesForge/SetDevice/set.py:79-83, 99`.
- `cores=0` ou `BF_MAX_CORES='0'` (la chaîne `'0'` est truthy, l. 80, donc `max_cores=0` et l. 83 retient 0) → `RuntimeError: … Could not determine process_index … you may need to uninstall the failing plugin package` — accuse un plugin imaginaire au lieu du device_count=0.
- `cores=-1` → `RuntimeError … vector::reserve` (erreur C++ brute).
- `cores=2.5` (ou toute valeur non entière, formatée brute l. 99) → XLA rejette le flag et **imprime son aide-mémoire complète** (centaines de lignes).
- `BF_MAX_CORES='auto'` → `ValueError: invalid literal for int() with base 10: 'auto'` sans nommer la variable en cause.
Aucun contrôle `isinstance(cores, int) and cores >= 1` au point d'entrée, alors que `float_precision`, lui, est validé proprement (l. 65–70) — la barre de qualité est inégale à l'intérieur d'une même fonction.

### MINEURES

**m1 — `gpu_index` jamais validé ; affichage de succès même quand l'effet est nul.** [RÉSOLU 0a28ed77]
`BayesForge/SetDevice/set.py:72-74`. `gpu_index='0,1'` (plusieurs GPU, contrairement à la docstring « The index of the GPU », l. 34–35) et `gpu_index=-3` sont acceptés tels quels (vérifié : `CUDA_VISIBLE_DEVICES='-3'`). Et le `print` de la l. 74 s'exécute même quand le backend GPU est déjà initialisé (2ᵉ `bf()`), annonçant une action qui n'aura pas lieu (cf. C1).

**m2 — `deallocate=True` après init : no-op silencieux, et irréversible.** [RÉSOLU 0a28ed77]
`BayesForge/SetDevice/set.py:5-7, 93-94`. Les flags `XLA_PYTHON_CLIENT_*` sont lus à l'init du client Python ; posés après (2ᵉ `bf(deallocate=True)` — vérifié : l'environnement change, le backend est déjà up), ils ne font rien. Dans le sens inverse, une fois posés au 1ᵉʳ appel, aucun chemin ne les retire : l'allocateur `platform` (non préemptable, sans cache) reste actif pour tout le processus, même si l'utilisateur veut s'en affranchir. La docstring (l. 37–39, « deallocate any existing devices ») décrit par ailleurs mal l'effet réel (choix de l'allocateur XLA).

**m3 — Docstring contractuellement fausse : « must be called before any JAX imports ».** [RÉSOLU 0a28ed77]
`BayesForge/SetDevice/set.py:49-50`. Via le package, c'est **impossible** : `BayesForge/__init__.py:22` importe `Main/main.py`, qui importe `jax`/`jax.numpy` (main.py:7–8) avant que l'utilisateur ait la main. La vraie contrainte est « avant l'**initialisation** du backend » (vérifié : l'import seul ne initialise pas, d'où un 1ᵉʳ `bf()` fonctionnel). Le vrai danger — la 2ᵉ exécution silencieusement ignorée (C1) — n'est documenté nulle part dans `setup_device` ; seule la docstring de `Parallel/simulate.py:12-14` le mentionne, ailleurs.

**m4 — L'utilisateur qui règle lui-même `XLA_FLAGS` est écrasé sans un mot.** [RÉSOLU 0a28ed77]
`BayesForge/SetDevice/set.py:97-99`. Le `re.sub` retire toute valeur existante de `--xla_force_host_platform_device_count` posée par l'utilisateur (ou un ordonnanceur) et la remplace sans avertissement.

**m5 — `float_precision` effectif à chaud ⇒ mélange de dtypes entre instances ; le backend tfp le dément.** [RÉSOLU 0a28ed77]
`BayesForge/SetDevice/set.py:106` + `Main/main.py:297, 562`. Contrairement aux autres leviers, x64 change en cours de processus (vérifié). Conséquence : `m1 = bf()` (float64, données converties en float64) puis `m2 = bf(float_precision=32)` → les tableaux déjà créés restent float64, les nouveaux sont float32, sans aucune alerte de mélange. Et `bf(backend='tfp', float_precision=64)` : main.py:297 force `jax_enable_x64=False` **après** que `setup_device` a appliqué 64 (et après init du backend, l. 109/180) — le paramètre demandé est silencieusement annulé (cohérent avec le `bit="32"` de main.py:562, mais rien ne le dit dans la docstring de `float_precision`, l. 41–46). *(tfp non installé ici : ce point est statique + ordre d'exécution vérifié ; l'absence de crash est garantie par le fait que l'update post-init est effectif en 0.11.2.)*

**m6 — Aucune sécurité vis-à-vis des threads.** [RÉSOLU 0a28ed77]
`BayesForge/SetDevice/set.py:73, 99` + `105-106`. `os.environ` et `jax.config` sont globaux au processus : deux threads qui appellent `bf()` concurrently se marchent sur les flags (course l. 97–99 notamment). Atténué par le choix `spawn` de `run_simulations` (simulate.py:851–853), mais `setup_device` est exporté publiquement (`BayesForge/__init__.py:23, 40`) sans mise en garde.

**m7 — Le compteur affiché peut mentir sur le platform demandé.** [RÉSOLU 0a28ed77]
`BayesForge/SetDevice/set.py:108-109`. `jax.local_device_count(backend=None)` compte le backend **par défaut déjà choisi** : avec `platform='gpu'` après un init CPU, il affiche « jax.local_device_count 2 » (les CPU) juste avant que le module ne découvre que GPU n'existe pas (vérifié, T4). Le libellé même (« jax.local_device_count ») n'indique ni le platform ni que ce sont des devices virtuels.

**m8 — `setup_multi_device` : validation et duplication.** [RÉSOLU 0a28ed77]
`BayesForge/SetDevice/set.py:180-182`. `jax.devices(platform)` lève un `RuntimeError` explicite sur platform inconnu (bien), mais `int(cores)` (l. 181) lève un `ValueError` brut sur `cores` non numérique ; la troncature `devices[:cores]` duplique `main.py:182-183` (deux implémentations à maintenir en phase). Le `(None, None, None)` de la l. 185 est documenté et correctement gardé côté Main (main.py:189, 1850).

---

## 3. Fragilités architecturales

1. **Le module écrit dans l'environnement d'un bus global figé à l'init, sans jamais vérifier ni déclarer l'état.** `setup_device` est présenté comme un setter idempotent mais est un *one-shot* : son exécution répétée est structurellement silencieuse (C1). La correction structurelle est une garde unique en tête de fonction (`backends_are_initialized()` → warning nommant chaque paramètre non appliqué, ou `RuntimeError` en mode strict), plus une mémoire de la config appliquée pour détecter les demandes divergentes.
2. **Pas de frontière de responsabilité avec Main.** L'ordre d'initialisation est implicite : `Main/main.py` importe `jax` au niveau module (main.py:7), appelle `setup_device` à chaque `bf()` (main.py:166), force x64 pour tfp (main.py:297) — trois décisions qui contredisent la docstring du module (m3, m5). Le contrat « qui configure JAX, quand, une fois » n'existe nulle part sous forme exécutable.
3. **Effets irréversibles cumulés par processus** : `XLA_FLAGS`, `CUDA_VISIBLE_DEVICES`, `XLA_PYTHON_CLIENT_*`, `jax_platform_name`, `jax_enable_x64` — aucun n'est annulable après init (m2, C1). D'où le contournement obligatoire par `spawn` dans `run_simulations` : l'architecture compense le module par des sous-processus, ce qui confirme que le module ne peut pas remplir sa promesse dans un processus unique.
4. **Validation d'entrée à deux vitesses** : `float_precision` est validé tôt et proprement (l. 65–70) ; `platform`, `cores`, `gpu_index`, `BF_MAX_CORES` sont délégués à XLA/JAX, qui échoue tard et en jargon (M1, M3, m1). Les messages de succès (l. 74, 109) ne sont pas conditionnés à l'effet réel.
5. **Flag déprécié** : `jax_platform_name` est marqué « Deprecated, use --jax_platforms » dans JAX 0.11.2 (`jax/_src/xla_bridge.py:73-77`) ; le module bâtit sa sélection de platform sur une API dont la disparition est annoncée.

## 4. Décompte par sévérité

| Sévérité | Nombre | Repères |
|---|---|---|
| Critique | **1** | C1 (no-op silencieux des 2ᵉ appels, reproductibilité dépendante de l'ordre) |
| Majeure | **3** | M1 (gpu sans GPU, crash tardif + message faux), M2 (`os.cpu_count` vs affinité), M3 (cores/BF_MAX_CORES non validés, crashs obscènes) |
| Mineure | **8** | m1 gpu_index, m2 deallocate, m3 docstring fausse, m4 XLA_FLAGS écrasé, m5 x64 à chaud / tfp, m6 threads, m7 compteur mensonger, m8 setup_multi_device |
| **Total** | **12** | |

**Verdict de release** : le module fonctionne dans le cas nominal unique (premier `bf()` d'un processus frais, CPU, `cores` absent ou valide) — cas vérifié de bout en bout. Il ne doit pas être qualifié de « configurable » tel quel : dès le deuxième appel dans un processus, il devient un no-op silencieux (C1), et trois chemins d'entrée courants (`platform='gpu'` sans GPU, `cores` extrême, nœud à affinité restreinte) produisent des échecs tardifs ou des surallocations. Les correctifs minimaux avant release sont concentrés et non invasifs : garde `backends_are_initialized()` + warning nommant les paramètres ignorés ; validation `platform`/`cores`/`gpu_index`/`BF_MAX_CORES` en tête de fonction ; `os.sched_getaffinity` à la place de `os.cpu_count` ; conditionner les `print` de succès à l'effet réel.

---

### Module R

**Périmètre** : `BayesForge/R/BF doc.py` (64 lignes, lu intégralement), avec vérifications croisées sur `BayesForge/R/` (`R warpper.R`, `RDoc.json`), `BayesForge/Build/`, `BayesForge/Main/main.py`, `BayesForge/Distributions/np_dists.py`, `pyproject.toml`, `MANIFEST.in`, wheel construit localement (`bayesforge-0.0.60-py3-none-any.whl`), `git ls-files` et grep d'ensemble de l'arbre principal (hors worktrees).

**Nature du fichier** : script de génération documentaire en cellules Jupyter/VS Code (marqueurs `#%%` L1, L41, L47), à exécution unique et manuelle : il instancie `bf()`, parcourt `dir(m.dist)` pour extraire les docstrings des distributions, écrit un JSON, puis contient en chaîne (L49–65) le prompt « Gemini 2 Pro » utilisé pour convertir ces docstrings en blocs roxygen2 R. Ce n'est pas un module : **aucune référence à `BF doc` nulle part** dans le package, les tests, `pyproject.toml`, `MANIFEST.in` ou la documentation (grep exhaustif : seul auto-référence, L43). Code mort livré.

---

## Problèmes détectés

### R-C1 — Critique — Le générateur est cassé contre l'API actuelle : plantage immédiat puis, s'il passait, sortie entièrement fausse `[RÉSOLU 3938c102]`
- **Emplacement** : `BayesForge/R/BF doc.py:L34` et `BayesForge/R/BF doc.py:L37`
- **Explication** : depuis la refactorisation proxy, `m.dist` est un `DistProxy` (`BayesForge/Main/main.py:255`, classe définie `BayesForge/Distributions/np_dists.py:122`) qui expose les distributions **uniquement via `__getattr__`** (L147–170) et ne définit **pas de `__dir__`**. Conséquence 1 : `dir(m.dist)` (L34) ne liste plus les ~99 distributions, seulement les membres internes du proxy (`__getattr__`, `__setattr__`, `_dist`, `_obs_names` et les dunders). Conséquence 2 : L37 indexe `m.dist.__dict__[name]`, dictionnaire d'**instance** qui ne contient que `_dist` et `_obs_names` ; le premier nom survivant du filtre `no` en ordre trié est `__getattr__` (attribut de classe, absent de `__dict__`) → **`KeyError` immédiat**. Le script ne peut plus tourner contre le code actuel.
- **Scénario déclencheur** : un mainteneur doit régénérer la doc R pour la 0.0.61, exécute le script → `KeyError: '__getattr__'` ligne 37. Il ne peut pas produire le JSON ; `RDoc.json` livré est un artefact gelé d'une version antérieure, **régénérable par aucune voie outillée**.

### R-C2 — Critique — Le fichier est pourtant embarqué et installé dans la roue (avec un nom de module illégal dans `site-packages`) `[RÉSOLU 3938c102]`
- **Emplacement** : `BayesForge/R/BF doc.py` (fichier) ; cause racine hors module : `[tool.setuptools.packages.find]` de `pyproject.toml:66-67`
- **Vérification empirique** : `pip wheel . --no-deps` produit `bayesforge-0.0.60-py3-none-any.whl` contenant **`BayesForge/R/BF doc.py`** (confirmé par `zipfile.namelist()`), malgré l'absence de `__init__.py` dans `R/` (confirmée) et l'absence de `BayesForge.R` dans `[tool.setuptools.package-data]` (qui ne liste que `Network` et `Resources`). Le même wheel contient **681 entrées sous `worktrees/`** (copies complètes des packages des 4 worktrees, ~9,5 Mo, installées en `site-packages/worktrees/npe/BayesForge/…`), plus `Test/` (93 entrées) et `Documentation/` en paquets de premier plan — 903 entrées, 38,9 Mo au total.
- **Risque** : un utilisateur qui fait `pip install BayesForge` se retrouve avec un fichier `.py` au nom non-importable dans son arborescence installée, et quatre copies concurrentes du package sous `worktrees.*` (collision d'imports, `import Test` parasite, audit/sécurité). Pour le module R : pire des deux mondes — le dossier n'est pas un paquet (donc pas de support officiel, pas de `package_data` explicite), mais ses fichiers fuillent malgré tout dans la roue.
- **Scénario** : installation PyPI propre → `site-packages/BayesForge/R/BF doc.py` présent ; `import worktrees` fonctionne et charge un vieux fork de BayesForge.
- **Note** : la cause racine (auto-découverte `where=["."]` sans `exclude`, worktrees sous la racine) relève du pyproject, pas du module R ; l'audit du module acte que **le fichier est livré installé** et que `MANIFEST.in` est lui aussi périmé (`recursive-include BF *` alors qu'aucun dossier `BF/` n'existe à la racine — Mineur, cf. R-m5).

### R-M1 — Majeure — Nom de fichier illégal : contenu inatteignable par toute voie outillée [RÉSOLU antérieurement 3938c102 — vérifié : tools/r_doc/bf_doc.py]
- **Emplacement** : `BayesForge/R/BF doc.py` (nom du fichier)
- **Explication** : `'BF doc'` n'est pas un identifiant Python valide (vérifié : `'BF doc'.isidentifier() == False`). `import BayesForge.R.BF doc` est une erreur de syntaxe ; sans `__init__.py` dans `R/`, il n'y a même pas de paquet à charger ; `importlib` exige des contournements (`importlib.machinery` + chemin manuel). Le fichier est invisible de pytest, des linters collectés par paquet, de `pydoc`, et de tout `__all__`.
- **Scénario** : un contributeur veut réutiliser la logique d'extraction pour la doc Julia (le commit `4cb177ef` ajoute des « api reference generation tools » pour Julia) : impossible de factoriser par import, il va copier-coller → divergence.

### R-M2 — Majeure — Pipeline de documentation à quatre noms de fichiers incohérents : provenance cassée, dérive confirmée [RÉSOLU e7673ca9]
- **Emplacement** : `BayesForge/R/BF doc.py:L43` (`pythonDoc.json`) ; `BayesForge/R/R warpper.R:11` (`Doc_R.json`) ; artefacts réels `BayesForge/Build/Doc_python.json`, `BayesForge/Build/Doc_R.json`, `BayesForge/R/RDoc.json`
- **Explication** : la chaîne documentée par le code est `BF doc.py → pythonDoc.json → (LLM) → Doc_R.json → R warpper.R`. Or : (i) `pythonDoc.json` n'existe nulle part (find exhaustif : 0 résultat) ; l'artefact réel s'appelle `Build/Doc_python.json` ; (ii) `R warpper.R` lit `Doc_R.json` **depuis le CWD**, fichier qui vit dans `Build/` ; (iii) un quatrième artefact `R/RDoc.json` (104 Ko) git-tracked n'est lu par **personne** (grep : aucune référence). Dérive mesurée : `Doc_python.json` a **92 clés**, `RDoc.json` en a **99** — les deux artefacts ne proviennent pas de la même version du code.
- **Scénario** : régénération semi-automatique (lancer `BF doc.py` depuis la racine, puis `R warpper.R` depuis `R/`) : le premier écrit `pythonDoc.json` dans le CWD (pollution ou erreur selon l'endroit), le second échoue (`Doc_R.json` introuvable) ou, pire, consomme silencieusement un vieux JSON resté dans le CWD → doc R produite à partir de docstrings périmées, sans que rien ne le signale.

### R-M3 — Majeure — Le livrable R (`RDoc.json`) est massivement dégradé : 93/99 entrées avec des descriptions de paramètres littéralement « None » [PARTIEL e7673ca9 — 0 « None » ; 4 non-lois retirées ; 33 descriptions remplies ; 170 params sans doc côté docstrings Python, marqués (undocumented)]
- **Emplacement** : `BayesForge/R/RDoc.json` (artefact produit par le pipeline de `BF doc.py`) ; défaut de la spécification de prompt en `BayesForge/R/BF doc.py:L56-L60`
- **Explication** : analyse programmatique : 93 des 99 entrées contiennent `@param <nom> None` (ex. `bernoulli` : `@param probs None`, `@param logits None`, `@param validate_args None`) — la conversion LLM a échoué à extraire les descriptions Google-style dès qu'elles n'étaient pas préfixées d'un type entre parenthèses. Deux entrées (`mask`, `plate`) valent `None` (docstring vide envoyée telle quelle au LLM). Quatre membres non-distribution (`biject_to`, `mask`, `plate`, `unit`) sont documentés comme des distributions, et le gabarit d'exemple obligatoire (`BF doc.py:L61-L64`, `bf.dist.[nom]([params], sample = TRUE)`) produit pour eux des exemples qui ne compilent pas. Des types Python (`jnp.ndarray`) fuilent dans l'aide R.
- **Scénario** : une utilisatrice R fait `?bf.dist.bernoulli` : la page d'aide affiche « probs: None », et les pages de `mask`/`plate` montrent un exemple `bf.dist.mask(..., sample = TRUE)` invalide.

### R-M4 — Majeure — Reproductibilité : génération par LLM non pinnée, sans trace ni revue, non rejouable [PARTIEL e7673ca9 — provenance consignée, post-traitement déterministe ; l'étape LLM reste non rejouable]
- **Emplacement** : `BayesForge/R/BF doc.py:L47-L65`
- **Explication** : le commentaire L48 (« withe the bellow prompt we I asked roxygen2 formatting from gemini 2 pro ») est la seule trace de provenance : pas de version/date du modèle, pas de température, pas de identifiant de conversation, pas de relecture humaine consignée, pas de script rejouant le prompt sur l'entrée. Le prompt lui-même est une chaîne morte (L49–65) jamais exécutée par le code. Combiné à R-C1 (générateur d'entrée cassé), **la doc R livrée n'est reproductible par aucune commande**.
- **Scénario** : litige sur une formule ou un paramètre dans l'aide R ; impossible de dire quelle version de la docstring Python et quel modèle ont produit le texte, ni de le régénérer à l'identique pour prouver la correction.

### R-m1 — Mineure — Effets de bord à l'exécution : import lourd, instanciation de modèle, écriture dans le CWD [RÉSOLU 3938c102, test daea811f]
- **Emplacement** : `BayesForge/R/BF doc.py:L2`, `L4`, `L43-L44`
- **Explication** : `from BayesForge import bf` (L2) charge jax/numpyro/arviz ; `m = bf()` (L4) instancie le moteur complet (affiche « bf v 0.0.60 package loaded », touche au device) ; `open("pythonDoc.json","w")` (L43) écrit un chemin **relatif au CWD** sans dossier garanti ni `if __name__ == "__main__"`. Tout outil qui collecterait ce fichier (collecteur de docs, `compileall` suivi d'exécution, notebook runner) déclenche une instanciation MCMC-side-effect et écrase un fichier dans le dossier courant.
- **Scénario** : lancé depuis la racine du dépôt, écrase tout `pythonDoc.json` existant ; lancé depuis `/`, `PermissionError` opaque.

### R-m2 — Mineure — Liste d'exclusion codée en dur, obsolète dans les deux sens [RÉSOLU daea811f]
- **Emplacement** : `BayesForge/R/BF doc.py:L6-L33`
- **Explication** : la liste `no` exclut `"sineskewed"` (L33) qui **n'existe plus** dans `np_dists.py` (grep : 0 occurrence) — exception manuelle pérennisée après un patch d'époque. En sens inverse elle omet les membres qui polluent aujourd'hui la sortie : dunders Python ≥3.12 (`__firstlineno__`, `__static_attributes__`), et les attributs du proxy (`_dist`, `_obs_names`, `__getattr__`, `__setattr__`) — cf. R-C1 — ainsi que les helpers non-distribution (`mask`, `plate`, `biject_to`, `unit`, visibles dans `RDoc.json`).
- **Scénario** : chaque ajout de dunder par CPython ou de helper sur `dist` casse ou pollue la sortie ; aucune détection, le filtre est une liste noire figée de 28 chaînes.

### R-m3 — Mineure — `json.dump` sans `ensure_ascii=False` ni `indent` : artefact invérifiable en revue [RÉSOLU daea811f]
- **Emplacement** : `BayesForge/R/BF doc.py:L43-L44`
- **Explication** : sortie mono-ligne avec `\uXXXX` pour tout caractère non-ASCII (les formules et accents des docstrings) : les diffs git de l'artefact sont illisibles, une modification involontaire de docstring est indétectable en revue.
- **Scénario** : un relecteur approuve un diff de `Doc_python.json` de 200 Ko mono-ligne sans pouvoir rien vérifier.

### R-m4 — Mineure — Duplication sans source unique avec `Documentation/*.qmd` [RÉSOLU daea811f — limite documentée dans tools/r_doc/README.md]
- **Emplacement** : `BayesForge/R/RDoc.json` vs `Documentation/*.qmd` (onglets « R »)
- **Explication** : les pages qmd contiennent des exemples R écrits à la main (`m$data(...)`, `bf.dist.exponential(...)`), indépendants des blocs roxygen générés depuis les docstrings Python. Deux sources de vérité pour l'API R, sans test de cohérence entre elles (les tests `lang:` couvrent les cas d'exécution, pas l'aide roxygen).
- **Scénario** : changement de signature d'une distribution : la page qmd est mise à jour, `RDoc.json` (gelé, cf. R-C1/R-M2) garde l'ancienne signature — contradiction publique entre le site et l'aide R.

### R-m5 — Mineure — `MANIFEST.in` périmé : `recursive-include BF *` cible un dossier inexistant [RÉSOLU 3938c102, test daea811f]
- **Emplacement** : `MANIFEST.in:1` (contexte packaging du module)
- **Explication** : le paquet s'appelle `BayesForge` ; aucun dossier `BF/` n'existe à la racine (vérifié). La règle ne matche rien ; l'intention d'origine (embarquer les artefacts) est perdue, et l'embarquage réel de `R/` dans la roue (R-C2) se fait par un chemin non déclaré — non intentionnel donc non maintenu.

---

## (a) Résumé du module

`BayesForge/R/` n'est pas un module Python mais un **atelier de fabrication de documentation R abandonné à mi-chemin** : un script-journal en cellules (`BF doc.py`) qui extrayait les docstrings de `m.dist`, un four LLM non rejouable (prompt Gemini 2 Pro collé en chaîne morte), un générateur de wrappers R (`R warpper.R`) qui lit un fichier sous un nom que personne n'écrit, et un artefact final (`RDoc.json`, 104 Ko) que plus rien ne lit. Le générateur d'entrée est **cassé contre l'API actuelle** (`DistProxy` sans `__dir__` → `KeyError` à la ligne 37), les quatre noms de fichiers du pipeline ne s'enchaînent pas (`pythonDoc.json` / `Doc_python.json` / `Doc_R.json` / `RDoc.json`), les artefacts ont dérivé (92 vs 99 clés), et le livrable R est dégradé à 94 % (`@param X None`). Personne ne référence `BF doc.py` : c'est du **code mort, non importable par nom légal, sans `__init__.py` — et pourtant embarqué et installé dans la roue 0.0.60**, en bonne place aux côtés de 681 fichiers de worktrees.

## (b) Fragilités architecturales

1. **Aucun seam testable** : la chaîne docstring-Python → JSON → LLM → roxygen → wrapper R n'existe que sous forme de gestes manuels ; aucun contrat de nom de fichier, aucun dossier d'artefacts (`Build/` vs `R/` vs CWD), aucune garde `__main__`. Rien ne peut être rejoué en CI.
2. **Couplage invisible à l'implémentation interne** : le générateur suppose `dir(m.dist)` == distributions et méthodes dans le `__dict__` d'instance ; le passage à `DistProxy` (légitime par ailleurs) l'a tué silencieusement, sans test d'API publique qui s'en aperçoive.
3. **Double source de vérité de la doc R** (docstrings Python via LLM vs onglets R des qmd) sans test de cohérence.
4. **Packaging par accident** : inclusion dans la roue non déclarée (auto-découverte `where=["."]` + worktrees sous la racine) ; `MANIFEST.in` et `package_data` ne décrivent pas la réalité livrée.
5. **Provenance LLM non gouvernée** : pas de pin de modèle, pas de revue consignée, pas de régénération déterministe — incompatible avec une release qui se veut vérifiable.

## (c) Décompte par sévérité

| Sévérité | Nombre | Issues |
|---|---|---|
| Critique | 2 | R-C1 (générateur cassé contre l'API actuelle), R-C2 (fichier installé dans la roue + worktrees embarqués) |
| Majeure | 4 | R-M1 (nom illégal), R-M2 (pipeline 4 noms, dérive 92≠99), R-M3 (93/99 `@param … None`), R-M4 (provenance LLM non rejouable) |
| Mineure | 5 | R-m1 (effets de bord/CWD), R-m2 (liste d'exclusion obsolète), R-m3 (json.dump brut), R-m4 (duplication qmd), R-m5 (MANIFEST périmé) |
| **Total** | **11** | |

**Recommandation de release** (sans modification de code, pour arbitrage) : soit sortir `BayesForge/R/` du dépôt du paquet (dépôt/tooling séparé) et corriger l'exclusion dans `pyproject.toml` (qui désamorce aussi R-C2 côté worktrees), soit le reconnaître explicitement comme outil de build (`tools/`, renommé `bf_doc.py`, gardé hors roue) et relancer le pipeline de bout en bout avec artefacts versionnés — dans l'état actuel, `RDoc.json` ne peut pas être régénéré et sa qualité (R-M3) ne permet pas de l'exposer comme aide R officielle.

---

---

## 2. Audit Inter-Modules

**Méthode** : quatre passes séquentielles (mêmes persona et discipline que le volet 1, lecture seule, vérifications empiriques quand possible) — ① trace du cycle de vie complet d'une analyse H1→H7 (`bf()` → données → modèle → inférence → postérieurs → diagnostics → persistance) ; ② coutures de construction de modèle (bibliothèque `Models/` ↔ singleton `dist` ↔ `Utils` ↔ `Network` ↔ `BNN` ↔ `NBDA`) ; ③ coutures runtime/persistance/documentation (`SetDevice` ↔ `Parallel` ↔ workers, `Resources` ↔ `Main`, `Build` ↔ `Distributions`, ArviZ, SBI, PostModel) ; ④ interface inter-langages R↔Python (reticulate).

**Résultat : 75 vulnérabilités d'intégration — 14 critiques, 35 majeures, 26 mineures — sur 102 coutures inventoriées (32 cassées, ~35 fragiles, ~31 saines).**

| Passe | Périmètre | Critique | Majeure | Mineure | Total | Coutures |
|---|---|---:|---:|---:|---:|---|
| 20 | Cycle de vie H1→H7 (V1–V18) | 4 | 7 | 7 | 18 | 27 (5 cassées) |
| 21 | Construction de modèle (W1–W16) | 3 | 9 | 4 | 16 | 24 (12 cassées) |
| 22 | Runtime / persistance / doc (X1–X17) | 1 | 5 | 11 | 17 | 35 (8 cassées) |
| 23 | Interface R↔Python | 6 | 14* | 4* | 24 | 16 (7 cassées) |
| — | **Total** | **14** | **35** | **26** | **75** | **102** |

\* Échelle d'origine Critique/Haute/Moyenne/Faible (6/6/8/4) mappée Critique/Majeure/Mineure (6/14/4).

### Thèmes transverses d'intégration

1. **Transitions d'état non revalidées — le cœur du risque critique.** `fit→svi` laisse trois attributs périmés et deux façons de ressusciter l'ancien fit (V1/V2) ; `fit→fit` resplatte les données du fit précédent sans validation de signature (V3) ; l'état d'erreur `"Error, no data found"` persiste dans `m.data_on_model` après l'échec (V4) ; `m.posteriors` **est** le cache du sampler numpyro (V5) — deux sources de vérité (attributs vs sampler) qui divergent dès qu'un fit est partiellement rafraîchi.
2. **Classe vs instance `dist` : la bibliothèque de modèles découplée de son singleton.** `SRM.py:9` et `NBDA.py:5` importent `UnifiedDist` **sans l'instancier** : les appels se lient en méthode non liée qui avale le premier argument → priors silencieusement décalées. NBDA, modèle phare, **aboutit de bout en bout avec des priors fausses** (W5) ; SRM crash en cascade (W3). Le recâblage des singletons par `bf()` fait du **dernier `bf()` créé le propriétaire exclusif** du RNG (W1) et oublie `model_effects2`/PCA/`survival_old` (W2).
3. **`sample=True` dans un contexte numpyro actif : l'unique chemin de l'audit vers un postérieur faux sans aucun message (X1).** Tirage hors `numpyro.sample`, re-tiré à chaque leapfrog, invisible du sampler, non contrôlé par la graine.
4. **Conventions d'indexation sans convertisseur.** `Data/index()` émet du 0-based documenté, `nodal_random_effects` exige du 1-based documenté : la combinaison des deux API officielles produit un gather `-1` → **dernier groupe**, effets décalés pour 100 % des observations, sans erreur (W7). Côté R : aucune conversion 1-based↔0-based nulle part, et `index1=True` **écrase la colonne intercept** (`SRM.py:142-143`).
5. **La « Built-in models API » est inutilisable par son chemin documenté** : 5 des 6 entrées de `m.models` crashent dans `pd_to_jax` (défauts `None`/`str` convertis en tenseurs — W9), objets sans contrat `__name__`/`__call__` (W10), hyperparamètres scalaires devenus colonnes obligatoires (W11), résolution `model_name` inexistante (W16).
6. **Angles morts de CI** : `Test/13.Parallel` (aucun `def test`) et `Test/16.SBI` hors de la liste pytest ; **zéro test de dérive** `Build/` → `Distributions/` (X3/X6) — la régénération d'un fichier de distributions par un mainteneur de bonne foi efface les correctifs manuels sans témoin.
7. **Pont R : trois générations divergentes.** Le générateur du repo meurt à sa ligne 10 (`import("BF")`, package réel `BayesForge`) et chaque porte corrigée en cache une autre (fences Markdown, `.bf` introuvable, `seed=as.integer(py_none())`) ; le pont **vivant** (package R v0.0.2, repo `BGN-for-ASNA/BFR`) est hors repo et testé pourtant par la CI des pages de doc. Le fossile est embarqué dans la roue. **Recommandation : retirer `BayesForge/R/` du dépôt et de la roue**, désigner BFR comme source unique, exposer `index1` (ou convertir à l'entrée) et le corriger.

### Priorités d'intégration (P0)

- Avertir quand `sample=True` s'exécute dans un contexte numpyro actif (X1).
- Rafraîchissement complet des attributs dans `svi` + copie/validation des clés `data_on_model` contre la signature du modèle en entrée de `fit`/`svi` (V1–V4).
- `dist = dist()` dans `SRM.py`/`NBDA.py` (ou passage par `parent.dist`) (W3/W5) ; recâblage des singletons avec garde multi-instances et liste complète (W1/W2).
- Convertisseur/erreur sur l'indexation 0-based vs 1-based à la couture `index()` → `nodal_random_effects` (W7) ; correctif intercept `index1` (`SRM.py:142-143`).
- Retrait du pont R fossile de la roue + test de parité d'API R↔Python en CI (passe 23).
- Trois lignes de garde : `cloudpickle.dumps` du résultat dans le filet `on_error` (X2) ; `XLA_PYTHON_CLIENT_PREALLOCATE=false` dans `_worker_env` (X5) ; `bind_diag_to_model` dans `_sbi_finish` (X4).

### Rapports des passes inter-modules

### 20 — Audit d'interfaces : cycle de vie complet H1→H7 (coutures)

> Passe « module-reviewer », racine `/home/sosa/work/BF`, chemin prioritaire **numpyro**.
> Méthode : lecture ciblée (`grep -n` + `read offset/limit`) + exécution empirique en lecture seule
> (régression 200 pts, 2 chaînes × 100 draws, CPU ; jax 0.11.2 x64 activé par `bf()`, numpyro 0.22.0,
> pandas 3.0.6, ArviZ 1.3.0, TFP absent → chemin tfp statique seulement).
> BayesForge v0.0.60. Aucune modification de source.

## 1. Cycle de vie — trace de bout en bout (tableau des sauts)

| # | Saut | Producteur `fichier:ligne` | Consommateur `fichier:ligne` | Données échangées (type / forme / clés) | Transformation | Contract-breakers potentiels |
|---|------|---------------------------|------------------------------|------------------------------------------|----------------|------------------------------|
| H1 | construction | `Main/main.py:123-297` (`bf.__init__`) | `SetDevice/set.py:9` (`setup_device`, appelé l.166) ; `Utils/rng.py:39` (`RNG`, l.206) ; `Distributions/np_dists.py:122` (`DistProxy`, l.255) | aucun en entrée ; en sortie : `m._rng:RNG(base)`, `m.dist:DistProxy(_dist:UnifiedDist)`, `m.data_on_model=None`, `m.posteriors_full=None`, `m.history={}` ; **`m.df` et `m.index_map` n'existent pas** | ordre : `manip.__init__`(163) → `setup_device`(166, x64=True si float_precision=64) → mesh multi-dev(189-191) → RNG+`seed_globals`(206-207) → singletons(228-249) → dist(251-258) → **mutation des singletons `dist` de module** (279-289) | tfp : `main.py:297` force `jax_enable_x64=False` (M-07) ; backend inconnu : ni branche ni `raise` (M-03) ; singletons partagés entre instances (M-08) |
| H2 | données | `Data/manip.py:33-62` (`data`) | `manip.py:64-238` (`OHE`/`index`/`scale`/`to_float`) puis `manip.py:240` (`pd_to_jax`) | entrée : CSV/DataFrame/dict ; sortie : `m.df` **copie** (l.60) `(n,p)` pandas ; `m.data_modification={}` réinitialisé (l.61) ; `m.index_map` créé **seulement** par `index()` (l.105) | `pd.read_csv` ou `pd.DataFrame(path).copy()` ; `scale` standardise en place dans `m.df` (l.158-163) ; `OHE` **concatène** `get_dummies` à `m.df` (l.86) | `OHE` duplique les colonnes non encodées dans `m.df` (C1, vérifié : `['x','g','x','g_a','g_b']`) ; `index` code NaN→-1 ; `m.df` absent avant tout `data()` → `AttributeError` en aval |
| H3 | modèle + données→jax | `Data/manip.py:240-283` (`pd_to_jax`, inspection de signature l.250) ; `Distributions/np_dists.py:5655` (`normal`) → `numpyro.sample` (l.5741) via `DistProxy.__getattr__` (l.147-170) | `Main/main.py:537-562` (`fit` résout `data_on_model`) | `pd_to_jax` : `dict {nom_arg: jax.Array (n,) float64}` pour les args sans défaut pris dans `m.df`, scalaires Python pour les défauts `int/float/bool` (l.274-277) ; le modèle émet des sites numpyro nommés (`name=` ou inférence bytecode LHS/`obs=`) | dtype map `pandas_to_jax_dtype_map` (l.15-30) ; `jnp.array(df[col], dtype=…)` (l.265) ; noms de sites inférés par bytecode (`_infer_caller_name` l.28-64) ou par identité `obs=y` (l.69-99) | colonne absente → **retourne la chaîne** `"Error, no data found"` (l.270, C6) ; colonnes dupliquées (post-OHE) → `AttributeError: 'DataFrame' object has no attribute 'dtype'` (l.263, vérifié) ; dtype hors map → `None` implicite ; collision de noms de sites → `ValueError` (main.py:667-676, sain) |
| H4 | inférence | `Main/main.py:386-727` (`fit`) → `Samplers/mcmc_numpyro.py:59-124` (`mcmc_numpyro` renvoie un `numpyro.infer.MCMC` brut, l.114) | `MCMC.run` appelé l.661-666 : `self.sampler.run(self._resolve_key(seed,"fit"), extra_fields=…, init_params=…, **run_data)` | entrée : clé PRNG (`RNG.key("fit")`, stable par label, rng.py:63-67), dict `data_on_model` splatté en **kwargs du modèle** ; défaut `extra_fields` = `('num_steps','energy','diverging','accept_prob','adapt_state.step_size')` pour nuts seulement (l.652-657) | `_as_count` (floats R→int, l.511-514) ; sharding éventuel sur copie locale `run_data` (l.634-645) ; `model_name`/`run_model_name` gérés l.549-555, 726-727 | splat `**run_data` collisionne avec la signature de `MCMC.run` si une clé de données s'appelle comme un de ses params (M-13) ; `obs=None` → **réutilisation silencieuse du `data_on_model` du fit précédent** (l.537-538, M-36, vérifié) ; branche `pca` : **transpose `data_on_model["X"]` en place** (l.578, M-02) ; tfp : l.689-717 (statique) |
| H5 | stockage postérieurs | `MCMC.get_samples()` / `get_samples(group_by_chain=True)` / `get_extra_fields()` lus dans `main.py:677-683` | attributs `m.posteriors`, `m.posteriors_by_chain`, `m.posteriors_full`, `m.posteriors_by_chain_full`, `m.extra_fields*`, `m.diag` (l.684-686), `m.history` (l.687) | `posteriors: dict {site: Array (chains·draws, …)}` — vérifié `(200,)` pour 2×100 ; `posteriors_by_chain: (2,100,…)` ; `extra_fields` pooled `(200,)` / by_chain `(2,100)` ; `diag:diagWIP` ; `history` 5 clés | `posteriors_full = dict(self.posteriors)` : **copie superficielle du dict** (mêmes objets Array, immuables → OK) ; `diagWIP.__init__` recroise `shape[0]` avec `sampler.num_chains` (Diag2.py:329-338, garde saine) | **`m.posteriors is m.sampler.get_samples()` → `True`** (cache interne numpyro, vérifié) : mutation in-place de `m.posteriors` atteindrait l'état du sampler ; svi (l.832-836) ne rafraîchit **que** `posteriors` (M-05) ; tfp (l.711-712) : `posteriors_by_chain = posteriors` même objet, forme `(chains, draws)` dans les deux vues (M-04) |
| H6 | extraction / diagnostics | `m.posteriors*` + `m.extra_fields_by_chain` + `m.sampler` + `m.data_on_model` | `main.py:1234` (`summary`) → `1323` (`summary_old`, lit `diag.trace` via `az.from_numpyro(sampler)`, l.1338-1357) ; `main.py:1147` (`sampler_stats`, lit `extra_fields_by_chain`) ; `Diag2.py:314/346/850` ; `patch_diag.py:712/1054` (binding) ; `ppc.py:60` (`get_yrep` ← `m.sample()` l.100) ; `PostModel/model_trace.py:231` (`trace_model(model, m.data_on_model)`) | `summary` : DataFrame arviz (index=paramètres, cols mean/sd/eti/ess/r_hat/mcse) — vient du **sampler**, pas de `m.posteriors` ; `sampler_stats` : dict (leapfrog, ebfmi, divergences…) ; PPC `m.sample()` : `{site: (draws, n_obs)}` vérifié `(200,200)` ; `to_az` : DataTree dims `{chain:2, draw:100}` | `summary(include=)` appelle `filter_posteriors` **en persistant** (l.1280-1281→1421) ; `bind_diag_to_model` remplace `diag.summary/rhat/ess/loo/…` sur l'instance (patch_diag.py:1054-1092) ; `_az_diag_for` cache `m._az_diag` indexé sur l'identité du sampler (patch_diag.py:560-567) | `sampler_stats` après `svi` lit les `extra_fields` **de l'ancien fit MCMC** (vérifié) ; `sampler_stats(per_chain=True)` sans `num_steps` → `TypeError: NoneType has no len()` (l.1225, vérifié) ; tfp `to_az` : `zip` sans `strict` sur 4 noms de stats vs `sampler.sample_stats` (Diag2.py:387, 393) ; `diagWIP.posterior_samples` (l.323) = instantané périmé si le diag survit à un refit |
| H7 | persistance | `main.py:1860-1886` (`save`, cloudpickle de `self` entier, `__getstate__` l.1841 neutralise `mesh/_data_sharding/_rep_sharding`) | `main.py:1888-1924` (`bf.load`, statique) ; alias `m.load` = `Resources/datasets.py:65-68` → délègue à `bf.load` (sain) | pickle complet : `model` (cloudpickle des closures), `sampler` (MCMC numpyro + état), `posteriors*`, `data_on_model`, `_rng` (graine 7 restaurée, vérifié), `diag`, `history` ; version estampillée `_saved_with_version` (l.1876) | `__setstate__` (l.1847-1858) reconstruit le mesh si `n_devices>1`, sinon `pass` (dégradable) ; warning si version différente (l.1918-1922) | nom par défaut `f"{self.model_name}_bf.pkl"` (l.1874) mais `fit` met `model_name=None` à la fin (l.727) → **toujours** `bf_model.pkl` (vérifié) ; `history` alias `data_on_model`/`posteriors` (l.1801-1807, vérifié `is` → True) → le snapshot se rétro-mute ; round-trip complet vérifié fonctionnel (summary/sampler_stats/sample OK après load) |

## 2. Trace empirique

Exécutions (scripts `/tmp/opencode/trace_lifecycle{,2,3}.py`, `/tmp/opencode/probe4.py`) — modèle linéaire `y ~ Normal(beta·x, sigma)`, 200 pts, NUTS 2×100, CPU :

| Étape | Observé | Écart vs lecture statique |
|---|---|---|
| H1 | `m.dist:DistProxy(_dist:UnifiedDist)`, `m._rng:RNG base=42`, x64 activé par `bf()` ; `hasattr(m,'df')=False`, `hasattr(m,'index_map')=False` | conforme |
| H2 | `m.data(df)` → copie (`m.df is df` False) ; `scale(['y'])` → `data_modification={'scale':['y']}`, y mean 0 / sd 1 | conforme |
| H3 | `pd_to_jax` → `{'x':Array(200,) float64, 'y':Array(200,) float64}` ; colonne absente → **`str` `'Error, no data found'`** | conforme (C6) |
| H4 | `m.sampler:MCMC` ; `posteriors {'beta':(200,), 'sigma':(200,)}` ; `by_chain (2,100)` ; `extra_fields` 5 clés pooled `(200,)` / by_chain `(2,100)` ; `m.diag:diagWIP num_chains=2` ; `m.model2=None`, `m.obs_args=None` (posés seulement par `sample()`) | conforme ; à noter : `obs_args`/`model2` **absents juste après fit** |
| H5 | `m.sampler.get_samples() is m.posteriors` → **True** ; `posteriors_full` dict distinct, mêmes arrays | identité avec le cache numpyro non documentée |
| H6 | `summary()` DataFrame arviz ; `sampler_stats()` ebfmi 0.977 ; `to_az()` DataTree `{chain:2, draw:100}` (+ observed_data) ; `m.sample()` → `{'y':(200,200)}` ; `get_yrep` → `(200,200)` ; `trace_model` sites `{beta:latent/Normal/(), sigma:…, y:observed/Normal/(200,)}` ; `to_latex` OK ; `loo` OK (ELPDData) | conforme ; `m.diag.summary/rhat` sont les méthodes bindées `_summary/_rhat` (patch_diag), pas celles de diagWIP |
| H6b | `summary(include='beta')` → `m.posteriors` réduit à `['beta']` **de façon persistante** ; `filter_posteriors()` le restaure depuis `_full` | conforme (documenté, mais effet de bord persistant) |
| H4b | 2ᵉ fit `extra_fields=('diverging',)` → `sampler_stats()` = `{'divergences':0}` (dégradation douce) ; **`per_chain=True` → `TypeError: NoneType has no len()`** (l.1225) | non documenté |
| H4c | `fit(model2)` (modèle sans `y`, `obs=None`) → `TypeError: model2() got an unexpected keyword argument 'y'` — le `data_on_model` de l'ancien fit est resplatté | M-36 confirmé ; message sans rapport avec la cause |
| H4d | `fit(model3)` (colonne `z` absente) → `TypeError: argument after ** must be a mapping, not str` **et `m.data_on_model` reste la chaîne d'erreur** après l'échec | M-09 confirmé + **état empoisonné persistant** (non listé dans l'audit Main) |
| H4e | `fit` → `svi` : `posteriors (60,)` mais `posteriors_full (100,)` et `posteriors_by_chain (2,50)` **de l'ancien fit** ; `sampler_stats()` après SVI renvoie leapfrog/ebfmi/step_size de l'ancien MCMC ; `filter_posteriors(include='beta')` → `beta (100,)` = **draws MCMC ressuscités** ; `m.summary()`/`m.diag.summary()` après SVI fonctionnent (SVI_BF duck-type `num_chains=1`, `group_by_chain` ajoute un axe 1 — svi_numpyro.py:11/62) | M-05 confirmé empiriquement + variante `sampler_stats` |
| H7 | `save`→`load` : posteriors/sampler/diag/dist/`_rng`(graine 7)/model/data_on_model tous survivants ; `summary`, `sampler_stats`, `sample` OK après load ; pkl 182 Ko ; `m.load(path)` fonctionne (alias délibéré datasets.py:65-68) | conforme ; défaut de nom : `model_name=None` après fit → toujours `bf_model.pkl` |
| H7b | `history['data'] is m.data_on_model` → True ; `history['posteriors'] is m.posteriors` → True | aliasing confirmé (déjà noté dans RELEASE_AUDIT:812) |
| log_prob | sans données : `TypeError: model() missing 2 required positional arguments` ; **avec** `**m.data_on_model` : `TypeError: Only integer scalar arrays can be converted to a scalar index` | M-01 confirmé : `model_args=(kwargs)` (main.py:1783) passe un dict en positionnel, jamais `model_kwargs=` |
| OHE→pd_to_jax | `m.df` après `OHE(['g'])` : `['x','g','x','g_a','g_b']` ; `pd_to_jax` → `AttributeError: 'DataFrame' object has no attribute 'dtype'` (manip.py:263) | C1 confirmé, mode de crash pandas 3.0 non listé tel quel |
| SampledData.hdi | entrée `(500,3)` → retourne un tuple de 2 scalaires (HDI calculé en mélangeant lignes et colonnes, sans erreur) | docstring « 1D » seulement ; aucun garde-fou |
| set_seed | `m.set_seed(99)` → `m._rng.base=99` **et** `m.dist._dist._rng.base=99` (objet partagé) | sain |

## 3. Vulnérabilités d'intégration

**V1 — Sévérité : Critique** — `Main/main.py:832-836` → `Main/main.py:1421` (svi → filter_posteriors/`posteriors_full`) `[RÉSOLU ec8795df]`
Risque : `svi()` ne rafraîchit que `m.posteriors` ; `posteriors_full`/`posteriors_by_chain_full`/`posteriors_by_chain` gardent le fit MCMC précédent. Tout appel ultérieur à `filter_posteriors` (ou `summary(include=)`, l.1280-1281) **remplace `m.posteriors` par les draws de l'ancien fit**, silencieusement.
Scénario (vérifié) : `m.fit(A)` 2×50 → `m.svi(B)` 60 draws → `m.filter_posteriors(include='beta')` → `m.posteriors['beta'].shape == (100,)` (MCMC de A) au lieu de `(60,)` (SVI de B). Confirmation empirique de M-05.

**V2 — Sévérité : Critique** — `Main/main.py:832-836` → `Main/main.py:1181-1231` (svi → sampler_stats) `[RÉSOLU ec8795df]`
Risque : `sampler_stats()` après un SVI lit `extra_fields_by_chain` **de l'ancien fit MCMC** (svi ne le vide pas) et rapporte leapfrog/e-BFMI/step_size d'un échantillonneur qui n'a pas tourné — diagnostics fantômes présentés comme ceux du dernier fit.
Scénario (vérifié) : fit → svi → `m.sampler_stats()` renvoie `{'leapfrog_total': 514, 'ebfmi': 1.094, 'step_size': 0.014, …}` alors que le dernier estimateur est variationnel.

**V3 — Sévérité : Critique** — `Main/main.py:537-538` → `Main/main.py:665` (fit → fit, `data_on_model` inter-fits) `[RÉSOLU ec8795df]`
Risque : `fit(model2)` avec `obs=None` réutilise le `data_on_model` du fit précédent sans validation contre la signature du nouveau modèle ; le splat `**run_data` produit un `TypeError` dont le message accuse le *modèle*, pas l'état périmé. Pire, si les deux modèles partagent des noms d'args aux **bonnes clés mais mauvais contenus** (ex. `y` d'un autre `m.df`), aucun appel d'erreur — postérieurs calculés sur les données du modèle précédent.
Scénario (vérifié, cas visible) : fit(`model(x,y)`) → `fit(model2(x))` → `TypeError: model2() got an unexpected keyword argument 'y'`. Confirmation empirique de M-36 ; le cas silencieux est la variante dangereuse.

**V4 — Sévérité : Critique** — `Data/manip.py:270` → `Main/main.py:560/665` (pd_to_jax → fit) + état résiduel `[RÉSOLU 84f84c87]`
Risque : colonne manquante → `pd_to_jax` retourne la **chaîne** `"Error, no data found"`, stockée dans `m.data_on_model`, splattée → `TypeError: argument after ** must be a mapping, not str`. **Après l'échec, la chaîne reste dans `m.data_on_model`** : toute lecture ultérieure qui s'en sert (`_observed_sites` l.1743-1749, `get_history`, `save`) consomme un état corrompu au lieu de `None`.
Scénario (vérifié) : fit d'un modèle avec une colonne `z` absente de `m.df` → TypeError sans lien avec la cause, puis `m.data_on_model == 'Error, no data found'` persistant. Confirmation empirique de M-09/C6, aggravée par la persistance.

**V5 — Sévérité : Majeure** — `Main/main.py:680` → cache interne `numpyro.infer.mcmc.MCMC` [RÉSOLU ba4e75d4]
Risque : `m.posteriors` **est** le dict mis en cache par `MCMC.get_samples()` (identité vérifiée : `m.sampler.get_samples() is m.posteriors` → True). Une mutation in-place (`m.posteriors['beta'] = …`, `del`, post-processing type non-centered→centered) mute l'état du sampler lui-même : `m.summary()` (qui repasse par `az.from_numpyro(self.sampler)`, l.1338-1357) et `m.sample()` (l.1653) verraient alors des valeurs différentes de `m.posteriors_full` — divergence silencieuse entre les deux vues documentées comme « copies ».
Scénario : `m.posteriors['sigma'] = jnp.log(m.posteriors['sigma'])` puis `m.summary()` : la table mélange le `sigma` transformé (via le sampler muté… en fait `az.from_numpyro` relit le cache muté) et les PPC de `m.sample()` qui lisent le même dict muté — plus aucune trace des draws originaux ailleurs que dans `_full`.

**V6 — Sévérité : Majeure** — `Main/main.py:652-657` → `Main/main.py:1224-1225` (extra_fields → sampler_stats per_chain) [RÉSOLU ba4e75d4]
Risque : `fit(extra_fields=('diverging',))` (ou gibbs → `extra_fields=()`) puis `sampler_stats(per_chain=True)` → `TypeError: object of type 'NoneType' has no len()` (`range(len(steps))` avec `steps=None`) ; le mode poolé se dégrade proprement, pas le mode per-chain.
Scénario (vérifié) : fit avec `extra_fields=('diverging',)` → `m.sampler_stats(per_chain=True)` plante.

**V7 — Sévérité : Majeure** — `Data/manip.py:86` → `Data/manip.py:263` (OHE → pd_to_jax) [RÉSOLU antérieurement — vérifié, test ba4e75d4]
Risque : `OHE` concatène le frame `get_dummies` (qui **recontient** les colonnes non encodées) à `m.df` → colonnes dupliquées ; `pd_to_jax` fait `self.df[arg]` qui renvoie un DataFrame → `AttributeError: 'DataFrame' object has no attribute 'dtype'` (pandas 3.0) — message sans lien avec la duplication.
Scénario (vérifié) : `m.data(df)` ; `m.OHE(['g'])` ; modèle avec arg `x` → crash l.263. Confirmation empirique de C1 avec son mode de crash exact.

**V8 — Sévérité : Majeure** — `Main/main.py:1783` → `numpyro.infer.util.initialize_model` [RÉSOLU antérieurement — vérifié, test ba4e75d4]
Risque : `model_args=(kwargs)` passe le dict des données en **argument positionnel unique** (les parenthèses ne font pas un tuple de kwargs) ; `m.log_prob` est cassé dans les deux usages : sans données → `TypeError: missing 2 required positional arguments` ; avec `**m.data_on_model` → `TypeError: Only integer scalar arrays can be converted to a scalar index`.
Scénario (vérifié) : les deux variantes exécutées. Confirmation empirique de M-01.

**V9 — Sévérité : Majeure** (statique, TFP absent) — `Samplers/mcmc_tfp.py:120-133` → `Main/main.py:711-712` → `Main/main.py:1653/1683` [RÉSOLU antérieurement — vérifié, test ba4e75d4]
Risque : `get_samples(group_by_chain=…)` TFP ignore son argument et renvoie toujours `(chains, draws, …)` ; `fit` affecte le même objet à `posteriors` et `posteriors_by_chain`. Le contrat aval (pool `(draws,)`) est enfreint : `m.sample()` passe un axe 0 = chaînes à `Predictive` (PPC à 2 dims de batch) et `m.sample(samples=n)` compte les **chaînes** (`ValueError: … only N = 4 draws`). Confirmation statique de M-04.

**V10 — Sévérité : Majeure** (statique) — `Main/main.py:710` → `Diagnostic/Diag2.py:384-397` (tfp → to_az) [RÉSOLU ba4e75d4]
Risque : `zip(sample_stats_name, self.sampler.sample_stats)` (l.387) et `zip(var_names, self.sampler.posterior)` (l.393) **sans `strict=`** : si `model_info`/`posterior`/`sample_stats` du sampler TFP ne sont pas exactement alignés sur les 4 noms codés en dur, la trace ArviZ est construite **tronquée silencieusement** (variables ou stats manquantes) et les diagnostics rhat/ess/summary tournent sur un sous-ensemble sans avertissement.

**V11 — Sévérité : Majeure** — `Main/main.py:1280-1281` → `Main/main.py:1421-1423` (summary → filter persistant) [RÉSOLU antérieurement — vérifié, test ba4e75d4]
Risque : `summary(include=/exclude=)` — une méthode de *lecture* — réduit de façon persistante `m.posteriors`/`m.posteriors_by_chain`. Seul `filter_posteriors()` sans argument rétablit. Combiné à V1, un `summary(include=…)` après un SVI ressuscite l'ancien fit (vérifié) ; combiné à V5, la vue « filtrée » et la vue sampler divergent.
Scénario (vérifié) : `m.summary(include='beta')` → `list(m.posteriors.keys()) == ['beta']` jusqu'au reset.

**V12 — Sévérité : Mineure** — `Main/main.py:1584` / `1338` (sample/summary avant fit) `[RÉSOLU ec5589f3 + 8c5b23ba — M-22/M-37]`
`m.sample()` sur instance neuve → `AttributeError: 'bf' object has no attribute 'model'` (l.1584, avant le message promis) ; `m.summary()` → `AttributeError: 'bf' object has no attribute 'diag'`. Vérifié. Confirmation empirique de M-22/M-37.

**V13 — Sévérité : Mineure** — `Main/main.py:1801-1807` (get_history → aliasing) `[RÉSOLU ec5589f3 — M-24]`
`history['data'] is m.data_on_model` et `history['posteriors'] is m.posteriors` (vérifié True/True) : le « snapshot » du fit se rétro-mute au fit suivant ou via `filter_posteriors`/M-02 — l'historique ne rejoue pas le fit d'origine.

**V14 — Sévérité : Mineure** — `Main/main.py:727` → `Main/main.py:1874` (fit → save défaut) `[RÉSOLU ec5589f3 — M-17]`
`fit` termine par `self.model_name = None` (l.727) ; `save()` sans chemin teste `self.model_name` (l.1874) → **toujours** `bf_model.pkl`, jamais `<modèle>_bf.pkl` comme documenté. Vérifié.

**V15 — Sévérité : Mineure** — `Utils/SampledData.py:588-608` (hdi, ndim non gardé) `[RÉSOLU 94e80673]`
`hdi` trie le tableau entier et fenètre sur l'axe 0 : sur une entrée 2D `(500,3)` il retourne un tuple de scalaires calculé en mélangeant lignes et colonnes, sans erreur (vérifié). La doc annonce « 1D » mais rien n'est validé — et `SampledData` est renvoyé par toutes les distributions `to_jax=False` (np_dists.py, 91 occurrences), y compris multidimensionnelles.

**V16 — Sévérité : Mineure** — `Diagnostic/Diag2.py:323` → méthodes non bindées (`Diag2.summary` l.850, `_summary_row`, `diagnose`) `[RÉSOLU 27d7ba49]`
`diagWIP.__init__` capture `posterior_samples` à la construction ; `bind_diag_to_model` (patch_diag.py:1054+) remplace summary/rhat/ess/loo/plots sur l'instance, masquant ces méthodes. Un utilisateur qui garde la référence `d = m.diag` avant un 2ᵉ fit et appelle une méthode non bindée lit l'instantané de l'ancien fit (le binding ne se re-exécute que si `fit` recrée `m.diag`).

**V17 — Sévérité : Mineure** — `Main/main.py:684-686` → `patch_diag.py:560-567` (cache `_az_diag`) `[RÉSOLU 27d7ba49]`
Le cache est bien invalidé par identité de sampler (sain, vérifié), mais `to_az` y est appelé dans un `except Exception: pass` (l.565-566) : la vraie cause d'un échec de trace (ex. tfp tronqué, V10) est avalée puis resurgit plus loin, déconnectée de sa source.

**V18 — Sévérité : Mineure** — `Main/main.py:652-657` → `Main/main.py:1181-1187` (gibbs → sampler_stats) `[RÉSOLU 27d7ba49]`
Les samplers composites (`mixed_hmc`, `discrete_gibbs`, `hmc_gibbs`) reçoivent `extra_fields=()` ; `sampler_stats()` lève alors un `RuntimeError` correct mais dont le message (« if extra_fields was overridden ») masque la cause réelle (choix du sampler). Documenté de biais.

## 4. Inventaire des coutures

| # | Couture (producteur → consommateur) | Statut |
|---|--------------------------------------|--------|
| S1 | `bf.__init__:166` → `SetDevice/set.py:9` (config x64/devices) | saine (numpyro) / fragile (tfp force x64=False, main.py:297) |
| S2 | `bf.__init__:279-289` → singletons `dist` des modules Effects/Gaussian/DPMM/GMM/surv | fragile (multi-instances, M-08) |
| S3 | `manip.data:60` → `m.df` (copie protectrice) | saine (vérifié) |
| S4 | `manip.OHE:86` / `index:111` / `scale:158` → `m.df` + `data_modification` | cassée (OHE duplique, C1) / fragile (index NaN→-1) |
| S5 | `m.df` → `manip.pd_to_jax:240-283` (signature-driven) | fragile (dtype hors map) / cassée (colonnes dupliquées → AttributeError l.263 ; colonne absente → `str` l.270) |
| S6 | `pd_to_jax:283` → `fit:560` (`data_on_model`) → splat `:665` | cassée (str splattée, V4) / fragile (réutilisation inter-fits V3, M-36) |
| S7 | modèle utilisateur + `DistProxy:147-170` → sites `numpyro.sample` (`np_dists.py:5741`) | fragile (inférence de noms par bytecode ; collision → `ValueError` main.py:667-676, garde saine) |
| S8 | `fit:597/661` → `mcmc_numpyro.py:114` → `MCMC.run(**run_data)` | fragile (collision clés de données / signature `run`, M-13) |
| S9 | `MCMC.get_samples()` → `m.posteriors` (main.py:680) | fragile (identité avec le cache du sampler, V5) |
| S10 | `fit:682-683` → `posteriors_full`/`_by_chain_full` (copies superficielles) | saine (arrays immuables) dans fit→fit ; cassée après svi (V1) |
| S11 | `fit:684` → `diagWIP.__init__:314-342` (cross-check num_chains l.329-338) | saine (garde explicite) |
| S12 | `fit:686` → `patch_diag.bind_diag_to_model:712/1054` (binding sur l'instance) | saine ; fragile (méthodes diagWIP masquées + instantané périmé, V16) |
| S13 | `summary_old:1338-1357` → `Diag2.to_az:379` (`az.from_numpyro(sampler)`) | saine (numpyro, vérifié) / cassée (tfp : zip sans strict, V10) |
| S14 | `summary:1280-1281` → `filter_posteriors:1421-1423` | fragile (effet de bord persistant, V11) |
| S15 | `fit:677-679` (`extra_fields*`) → `sampler_stats:1181-1231` | fragile (périmé après svi V2 ; `per_chain` sans `num_steps` V6 ; gibbs V18) |
| S16 | `sample:1724-1730` (`Predictive`) → `ppc.get_yrep:100` → plots PPC | saine (vérifié `(200,200)` ; `_ExpandObserved:87-119` corrige les scalaires broadcastés) |
| S17 | `_observed_sites:1731-1765` → `m.obs_args` → PPC/`_resolve_obs_y` (patch_diag:586) | saine ; fragile (`obs_args=None` jusqu'au premier `sample()`) |
| S18 | `m.data_on_model` → `PostModel/model_trace.trace_model:231` / `to_latex` | saine (vérifié sites + LaTeX) |
| S19 | `save:1860`/`__getstate__:1841` → cloudpickle → `load:1889`/`__setstate__:1847` | saine (round-trip complet vérifié) ; fragile (nom par défaut mort V14 ; `history` aliasé V13) |
| S20 | `get_history:1801-1807` → `m.history` | fragile (aliasing, V13) |
| S21 | `svi:832-836` → attributs post-fit (`posteriors_full`, `by_chain`, `extra_fields`) | cassée (V1+V2, vérifié) |
| S22 | `fit:711-712` (tfp) → contrat postérieurs commun | cassée (statique, V9/M-04) |
| S23 | `SVI_BF` (svi_numpyro.py:11/62) → `diagWIP`/`to_az` (`az_from_dict` Diag2:370-377) | saine (duck-typing chaîne-unique vérifié : summary OK après svi) |
| S24 | `log_prob:1783` → `numpyro.initialize_model` | cassée (V8/M-01, vérifié dans les deux usages) |
| S25 | distributions `sample=True, to_jax=False` (np_dists, 91×) → `SampledData.hdi:588` | fragile (ndim non gardé, V15) |
| S26 | `m.load` (instance `Resources.datasets.load`, main.py:249) → `bf.load` (datasets.py:65-68) | saine (alias délibéré, vérifié) |
| S27 | `set_seed:330` → `m._rng` partagé avec `dist` singleton | saine (propagation vérifiée) |

## 5. Décompte par sévérité

| Sévérité | Nombre | IDs |
|---|---|---|
| Critique | 4 | V1, V2, V3, V4 |
| Majeure | 7 | V5, V6, V7, V8, V9, V10, V11 |
| Mineure | 7 | V12, V13, V14, V15, V16, V17, V18 |
| **Total** | **18** | |

Coutures : 27 inventoriées — **19 saines** (dont 6 avec réserve « fragile » légère comptée dans leur ligne), **8 fragiles**, **5 cassées** (S4-OHE, S5/S6-pd_to_jax, S21-svi, S22-tfp, S24-log_prob).

**Constat de synthèse.** Le chemin nominal `bf() → m.data → scale → fit → summary → save/load` est **solide de bout en bout** (vérifié empiriquement, y compris le round-trip complet et les PPC). Les coutures cassées sont toutes sur les **transitions d'état** : (a) entre estimateurs (fit→svi : trois attributs périmés, deux façons de ressusciter l'ancien fit — V1/V2) ; (b) entre fits (données non revalidées — V3 ; état d'erreur persistant — V4) ; (c) entre émetteur numpyro et consommateurs qui relisent le **sampler** plutôt que les attributs (`summary`, `sample`) alors que d'autres lisent les attributs (`sampler_stats`, plots bindés, `filter_posteriors`) — deux sources de vérité qui ne coïncident plus dès qu'un fit partiellement rafraîchi a eu lieu (V5/V11). Un rafraîchissement complet des attributs post-estimation dans `svi` (comme le fait `fit` l.677-687) et une copie (`dict(...)`, tableau des clés) à l'entrée de `fit`/`svi` fermeraient les quatre critiques.

---

### 21 — Audit d'interfaces : coutures de construction de modèle (Models/* et bibliothèque)

> Passe « module-reviewer », racine `/home/sosa/work/BF`, chemin prioritaire numpyro.
> Complément de `20_Lifecycle.md` (chemin nominal data→fit) : cette passe porte exclusivement sur
> les **coutures de construction de modèle** C1→C8, hors chemin nominal.
> Méthode : `grep -n` systématique + lecture ciblée + vérifications empiriques légères
> (AST, `inspect.signature().bind`, `numpyro.handlers.trace`, `pd_to_jax` réel sur les objets du
> registre, deux instances `bf()` sans fit). BayesForge v0.0.60. Aucune modification de source.
> Les bugs internes déjà documentés (M-xx, C-xx, V1-V18 de `20_Lifecycle.md`, NameError connus)
> ne sont re-listés que lorsque **la couture elle-même les aggrave**.

## 1. Tableau des coutures C1→C8

| # | Couture producteur `fichier:ligne` → consommateur `fichier:ligne` | Contrat supposé | Contrat réel | Statut |
|---|---|---|---|---|
| C1a | `Main/main.py:279-289` (recâblage `.seed`/`._rng` des singletons `dist` de Effects/Gaussian/model_effects/DPMM/GMM/surv) → appels `dist.xxx()` dans ces modules (référence au **global module**, résolue à l'appel) | le singleton de module pointe vers l'instance bf qui construit le modèle ; un modèle construit avant `bf()` voit le recâblage | liaison tardive OK (le recâblage est effectif pour tout appel après `bf()`) ; mais **la dernière instance bf gagne** : après `bf()` #2, le singleton suit `_rng` de bf2 ; `m1.set_seed()` ne contrôle plus les DGP des modèles de bibliothèque utilisés via `m1` (vérifié : `GMM.dist._rng is m2._rng` → True, `m1.set_seed(11)` laisse `base=2`) | **cassée** (multi-instances) / saine (mono-instance) |
| C1b | `Main/main.py:279-289` (liste de recâblage) → `Network/model_effects2.py:57` (`dist = _UnifiedDist()`), `Models/PCA.py:32` (`self.dist = dist()`), `Models/surv.py:71` (`survival_old.self.dist`) | tous les singletons `UnifiedDist` du package sont recâblés par `bf()` | **hors liste** : `model_effects2` (donc `m.net2`), PCA (instance propre par objet), `survival_old` gardent `_rng=None`, `seed=True` → `time_ns()` par tirage (vérifié : deux tirages `sample=True` de `ME2.dist` différents malgré un seed posé) | **cassée** (DGP non reproductible, même avec un seul bf) |
| C1c | `Main/main.py:251-258` (`self.dist = DistProxy(_unified)` ; `self.bnn.dist = _unified`) → modèles utilisant `m.dist`/`m.bnn` | les modèles pilotés par `m` partagent le RNG de `m` | vérifié `m.bnn.dist is m.dist._dist` → True (numpyro) ; branche tfp (`main.py:291-297`) : `self.dist` = **classe** `tfp_dists`, singletons de module non touchés → les modèles de bibliothèque émettraient des sites numpyro dans un sampler tfp (statique, TFP absent) | saine (numpyro) / fragile (tfp, statique) |
| C2a | `Models/SRM.py:9` (`from ...np_dists import UnifiedDist as dist` — **classe**, jamais instanciée ; shadowe l'import numpyro de la ligne 7) → `SRM.py:199,296,415-516` (`dist.normal/truncated_normal/bernoulli/deterministic`) | `dist` = instance `UnifiedDist` (contrat de tous les autres modules) | appels positionnels = méthode non liée : `self` avale le 1ᵉʳ argument → **priors silencieusement décalées** (vérifié via trace : `Normal loc=[2.5…] scale=[1.]` au lieu de `Normal(0, 2.5)`) ; appels en mots-clés (`dist.bernoulli(logits=, obs=)`) → `TypeError: missing 'self'` ; `dist.deterministic` → `AttributeError` (la méthode s'appelle `track`, np_dists.py:182) — l.515-516 **inconditionnelles** | **cassée** |
| C2b | `Network/util.py:163-181` `mat_to_edgl` (dyades **non ordonnées** i<j via `triu(k=1)`, 2 colonnes directionnelles `[M[i,j], M[j,i]]`, 0-based) → `Models/SRM.py:312-330` `numpyro_bsrm_model` (dyades **ordonnées** `meshgrid(S≠R)`, N(N−1) lignes, ordre row-major) | une sortie `mat_to_edgl` (ex. `srm.network`, SRM.py:23) se consomme telle quelle dans `numpyro_bsrm_model(network=…)` | deux conventions dyadiques incompatibles dans la même classe : (N(N−1)/2, 2) triu vs (N(N−1),) meshgrid ; `srm.numpyro_bsrm_model(network=srm.network)` décale lignes et directions | **cassée** |
| C2c | `Network/util.py:163` `mat_to_edgl` → `Network/model_effects.py:185-199` (`prepare_dyadic_effect`), `:292-307` (`block_prior_to_edglelist`), `SRM.py:93,283` | même ordre triu i<j, colonnes [ij,ji], indices 0-based | cohérent partout (`block_prior_to_edglelist` reconstruit `[b[v_i,v_j], b[v_j,v_i]]` dans l'ordre triu ; `SRM.import_block_predictors` stocke 0-based, option `index1` soustrait 1 pour utilisateurs R) | **saine** |
| C2d | `Network/model_effects.py` (edgelist, singleton recâblé) → `Network/model_effects2.py:62` `NeteffectMatrix(Neteffect)` (override `block_model`/`sender_receiver`/`dyadic_effect` en (N,N), propre singleton l.57 **non recâblé**) | `m.net` et `m.net2` sont deux implémentations équivalentes du même contrat | divergence documentée (sharding) et mathématiquement alignée (docstring model_effects2.py:1-40) ; mais `net2` **hérite** des méthodes edgelist de `Neteffect` → piège de mélange de formats sur le même objet ; RNG du singleton propre hors recâblage (voir C1b) | fragile |
| C3a | `Utils/Effects.py:443-480` `varying_slope_non_centered` (corps : `m.dist.*`, l.458/462/467 — `m` inexistant au module, vérifié AST : seuls usages hors docstring) → `PostModel/to_latex.py:530`, `PostModel/model_graph.py:213` (whitelist de l'API `effects.*`) | les fonctions listées par PostModel comme API supportée sont appelables | `NameError: name 'm' is not defined` **à chaque appel** (vérifié) ; les fonctions sœurs (`varying_intercept_non_centered`, `varying_intercept_slope_non_centered`) utilisent `dist` correctement et tracent OK | **cassée** |
| C3b | `Data/manip.py:95-125` `index()` (codes **0-based**, documenté) → `Network/model_effects.py:388-427` `nodal_random_effects(random_group_index=…)` (**1-based**, documenté `Documentation/reference/net.qmd:627`) | les deux conventions de la doc officielle se combinent | aucun convertisseur ; codes 0-based → `group_idx_0 = idx − 1` → **−1** → gather JAX qui **boucle sur le dernier groupe** (vérifié : codes `[0,1,2]` → effets `[30,10,20]`) — groupement faux, aucune erreur | **cassée** |
| C3c | `Models/*` → `Utils/Effects.py` | les modèles de bibliothèque appellent les helpers d'effets | **auc appel** (vérifié grep) ; la couture réelle est utilisateur → `m.effects` (main.py:230) + whitelist PostModel (C3a) | saine (inexistante) |
| C4a | `Utils/link.py:52-66` `inv_logit_scale` (`(sigmoid(x)−0.5)·2`) → `NBDA/NBDA.py:605,632,658` (`social_influence_weight`) | docstring + `Documentation/reference/link.qmd:100-118` : « scaling between 0 and 1 » (et ligne Returns = `1/(1+exp(−x))`, formule d'`inv_logit` !) | image réelle **(−1, 1)** : poids d'influence sociale **négatif possible** (anti-conformité) et amplitude ≠ papier cité (sciadv.aax9070) ; seul consommateur du package : NBDA | **cassée** (doc vs espace réel) |
| C4b | `Utils/link.py:27` `logit` sans clipping → modèles de bibliothèque | les liens de `m.link` sont utilisables dans les graphes | dans la bibliothèque, `logit` n'est appliqué qu'à des **constantes** hors gradient (SRM.py:281 `Mu = logit(dat_B…)` calculé à l'import des données ; `Neteffect.logit(block_build_mu_ij)` sur base_rate/ss > 0) → pas d'exposition gradient dans les modèles livrés ; trois implémentations dupliquées (link.py:27, model_effects.py:35, SRM.py:12) ; le risque NaN reste entier pour les modèles utilisateur via `m.link` | fragile (connu, non aggravé par la couture) |
| C5a | `Models/BNNC.py:52-61` (`m.dist.normal(…, shape=(D_X,D_H1))` puis `m.bnn.layer_linear(X, dist=W)`) → `BNN/bnn.py:46-58` (`jnp.matmul(X, dist) + bias`) | convention de poids (in, out) | cohérent : `X (N,D_X) @ W (D_X,D_H1)` ; `m.bnn.dist` partagé avec `m.dist` (vérifié) ; sites nommés explicitement | **saine** |
| C5b | `Main/main.py:245` (`self.bnn = bnn()` **avant** la branche backend) → `BNN/bnn.py:16` (`self.dist = dist(rand_seed)`) ; `Samplers/SBI/npe.py:37` (`_BNN = bnn()` module) | `bnn.dist` = backend courant de `m` | numpyro : écrasé sainement l.254 ; tfp : `bnn.dist` reste un `UnifiedDist` numpyro `time_ns` (statique) ; `_BNN.dist` non recâblé (`_rng=None`, vérifié) mais **inoffensif** : npe n'utilise que `layer_linear`/`layer_mixture_normal(create_obj)`/sites `sample=False` (npe.py:59,81,271 — RNG tenu par numpyro/SVI) | fragile (tfp) / saine (numpyro) |
| C5c | modèles successifs → `m.bnn` (bâtisseur **à état** : `layer_attention`/`layer_toeplitz`/`layer_compound_symmetry`/`layer_diagonal` stockent `self.emb_q`, `self.log_sigma`, … sur l'objet partagé par bf) | `m.bnn` est une fabrique sans mémoire | deux modèles construits avec le même `m.bnn` **s'écrasent mutuellement** les attributs de couche ; les tirages `sample=True` de la 2ᵉ construction remplacent les sites de la 1ᵉʳᵉ dans l'objet | fragile |
| C6a | `NBDA/NBDA.py:5` (`UnifiedDist as dist`, **classe**, jamais instanciée) → `NBDA.py:492-549` (`dist.uniform(0.0001, 0.9999, name='ces_alpha')`, `dist.normal(0, 2, …)`) | `dist` = instance ; priors Normal(0,2), Uniform(0.0001, 0.9999) | méthode non liée : `self=0` → **Normal(loc=2, scale=1)** partout ; `uniform` → `Uniform(0.9999, 1.0)` dégénéré ; le site observé utilise `numpyro.sample` directement (l.589) → **le modèle tourne au bout sans erreur** avec des priors faux (mécanisme vérifié par `signature().bind`, cf. C2a) | **cassée** |
| C6b | `Main/main.py:214` (`self.nbdaModel = False`, jamais mis à True) et `:543-547` (`self.model = self.nbda.model` — attribut `self.nbda` **inexistant**) → `Models/models.py:26` (`self.nbda = NBDA.model`, fonction **non liée**) | `fit(model=None)` bascule sur le modèle NBDA ; `m.models.nbda` est un modèle appelable | branche morte (rien ne pose `bf().nbdaModel=True`) ; `fit(model=None)` affiche « Argument model can't be None » puis **plante** dans `pd_to_jax(None)` ; `m.pd_to_jax(m.models.nbda)` → `'Error, no data found'` (colonne `self` requise, vérifié) ; l'usage réel `m.fit(nbda.model)` (méthode liée) fonctionne mais `model.__name__` = `"model"` → `run_model_name`/dispatch plot/nom de save faux | **cassée** |
| C6c | `NBDA/NBDA.py:556-589` `model()` (données lues sur `self.status/self.network/self.T_*`) → `Main/main.py:557-562` (`data_on_model = pd_to_jax(model)`) | les données du modèle passent par `m.df`/`data_on_model` | `pd_to_jax(nbda.model)` → `{'CES': True}` uniquement (le `self` de la méthode liée est masqué, `CES` a un défaut) : les données voyagent dans l'instance NBDA, `m.data()`/`m.df` sont hors circuit — PPC OK car le modèle relit `self`, mais `m.data_on_model` ne décrit plus le fit (sauvegardé/rejoué, il manque tout) | fragile |
| C7a | `Models/models.py:18-35` (registre `m.models.*`, documenté « Built-in models API », `Documentation/reference/models.qmd`) → `Main/main.py:557-562` → `Data/manip.py:272-279` (défauts de signature → `jnp.array(défaut)`) | `m.fit(m.models.X)` construit et ajuste le modèle de la bibliothèque | **5 des 6 entrées crashent dans `pd_to_jax`** (vérifié sur les objets réels) : `gmm` (`initial_means=None` → `ValueError: None is not a valid value for jnp.array`), `dpmm` (`method='marginal'` → `TypeError: <U8 not a valid JAX array type`), `pca` (`X=None` → ValueError), `survival.model` (`censoring=None`), `nbda` (voir C6b) ; seul `bnnc` passe, mais avec `data` **1D** (colonne) là où le modèle exige `(N, D_X)` | **cassée** |
| C7b | `Models/models.py:24` (`self.pca = pca` — **classe**) → `Main/main.py:571-573` (`model.get_model(model.type)`) ; `models.py:27` (`self.survival = survival(parent)` — instance **sans `__name__` ni `__call__`**) → `main.py:555` (`model.__name__`) | les entrées du registre sont des modèles appelables nommés | `m.models.pca` (classe) : pas d'attribut `type` → `AttributeError` au special-case (vérifié `hasattr(pca,'type')=False`) ; l'instance `pca(...)` fonctionne (factory `__call__` l.59-66, `__name__` posé l.30) ; `m.models.survival` : `__name__` absent → `AttributeError` dans `fit`, et `callable()` False (vérifié) → entrée inutilisable telle quelle | **cassée** |
| C7c | `Models/GMM.py:18` `gmm(data, K, initial_means=None)` → `Data/manip.py:250-270` (args **sans défaut = colonnes**) + `Main/main.py:564-569` (KMEANS avec `data_on_model["K"]`) | `K` et `data` se déduisent de `m.df` | `K` sans défaut ⇒ **colonne nommée « K »** obligatoire, produite en `(N,)` là où `plate("components", K)`/`n_clusters=` exigent un scalaire ; `data` multivarié `(N,D)` impossible depuis une colonne 1D ; le special-case KMEANS exige en plus la clé `K` dans `data_on_model` (KeyError sinon si `obs=` partiel) | **cassée** |
| C7d | `Main/main.py:386-391` (`fit(model, obs, …)` — **pas de paramètre `model_name`**) + `models.py:29-35` (`available` = dict sans consommateur, vérifié grep) → résolution par `model.__name__` (l.548-555) | `fit(model_name="gmm")` résout le nom via le registre | la résolution nom→fonction n'existe pas : `model_name` est un **attribut d'état** déduit du `__name__` du callable, consommé par les special-cases (l.564, 571) et le dispatch de plot (l.1822-1833) ; `models.available` est du décoratif mort | fragile (contrat supposé inexistant) |
| C8a | `Models/GMM.py:41-58` / `DPMM.py:80-137` (`plate("components", K)`, `event=1`, `mixture_same_family(name="obs", obs=data)` — obs **event_dim=1**) → `Main/main.py:1653+` `sample()` (`Predictive`) → `Diagnostic/ppc.py:60-115` `get_yrep` (contrat `(S, n_obs)`) | yrep `(draws, n_obs)` 1D | `Predictive` renvoie `(S, N, D)` ; `get_yrep` **aplatit** (ppc.py:106 `reshape(S, −1)`) → `(S, N·D)` : les PPC mélangent les dimensions de la variable dans un même « obs » (pas de crash, sémantique douteuse) ; `_ExpandObserved` (main.py:87-119) gère bien `event_shape` | fragile (garde-foi aplatissant) |
| C8b | conventions `event`/`expand`/`plate` de `Distributions/np_dists.py` (`expand` l.406+, `plate` l.259) → usage dans `Models/*` | les modèles respectent batch/event | usage cohérent (GMM : `sigma event=1` dans `plate(K)` → `scale_tril (K,D,D)` ; DPMM idem ; surv : `poisson` batch 2D sans event ; PCA : `normal(obs=X)` 2D event=0) ; pas d'`expand=` dans les modèles | **saine** |

## 2. Vulnérabilités d'intégration

**W1 — Sévérité : Critique** — `Main/main.py:279-289` → `Models/GMM.py:19`, `Models/DPMM.py:19`, `Models/surv.py:9`, `Utils/Effects.py:9`, `Utils/Gaussian.py:5`, `Network/model_effects.py:10` `[RÉSOLU ec8795df]`
Risque : le recâblage des singletons par mutation fait du **dernier `bf()` créé le propriétaire exclusif** du RNG des DGP de la bibliothèque. Deux instances (mêmes ou deux backends) : les tirages `sample=True` des modèles de bibliothèque utilisés via `m1` suivent `m2` ; `m1.set_seed()` n'a plus d'effet sur eux ; un `bf(backend="tfp")` créé ensuite laisse les singletons pointer vers l'instance numpyro (et ne recâble rien pour sa propre branche).
Scénario (vérifié) : `m1 = bf(rand_seed=1)` ; `m2 = bf(rand_seed=2)` ; `m1.set_seed(11)` → `GMM.dist._rng.base == 2` ; deux tirages `GMM.dist.normal(sample=True)` encadrés de `m1.set_seed(99)` diffèrent. Toute la reproductibilité « DGP » documentée par le commentaire main.py:258-270 est conditionnée à l'existence d'une seule instance bf.

**W2 — Sévérité : Majeure** — `Main/main.py:279-289` (liste de recâblage) → `Network/model_effects2.py:57`, `Models/PCA.py:32`, `Models/surv.py:71` [RÉSOLU 456039cf]
Risque : trois singletons `UnifiedDist` du package sont **absents de la liste** de recâblage : `model_effects2` (donc tout `m.net2`, voie « shard-compatible » documentée), chaque instance `pca`, `survival_old`. Leurs tirages `sample=True` partent de `rand_seed=True` → `time_ns()` par appel : DGP **non reproductible même avec un seul bf** et `set_seed`, et non décorrélation garantie par `_rng.stream()`.
Scénario (vérifié) : `m = bf(rand_seed=7)` ; deux appels `m.net2`-style via `ME2.dist.normal(0,1,sample=True)` → valeurs différentes ; `p.dist._rng is None`.

**W3 — Sévérité : Critique** — `Models/SRM.py:9` → `Models/SRM.py:199,296,415-434,455,492-516` `[RÉSOLU a22aeae6]`
Risque : `dist` y est la **classe** `UnifiedDist` (l'import de la ligne 9 shadowe `numpyro.distributions` de la ligne 7 ; contrairement à DPMM/GMM/surv, aucune ligne `dist = dist()`). Conséquences en trois étages, tous vérifiés : (a) appels positionnels = méthode non liée, le 1ᵉʳ argument est avalé comme `self` → **priors décalées en silence** (`truncated_normal(0.0, 2.5, low=0)` → `TruncNormal(loc=2.5, scale=1.0)` ; `normal(mu_vec, sd_vec)` → `Normal(loc=sd_vec, scale=1.0)` — trace numpyro produite) ; (b) appels en mots-clés (`dist.bernoulli(logits=…, obs=…)` l.492, binomial/poisson/normal l.498-510) → `TypeError: missing 1 required positional argument: 'self'` ; (c) `dist.deterministic` (l.494-516, dont **515-516 inconditionnelles**) → `AttributeError` (la méthode s'appelle `track`). `SRM.numpyro_bsrm_model` et `SRM.model` ne peuvent donc **jamais** s'exécuter, et `block_prior`/`block_effect_srm_class` (l.199/296) corrompraient les priors si on les branchait hors du chemin crashé.
Scénario : `SRM(A).numpyro_bsrm_model(...)` passé à `m.fit` → `AttributeError: type object 'UnifiedDist' has no attribute 'deterministic'` dans le premier trace du modèle — message sans rapport avec l'omission de `dist = dist()`. Classe absente du registre et de la doc d'usage (seul `m.net` est documenté) : modèle orphelin, mais public (`BayesForge.Models.SRM`).

**W4 — Sévérité : Majeure** — `Network/util.py:163-181` (`mat_to_edgl`, N(N−1)/2 dyades triu, colonnes [ij,ji]) → `Models/SRM.py:23` vs `Models/SRM.py:312-330` (`numpyro_bsrm_model`, N(N−1) dyades ordonnées meshgrid) [RÉSOLU 456039cf]
Risque : la même classe produit (`__init__` : `self.network = mat_to_edgl(network)`) et consomme (`numpyro_bsrm_model(network=…)` attend l'edgelist des **observations**, indexées sur les dyades ordonnées) deux conventions dyadiques incompatibles — nombre de lignes (÷2), ordre, et sens des colonnes. Brancher la sortie de l'un dans l'autre décale silencieusement lignes et directions i→j/j→i (le crash par broadcast n'arrive qu'après les sites latents, cf. W3).
Scénario : `s = SRM(A); s.numpyro_bsrm_model(s.network, …)` → `N_dyads = s.network.shape[0]` faux de moitié vs `long_ids_int` ; même corrigé, l'ordre triu ≠ ordre meshgrid.

**W5 — Sévérité : Majeure** — `NBDA/NBDA.py:5` → `NBDA/NBDA.py:492-549` (priors) → `Main/main.py:661-666` (`fit`) [RÉSOLU antérieurement — vérifié, test 456039cf]
Risque : même couture classe/instance que W3, mais ici **aucun crash** : les sites latents passent par la méthode non liée avec `self` avalé (les priors `sample=False` ne touchent jamais `self`), le site observé utilise `numpyro.sample` brut (l.589). `m.fit(nbda.model)` aboutit donc **de bout en bout** avec toutes les priors fausses : `Normal(0,2)` → `Normal(loc=2, scale=1)` (l.495-549), `Uniform(0.0001, 0.9999)` → `Uniform(0.9999, 1.0)` dégénéré (l.492/526). L'audit 11_NBDA connaît « priors fausses » ; la couture l'aggrave : le chemin `bf.fit` complet est silencieux (postérieurs publiés, `save()` round-trip, PPC cohérents avec les priors fausses).
Scénario : diffusion NBDA standard (`NBDA(network, status)` → `import_covDF` → `m.fit(nbda.model)`) → inférence aboutie, biais systématique des taux de transmission (priors centrées sur 2 au lieu de 0, échelle 1 au lieu de 2, `ces_alpha` pincé à ~1).

**W6 — Sévérité : Majeure** — `Utils/Effects.py:458-467` (`m.dist` hors tout contexte — seuls usages `m.` du fichier hors docstring, vérifié AST) → `PostModel/to_latex.py:530`, `PostModel/model_graph.py:213` [RÉSOLU antérieurement — vérifié, test 456039cf]
Risque : `varying_slope_non_centered` lève `NameError: name 'm' is not defined` à **chaque appel** (vérifié ; les deux fonctions sœurs non_centered tracent OK), alors que PostModel la traite comme API supportée (dispatch LaTeX et graphe la listent explicitement, donc la documentation générée la présente comme un composant du modèle). L'utilisateur guidé par `to_latex`/`model_graph` vers les helpers `m.effects.*` tombe sur la seule variante cassée.
Scénario (vérifié) : `m.effects.varying_slope_non_centered(cat_idx, grp_idx, 2, 2)` → NameError ; `m.to_latex()` sur un modèle qui l'utilise ne l'atteint jamais (le modèle crache avant).

**W7 — Sévérité : Majeure** — `Data/manip.py:95-125` (`index()`, codes **0-based**, documenté) → `Network/model_effects.py:388-427` (`nodal_random_effects`, `random_group_index` **1-based**, documenté `net.qmd:627`) [RÉSOLU 456039cf]
Risque : les deux conventions officielles se rencontrent sans convertisseur : passer une colonne `index_<col>` dans `random_group_index` fait `idx − 1` → **−1** pour le groupe 0, que JAX résout en **dernier groupe** (vérifié : codes `[0,1,2,0,1]` → effets `[30,10,20,30,10]`). Effets hiérarchiques attribués au mauvais groupe pour 100 % des observations, sans erreur ni warning. Inversement, des indices 1-based corrects fonctionnent — le piège est exactement la combinaison documentée des deux API.
Scénario : `m.index('sex')` → `m.fit(mon_model)` où `mon_model` appelle `m.net.nodal_random_effects(design, index_sex, …)` → groupement décalé, `num_random_groups` cohérent avec `len(index_map)` donc le −1 reste dans les bornes.

**W8 — Sévérité : Majeure** — `Utils/link.py:52-66` (`inv_logit_scale = (sigmoid−0.5)·2`, image **(−1,1)**) → `NBDA/NBDA.py:605,632,658` (`social_influence_weight`) + `Documentation/reference/link.qmd:100-118` [RÉSOLU antérieurement — vérifié, test 456039cf]
Risque : la docstring et la référence affirment « scaling between 0 and 1 » (et rendent même la formule d'`inv_logit` dans la ligne Returns). Le poids d'influence sociale de NBDA — modèle phare, référence sciadv.aax9070 citée — vit en réalité dans (−1,1) : une inférence peut donner une influence **négative** (anti-conformité) là où le modèle théorique l'admet en [0,1] borné ; toute lecture de posterior/`ces` fondée sur la doc ([0,1]) est fausse d'un facteur 2 et d'un support.
Scénario : posterior de `social_influence_weight` échantillonné autour de −0.3 → interprété « influence nulle » par la doc ([0,1] ne peut pas être négatif) alors que le modèle a inféré une répulsion.

**W9 — Sévérité : Majeure** — `Models/models.py:18-27` (objets du registre) → `Data/manip.py:272-279` (`jnp.array(défaut)`) [RÉSOLU antérieurement — vérifié, test 6900415f]
Risque : `pd_to_jax` convertit **tout** défaut non `int/float/bool` via `jnp.array(val, dtype=map.get(type_name))` ; `None` → `ValueError: None is not a valid value for jnp.array`, `str` → `TypeError: <U8 … not a valid JAX array type`. Or les signatures des modèles livrés regorgent de `None`/`str` (`initial_means=None`, `alpha=None`, `method='marginal'`, `censoring=None`, `X=None`). Vérifié sur les objets réels : `m.pd_to_jax` crashe pour `gmm`, `dpmm`, `pca`, et `survival.model` (via sa méthode), `nbda` renvoie la chaîne d'erreur — **5 des 6 entrées de la « Built-in models API » documentée** (`models.qmd`, `reference/index.qmd:309`) sont inutilisables par le chemin `m.data()` → `m.fit(m.models.X)` ; le message d'erreur accuse jax, pas la couture.
Scénario (vérifié) : `m.data(df)` avec colonnes `data`,`K` ; `m.fit(m.models.gmm)` → `ValueError: None is not a valid value for jnp.array` dans `pd_to_jax`.

**W10 — Sévérité : Majeure** — `Models/models.py:24,27` → `Main/main.py:555,571` [RÉSOLU 6900415f]
Risque : le registre expose des objets hétéroclites que `fit` ne sait pas consommer : `m.models.pca` est la **classe** (pas d'attribut `type` → `AttributeError` au special-case PCA, vérifié) ; `m.models.survival` est une instance **sans `__name__` ni `__call__`** (vérifié : `callable()` False) → `AttributeError` sur `model.__name__` dès l'entrée de `fit`. Seuls `gmm` (fonction), `dpmm`/`bnnc` (instances avec `__name__`+`__call__`) et l'instance `pca(...)` construite à la main satisfont le contrat implicite `__name__` + `__call__` + signature jax-compatible.
Scénario (vérifié) : `m.fit(m.models.survival)` → `AttributeError: 'survival' object has no attribute '__name__'` ; `m.fit(m.models.pca)` → crash `pd_to_jax` (W9) avant même le special-case.

**W11 — Sévérité : Majeure** — `Models/GMM.py:18` (`K` sans défaut) → `Data/manip.py:250-270` (colonne requise, tableau `(N,)`) → `Main/main.py:564-569` (`n_clusters=data_on_model["K"]`) + `Models/GMM.py:43` (`plate("components", K)`) [RÉSOLU 6900415f]
Risque : le contrat « args sans défaut = colonnes » transforme un **hyperparamètre scalaire** en contrainte de dataframe : il faut une colonne numérique `K` (absurde mais possible), qui arrive en `(N,)` là où `plate`/`n_clusters`/`jnp.ones(K)` exigent un int Python — `pd_to_jax` ne peut pas produire de scalaire depuis une colonne, et ne peut pas produire le `data (N,D)` multivarié depuis une colonne 1D. Le special-case KMEANS exige en plus la clé `K` dans `data_on_model` (KeyError si `obs={'data': X}`).
Scénario : `df` avec colonne `K=3` partout → `data_on_model['K']` = `Array(20,)` → `KMEANS(n_clusters=Array)` (sklearn) ou `plate("components", Array)` → crash ; sans colonne → « Error, no data found » (V4 de la passe cycle de vie).

**W12 — Sévérité : Majeure** — `Main/main.py:543-547` (`self.nbdaModel`/`self.nbda`) → `Models/models.py:26` (`NBDA.model` non lié) [RÉSOLU 6900415f]
Risque : triple rupture du contrat NBDA↔Main : (a) `bf.nbdaModel` n'est jamais mis à True (le `self.nbdaModel = True` de NBDA.py:53 est posé sur l'instance **NBDA**, jamais lue par `fit`) → la branche `fit(model=None)` est morte et, pire, **continue** après son `print` pour planter dans `pd_to_jax(None)` ; (b) `m.models.nbda` = fonction non liée dont la signature exige la colonne `self` → `'Error, no data found'` (vérifié) ; (c) l'usage réel `m.fit(nbda.model)` donne `model.__name__ == "model"` → `run_model_name` faux → dispatch `m.plot()` (l.1822-1833) muet et `save()` nommé `model_bf.pkl`.
Scénario (vérifié pour a/b) : `m.fit()` sans modèle → print puis `TypeError` dans `inspect.signature(None)` ; `m.pd_to_jax(m.models.nbda)` → chaîne d'erreur.

**W13 — Sévérité : Mineure** — `Main/main.py:245` (`self.bnn = bnn()` avant la branche backend) → `BNN/bnn.py:16` ; `BNN/bnn.py:149-281` (couches à état) `[RÉSOLU 27d7ba49]`
Risque : (a) sous tfp, `self.bnn.dist` reste un `UnifiedDist` **numpyro** `time_ns` (le recâblage l.254 est dans la branche numpyro) — les couches BNN d'un bf tfp émettraient des sites numpyro (statique, TFP absent) ; (b) `layer_attention`/`layer_toeplitz`/`layer_compound_symmetry`/`layer_diagonal` écrivent leurs poids en **attributs d'instance** sur le `m.bnn` partagé : deux modèles construits coup sur coup avec le même `m.bnn` se sur-écrivent leurs couches (et un `save()` pickle l'état de la dernière construction).
Scénario : `m.bnn.layer_attention(...)` pour le modèle A puis B → `m.bnn.emb_q` est celui de B ; le graphe d'A rejoué plus tard (PPC, `log_prob`) voit les tenseurs de B.

**W14 — Sévérité : Mineure** — `Models/GMM.py:55-58`, `Models/DPMM.py:132-137` (site `obs` event_dim=1) → `Main/main.py:1653+` (`sample()`) → `Diagnostic/ppc.py:106` (`reshape(S, −1)`) `[RÉSOLU 27d7ba49 — avertissement + doc ; aplatissage conservé]`
Risque : le contrat PPC `(S, n_obs)` est tenu par un **aplatissage** des event dims : les PPC d'un GMM/DPMM multivarié `(S, N, D)` deviennent `(S, N·D)` — les statistiques de PPC mélangent les D dimensions de la variable dans un même échantillon d'« observations » (p. ex. `mean` confondu sur dimensions et individus). Pas de crash, `_ExpandObserved` gère correctement `event_shape` pour la simulation ; seul le consommateur aval suppose la 1D. PCA (`obs=X` 2D event=0, PCA.py:131/155/187/229) : même aplatissage.
Scénario : `m.fit(m.models.gmm, obs={'data': X, 'K': 3, 'initial_means': M})` puis `m.sample()` → yrep `(S, N·D)` ; un `ppc_plot` compare des distributions hétérogènes.

**W15 — Sévérité : Mineure** — `Main/main.py:55-57,228` (`m.gaussian` exposé) → `Utils/Gaussian.py:323-332` (`Mgaussian` jamais défini/importé) `[RÉSOLU 94e80673 — Mgaussian supprimé, kernel sq_exp fonctionnel]`
Risque : le module Gaussian est **orphelin** (aucun consommateur dans le package, vérifié grep) mais exposé via `m.gaussian` ; son unique chemin utile (`kernel` avec `sq_exp`) lève `NameError: Mgaussian`, et `dist.Exponential` (l.315/319, casse majuscule sur l'instance recâblée) est une 2ᵉ bombe. L'exposer dans l'instance bf laisse croire à une API vivante.
Scénario : `m.gaussian.kernel(Dmat, 'sq_exp', kernel_params={'z': z})` → NameError.

**W16 — Sévérité : Mineure** — `Main/main.py:386-391` + `Models/models.py:29-35` `[RÉSOLU 27d7ba49 — convention documentée + UserWarning pour un lambda]`
Risque : le contrat supposé `fit(model_name="gmm")` n'existe pas : `fit` n'a pas de paramètre `model_name`, le dict `models.available` n'est consommé par personne (grep : zéro lecteur), et la résolution se fait par l'attribut `__name__` du callable — convention non documentée que chaque classe de la bibliothèque implémente inégalement (`bnnc`/`dpmm`/`pca` : oui ; `survival` : non, cf. W10). Un utilisateur qui passe `m.models.available["gmm"]` ou un lambda/fermeture obtient un `model_name` vide ou mensonger, désactivant silencieusement les special-cases (KMEANS de GMM, transpose PCA) et le dispatch de plot.
Scénario : `m.fit(lambda **kw: ...)` OK mais `m.plot()` après fit ne trouve plus le modèle ; `m.fit(functools.partial(gmm, K=3))` → `model_name="gmm"` mais `data_on_model` sans clé `K` → KeyError l.567.

## 3. Décompte par sévérité

| Sévérité | Nombre | IDs |
|---|---|---|
| Critique | 3 | W1, W3, W5 |
| Majeure | 9 | W2, W4, W6, W7, W8, W9, W10, W11, W12 |
| Mineure | 4 | W13, W14, W15, W16 |
| **Total** | **16** | |

**Coutures : 24 lignes inventoriées (C1a→C8b) — 4 saines (C2c, C3c, C5a, C8b), 8 fragiles (C1c, C2d, C4b, C5b, C5c, C6c, C7d, C8a), 12 cassées (C1a, C1b, C2a, C2b, C3a, C3b, C4a, C6a, C6b, C7a, C7b, C7c).**

## 4. Contrôle croisé (revue indépendante Server51, Qwen3-Coder-30B)

Une revue texte-seul des six mécanismes a été déléguée à `server51-subagent` (mode coder). Verdict et arbitrage :

| Mécanisme soumis | Verdict du relecteur | Arbitrage (preuve locale) |
|---|---|---|
| Liaison non liée `dist.normal(0, 2, …)` sur la **classe** (C2a/C6a) | « Faux : loc=0, scale=2 » | **Rejeté** — le relecteur raisonne sur une méthode liée ; `dist` est la classe : `inspect.signature(UnifiedDist.normal).bind(0, 2, name='x')` → `self=0, loc=2, scale=1.0`, et `numpyro.handlers.trace` confirme `Normal loc=[2.5…] scale=[1.]` pour l'appel `normal(zeros, full(2.5))` de SRM.py:424 |
| `dist.deterministic` → AttributeError (C2a) | Correct | Confirmé empiriquement (pas de `__getattr__`/métaclass de secours : l'`AttributeError` est le comportement observé) |
| Recâblage = dernier bf gagne (C1a) | Correct | Confirmé empiriquement (W1) |
| `jnp.array(None)`/`jnp.array('marginal')` (C7a) | « Faux » | **Rejeté** — exécuté : `ValueError: None is not a valid value for jnp.array` et `TypeError: <U8 … not a valid JAX array type` ; crashs reproduits sur `m.pd_to_jax(m.models.gmm/dpmm/pca)` |
| Gather d'indice −1 = wrap silencieux (C3b) | « Douteux : comportement standard » | Confirmé comme **comportement JAX standard** — c'est précisément le risque : la couture 0-based/1-based n'est signalée par rien (`[30,10,20]` observés) |
| Predictive event_dim=1 → reshape aval (C8a) | « Douteux : piège, pas erreur » | Confirmé comme piège — classé Mineure (W14) en conséquence |

Aucune échappatoire (`__getattr__` de métaclass, binding dynamique) n'a survécu aux tests d'exécution déjà présentés dans les sections 1-2 ; la revue ne modifie aucun statut ni sévérité.

**Constat de synthèse.** Le chemin nominal (modèle utilisateur + `m.dist` DistProxy) est sain ; **la bibliothèque de modèles est presque entièrement découplée du singleton qu'elle croit utiliser**. Trois familles de ruptures : (1) **classe vs instance** — SRM et NBDA importent `UnifiedDist` sans l'instancier : les appels se lient en méthode non liée, avalant le premier argument → priors silencieusement fausses (NBDA, modèle phare, aboutit de bout en bout — W5) ou crashs en cascade (SRM — W3) ; (2) **registre vs `fit`** — la « Built-in models API » (`m.models.*`) est inutilisable par le chemin documenté pour 5 des 6 modèles : `pd_to_jax` bute sur les défauts `None`/`str` (W9), les objets n'ont pas le contrat `__name__`/`__call__`/`type` (W10), les scalaires d'hyperparamètres deviennent des colonnes (W11), et la résolution par nom n'existe pas (W16) ; (3) **RNG des singletons** — le recâblage par mutation ne survit pas à une deuxième instance bf (W1) et oublie `model_effects2`/PCA/`survival_old` (W2). Les deux corrections à plus fort rendement : instancier `dist = dist()` dans SRM.py/NBDA.py (ou mieux, passer par `parent.dist` comme la nouvelle classe `survival`), et rendre `pd_to_jax` tolérant aux défauts non numériques (conserver la valeur Python telle quelle quand `jnp.array` échoue) — ce qui rendrait la bibliothèque de modèles réellement appelable par son API documentée.

---

### 22 — Audit d'interfaces : coutures runtime / persistance / documentation (C9→C15)

Passe complémentaire des passes 20 (`20_Lifecycle.md`, cycle de vie) et 21 (`21_Seam_Models.md`, construction de modèle) — les vulnérabilités V1–V18 et W1–W16 ne sont pas re-listées, seulement leurs aggravations par les coutures ici examinées. Enveloppe d'exécution : Python 3.14, ArviZ 1.3.0, JAX sans CUDA local, hôte 16 CPU. Vérifications empiriques légères autorisées effectuées (sonde d'environnement worker `run_simulations`, 2 workers sans fit ; test de picklabilité ; test `on_error` ; `pd.DataFrame(Bunch)` ; signatures ArviZ). Aucune régénération Build lancée, aucun MCMC long, aucune modification du code.

## 1. Tableau des coutures — runtime, persistance, documentation

| # | Couture (producteur `fichier:ligne` → consommateur `fichier:ligne`) | Contrat supposé | Contrat réel | Statut |
|---|---|---|---|---|
| C9a | `SetDevice/set.py:99-101` (XLA_FLAGS one-shot) → `Parallel/simulate.py:186` (`_worker_env`) → worker | le worker hérite du budget JAX du parent | **remplacé** par un budget propre : `XLA_FLAGS`/`BF_MAX_CORES`/`OMP_*` écrasés avant le premier usage du backend (spawn copie l'environnement à la création du worker, dans la fenêtre `_env_override` `simulate.py:194-212`). Mesuré : parent `device_count=4`, worker `XLA_FLAGS=…count=1`, `devices=1` | saine |
| C9b | `bf.__init__` (`main.py:166`, `float_precision`) → worker (`simulate.py:176-191`) | la config `bf()` du parent vaut pour les workers | `float_precision` **voyage par aucun canal** : parent `jax_enable_x64=False` (float_precision=32), worker `bf()` par défaut → `x64=True` (mesuré). Un `fn` qui n'appelle pas `bf()` et utilise jax directement reçoit le défaut jax x32, pas la config parent | fragile |
| C9c | `set.py:78` (`CUDA_VISIBLE_DEVICES`), `set.py:12-14` (`XLA_PYTHON_CLIENT_*`) → workers spawn | les workers se partagent le GPU comme les CPU | `CUDA_VISIBLE_DEVICES` est hérité (tous les workers sur le même GPU) mais `_worker_env` ne pose **jamais** `XLA_PYTHON_CLIENT_PREALLOCATE=false` : chaque processus JAX préalloue ~90 % de la VRAM au premier appel → le 2ᵉ worker OOM. `set_deallocate()` existe (`set.py:11`) mais n'est jamais appelé par `run_simulations`. Sur CPU : sans effet, d'où le test CI muet | fragile (GPU) |
| C9d | `simulate.py:219-258` (`_pin_worker`, `sched_setaffinity`) → budget JAX (`XLA_FLAGS` du worker) | l'épinglage borne les threads XLA, disjoint par worker | vérifié mesuré : 16 CPU / 2 workers → affinité 8 par worker = `block_size`. Mais le bloc vient de `block = i % workers` (`simulate.py:905`) et suppose que le 1ᵉʳ lot de tâches tombe 1:1 sur les workers — hypothèse non garantie par `ProcessPoolExecutor` (démarrage asynchrone des workers) : deux workers peuvent revendiquer le même bloc, d'autres rester non épinglés. Performance seulement | fragile |
| C9e | `run_simulations(fn)` → `cloudpickle.dumps(fn)` (`simulate.py:847`) | fermetures capturant un `bf`/sampler/device transportables | vérifié : une fermeture sur une instance `bf` passe (2 lignes, 0 erreur) — `bf.__getstate__` (`main.py:1841`) neutralise mesh/shardings ; cloudpickle sérialise les arrays jax. `fn_kwargs` désérialisé une fois par worker **après** épinglage (`simulate.py:270-279, 288-292`) — conception soignée | saine |
| C9f | `_run_one`/`_normalise_result` (`simulate.py:125-163, 318-348`) → DataFrame (`simulate.py:622-633`) | assemblage ordre/axes/comptage stables | meta-scénario d'abord, clés du résultat gagnantes ; tri stable par `sim` ; une ligne par dict retourné ; `bf` retourné = erreur explicite (`simulate.py:127-133`). Vérifié sur 2×2 runs | saine |
| C9g | `on_error='record'` (`simulate.py:611`) → `cloudpickle.dumps((rows, error))` (`simulate.py:307`) | toute erreur de simulation est *enregistrée* dans la colonne `error` | l'erreur de **sérialisation de la valeur retournée** échappe au filet : `_run_one` n'entoure que `fn(**call)` ; le picklage du résultat se fait après, dans `_worker_entry` → `future.result()` relance la `TypeError` dans le parent et **tue tout le run**. Mesuré : `on_error='record'`, retour d'un `threading.RLock` → « RUN CRASHED … cannot pickle '_thread.RLock' object ». Avec `backend='serial'` (documenté comme équivalent de débogage, `simulate.py:481`) le même code passe : les deux backends divergent | cassée |
| C9h | `bf.__setstate__` (`main.py:1847-1858`) → worker (`XLA_FLAGS`=1 device) | un `bf` rechargé dans un worker retrouve son mesh | `setup_multi_device` y renvoie `(None, None, None)` (1 device) → `m.shard()` y est mort sans message lié à la cause. Edge-case (recharger un objet shardé dans un worker) | fragile |
| C9i | `Test/13.Parallel/test_run_simulations.py` → CI (`.github/workflows/tests.yml:36`) | la suite de tests protège la couture | le fichier ne contient **aucun** `def test` (script manuel, en-tête pointant vers un chemin périmé `Test/Parallel/`) et `Test/13.Parallel` n'est pas dans la liste pytest de la CI : `run_simulations` n'a **zéro** couverture CI | fragile |
| C10a | `Resources/datasets.py:50-68` (`class load`, `__call__`) → `main.py:249` (`self.load = load()`) / `main.py:1889` (`bf.load` staticmethod) / `__init__.py:25` (export `load`) | « load » désigne une opération | triple identité : `m.load.howell1()` = jeu de données ; `m.load(path)` = session sauvegardée (délégation à `bf.load`) ; `bf.load` = session ; `BayesForge.load` = la **classe** (à instancier). Fonctionnel (S26 validait l'alias) mais le même nom change de sens selon le récepteur | fragile |
| C10b | chargeurs `frame=True` par défaut (`datasets.py:70-110`) → `manip.data` (`manip.py:33-63`) | le DataFrame sortant rentre dans `m.data` | oui : `pd.DataFrame(df).copy()` ; chemin nominal vérifié | saine |
| C10c | chargeurs `frame=False` → `Bunch` (`datasets.py:22-48, 108-114, 578-626`) → `m.data` | `m.data` accepte « tout ce que `pandas.DataFrame()` accepte » (docstring `manip.py:38-42`) | `Bunch` est un `dict`, passe le `isinstance`, et **casse** : mesuré `ValueError: Per-column arrays must each be 1-dimensional` (le `data` 2D est pris pour une colonne). Les deux styles de Bunch (sklearn-like et `data=/columns=` des chargeurs phylo) échouent identiquement. Message illisible, aucune mention « passez `frame=True` » | cassée |
| C10d | `datasets.py:219-241` (`trolley()`) → package data (`pyproject.toml:69-71`) | les 24 chargeurs livrés ont leur CSV | `Trolley.csv` référencé mais **absent** de `BayesForge/Resources/` (inventaire croisé 24 refs ↔ fichiers) → `m.load.trolley()` = `FileNotFoundError` même après `pip install` (le glob `*.csv` ne peut embarquer un fichier inexistant) | cassée |
| C10e | `phylo_L_*` `frame=False` (`datasets.py:572-626`, `jnp.array(df.values)`) → modèles phylogénétiques | le facteur de Cholesky garde la précision du CSV | `jnp.array` suit `jax.config` **au moment de l'appel** : dans une session `bf()` x64 → float64 OK ; en usage nu (import direct du module, pas de `setup_device`) → float32, et les valeurs type `0.99999999998205` du CSV s'arrondissent à `1.0`. Le chemin par défaut (`frame=True`) est épargné | fragile |
| C10f | docstrings/emballage : `datasets.py:12-14` (`from BayesForge.resources.data import datasets`, `datasets.load_howell1()`) ; `MANIFEST.in` (`recursive-include BF *`) ; `KosterLeckie copy.rda` | la doc montre l'API réelle ; le sdist embarque les données | chemin de module inexistant (casse + nom `data`≠`Resources.datasets`), API sklearn `load_*` inexistante ; le glob MANIFEST vise un répertoire `BF` qui n'existe pas à la racine (le paquet s'appelle `BayesForge`) — sans effet grâce à `package-data` pyproject, mais le sdist ne doit son salut qu'à un autre mécanisme ; un fichier « copy.rda » est embarqué par le glob `*.rda` | mineure (doc/packaging) |
| C11a | `Build/Build_NP_light.py:136` (`open("dists.py","w")`, CWD) → `Distributions/np_dists.py` (production) | régénérer = reproduire la production | le générateur écrit `dists.py` dans le **CWD** (pas d'écrasement direct par `pip install` — il n'est jamais exécuté à l'installation), mais son propre docstring (l.4-11) assume 4 classes de **corrections manuelles** (« edited by hand ») puis « `dists.py` is then renamed to `np_dists.py` » : la procédure documentée est un `mv` sans garde, qui écrase silencieusement les correctifs (doublons `mask`, ordre `*args`, snake_case `lkj`, 100 % des docstrings) | cassée (processus) |
| C11b | `Build/Build_TFP_light.py:19` (`open("tfp_dists.py","w")`) → `Distributions/tfp_dists.py` | idem | pis : le générateur écrit sous le **nom exact** du fichier de production. Lancé avec CWD=`BayesForge/Distributions/`, il écrase la production en direct, sans sauvegarde ni diff | cassée |
| C11c | `Build_NP_light.py:17` (`import ollama`) → régénération | régénérer est reproductible | dépendance `ollama` absente de `pyproject.toml` (ni runtime ni extras) ; les docstrings sont **générées par LLM** → sortie non déterministe, impossible à reviewer en diff. La régénération est de fait impraticable, ce qui rend la dérive inévitable | fragile |
| C11d | production ↔ générateur → CI (`tests.yml:36`) | un test détecte la dérive | **aucun** test ne référence `Build_*` ni ne compare un régénéré au fichier livré ; la CI ne lance que `Test/0.Core` + docs. La dérive production↔générateur (établie passe 13) est indétectée et non résorbable (C11c) | cassée |
| C11e | `Build/Build_test_file.py:84` | l'outillage de build est vivant | le générateur de tests émet `print(erros):` (deux-point en fin d'instruction) : code généré syntaxiquement invalide — outil mort, sans conséquence runtime mais preuve que rien ne teste les générateurs | mineure |
| C12a | `pyproject.toml:44` (`arviz>=1.2.0`) → sondes de signatures (`main.py:1345-1347`, `Diag2.py:64-75`, `Diag.py:95-96`) | le pin protège la surface ArviZ | vérifié sur ArviZ 1.3.0 installé : `az.summary` a `ci_prob` et plus `hdi_prob` (sonde OK) ; `az.from_dict(posterior=…)` **lève** `TypeError` (mesuré) — la branche `az.from_dict(groups)` de `_az_from_dict` est bien celle qui porte ; `az.waic` supprimé en 1.x, contourné nativement (`Diag2.py:430`). Les sondes « 0.x » sont mortes sous le pin mais inoffensives | saine |
| C12b | ArviZ ≥1.3 (lazy_loader) → `numpyro.infer.initialization` → `BayesForge/__init__.py:10-19` | le patch garde `az.from_numpyro` vivant | le patch d'attache du sous-module (cicatrisation d'une vraie régression ArviZ 1.3, avouée en commentaire) est enveloppé `try/except ImportError` : si ArviZ 1.4/2.0 renomme `io_numpyro.infer_dims` ou `initialization`, le patch devient un **no-op silencieux** et la panne réapparaît au premier `summary()`, avec une `AttributeError` sans lien avec la cause | fragile |
| C12c | `patch_diag.bind_diag_to_model` (`patch_diag.py:1051-1092`, binding sur l'**instance**) → métriques de convergence ArviZ | un upgrade ArviZ casse les plots | non : les plots sont du plotly maison (indépendants d'ArviZ) ; seules summary/rhat/ess/mcse/loo/waic délèguent, via `_az_diag_for` (`patch_diag.py:553-568`) dont le cache est invalidé par identité du sampler (`cached.sampler is not m.sampler`) — un refit ou un svi invalide bien l'instance ArviZ. Surface de défaillance sur upgrade = les délégations Diag.py, pas le binding | saine |
| C12d | `_az_diag_for` `except Exception: pass` (`patch_diag.py:565-566`) → `_ensure_trace()` (`Diag2.py:403-407`) | la construction du trace peut échouer | l'échec est **différé, pas avalé** : `trace` reste `None`, `_ensure_trace` rejoue `to_az` au premier usage et l'erreur remonte — mais depuis le site d'appel de l'utilisateur (`m.diag.mcse()`…), pas depuis la cause ; et chaque appel suivant rejoue la construction ratée | mineure |
| C12e | `summary_old` (`main.py:1338-1341` → `diag.trace` = `az.from_numpyro`, `Diag2.py:379`) vs `patch_diag._summary` backend arviz (`_az_diag_for` → `Diag.py:62` `az.from_numpyro`) vs backend jax (`m.posteriors`) | un seul chemin vers « le » summary | les deux chemins ArviZ partent du **même sampler** `az.from_numpyro` → noms/dims/coords identiques, cohérents. Le chemin jax lit `m.posteriors` avec des labels plats étendus (`mu[0,1]`) là où ArviZ rend `mu` + dims : deux surfaces d'étiquettes pour la même table, et (rappel V5) deux sources de vérité dès qu'un fit partiellement rafraîchi a eu lieu | fragile |
| C13a | utilisateur → `m.abc`/`m.abc_rf`/`m.npe`/`m.npe_posterior` (`main.py:879, 953, 1016, 1089`) | le SBI a des méthodes bf dédiées | oui, les quatre existent, avec `_sbi_setup` partagé (`main.py:842-867`) qui résout modèle/données comme `fit`. Contrat simulateur : le modèle numpyro **est** le simulateur (`_uncondition` `SBI/simulate.py:25-44` retourne les sites `obs=` en latents) ; seed via `_resolve_key` (`main.py:299-306`) | saine |
| C13b | `_sbi_finish` (`main.py:869-878`) → `m.diag` | le contrat `m.diag` est le même quel que soit l'estimateur | `fit` appelle `bind_diag_to_model` (`main.py:685-686, 715-716`) ; **`_sbi_finish` ne l'appelle pas** : après `m.abc(...)`, `m.diag` reste le `diagWIP` brut — `m.diag.density(include="mu")` (documenté, signature patchée `patch_diag.py:1054+`) lève `TypeError` alors qu'après `fit` il passe ; `Diag2.density` n'accepte que `var_names` (`Diag2.py:1743`). La surface `m.diag` change de contrat selon l'estimateur | cassée |
| C13c | `SBIPosterior` (`SBI/simulate.py:306-338`) → `Diag2.to_az` (`Diag2.py:370-377`) | les postérieurs SBI atterrissent dans `m.posteriors`/diag | oui : `_sbi_finish` remplit `posteriors`/`by_chain`/`_full`, et le flag `az_from_dict=True` route vers `_az_from_dict` (qui survit au changement de signature `from_dict` d'ArviZ 1.3, C12a). Attention : `num_chains` **pseudo-chaînes** (l.324-329) → R-hat≈1 et ESS≈n garantis par construction, documenté mais trompeur pour un non-initié | saine (avec réserve) |
| C13d | `Simulator.__init__` trace sous `seed(model, 0)` (`SBI/simulate.py:113`) et `prior_log_density_u` `PRNGKey(0)` (`SBI/simulate.py:268`) → densité a priori | la clé du log-density ne dépend pas du hasard | déterministe par construction, mais les latents **nuisance** (non substitués) sont tirés sur cette clé fixe et, sous `vmap`, identiques sur toutes les lignes : la densité est conditionnée à un tirage nuisance figé. Déjà compté au RELEASE_AUDIT (repro #1) ; ici aggravé par le fait que la fonction est cachée derrière `m.abc` | mineure |
| C13e | `npe.py:37` (`_BNN = bnn()` module-level) → `BayesForge.BNN.bnn` | l'instance de module pollue l'état bf | vérifié : usage **pur** — `_BNN.layer_linear` (paramètre homonyme `dist` = matrice de poids, `bnn.py:46`) et `_activation_map` n'appellent jamais `self.dist` ; pas de RNG partagé avec un `bf`. W13 ne se propage pas ici | saine |
| C13f | `Test/16.SBI/test_sbi.py` (7 tests de récupération analytique) → CI (`tests.yml:36`) | les tests SBI gardent la couture | tests réels et sérieux, mais `Test/16.SBI` n'est pas dans la liste pytest de la CI : jamais exécutés automatiquement | fragile |
| C14a | `m.to_latex`/`m.graph` (`main.py:348-354, 375-380`) → `PostModel/to_latex.py:897`, `model_graph.py:308` | PostModel lit l'instance bf | oui : `self.model` + `self.data_on_model` de l'**instance**, passés en paramètres ; le parseur AST reconnaît tout préfixe `.dist.` (`model_graph.py:100-106`), donc `m.dist.x` (instance) et `dist.x` (singleton de module) sont vus pareil. Pas de nouvelle rupture W1/W2 à cette couture ; les modèles bibliothèque qui crashent sous `fit` (W3/W5) crashent identiquement sous `m.graph()` — l'aval hérite, n'aggrave pas | saine |
| C14b | `model_trace.trace_model` (`model_trace.py:231-247, 302-305`) exécuté **hors** `fit` | tracer le graphe est sans effet | `handlers.seed(seed=0)+trace` exécute le modèle pour de vrai : les sites `sample=True` (invisibles d'numpyro, C15b) y tirent sur le **flux partagé** `self._rng` du singleton → `m.graph()`/`m.to_latex()` **avancent le RNG de la session** (les tirages `sample=True` ultérieurs sont décalés) et n'apparaissent pas dans le graphe (nœus manquants). Un modèle avec effets de bord (print, mutation) les rejoue à chaque rendu | fragile |
| C15a | `Samplers/mcmc_numpyro.py:36-56` (`NUTS(model)` nu) → branche « modèle » de `np_dists` (`np_dists.py:414-421`, `numpyro.sample(name, d, obs=…)`) | le modèle s'exécute sous les handlers numpyro | oui : les sites passent par `numpyro.sample`, y compris le `mask=` qui s'enveloppe dans son propre `numpyro.handlers.mask` (compose par produit avec un `m.dist.mask()` externe, sémantique correcte). `create_obj=True` ne crée délibérément pas de site (contrat documenté) | saine |
| C15b | `sample=True` dans un modèle passé à `fit` → `_draw_key` (`np_dists.py:198-209`) → `self._rng.stream()` | un tirage dans le modèle fait partie du modèle | **non** : le tirage sort du flux RNG partagé du singleton, hors `numpyro.sample` → invisible du sampler, **re-tiré à chaque pas de leapfrog** (le flux avance), valeur non contrôlée par la graine numpyro. Postérieur invalide, non reproductible, **sans aucun avertissement** : grep confirme qu'aucun code de `np_dists.py` n'inspecte le contexte numpyro (`_context`/`get_mask` absents) pour avertir. Même racine sous `handlers.substitute` (SBI, `Predictive`, `model_trace`) : la valeur substituée est ignorée par les sites `sample=True` | cassée |

## 2. Vulnérabilités d'intégration

**X1 — Sévérité : Critique** — `Samplers/mcmc_numpyro.py:36` → `Distributions/np_dists.py:198-209, 390-403` (C15b/C15c) `[RÉSOLU ec8795df]`
`fit` passe le modèle nu à `NUTS` ; la façade `m.dist.*` offre `sample=True` (tirage direct hors `numpyro.sample`) sans aucune garde de contexte. Risque : postérieur **faux silencieusement** — le « paramètre » tiré en `sample=True` change à chaque évaluation de la densité et n'est ni échantillonné ni consigné. Scénario : `def model(): z = m.dist.normal(0,1,sample=True); m.dist.normal(z, 1, obs=y)` — R-hat correct, trace propre, conclusion publication fausse ; le même code sous `m.abc()` rend de plus le simulateur non reproductible depuis la clé `_resolve_key` (les substitutions `substitute`/`Predictive` n'atteignent pas ces sites). Un avertissement unique quand `sample=True` est appelé dans un contexte numpyro actif (`numpyro.primitives._context`) fermerait la porte.

**X2 — Sévérité : Majeure** — `Parallel/simulate.py:307` → `Parallel/simulate.py:916` (C9g) [RÉSOLU b9884f17]
Contrat `on_error='record'` : « continue et mets le traceback dans la colonne `error` ». Le picklage de la valeur retournée hors du filet `_run_one` transforme une simulation au retour non cloudpicklable en **crash du run entier** (mesuré : `TypeError: cannot pickle '_thread.RLock' object`), avec perte de toutes les simulations déjà terminées (elles ne sont agrégées qu'à travers la boucle `wait`). Et `backend='serial'`, documenté comme équivalent de débogage, accepte le même retour : le test en serial passe, le run process meurt. Scénario : un `fn` qui retourne `{"fit": m.sampler}` ou un figure avec verrou — l'étude de 500 sims tombe à la 300ᵉ.

**X3 — Sévérité : Majeure** — `Build/Build_TFP_light.py:19` / `Build/Build_NP_light.py:136` → `Distributions/{tfp_dists,np_dists}.py` + `.github/workflows/tests.yml:36` (C11a/b/c/d/e) [RÉSOLU 5c4efccd — détecteur de dérive par noms de méthodes (la production est maintenue à la main : comparaison octet impossible)]
La chaîne générateur→production n'a ni garde ni détecteur : `Build_TFP_light.py` écrit sous le nom exact du fichier de production (écrasement direct selon le CWD) ; la procédure documentée pour NP est un `mv` qui efface les 4 classes de correctifs manuels listées par le générateur lui-même ; `ollama` (généré par LLM, non déterministe) rend la régénération impraticable et la revue en diff impossible ; **aucun test CI** ne compare production et générateur (et `Build_test_file.py:84` prouve que les générateurs ne sont plus exécutés depuis longtemps). Scénario : un nouveau mainteneur « régénère pour ajouter une loi » et perd en un `mv` les corrections de doublons `mask` et d'ordre `*args` — la CI (qui ne teste que le runtime nominal) ne dit rien tant que les 90 méthodes existantes continuent de tourner.

**X4 — Sévérité : Majeure** — `Main/main.py:869-878` → `Diagnostic/patch_diag.py:1051-1092` (C13b) [RÉSOLU 365eb5fe]
`_sbi_finish` construit `m.diag` sans `bind_diag_to_model` : après `m.abc/npe`, la surface `m.diag` est la classe brute (`Diag2`) avec d'autres signatures (`density(var_names=)` vs `density(include=, exclude=, filtered=, …)`, `summary(kind="stats")` vs `kind="all"`) et sans les PPC/sensibilité bindées. Scénario : un tutoriel « mêmes diagnostics pour tout le monde » — `m.diag.ppc_density()` (documenté au registre H6) lève `AttributeError` après ABC alors qu'il marche après `fit` ; l'utilisateur conclut à tort que « les outils diag ne marchent pas sur SBI » — prediction confirmée par la brève elle-même : l'impasse est réelle, mais c'est l'oubli du binding, pas SBI.

**X5 — Sévérité : Majeure** — `Parallel/simulate.py:176-191` → workers GPU héritant de `CUDA_VISIBLE_DEVICES` (`set.py:78`) (C9c) [RÉSOLU b9884f17]
`_worker_env` borne CPU/threads mais pas la VRAM : N workers spawn sur le même GPU, chacun avec `XLA_PYTHON_CLIENT_PREALLOCATE` par défaut (~90 % de la VRAM au premier appel) → OOM du 2ᵉ worker, message « failed to initialize cuDNN/memory » sans lien avec le parallelisme. `set_deallocate()` est sous la main et n'est jamais appelé. Scénario : `run_simulations(..., workers=8)` sur une machine GPU — le run meurt ou ne tourne qu'à 1 worker utile. Un `"XLA_PYTHON_CLIENT_PREALLOCATE": "false"` dans `_worker_env` (ou un avertissement quand `CUDA_VISIBLE_DEVICES` est posé) ferme la couture.

**X6 — Sévérité : Majeure** — `Test/13.Parallel/test_run_simulations.py` + `Test/16.SBI/test_sbi.py` → `.github/workflows/tests.yml:36` (C9i/C13f) [RÉSOLU b9884f17]
Les deux seules suites qui gardent les coutures C9/C13 sont hors CI ; de plus `test_run_simulations.py` ne contient aucun `def test` (script manuel, en-tête vers un chemin périmé). Scénario : le changement de `_env_override`→`_worker_env` (déjà réécrit au moins une fois au vu des commentaires de mesure) régresse — XLA_FLAGS hérité du parent, pinning désactivé — rien ne rougit ; les 7 tests de récupération analytique SBI ne protègent rien tant que la CI ne les collecte pas.

**X7 — Sévérité : Mineure** — `Main/main.py:166` → `Parallel/simulate.py:176-191` (C9b) `[RÉSOLU 27d7ba49]`
`float_precision` du parent non propagé aux workers (mesuré 32→64) : une étude calibrée en float32 tourne en float64 dans les workers (lente, numériquement différente), ou l'inverse en usage jax nu (x32). Le docstring de `run_simulations` détaille graines et cœurs mais pas la précision.

**X8 — Sévérité : Mineure** — `Parallel/simulate.py:905` → `_pin_worker` (`simulate.py:219-258`) (C9d) `[DOCUMENTÉ, NON CORRIGÉ — limite connue décrite dans la docstring de _pin_worker ; performance seule, résultats intacts]`
L'attribution `i % workers` du bloc CPU suppose le premier lot de tâches réparti 1:1 dans l'ordre de soumission ; le démarrage asynchrone des workers du pool peut faire revendiquer le même bloc à deux workers (sur-souscription ~2× sur un bloc, un bloc oisif). Performance uniquement, résultats intacts.

**X9 — Sévérité : Mineure** — `BayesForge/__init__.py:10-19` → ArviZ ≥1.3 (C12b) `[RÉSOLU f09f38cb]`
Le monkey-patch d'attache `numpyro.infer.initialization` (cicatrice d'une vraie régression ArviZ 1.3, vérifiée nécessaire : `az.from_numpyro` est le chemin unique de `summary_old` et `_az_diag_for`) est `try/except ImportError` : muet si d'ici là ArviZ déplace le module, donc la panne d'origine revient sans le patch ni message. Le pin `arviz>=1.2` n'a pas de plafond : ArviZ 2.0 est dans l'intervalle autorisé.

**X10 — Sévérité : Mineure** — `Diagnostic/patch_diag.py:565-566` → `Diag2.py:403-407` (C12d) `[RÉSOLU 27d7ba49]`
`except Exception: pass` : l'échec de construction de l'InferenceData est rejoué à chaque appel et remonte depuis le site d'usage (`m.diag.mcse()`), jamais depuis sa cause ; un bug BF (vs ArviZ) se déguise en erreur ArviZ différée.

**X11 — Sévérité : Mineure** — `Main/main.py:1280-1303` (C12e) `[RÉSOLU 27d7ba49 — libellés identiques vérifiés ; sources différentes documentées]`
Les résumés arviz (`mu` + dims) et jax (`mu[0,1]`, labels plats) du même échantillon n'ont pas le même vocabulaire d'index : un script `df.loc["mu"]` marche avec un backend et pas l'autre ; de plus les deux backends ne lisent pas la même source (`trace` du sampler vs `m.posteriors`) — le terrain de V5, vu sous l'angle API.

**X12 — Sévérité : Mineure** — `Resources/datasets.py:219-241` → package data (C10d) `[RÉSOLU 3b98e6c1 — chargeur trolley supprimé]`
`m.load.trolley()` : `FileNotFoundError` garanti (CSV jamais livré). Le seul test d'installation qui itérerait sur les chargeurs le verrait — il n'existe pas (X6).

**X13 — Sévérité : Mineure** — `Resources/datasets.py:22-48, 578-626` → `Data/manip.py:33-63` (C10c/C10e) `[RÉSOLU 2906f8dd (float64) + 4618e249 (m.data(bunch))]`
`m.data(bunch)` (sortie `frame=False`, documentée) plante avec `ValueError: Per-column arrays must each be 1-dimensional` au lieu d'un « passez `frame=True` » ou d'une conversion `bunch["frame"]` ; et les facteurs de Cholesky `phylo_L_*` en `frame=False` hors session x64 s'arrondissent en float32 (`0.99999999998205 → 1.0`). Le chemin par défaut (`frame=True`) est sain — c'est la branche secondaire documentée qui ne l'est pas.

**X14 — Sévérité : Mineure** — `Resources/datasets.py:12-14` + `MANIFEST.in` (C10f) `[RÉSOLU 1a285825 + 3938c102 + 6f200f67]`
Docstring d'exemple pointant un module inexistant (`BayesForge.resources.data`) et une API `load_howell1()` qui n'existe pas ; `recursive-include BF *` vise un répertoire fantôme (le paquet est `BayesForge`) — la livraison des données ne tient qu'à `pyproject.package-data` ; un `KosterLeckie copy.rda` est embarqué dans la roue par le glob `*.rda`.

**X15 — Sévérité : Mineure** — `PostModel/model_trace.py:231-247` → flux RNG du singleton (C14b) `[RÉSOLU 27d7ba49]`
`m.graph()`/`m.to_latex()` exécutent le modèle (seed 0) : les sites `sample=True` tirent sur le flux partagé et **avancent l'état reproductible de la session** — deux scripts identiques sauf un appel `m.graph()` intermédiaire divergent dans leurs tirages `sample=True` suivants ; les mêmes sites sont absents du graphe.

**X16 — Sévérité : Mineure** — `Main/main.py:1847-1858` → worker spawné (C9h) `[RÉSOLU 27d7ba49 — RuntimeWarning quand le mesh rechargé diffère]`
`__setstate__` reconstruit le mesh depuis les devices **du worker** (1 par budget `cores_per_worker`) → `m.shard()` silencieusement indisponible sur un objet rechargé dans un worker ; le `except RuntimeError: pass` (machine sans GPU) masque aussi ce cas.

**X17 — Sévérité : Mineure** — `Samplers/SBI/simulate.py:268` (C13d, renvoi RELEASE_AUDIT §1) `[RÉSOLU 19f93e5d — limite documentée + UserWarning à la construction]`
`PRNGKey(0)` de `prior_log_density_u` : déterminisme oui, mais latents nuisance figés et identiques sous `vmap` — la densité a priori renvoyée à `m.abc(smc)` est conditionnée à un tirage nuisance arbitraire. Déjà listé au RELEASE_AUDIT ; compté ici seulement parce que la couture `m.abc → simulate` l'expose sans le documenter.

## 3. Décompte par sévérité

| Sévérité | Nombre | IDs |
|---|---|---|
| Critique | 1 | X1 |
| Majeure | 5 | X2, X3, X4, X5, X6 |
| Mineure | 11 | X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17 |
| **Total** | **17** | |

Coutures : 35 lignes inventoriées (C9a→C15b) — **11 saines** (C9a, C9e, C9f, C10b, C12a, C12c, C13a, C13c, C13e, C14a, C15a), **12 fragiles** (C9b, C9c, C9d, C9h, C9i, C10a, C10e, C11c, C12b, C12e, C13f, C14b), **8 cassées** (C9g, C10c, C10d, C11a, C11b, C11d, C13b, C15b — l'API SBI et la couture `sample=True` inclues), **4 étiquetées mineures dans leur ligne** (C10f, C11e, C12d, C13d).

**Constat de synthèse.** La couture runtime phare, `run_simulations`, est **la mieux ingénierée du paquet** (environnement posé avant le backend, épinglage effectif mesuré, cloudpickle choisi pour les fermetures sur `bf`, seeds sha256 indépendants de l'ordre d'achèvement, `__getstate__` qui neutralise le mesh) — et pourtant c'est elle qui cumule les angles morts : tout ce qui borde la couture (précision flottante, VRAM GPU, picklabilité du retour, CI) est hors contrat. Trois corrections à fort rendement : (1) refuser/avertir `sample=True` quand un contexte numpyro est actif (X1, le seul chemin de l'audit vers un postérieur **faux sans message**) ; (2) envelopper le `cloudpickle.dumps` du résultat dans le filet `on_error` (X2, trois lignes) ; (3) ajouter `Test/13.Parallel` (après en avoir fait de vrais pytest) et `Test/16.SBI` à la liste CI (X6) — sans ce troisième point, les deux premiers pourraient régresser sans témoin. Côté documentation/persistance, la chaîne générateur→production (C11) est une épée de Damoclès de maintenance : ni garde, ni reproductibilité, ni détecteur — le seul statut où `pip install` du dépôt protège l'utilisateur… jusqu'au prochain `mv` d'un mainteneur de bonne foi.

---

### 23 — Couture inter-langages R↔Python (reticulate) — BayesForge

**Périmètre** : `BayesForge/R/R warpper.R` (169 l., lu intégralement), `BayesForge/R/BF doc.py` (64 l., lu intégralement), échantillons `R/RDoc.json` et `Build/Doc_R.json` (via python3/json), `BayesForge/__init__.py`, `Distributions/np_dists.py` (signatures réelles, `DistProxy`, `normalize_seed`), `Models/SRM.py` (`index1`), `Data/manip.py` (`index()`), `Documentation/reference/*.qmd`, `Test/3.Documentation/R/`, et — découverte de la passe — le package R **vivant** installé `~/R/lib/BayesForge` v0.0.2 (repo GitHub distinct `BGN-for-ASNA/BFR`, dumpé par `lazyLoad` sans exécution). R **est** installé (`/usr/bin/Rscript`, reticulate 1.47.0) ; la simulation du code généré a été faite par déroulement manuel + vérifications R ciblées.

**DÉCOUVERTE CADRE** : il existe **trois générations** de pont R et le repo ne contient que la plus vieille.
1. **Fossile (ce repo)** : `BayesForge/R/R warpper.R` (26 mai 2026) — générateur cassé contre l'API actuelle (détaillé ci-dessous).
2. **Artefact intermédiaire committé** : `BayesForge/Utils/R/asymmetric_laplace.R` — sortie d'un générateur variante (cible `.BF$dist$…`, `seed=0`, sans `to_jax`), toujours clôturée en fence ```` ```r ````.
3. **Vivant (hors repo)** : package R `BayesForge` v0.0.2, construit le 2026‑11‑03 depuis `BGN-for-ASNA/BFR@dev`, installé dans `~/R/lib/BayesForge` — 106 objets dont ~99 `bf.dist.*` qui, eux, ciblent correctement l'instance (`reticulate::import("BayesForge")$bf`, `.BF_env$.bf_instance$dist$<nom>(…)`, garde `is_none` sur `seed`, `jnp.asarray` sur `loc`/`scale`, S3 `jax.Array`↔R, `importFrom(reticulate, py_none, tuple)`). C'est **lui** que testent `Test/3.Documentation/R/test_r_model_pages.py|test_r_start_pages.py` (les pages de doc appellent `importBF('cpu')`, fonction absente du repo). `Documentation/reference/dist.qmd` n'a **aucun onglet R** (grep `{r`/`tab:R` : 0).

---

### Interface inter-langages R↔Python

#### 1. Tableau des coutures (producteur → consommateur)

| # | Producteur → Consommateur | Contrat supposé | Contrat réel | Statut |
|---|---|---|---|---|
| S1 | `BF doc.py` → `pythonDoc.json` → LLM → `Doc_R.json` → `R warpper.R:11` | chaîne de régénération doc R reproductible | chaîne cassée de bout en bout (R‑C1 : `m.dist.__dict__[name]` → `KeyError` contre `DistProxy` ; R‑M2 : `Doc_R.json` lu dans le CWD, l'artefact vit dans `Build/` ; `R/RDoc.json` lu par personne) — **rappel, non re-compté** | ☠️ établi |
| S2 | `R warpper.R:10` `import("BF")` → roue PyPI | module Python top-level `BF` | le paquet s'appelle `BayesForge` ; `BF` n'y est qu'un **alias de classe** (`Main/main.py:2247` `BF = bf`, exporté dans `__all__`) ; aucun module `BF` (ni dans la roue, ni à la racine du repo) | ☠️ Critique `[RÉSOLU 3938c102]` |
| S3 | `R warpper.R:115` `BF$dist` → attribut de module | `dist` exporté au niveau **module**, contenant des **fonctions libres** | `dist` est un attribut d'**instance** de `bf()` (un `DistProxy`, `np_dists.py:122`) ; le module ne l'expose pas | ☠️ Critique `[RÉSOLU 3938c102]` |
| S4 | `R warpper.R:148` `BF$dist[[a]]` (`[[` = `py_get_item`) → `DistProxy.__getitem__` | accès par élément | `DistProxy` n'a **pas** `__getitem__` → exception → `tryCatch(… NULL)` → **les 99 distributions sont sautées en silence**, zéro fichier produit, zéro erreur | ☠️ Critique (silencieux) `[RÉSOLU 3938c102]` |
| S5 | `R warpper.R:151` regex `<function\s+[\w\.]+\.(\w+)\s+at\s+0x…>` → `str()` de l'attribut | `str` d'une fonction libre (`<function normal at 0x…>`) — API ancienne module‑centrée | accès sur instance → **bound method** : `<bound method DistProxy.normal of <…DistProxy object at 0x…>>` → regex ne matche jamais → `func_name` = le repr brut → `func_body` commence par du texte illégal → `parse` L98 ** tue la boucle à la 1re distribution | ☠️ Critique `[RÉSOLU 3938c102]` |
| S6 | Fonction générée L77 `.bf$dist$<nom>(…)` → environnement R | un objet R `.bf` (module Python) existerait | rien ne définit jamais `.bf` ; le pont vivant résout `.BF_env$.bf_instance$dist$<nom>` | ☠️ Critique `[RÉSOLU 3938c102]` |
| S7 | Doc insérée L108 ← `Build/Doc_R.json` | bloc roxygen brut | **92/92 entrées encadrées par le fence Markdown** ```` ```r ```` … ```` ``` ```` → chaque fichier `.R` généré commence et finit par du Markdown → `source()` échoue ; de plus les `@examples` embarqués appellent `BF.dist.normal(…)` (B majuscule) alors que la fonction générée s'appelle `bf.dist.normal` | ☠️ Critique `[RÉSOLU 3938c102]` |
| S8 | Défaut généré `seed=py_none()` → `normalize_seed` (`np_dists.py:211`) | `None` passe tel quel | le corps généré fait `seed=as.integer(seed)` : `as.integer(py_none())` → `integer(0)` → ndarray vide côté Python → ni `None`, ni `bool`, ni `int` → **`ValueError: Invalid seed type: <class 'numpy.ndarray'>`** ; le package vivant a justement ajouté la garde `is_none` (preuve par l'évolution) | 🔴 Haut |
| S9 | Défaut généré `name='x'` passé **explicitement** → inférence de nom `DistProxy.__getattr__` (`np_dists.py:147`) | passe-plat neutre | passer `name` explicitement **désactive** l'inférence Python (affectation/`obs`) → tous les sites nommés `"x"` → erreur numpyro « duplicate site » dès 2 sites ; le vivant réplique l'inférence via `.bf_obs_name(substitute(obs))` | 🟠 Moyen |
| S10 | Fichier généré → portée de `py_none()`/`tuple` | le `@importFrom reticulate py_none tuple` du doc suffirait | inertie sans build de package R (aucun DESCRIPTION/NAMESPACE dans ce repo) ; les défauts sont évalués dans la fonction → « could not find function » sauf si l'utilisateur a `library(reticulate)` | 🟠 Moyen |
| S11 | Utilisateur R codes de groupes `1..K` → `SRM.import_block_predictors(2)(index1=…)` | le pont R convertirait 1‑based→0‑based ou exposerait `index1` | **aucune conversion nulle part** (ni fossile, ni package vivant : grep `index1|1-based` dans les 106 objets installés → 0) ; `index1` n'est pas documenté côté R ; `jnp` **borne les index hors plage au lieu d'errer** → corruption silencieuse | 🔴 Haut |
| S12 | `SRM.import_block_predictors2` option `index1` (l'option « for R users », L225) | `index1=True` = même modèle, codes décrémentés | L139‑141 concatène la colonne intercept, puis L142‑143 `if index1: self.block_predictors = block_predictors - 1` **écrase** avec l'array **sans intercept** → `index1=True` supprime silencieusement l'intercept (bug côté Python, dans l'option destinée aux R‑users) | 🔴 Haut |
| S13 | Factor R → `Data/manip.py index()` | factor arrive comme pandas `Categorical`, niveaux ordonnés conservés | conforme : `select_dtypes(['object','category'])`, codes 0‑based, `index_map` documenté ; commenté explicitement pour les factors R | ✅ OK |
| S14 | `data.frame` R → `bf.load`/DataFrame | conversion reticulate transparente | documenté (`data.qmd:39`) ; reticulate convertit data.frame→DataFrame, NULL→None, TRUE→True, scalaire length‑1→scalaire Python, vecteur→ndarray 1‑D | ✅ OK |
| S15 | Retour Python → R (`jax.Array`) | (fossile) objet brut exploitable | fossile : aucune conversion ; vivant : S3 `as.matrix`/`as.vector`/`Math`/`mean` sur `jax.Array` — le fossile est inexploitable en aval | 🟠 Moyen |
| S16 | Package vivant BFR ↔ ce repo | le repo devrait contenir/regénérer l'interface R livrée | source BFR **hors repo** ; le générateur du repo ne peut pas reproduire l'interface livrée (3 générations divergentes) ; la roue embarque le fossile `R/` (R‑C2, rappel) | 🔴 Haut |

#### 2. Code R réellement généré (simulation déroulée)

Signatures Python réelles (`np_dists.py:5655`, `:7502`) — **sans annotations, sans `*args`/`**kwargs`** :
`normal(self, loc=0.0, scale=1.0, validate_args=None, name='x', obs=None, mask=None, infer=None, sample=False, seed=None, shape=(), event=0, create_obj=False, to_jax=True)`

Déroulé `normal` : `str(inspect.Parameter)` n'a pas de guillemets doubles → `extract_values` (L13‑17) laisse `loc=0.0` intact ; `name='x'` (guillemets **simples**) survit ; L35 `shape=()` → `shape=c()` ; L37 `None`→`py_none()` ; L38‑39 `False/True`→`FALSE/TRUE` ; pas de `kwargs` → branche L53 ; `grepl('shape')` → **`shape_inside=TRUE` pour les 99 lois** (toutes ont un paramètre `shape`) → la branche `else` L86‑94 est du **code mort** ; L60‑66 produit les doubles espaces caractéristiques.

**Cas 1 — `normal`**, `func_body` final (L72‑78), tel qu'écrit dans `./R/normal.R` après la doc clôturée :

```r
bf.dist.normal=function(loc=0.0, scale=1.0, validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), infer=py_none(), sample=FALSE, seed=py_none(), shape=c(), event=0, create_obj=FALSE, to_jax=TRUE) { 
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bf$dist$normal(loc=loc,  scale= scale,  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  infer= infer,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,  to_jax= to_jax)
}
```

Le fichier complet = ```` ```r ```` + roxygen (dont `@examples` appelant `BF.dist.normal`, nom inexistant) + ```` ``` ```` + ce bloc. Triple condamnation à l'usage : `source()` échoue (fences) ; même sans fences, `seed` omis → `as.integer(py_none())` = `integer(0)` → `ValueError` Python ; `shape`/`seed` fournis → `.bf` introuvable.

**Cas 2 — `bernoulli`** (`probs=None, logits=None, …, enum=False, …`) :

```r
bf.dist.bernoulli=function(probs=py_none(), logits=py_none(), validate_args=py_none(), name='x', obs=py_none(), mask=py_none(), infer=py_none(), enum=FALSE, sample=FALSE, seed=py_none(), shape=c(), event=0, create_obj=FALSE, to_jax=TRUE) { 
     shape=do.call(tuple, as.list(as.integer(shape)))
     seed=as.integer(seed);
     .bf$dist$bernoulli(probs= probs,  logits= logits,  validate_args= validate_args,  name= name,  obs= obs,  mask= mask,  infer= infer,  enum= enum,  sample= sample,  seed= seed,  shape= shape,  event= event,  create_obj= create_obj,  to_jax= to_jax)
}
```

**Branche morte `else` (L88)** — la concaténation `"BF$" + foo` avec `foo = paste(".", gsub("\\.","\\$","bf.dist.normal"))` = `".bf$dist$normal"` produit la chaîne `BF$.bf$dist$normal(…)` : R l'analyse comme `BF[[".bf"]][["dist"]][["normal"]]` — syntaxiquement valide, sémantiquement mort (attribut `.bf` inexistant). Jamais exécutée en pratique (S3 note : `shape_inside` toujours TRUE).

**Cas 3 — défauts « exotiques »** : aucun défaut fonction (`<function…>` → L40 mort), aucun défaut tuple/liste avec virgule (`shape=()` est le seul défaut conteneur, sans virgule) → L40 et la corruption L60 sont **latents** ; les rustines L80‑84 (`loc: jax.Array`, `covariance_row: jax.Array`) sont mortes : **plus aucune signature annotée** dans `np_dists.py` (grep `def x(self, y: ` → 0) — fossiles d'une API intermédiaire.

**Référence empirique** — `Utils/R/asymmetric_laplace.R` (committé) confirme le gabarit (mêmes doubles espaces, même `do.call(tuple,…)`) avec `.BF$dist$…` et `seed=0` ; le package installé v0.0.2 montre la génération corrigée :

```r
function (loc = 0, scale = 1, validate_args = py_none(), name = "x", obs = py_none(), mask = py_none(),
          sample = FALSE, seed = py_none(), shape = c(), event = 0, create_obj = FALSE, to_jax = TRUE) {
    if (missing(name) && !missing(obs)) name <- .bf_obs_name(substitute(obs))
    shape = do.call(tuple, as.list(as.integer(shape)))
    if (!.BF_env$.py$is_none(seed)) { seed = as.integer(seed) }
    .BF_env$.bf_instance$dist$normal(loc = .BF_env$jnp$asarray(loc), scale = .BF_env$jnp$asarray(scale), …)
}
```

#### 3. Vulnérabilités

**Criticité : ☠️ Critique | 🔴 Haute | 🟠 Moyenne | 🟢 Faible**

| Sév. | Emplacement | Risque | Scénario déclencheur |
|---|---|---|---|
| ☠️ | `R warpper.R:L10` | `import("BF")` : pas de module `BF` (paquet = `BayesForge`, `BF` = alias de classe, `main.py:2247`) | toute exécution du script → `ModuleNotFoundError` immédiate `[RÉSOLU 3938c102]` |
| ☠️ | `R warpper.R:L115` | `BF$dist` au niveau module : `dist` n'existe qu'on l'instance `bf()` (`DistProxy`) | même corrigé en `import("BayesForge")` → erreur d'attribut au 1er appel utile `[RÉSOLU 3938c102]` |
| ☠️ | `R warpper.R:L148` | `[[` = `__getitem__` que `DistProxy` n'a pas ; `tryCatch(NULL)` avale tout | corrigé S2‑S3 → boucle « réussie » qui ne génère **rien**, sans un bruit `[RÉSOLU 3938c102]` |
| ☠️ | `R warpper.R:L151` + `L98` | regex calée sur `<function … at 0x…>` (API module ancienne) ; l'instance donne `<bound method …>` → nom de fonction = repr brut → `parse` échoue, aucune protection | corrigé S4 (`$`) → `Error: unexpected '<'` à la 1re distribution, boucle tuée `[RÉSOLU 3938c102]` |
| ☠️ | `R warpper.R:L77`/`L152` | cible d'appel générée `.bf$dist$<nom>` : objet `.bf` jamais défini (vivant : `.BF_env$.bf_instance`) | toute invocation d'une fonction générée → `object '.bf' not found` `[RÉSOLU 3938c102]` |
| ☠️ | `R warpper.R:L108` + `Build/Doc_R.json` (92/92) | doc embarquée fenceée ```` ```r ```` → fichiers générés non « sourceables » ; `@examples` nomment `BF.dist.*` vs fonctions `bf.dist.*` | `source("normal.R")` → `Error: unexpected input` ligne 1 `[RÉSOLU 3938c102]` |
| 🔴 | `R warpper.R:L76` | `seed=as.integer(seed)` avec défaut `py_none()` → `integer(0)` → `ValueError` dans `normalize_seed` (`np_dists.py:224`) | `bf.dist.normal(loc=0, scale=1)` sans seed → crash Python ; le vivant a ajouté la garde `is_none` |
| 🔴 | `R warpper.R:L144` | `attrs[30:length(attrs)]` : offset magique vs 27 dunders réels → **ampute les 2 premières distributions** triées ; slice décroissante (`30:29`) si <30 attributs ; redondant avec la liste `no` (28 noms) | tout changement d'API (dunder ajouté/retiré) décale silencieusement la fenêtre |
| 🔴 | pont entier (`R warpper.R`, package BFR) + `SRM.py` | **aucune conversion 1‑based→0‑based** ; `index1` (L224‑225 « for R users ») non exposé/non documenté côté R ; `jnp` clamp les index hors plage | utilisateur R passe des codes `1..K` avec `index1` par défaut `False` → décalage d'un niveau ou clamp sur K‑1 → **contrastes faux sans erreur** |
| 🔴 | `Models/SRM.py:L139‑143` | `import_block_predictors2` : `if index1: self.block_predictors = block_predictors - 1` écrase le tableau avec intercept construit 2 lignes plus haut | R‑user qui fait **correctement** `index1=TRUE` perd l'intercept, silencieusement |
| 🔴 | repo vs `~/R/lib/BayesForge` v0.0.2 (BFR) | la source de l'interface R réellement livrée/testée est hors repo ; le repo livre un générateur incapable de la reproduire, embarqué dans la roue (R‑C2, rappel) | régénération de la doc R pour une release → impossible depuis ce repo ; dérive repo↔BFR non détectée |
| 🔴 | `R warpper.R:L98‑99` | `eval(parse(...))` + `assign(...)` dans le frame local de `build_function` → rien ne persiste en globalenv ; double parse de `func_body` | usage « interactif » du script (fonctions censées être disponibles en session) → néant |
| 🟠 | `R warpper.R:L60` | `strsplit(default_params, ",")` casse tout défaut contenant une virgule (tuple `(2, 3)`, dict, chaîne à virgule) | latent aujourd'hui ; l'ajout d'un défaut tuple dans une signature corrompe signature **et** appel sans erreur |
| 🟠 | `R warpper.R:L37` | `gsub("None", "py_none()")` non ancré réécrit les sous‑chaînes (`name='None'` → `name='py_none()'` ; annotations `… \| None` déjà massacrées jadis — rustines L80‑84) | défaut ou annotation contenant `None` en sous‑chaîne |
| 🟠 | `R warpper.R:L49/L55` | `grepl('shape', default_params)` matche tout paramètre contenant « shape » → `shape_inside` toujours TRUE → branche `else` L86‑94 (et ses rustines L91‑93) code mort | toute loi ; masque aussi qu'un paramètre `reshape`/`shape_x` déclencherait la branche tuple à tort |
| 🟠 | `R warpper.R:L25` vs `L108` | paramètre `docString` jamais lu ; `writeLines(doc, …)` lit la **globale** `doc` assignée L155 dans la boucle | `build_function` appelée hors boucle → `object 'doc' not found` ; appel avec `docString` ≠ doc → doc silencieusement faussée |
| 🟠 | `R warpper.R:L73` (défaut `name='x'`) | passer `name` explicitement désactive l'inférence de nom du `DistProxy` (affectation/`obs`) | deux sites dans un modèle R → tous nommés `"x"` → erreur numpyro « duplicate site » |
| 🟠 | fichiers générés (portée) | `py_none()`/`tuple` non importés dans le fichier écrit ; `@importFrom` inertes (pas de package R dans ce repo) | `source()` + appel sans `library(reticulate)` → « could not find function » |
| 🟠 | `R warpper.R:L5`, `L8`/`L27`, `L104` | `py_install('BayesForge')` à chaque source (réseau, déprécié) ; écritures `./R/*.R` dans le CWD sans cleanup ni verrou ; `print(file_path)` ×99 | exécution depuis un CWD quelconque pollue (`./R/` créé n'importe où) ; réexécution écrase sans crier |
| 🟠 | fossile, sens retour | aucune conversion `jax.Array`→R (le vivant a S3 `as.matrix`/`Math`/`mean`) | post‑traitement R d'un échantillon → objet Python brut inexploitable |
| 🟢 | `R warpper.R:L150`, `L153`, `L7` | `py_has_attr` calculé puis jeté (garde « callable ? » jamais appliquée) ; `func_name3` mort ; `jnp` importé inutilisé | lecture/maintenance seulement |
| 🟢 | `R warpper.R:L30` | `reticulate:::py_dict_get_keys_as_str` : fonction **interne non exportée** (présente en 1.47.0, aucune stabilité d'API garantie) | mise à jour de reticulate |
| 🟢 | `R warpper.R:L38‑39` | `gsub("True"/"False")` non ancré (sous‑chaînes) | latent, aucun défaut actuel déclencheur |
| 🟢 | `R warpper.R:L147` | `!a %in% no` — **vérifié correct** (les opérateurs `%any%` lient plus fort que `!` en R) ; noté pour éviter un faux positif futur | — |

(R‑C1, R‑C2, R‑M2, R‑M3 des passes 21/22 : rappelés dans le tableau S1/S16, non re‑comptés.)

#### 4. Verdict

**Le pipeline R du repo n'est pas exécutable, et ne l'a jamais été contre l'API actuelle.** Il meurt à la ligne 10 (`import("BF")`) ; derrière chaque porte corrigée il y en a une autre : `Doc_R.json` absent du CWD (L11), `BF$dist` inexistant au niveau module (L115), `[[` sans `__getitem__` qui fait tout sauter en silence (L148), regex `<function…>` qui explose sur les bound methods (L151→L98), et — une fois tout cela réparé — des fichiers fenceés non « sourceables », un objet `.bf` introuvable, et un `seed` par défaut qui déclenche un `ValueError` Python. Il n'est **pas reproductible** non plus : sa chaîne amont (`BF doc.py` → LLM → `Doc_R.json`) est cassée (R‑C1/R‑M2) et ses artefacts ont dérivé en trois générations.

**Mais le pont R réel existe et fonctionne** : le package `BayesForge` v0.0.2 (repo `BGN-for-ASNA/BFR@dev`, hors de ce repo) dont les `bf.dist.*` ciblent correctement l'instance `DistProxy`, gèrent `seed`/`name`/les types, et sont couverts par la suite de tests R des pages de documentation. Le fossile du repo est donc **superfétatoire et trompeur**.

**Recommandation : retirer, pas réparer.**
1. Supprimer `BayesForge/R/` (`R warpper.R`, `BF doc.py`, `RDoc.json`) du dépôt **et** de la roue (corrige R‑C2 d'un coup), plus l'artefact `BayesForge/Utils/R/asymmetric_laplace.R` ; au besoin les déplacer dans `attic/r-legacy/` avec un README pointant vers `BGN-for-ASNA/BFR` comme source unique de vérité du pont R.
2. Ne pas tenter de ressusciter le générateur : l'écart fossile↔vivant (`.bf` vs `.BF_env$.bf_instance`, garde `is_none`, `jnp.asarray`, inférence de nom, S3 `jax.Array`) montre que la bonne implémentation est déjà faite, ailleurs et mieux.
3. Traiter les deux vrais bugs de **couture** côté Python/repo : exposer et documenter `index1` pour les utilisateurs R (ou convertir à l'entrée des `import_block_predictors`), et corriger `SRM.py:L142‑143` où `index1=True` écrase la colonne intercept (à faire dans une passe Models, avec test : `index1=True` doit donner le même `self.block_predictors` que codes 0‑based + intercept).
4. Tant que BFR n'est pas versionné ensemble, ajouter dans la CI un test de **parité d'API** : les noms `bf.dist.*` du package R installable doivent couvrir les 99 méthodes de `DistProxy` (le fossile masquait cette dérive par son offset `30:`).

#### 5. Décompte par sévérité

| Sévérité | Nombre |
|---|---|
| ☠️ Critique | 6 |
| 🔴 Haute | 6 |
| 🟠 Moyenne | 8 |
| 🟢 Faible | 4 |
| **Total (nouvelles)** | **24** |

(Rappels hors décompte : R‑C1, R‑C2, R‑M2, R‑M3.)

---

---

## 3. Validation Empirique & Couverture Test

**Méthode** : trois passes complémentaires (mêmes personas, lecture seule du code source). ① Croisement des 615 constats des sections 1-2 avec le contenu réel de `Test/` (rapport 24) ; ② audit de la rigueur des comparaisons inter-bibliothèques (rapport 25) ; ③ script de torture temporaire (`tmp_audit_repro.py`, 22 sondes) **exécuté sur le cluster ecocn20** (réservation `d268f645`, source 0.0.60 injectée par `PYTHONPATH`, jax 0.10.1 / numpyro 0.21.0 / arviz 1.2.0 / pandas 3.0.3, CPU 24 cœurs) + sonde R du pont inter-langages ; scripts supprimés après exécution, réservation libérée.

**Résultat en chef** :

| Question | Réponse |
|---|---|
| Un test du dépôt échouerait-il sur l'un des 82 bugs critiques ? | **Non — 0/82.** 10 sont « couverts-passe » (l'assertion ne peut pas voir le bug), 72 (87,8 %) ne sont atteints par aucun test. La CI verte est parfaitement cohérente avec l'audit. |
| Les comparaisons inter-bibliothèques sont-elles rigoureuses ? | **Bimodal** : `Test/0.Core` (CI) est de niveau B+/A− (références analytiques, harnais z/MCSE) ; dès que la comparaison devient externe (sklearn, Stan, TFP), elle quitte pytest, la CI **et la capacité d'échouer** (try/except qui avale, print sans assertion, exit 0 garanti) — les 3 suites « externes » sont notées D. |
| Les bugs confirmés par reproduction ? | **15 confirmés à l'identique** (dont les 5 prioritaires : W5, W7, V1/V2, V3, Data C1/C8), **1 infirmé** (T14b : le sampler ne reste pas bloqué sur valeurs extrêmes), **3 sains** (N=1, save/load, `bf(rand_seed=42)`), **1 reclassement majeur** (X1) et **1 découverte nouvelle** (R1). |

**Les deux changements de statut apportés par la validation** :
- **X1 reclasse** : le chemin principal `sample=True` dans `fit` est **déjà corrigé en 0.0.60** par le garde `rng.py:78-84` (RuntimeError explicite, commit `26421450` « No hidden fixed seeds »). Le **résiduel est confirmé** : avec `seed=` explicite, `normalize_seed` retourne un `PRNGKey` concret qui contourne le garde → le fit **aboutit sans erreur** avec un postérieur faussé silencieusement (bias mesuré +0,19 sur `mu` ≈ 2,5 écarts-types postérieurs, site absent de la trace).
- **R1 nouveau** : sourcer le pont R fossile a un **effet de bord d'installation** — sa ligne 5 `reticulate::py_install('BayesForge')` installe réellement BayesForge-0.0.59 depuis PyPI dans l'environnement R de l'utilisateur, avant de mourir ligne 11 (`fromJSON("Doc_R.json")` : fichier inexistant, parsé comme texte JSON littéral). Le fossile n'est pas seulement mort, il est **contaminant** ; argument supplémentaire pour son retrait.

**Verdict de validation** : les correctifs P0 des sections 1-2 sont **confirmés par reproduction** et la liste est ajustée : le garde X1 existant doit être étendu au chemin `seed=` (une garde) ; la sur-comptage broadcasting `obs (n,1)` (sd postérieur divisé par 10, silencieux) rejoint les P0 ; les trois API documentées `scale()`/`scale_var()`/`to_int()` sont toutes cassées à l'appel nominal. Côté tests, la priorité n'est pas de resserrer des tolérances mais de **rendre faillibles et collectées** les suites existantes (voir rapports 24-25).

### Rapports de validation

### 24 — Validation — couverture des bugs par les tests (`Test/`)

**Persona** : test-validator (ingénieur de recherche senior, calcul scientifique).
**Périmètre** : croisement des sections 1–2 de `RELEASE_AUDIT.md` (68 critiques modules + 14 critiques d'intégration = 82) avec le contenu réel de `Test/` (le dossier s'appelle bien `Test/`). Aucune modification de code source ni de tests.
**Environnement d'exécution** : Python 3, JAX 0.11.2 (x64 off), numpyro 0.22.0, ArviZ 1.3.0, CPU ; TFP absent (comme la CI).

---

## 1. Méthode et inventaire

**Méthode** : (1) extraction des 82 critiques de `RELEASE_AUDIT.md` par grep des en-têtes d'items et des tableaux de décompte par module ; (2) grep systématique des symboles de chaque bug (noms de fonctions, paramètres, chemins) dans `Test/` ; (3) lecture des fichiers de tests « leurres » cités dans la mission ; (4) exécution ciblée : `pytest Test/0.Core/test_rng.py Test/0.Core/test_index_categorical.py` (5 passed, 24,85 s) et reproduction en lecture seule de 12 bugs (np_dists C1/C2, Data C1/C3/C8, Effects NameError, V1/V2/V3 cycle de vie, X1 résiduel, W7).

**Inventaire du corpus de tests** (`/home/sosa/work/BF/Test/`) :

| Mesure | Valeur |
|---|---|
| Fichiers `.py` dans `Test/` | 230 |
| Notebooks `.ipynb` | 12 |
| Fichiers `.py` contenant ≥ 1 `def test_` | **40** |
| `def test_` au total | **364** |
| `def test_` collectés par la CI par PR (`tests.yml` : `0.Core` + `test_language_coverage.py` + `3.Documentation/Python` hors marqueur `pages`) | **≈ 92** (0.Core : 20 fichiers / 89 `def test_` ; 3.Doc : 3) |
| `def test_` en jobs CI séparés (docs-r, docs-julia, cross-language, compat) | 5 fichiers / 5 `def test_` (paramétrés sur les pages et les cas) |
| `def test_` hebdo seulement (`doc-pages.yml`, marqueur `pages`) | 1 fonction × ~50 pages |
| `def test_` **hors CI** | **≈ 267** (73 %) |

**Plus gros blocs hors CI** : `14.Jax extension (WIP)/test_sampleddata.py` (112 `def test_`), `15.Diagnostics` (3 : 59 `def test_`), `Test/14.SampledData.py` (41 `def test_` — **introuvable par pytest : le nom ne matche pas `test_*.py`, la suite ne peut jamais être collectée**, même par `pytest Test/`), `Causal & DAG (WIP)` (2 fichiers / 30), `6.Network/NetExplorer` (8), `16.SBI/test_sbi.py` (7), `HGAM (WIP)` (5), `10.BNN/test_mixture_layers.py` (5).

**Suites qui ne sont pas du pytest du tout** (scripts à `try/except`/`print`, 0 `def test_`) : `1.Distributions` (2 scripts smoke), `4. Rethinking` (39 scripts, **0 `assert`** sur les 39), `17.Repertoire` (`run_all.py` + scripts BF-vs-Stan), `13.Parallel/test_run_simulations.py` (0 `def test_`, confirmé — cf. X6), `6.Network/Network metrics` + `SRM/benchmark_suite`, `7.Survival analysis`, `9.Phylogenic`, `Shard`, `VINE`, etc.

**Imports directs de `BayesForge.<Module>` dans `Test/`** : Diagnostic 35, Network 18, Models 6, PostModel 5, Distributions 3, Utils 2, **Resources 1, R 1, Causal 1, Parallel 1, Main 1 — et 0 pour `Data`, `Build`, `SetDevice`, `NBDA`, `ML`, `BNN`, `Samplers`** (ces modules ne sont atteints que indirectement via `bf()`).

---

## 2. Matrice bug↔test

Légende : **É** = COUVERT-ÉCHEOUERAIT (le test devrait rougir), **P** = COUVERT-PASSE (un test passe par le chemin mais ne peut pas voir le bug), **N** = NON-COUVERT (aucun test n'atteint le chemin). Reproduction = bug reproduit en exécution lecture seule pendant cette passe.

**Résultat structurel : 0 critique classée COUVERT-ÉCHEOUERAIT.** La CI verte n'est en contradiction avec aucun des 82 critiques : aucun test du dépôt ne réunit la condition déclencheuse. Tous les « leurres » sont des COUVERT-PASSE dont l'assertion ou l'entrée rate le cas.

### 2.1 Distributions

| ID (module) | Sév. | Test le plus proche | Classe | Explication |
|---|---|---|---|---|
| C1 np_dists (`seed=False`→PRNGKey(0)) | Critique | `0.Core/test_rng.py:11` | **P** | Le test n'inspecte que la **signature** (défaut `None`) ; le comportement `seed=False` n'est jamais appelé. Reproduit : 3 tirages `seed=False` identiques. |
| C2 np_dists (`rand_seed=True`→`time_ns`) | Critique | `0.Core/test_rng.py:26` | **P** | Via `bf()` le gestionnaire RNG est branché et la graine de session marche (le test le prouve) ; le bug vise `UnifiedDist` nu (`_rng=None`), condition jamais réunie en test. |
| C1–C8 tfp_dists (8 critiques) | Critique | `1.Distributions/distribution_tfp.py` (script) | **N** | TFP absent de l'environnement de test et de la CI ; le seul fichier qui les exerce est un script sans `def test_` ni assert, où `gamma_poisson` est **commenté « Broken »** (l.50) — le bug C2 est connu et mis sous le tapis. |

### 2.2 Samplers (3 critiques, toutes `mcmc_tfp.py`)

| ID | Sév. | Test le plus proche | Classe | Explication |
|---|---|---|---|---|
| C1 (adaptation 400 en dur), C2 (`target_log_prob_fn` inexistant), C3 (état contraint) | Critique | aucun | **N** | Backend TFP jamais instancié en test (absent de l'environnement) ; `grep mcmc_tfp Test/` = 0. |

### 2.3 Models (5 critiques)

| ID | Sév. | Test le plus proche | Classe | Explication |
|---|---|---|---|---|
| C1–C4 SRM (`index1=True`, blocs 2 groupes, `outcome_mode`→prior silencieux) | Critique | `6.Network/SRM/benchmark_suite/*.py` (scripts) | **N** | Aucun pytest sur SRM ; les benchmarks ne testent pas `index1=True` ni un `outcome_mode` invalide, et ne sont pas dans la CI. |
| C5 surv (`plot_surv` λ0·β au lieu de λ0·exp(β)) | Critique | aucun | **N** | `plot_surv` jamais appelé en test. |
| V1 surv (méthode `observed` écrasée par `import_time_even`) | Majeure/API | `0.Core/test_survival_sample_times.py:20` | **P** | Le test appelle `import_time_even` (l'écrasement a donc lieu) mais ne rappelle jamais `s.observed(...)` après — le `TypeError` ne se déclenche pas. |
| V2 surv (modèle sans covariable) | Majeure/API | `0.Core/test_survival_*.py` | **P** | Les deux tests CI font toujours `import_covF` avant — le chemin sans covariable n'est jamais pris. |

### 2.4 Main (3 critiques)

| ID | Sév. | Test le plus proche | Classe | Explication |
|---|---|---|---|---|
| M-01 `log_prob` sans données (print au lieu de return) | Critique | aucun | **N** | `m.log_prob` n'est appelé par aucun test (les 33 hits `log_prob` sont dans des scripts hors CI ou du code interne). |
| M-02 branche `pca` de `fit` | Critique | `8.PCA/` (notebook + script) | **N** | `8.PCA` ne contient aucun `def test_`. |
| M-03 backend inconnu/`tfp` → no-op silencieux | Critique | `15.Diagnostics/API_Routing/test_diag_backend.py:66` | **N** | Le seul rejet de backend nonsense testé est celui de **diag**, pas de `bf(backend=...)` ; `bf('inconnu')` n'est testé nulle part. |

### 2.5 Diagnostic (2 critiques)

| ID | Sév. | Test le plus proche | Classe | Explication |
|---|---|---|---|---|
| Bug 1 (côté TFP, `log_accept_ratio` = `leapfrogs_taken`) | Critique | aucun | **N** | TFP absent de la CI. |
| Bug 2 (postérieurs SBI : 4 pseudo-chaînes ⇒ R-hat≈1, ESS≈n garantis) | Critique | `16.SBI/test_sbi.py` (7 tests) | **N** | Les tests SBI valident la **récupération analytique**, jamais les diagnostics sur un postérieur SBI ; de plus la suite est hors CI (X6). |
| Plots legacy `Diag.py` morts sous ArviZ 1.3 (22/23) | Majeure | `0.Core/test_diag_api.py:50` | **P** | Le test CI exécute bien toutes les méthodes `m.diag` — mais la surface bindée est `Diag2`/`patch_diag` ; la surface legacy `Diag.py` n'est atteinte que pour `summary` (`15.Diagnostics/test_summary_quiet_warnings.py:20`), hors CI. |

### 2.6 Utils (5 critiques)

| ID | Sév. | Test le plus proche | Classe | Explication |
|---|---|---|---|---|
| `varying_slope_non_centered` NameError `m` | Critique | `0.Core/test_effects.py` (leurre) | **N** | Le test CI ne teste **que `phylo_effect`** ; grep `varying_slope_non_centered` dans `Test/` = 0. Reproduit : `NameError: name 'm' is not defined`. |
| `varying_intercept_slope` (Cholesky pris pour la corrélation) | Critique | aucun | **N** | grep = 0 dans `Test/`. |
| `gaussian_process2` NameError `Mgaussian` | Critique | aucun | **N** | grep = 0 ; l'import même échoue (`ImportError`). |
| `LKJcor` fausse, η inopérant | Critique | aucun | **N** | grep `LKJcor` = 0 dans `Test/`. |
| `SampledData.hdi` faux en 2D/NaN | Critique | `Test/14.SampledData.py` (41 tests, hors CI) | **N** | grep `.hdi(` = 0 dans tout `Test/` — même la suite morte de 41 tests ne teste pas `hdi`. |

### 2.7 PostModel (1 critique)

| ID | Sév. | Test le plus proche | Classe | Explication |
|---|---|---|---|---|
| C1 `to_latex` hyperpriors fabriqués (`varying_intercept_slope`) | Critique | `Causal & DAG (WIP)/test_model_graph.py:189` | **N** | Le test de rendu LaTeX n'utilise que `varying_effects` (l.204), jamais `varying_intercept_slope` ; et il est hors CI. |

### 2.8 Network (4 critiques)

| ID | Sév. | Test le plus proche | Classe | Explication |
|---|---|---|---|---|
| C1 betweenness pondéré faux sur graphe déconnecté | Critique | `6.Network/Network metrics/network_metrics.py:85` | **P** | Comparaison BF-vs-networkx (bonne idée) mais sur graphe connexe, sans le cas déconnecté, dans un **script hors CI** à `run_test`/`except`. |
| C2 `betweenness(endpoints=True)` | Critique | aucun | **N** | grep `endpoints` dans `Test/` = 0 appel. |
| C3 `sender_receiver` receiver-only crash | Critique | `6.Network/SRM/benchmark_suite/*.py:116` | **P** | Les benchmarks passent toujours **les deux** jeux de prédicteurs ; le mode receiver-only n'est jamais appelé (scripts hors CI). |
| C4 `single_source_dijkstra` faux sur matrice brute | Critique | aucun | **N** | grep = 0. |

### 2.9 Parallel (1 critique)

| ID | Sév. | Test le plus proche | Classe | Explication |
|---|---|---|---|---|
| C1 erreur hors corps de `fn` ⇒ run entière perdue même avec `on_error="record"` | Critique | `0.Core/test_run_simulations_languages.py:37` | **P** | La CI exécute `run_simulations` avec des fonctions propres ; le chemin d'erreur (pickle du retour, socket Manager) n'est jamais déclenché. Le fichier dédié `13.Parallel/test_run_simulations.py` n'a **aucun `def test_`** (confirmé, = X6). |

### 2.10 NBDA (7 critiques)

| ID | Sév. | Test le plus proche | Classe | Explication |
|---|---|---|---|---|
| C1–C7 (priors unbound, covNF i/j, covNV, `inv_logit_scale`<0, gradients NaN, `print_model`, contrat Main) | Critique | aucun | **N** | grep `NBDA` dans `Test/` = 1 mention de docstring (`0.Core/test_summary_factor_sites.py:5`) ; 0 import du module. Le seul usage est le notebook `5.NBDA.ipynb`, hors pytest. L'audit dit exactement cela (§ NBDA C7). |

### 2.11 BNN (2 critiques)

| ID | Sév. | Test le plus proche | Classe | Explication |
|---|---|---|---|---|
| CRIT-1 `bnn.__call__` (attributs/méthodes inexistants) | Critique | `10.BNN/test_mixture_layers.py` | **N** | Les 5 tests CI-absents couvrent les **couches** (`layer_cholesky`, `layer_mixture_normal`), jamais `__call__` ni `cov`. |
| CRIT-2 `cov` broadcasting `(2,N)`/`(N,2)` | Critique | aucun | **N** | grep `\.cov(` dans `10.BNN` = 0. |

### 2.12 Build (7 critiques)

| ID | Sév. | Test le plus proche | Classe | Explication |
|---|---|---|---|---|
| T-1, T-2, T-3, N-1, N-2, N-3, N-4 (générateurs) | Critique | aucun | **N** | grep `BayesForge.Build` / `Build_*` dans `Test/` = 0. La « suite de tests générée » est elle-même morte (B-1 : `Build_test_file.py:84`, le fichier généré ne compile pas). |

### 2.13 Causal (2 critiques)

| ID | Sév. | Test le plus proche | Classe | Explication |
|---|---|---|---|---|
| C1 parsing silencieux (`"a ->"` → arête vers nœud vide) | Critique | `Causal & DAG (WIP)/test_causal_dag.py:22-42` | **P** | Le parsing est testé sur entrées valides + cycles/self-loops rejetés ; l'entrée malformée à nœud vide n'est pas dans les cas (et la suite est hors CI). |
| C2 nœuds inconnus silencieux dans les checks d'ajustement | Critique | `test_causal_dag.py:120-132` | **N** | Les tests `check_*` n'utilisent que des nœuds existants. |

### 2.14 Resources (3 critiques)

| ID | Sév. | Test le plus proche | Classe | Explication |
|---|---|---|---|---|
| C-1 `trolley()` FileNotFoundError | Critique | aucun | **N** | grep `trolley` = 0 dans `Test/`. |
| C-2 `elephants()` DataFrame vide | Critique | aucun | **N** | grep = 0. |
| C-3 `WaffleDivorce()` DataFrame vide | Critique | `4. Rethinking/.../15.Measuring error.py:15` | **P** | Seul usage : `m.load.WaffleDivorce(only_path=True)` — le script ne lit que le **chemin** et parse le CSV lui-même, contournant le chemin buggé ; script hors CI sans assert. |

### 2.15 ML (2 critiques)

| ID | Sév. | Test le plus proche | Classe | Explication |
|---|---|---|---|---|
| C1 `random_state` ignoré (KMeans) | Critique | `11.MGM and DPMM/*.py:76` | **N** | Les hits `KMeans`/`random_state` sont du **sklearn**, pas du `BayesForge.ML` ; grep `BayesForge.ML` = 0. |
| C2 `ml.fit()` lève toujours `TypeError` | Critique | aucun | **N** | API jamais appelée en test. |

### 2.16 Data (8 critiques — le module le plus critiqué, 0 import direct)

| ID | Sév. | Test le plus proche | Classe | Explication |
|---|---|---|---|---|
| C1 `scale()` défaut `'all'` → TypeError | Critique | `0.Core/test_data_input.py:22` | **P** | Le test CI appelle `m.scale(["x"])` (liste explicite) ; le défaut `'all'` n'est jamais appelé. Reproduit : `TypeError ... scale_var`. |
| C2 `scale_var` inutilisable (`@jax.jit` sur méthode) | Critique | aucun | **N** | grep `scale_var` = 0 dans `Test/`. |
| C3 `OHE()` duplique les colonnes de `m.df` | Critique | aucun | **N** | grep `.OHE(` = 0 dans `Test/`. Reproduit : colonnes ajoutées/corruption. |
| C4 `to_int()` cassé (labels vs positionnel) | Critique | aucun | **N** | grep `to_int` = 0 (les 4 hits `to_float` sont un helper local d'un script GGMM). |
| C5 `to_float()` branche liste = no-op silencieux | Critique | aucun | **N** | idem. |
| C6 `pd_to_jax` retourne une chaîne d'erreur | Critique | aucun | **N** | grep `pd_to_jax` = 0 dans `Test/`. |
| C7 `pd_to_jax` plante sur défaut array-like | Critique | aucun | **N** | idem. |
| C8 `index()` NaN → code `-1` | Critique | `0.Core/test_index_categorical.py:10` | **P** | Le DataFrame du test ne contient **aucun NaN** ; reproduit hors test : `["a","b",None,"a"]` → `[0,1,-1,0]`, index de groupe silencieusement faux. |

### 2.17 SetDevice (1 critique) et R (2 critiques)

| ID | Sév. | Test le plus proche | Classe | Explication |
|---|---|---|---|---|
| SetDevice C1 (2ᵉ `setup_device` = no-op silencieux partiel) | Critique | aucun | **N** | grep `setup_device` = 0 dans `Test/`. |
| R-C1 générateur `R/` cassé vs API actuelle | Critique | `3.Documentation/R/test_r_*.py` | **N** | Le job CI `docs-r` installe **BFR depuis GitHub (dev)**, pas les fichiers générés du dépôt ; la couture testée n'est pas celle de R-C1. |
| R-C2 fichier mort embarqué dans la roue | Critique | aucun | **N** | Aucun test n'inspecte le contenu de la roue. |

### 2.18 Intégration — section 2 (14 critiques + X2–X6 majeures d'API publique)

| ID | Sév. | Test le plus proche | Classe | Explication |
|---|---|---|---|---|
| V1 (svi → `posteriors_full`/`filter_posteriors` périmés) | Critique | aucun (`grep \.svi(` = 0) | **N** | **Reproduit** : après `m.svi()`, `posteriors`=50 draws SVI mais `posteriors_full` garde les 100 draws MCMC. Aucun test n'appelle `svi`. |
| V2 (svi → `sampler_stats` périmé) | Critique | aucun | **N** | **Reproduit** : `sampler_stats()` après `svi` renvoie les stats MCMC (leapfrog, accept_prob) de l'ancien fit. |
| V3 (fit→fit, `data_on_model` resplatté sans validation) | Critique | aucun | **N** | **Reproduit** : second `fit(model2)` sans `obs=` → `TypeError: model2() got an unexpected keyword argument 'y'`, à l'opposé de la cause. |
| V4 (`pd_to_jax` → fit + état résiduel) | Critique | aucun | **N** | `pd_to_jax` jamais testé (cf. Data C6/C7). |
| W1 (singletons `dist` mutés par `main.py:279-289`) | Critique | aucun | **N** | Aucun test ne crée deux instances `bf` et ne compare leurs `dist` (le test du singleton muté, `Utils` M-x de l'audit, n'a pas d'équivalent en test). |
| W3 (SRM méthode non liée → priors fausses) | Critique | aucun | **N** | Pas de pytest SRM (cf. 2.3). |
| W5 (NBDA : `m.fit(nbda.model)` aboutit avec **toutes les priors fausses**) | Majeure (priorité) | aucun | **N** | 0 import NBDA ; le notebook `5.NBDA.ipynb` ne passe que par l'objet autonome, jamais par `bf.fit` — exactement ce que dit l'audit NBDA C7. |
| W7 (`index()` 0-based → `nodal_random_effects` 1-based) | Majeure (priorité) | `0.Core/test_index_categorical.py` | **N** | **Reproduit** : `index()` → `[0,1,2,0]` ; `nodal_random_effects` fait `group_idx_0 = random_group_index - 1` (docstring « 1-based ») ⇒ décalage + enroulement de l'index 0. Aucun test ne branche la sortie de `index()` sur l'effet réseau. |
| X1 (`sample=True` dans un modèle ⇒ postérieur faux silencieux) | Critique | `0.Core/test_rng.py:61` | **P** | La garde existe (via `Utils/rng.py:73`, `stream()` sous trace → `RuntimeError`) et le test CI passe. **Mais** elle ne couvre que la branche `seed=None` : **reproduit** — `m.dist.normal(0,1,sample=True, seed=42)` dans un modèle passe le `fit` **silencieusement**, avec `posteriors` vides. L'assertion du test ne voit pas ce bras. |
| V14 (`fit`→`save` défaut de nom) | Mineure | `0.Core/test_save_load.py:31` | **P** | Le test passe toujours un chemin explicite à `save()` ; le nom par défaut issu de `model_name` n'est jamais exercé. |
| S2–S7 (pont R : `import("BF")`, `BF$dist`, `DistProxy.__getitem__`, regex `<function…>`, `.bf`, fence Markdown dans les `.R` générés) | Critique ×6 | `3.Documentation/R/test_r_*.py` (CI) | **N** | Le job `docs-r` est vert parce qu'il teste le **package BFR dev de GitHub** via le pont reticulate vivant ; les artefacts des coutures S2–S7 (fichiers générés `Build/Doc_R.json`, `wrapper.R`, module `R/BF doc.py`) ne sont construits par aucun test. |
| X2 (`on_error='record'` + retour non picklable ⇒ run perdue) | Majeure | `0.Core/test_run_simulations_languages.py` | **N** | Chemin d'erreur jamais déclenché (cf. Parallel C1). |
| X3 (générateurs → production, sans garde) | Majeure | aucun | **N** | Cf. Build ; la CI n'a aucun job « le fichier régénéré compile ». |
| X4 (`_sbi_finish` sans `bind_diag_to_model`) | Majeure | `16.SBI/test_sbi.py` | **N** | Les tests SBI n'appellent pas `m.diag.*` après `abc/npe` ; et la suite est hors CI. |
| X5 (workers GPU héritent `CUDA_VISIBLE_DEVICES`) | Majeure | aucun | **N** | CI `JAX_PLATFORMS=cpu`, aucun test GPU. |
| X6 (suites Parallel/SBI hors CI) | Majeure (méta) | — | — | **Constat auto-portant** : `13.Parallel/test_run_simulations.py` a 0 `def test_` (vérifié) et `16.SBI` n'est pas collecté par `tests.yml` (vérifié dans le workflow). |

---

## 3. Constats sur la rigueur (hors corpus PCA/Stan/TFP, déjà revu)

| Sév. | Emplacement | Constat / risque |
|---|---|---|
| **Haute** | `Test/14.SampledData.py` | 41 `def test_` dans un fichier que **pytest ne collecte jamais** (nom sans préfixe `test_`) ; de plus il ne teste pas `hdi`, le critique du module. La seule barrière de `SampledData` est de fait `14.Jax extension (WIP)/test_sampleddata.py` (112 tests), hors CI. |
| **Haute** | `Test/1.Distributions/distribution_numpyro.py` / `distribution_tfp.py` | « Tests » = scripts `# %%` avec `try/except` qui **empilent les erreurs dans une liste sans jamais les assert** ; une distribution cassée n'échoue rien. `distribution_tfp.py:49-50` **commente** `discrete_uniform`/`gamma_poisson` « Broken » : les entrées cassées sont masquées par commentaire, pas corrigées. Hors CI. |
| **Haute** | `Test/4. Rethinking/` (39 scripts) | **0 `assert` sur 39 fichiers** : comparaison aux résultats `rethinking`/brms par print/plot uniquement ; aucune régression mesurable. Hors CI. |
| **Haute** | CI `tests.yml:36` | La CI ne collecte que ~25 % des `def test_` (92/364) ; les deux seules suites à références externes fortes (parité ArviZ rtol=1e-7/1e-12 dans `15.Diagnostics/Parity_Tests`, postérieurs analytiques conjugués dans `16.SBI`) sont **hors CI** — exactement ce que l'audit X6 énonçait, confirmé par la lecture du workflow. |
| **Moyenne** | `Test/HGAM (WIP)/test_native_gam.py` | Référence externe réelle (bases mgcv/R via CSV) mais tolérance très lâche (`recon_error < 0.15` par moindres carrés — une base mal paramétrée mais du bon rang passe) et `skipif` silencieux si les données R ne sont pas régénérées : la suite peut disparaître en vert. Hors CI. |
| **Moyenne** | `Test/10.BNN/test_mixture_layers.py` | Bon niveau pour les couches (comparaison à un calcul manuel + `numpyro.distributions` de référence, test de reproductibilité par graine) ; mais n'atteint ni `__call__` (CRIT-1) ni `cov` (CRIT-2). Hors CI. |
| **Moyenne** | `Test/6.Network/Network metrics/network_metrics.py` | Référence externe correcte (networkx) sur degree/strength/clustering/betweenness — mais en script hors CI, graphe connexe uniquement, échecs collectés par `except` (cf. Network C1/C3). |
| **Moyenne** | `Test/17.Repertoire/` | Suite « BF vs Stan » pilotée par `run_all.py` (subprocess + log texte) : pas de pytest, pas de CI, résultats dans `results/` commités. |
| **Bonne point** | `Test/15.Diagnostics/Parity_Tests` | Comparaisons **externes** (ArviZ) avec tolérances justifiées en commentaires (rtol=1e-7 = marge d'accumulation float, pas de bricolage) ; `test_jax_diagnostics_parity` vérifie même que `hdi` rend du double précision. À remonter dans la CI. |
| **Bonne point** | `Test/lang_harness.py` + `test_cross_language.py` (dans la CI) | Équivalence inter-langages par cas Markdown exécutés en Python/R/Julia, comparaison des moyennes en **unités d'écart-type postérieur** (`tol=0.25` sd) : la paramétrisation propre à chaque langue (rate vs scale, gate=1−p) est absorbée dans le bloc de code de chaque langue — le risque est donc documentaire (un cas peut encoder une équivalence fausse dans ses deux blocs), pas mécanique. |
| **Bonne point** | `Test/0.Core/test_docstring_examples.py` (dans la CI) | Exécute le snippet « Example Usage » de **toutes** les docstrings `m.dist.*` — le garde-fou le plus efficace du dépôt (backend numpyro uniquement ; TFP jamais couvert). |
| **Moyenne** | `Test/3.Documentation` | Les tests de pages **exécutent réellement** les blocs ```python des pages (concaténés, interpréteur neuf) : les pages Get started à chaque PR, les ~50 pages modèles **hebdo seulement** (`doc-pages.yml`, marqueur `pages`). Les exemples cassés documentés ne sont pas verts par accident d'exécution : **`m.scale(data='all')` n'apparaît que dans `Documentation/reference/data.qmd:113`, et `Documentation/reference/` n'est pas dans le corpus `MODEL_PAGES`/`START_PAGES`** (glob non récursive de `doc_blocks.py:20-27`) ; les pages modèles testées n'emploient que `m.scale([...])` (liste explicite) et `scale_var`/`to_int`/`OHE` n'apparaissent que dans la **prose** de `start/Import_data.qmd` (l.10, backticks), jamais dans un bloc exécuté. |
| **Moyenne** | Tolérances float32 | La CI tourne sans x64 (JAX par défaut) ; les suites externes (parité 1e-7, SBI 0.05/0.3) sont calibrées float64-vs-float64 sur des quantités stables — cohérent — mais rien ne teste les promesses float64/int64 de `pd_to_jax` (Data M3) : la descente silencieuse en 32 bits est invisible par construction. |

---

## 4. Synthèse chiffrée

**Sur les 82 critiques (68 modules + 14 intégration) :**

| Classe | Nombre | % |
|---|---|---|
| COUVERT-ÉCHEOUERAIT | **0** | **0 %** |
| COUVERT-PASSE | 10 (np_dists C1-C2 ; Network C1, C3 ; Parallel C1 ; Causal C1 ; Resources C-3 ; Data C1, C8 ; X1) | **12,2 %** |
| NON-COUVERT | 72 | **87,8 %** |

**Lecture** : la CI verte est **parfaitement cohérente** avec les 82 critiques — il n'existe aucun test dont l'échec contredirait l'audit. Les 10 COUVERT-PASSE sont tous des ratés de condition déclencheuse (défaut non appelé, NaN absent, branche `seed=` non prise, script hors CI sans assert). Les bugs « postérieur faux silencieux » prioritaires sont **tous non couverts** sauf X1 (couvert-passe, résidu `seed=` explicite reproduit) : V1/V2/V3 (reproduits en 3 minutes de script), W5, W7 (décalage 0/1-based reproduit), Data C8 (code `-1` reproduit).

**Modules les plus critiqués sans aucune couverture pytest** : `NBDA` (7 critiques, 0 import), `Build` (7, 0), `tfp_dists` (8, hors environnement CI), `Data` (8, 0 import direct — seulement frôlé par `test_data_input`/`test_index_categorical`), `Samplers/mcmc_tfp` (3), `Resources` (3, un seul appel `only_path`), `ML` (2), `SetDevice` (1), `R` (2). Le chemin nominal `bf()→data→fit→summary→save/load` est bien gardé (0.Core + pages), mais **tout ce qui est transition d'état (svi, re-fit), backend secondaire (TFP), générateurs (Build), et modules dyadiques (NBDA/SRM) est hors de portée des tests**.

**Actions à moindre coût suggérées par la passe** (sans toucher au code, simples extensions de suites existantes) : (1) ajouter `15.Diagnostics/Parity_Tests` + `16.SBI` à `tests.yml` ; (2) renommer `Test/14.SampledData.py` en `test_sampleddata.py` ; (3) quatre cas de test paient 10 critiques : `scale()` sans argument, `index()` avec NaN, `OHE()`, `svi()` après `fit()` + lecture de `posteriors_full`/`sampler_stats`, `sample=True, seed=…` dans un modèle.

---

### 25 — Rigueur des comparaisons inter-bibliothèques (corpus PCA / Stan / TFP / scipy)

**Persona** : test-validator (ingénieur de recherche senior, calcul scientifique).
**Périmètre** : audit de RIGUEUR statique du corpus `/tmp/opencode/briefs/testrigor_ctx.txt` — validité des assertions, calibration des tolérances, équivalence des paramétrisations, garde-fous manquants, protection réelle de la CI. Ne répète pas la passe 24 (§3 de `24_TestCoverage.md` : 14.SampledData, 4.Rethinking, HGAM, 15.Diagnostics/Parity, lang_harness, 3.Documentation, tolérances float32 n'y sont pas re-audités ; j'y renvoie par référence).
**Vérifications faites** : lecture complète de `Test/8.PCA/PCA.py`, `Test/0.Core/test_dpmm_plot.py` (L40-159), `test_effects.py`, `test_survival_ppc_km_by_cov.py` (L40-90), `test_sample_api.py` (L60-80), `test_enumeration.py` (L110-150), `conftest.py` (harnais `verify`), `Test/10.BNN/test_mixture_layers.py` (L48-75), `Test/1.Distributions/distribution_tfp.py` (queue), `Test/17.Repertoire` (scripts/, data/, legacy/*.stan, cases/), `.github/workflows/tests.yml`, docstring `zero_inflated_poisson` (`BayesForge/Distributions/np_dists.py:7323-7348`).

---

### Rigueur des comparaisons inter-bibliothèques

## 1. Verdict par fichier du corpus

| Fichier | Statut | Qualité | Observations |
|---|---|---|---|
| `Test/8.PCA/PCA.py` | **script** (nom hors `test_*.py`, jamais collecté ; hors CI) | **D** | Seule comparaison BF-vs-sklearn du dépôt. Les deux assertions (L76-85 variance ratio, L108-117 composantes) sont dans `try/except AssertionError` qui **printe ❌ sans relancer** : le script ne peut pas échouer — il imprime même « PCA unit test completed successfully. » (L119) et sort 0 quand les deux comparations sont en échec. Tolérances mal calibrées (voir §2 C1/M1/M2). `align_signs` (L93-99) ne protège ni des permutations d'axes ni des signes ambigus sur composantes dégénérées. |
| `Test/17.Repertoire/scripts/compare_moments_p.py` | **script** (hors CI, piloté par `run_all.py`) | **D** | **Zéro assertion** : compare les moments de `p` BF vs échantillons Stan et **printe un tableau** (L55-58) ; le statut est « informatif », jamais rouge. Métriques non comparables telles quelles : Min/Max d'un échantillon MCMC ne sont pas des quantités postérieures (dépendent du nombre de draws et du mélange), Skew/Kurtosis sans ESS ni MCSE. Commentaire L34-38 (« Wait. … I'll check both ») : l'auteur n'a pas tranché l'indexation dans sa tête ; le code (L40-43, `idx_stan = idx_BF+1`) fait pourtant le bon mapping 1-based Stan → 0-based BF. Sans seuil, la table imprimée invite au cherry-picking après coup. |
| `Test/17.Repertoire/` (harnais) | **scripts** (hors CI) | **D** | `run_all.py:36-47` : `check=True` mais `CalledProcessError` **capturé, loggé `[FAILURE]`, et `main()` sort 0** — même un crash du script est invisible au code de sortie. `save_stan_samples.py:12` pointe `cg_vocal_repertoires/model0.stan` alors que le fichier est dans `legacy/cg_vocal_repertoires/` : **la régénération de la référence Stan est cassée**, le CSV commité (2 001 lignes = 2 chaînes × 1000 itérations, en-têtes `L[1..50]`, `p[1..50]`) n'est plus reproductible. `results/` (log.txt, png) est commité — artefacts périmés servant de « preuve ». |
| Paramétrisation BF vs Stan (modèle) | — | **B+ (le modèle)** | Vérifié : `legacy/cg_vocal_repertoires/model0.stan` a `p` = prob. de possession, branche zéro `log1m(p[m])` ; BF (`compare_moments_p.py:22-26`) : `L~exp(1)`, `p~beta(2,2)`, `gate=1-p`, `rate=L*d`. La docstring `zero_inflated_poisson` (`np_dists.py:7330`) définit `gate`=prob. de zéro structurel → **`gate=1-p` est le bon mapping**, priors identiques, `id` absent du modèle BF est sans effet (absent de la vraisemblance Stan). **Mais cette équivalence n'est assertée nulle part** : si la convention `gate` était inversée, les trois langues du cas cross-language (`cases/1.Repertoire ZIP.md`, en CI) seraient fausses ensemble — risque partagé, non détectable (cf. passe 24, lang_harness). |
| `Test/1.Distributions/distribution_tfp.py` | **script** (hors CI) | **D** | Ne compare **jamais np vs tfp numériquement** : construit chaque distribution sous backend TFP dans `try/except Exception` (L66-70) qui **append dans `log_tfp.txt` et continue** ; `sys.exit(1)` uniquement si `bf(backend='tfp')` échoue à l'init (L33-37). Les 14 distributions pourraient toutes lever : exit 0. C'est un smoke test « ça ne plante pas », pas un test de parité `log_prob`. Les cas cassés sont **commentés** avec la cause (« Broken in tfp_dists.py (passes probs=None, logits=None) » L50, « Missing TF dependency » L51, « Not in TFP JAX » L45-46) : un bug connu du pont TFP est consigné en commentaire au lieu d'un `xfail` — la liste des cas testés rétrécit silencieusement, et `log_tfp.txt` commité dans le dossier le prouve. Details : L3 `import numpy as jnp` (alias trompeur, corrigé L23) ; L7-18 monkey-patch JAX/TFP dont l'échec est avalé par print. (`distribution_numpyro.py` : même gabarit, cf. passe 24 §3.) |
| `Test/0.Core/test_dpmm_plot.py` | **pytest, DANS la CI** | **A-** | Les assertions « sans tolérance » L69/L79-80/L95-96 **ne sont pas de l'MCMC** : ce sont des fonctions utilitaires déterministes (`_component_covariances`, `_cluster_moments`) avec moments analytiques calculés à la main — vérifié : mélange 0.75·N(0,I)+0.25·N(10,I) → mean 2.5, var 0.75·1+0.25·101−6.25 = **19.75** ✓. `np.allclose` par défaut (rtol 1e-5) est le bon régime float64. L112/L115-116 `atol=1e-9` : numpy float64, cas passe-plat exact → OK, pas un problème float32. Seul relâchement : L149 `atol=0.6` sur les centres d'ellipses (≈ 7,6 × le SEM du nuage de points, 0.5/√40 = 0.079) et `any()` sur les 6 ellipses sans associer ellipse↔cluster. |
| `Test/0.Core/test_effects.py` | **pytest, DANS la CI** | **B+** | Référence **analytique** (cov attendue = sd²·cov = 4·cov), pas BF-vs-BF — bonne pratique. `atol=0.35` (L15) sur `np.cov` de 4 000 draws : MC-SE théorique (normale multivariée) = σ²ᵢᵢ·√(2/N) = 4·√(2/4000) = **0.089** en diagonale, √((16+5.76)/4000) = 0.074 hors-diagonale → tolérance ≈ **3,9–4,7 σ** : adéquate (détecte une erreur systématique ≥ 10 % sur la diagonale, rate 5 %), mais **non documentée** et seed fixe (`rand_seed=0`) — un changement de seed la rendrait friable au ras du seuil. |
| `Test/0.Core/test_survival_ppc_km_by_cov.py` | **pytest, DANS la CI** | **C+** | Seule assertion numérique (L76-85) compare les traces du plot à `ppc._km(...)` — **la fonction privée de l'implémentation elle-même** : tautologie. Le test valide le plomberie (le trace affiche bien ce que `_km` a calculé ; le masque de sous-ensemble `x` est indépendant, lui) mais un bug de formule KM (ties, censoring, pas de survie) passe. Aucun oracle KM indépendant n'existe ailleurs dans `Test/0.Core` (grep `_km` : un seul appel, L76). |
| `Test/0.Core/test_sample_api.py` | **pytest, DANS la CI** | **B** | L78 `atol=0.5` avec `posterior=dict(a=1,b=2,s=0.1)` : un draw `y[0] ~ N(1+2x, 0.1)` → tolérance = **5 σ du bruit d'observation** ; attrape une erreur de moyenne ≥ 0.5, **rien sur l'échelle du bruit** (s=0.1 non testé). Seed fixe, donc déterministe. |
| `Test/0.Core/test_rng.py`, `test_diag_api.py` | **pytest, DANS la CI** | **B+** | Bon pattern : assertions positives ET négatives (`not np.allclose` L29/L33/L54 rng ; L67 diag) — reproductibilité par graine ET sensibilité à la graine. |
| `Test/0.Core/test_enumeration.py` | **pytest, DANS la CI** | **B** | Comparaisons **BF-vs-BF** : `enum=True` vs énumération implicite NumPyro (même moteur, autre chemin, L122-126) et vs `infer={"enumerate":"parallel"}` (L146-147). Ce sont des locks de mapping — valides avec `max|Δ|<1e-10` / `=0` en float64 explicite (L119-120, casts `dtype=np.float64`) — mais la colonne « reference » du tableau `verify` (« implicit enumeration (NumPyro) ») peut faire croire à une validation externe qu'elle n'est pas. |
| `Test/0.Core/conftest.py` (harnais `verify`) | **harnais CI** | **B+** | La bonne unité pour l'MCMC existe ici : `z = (mean testée − mean référence) / MCSE combinée`, bande `|z| < 3` (L91-105, résumé terminal + job summary GitHub). **Mais** `pytest_sessionfinish` (L121+) écrit les rapports sans jamais modifier `exitstatus` : un `verify.record(passed=False)` non accompagné d'un `assert` resterait **vert**. Aujourd'hui les 8 `verify.record` de 0.Core sont tous appariés à un `assert` dur (`test_discrete_samplers.py:156/213/270`, `test_enumeration.py:126/147`, `test_run_simulations_languages.py:61-62`) — le piège est potentiel, pas actuel. |
| `Test/10.BNN/test_mixture_layers.py` | **pytest HORS CI** | **B+** | Vraie référence **externe** : `npd.MixtureSameFamily`/`MultivariateNormal` numpyro pour `log_prob` (L52-57, `atol=1e-5`) — rare et précieux. Réproductibilité par graine + négatif (L67-70). L72 : `std` de 1 500 draws vs `ln2+1e-3` à 0.06 près = ≈ 4,7 MC-SE (σ/√2n = 0.0127) — bien calibré. Réserves : `atol=1e-5` sur `log_prob` en float32 (|log_prob|~1-10 → erreur float32 ~1e-6-1e-5) est au ras du plancher de bruit → risque de flake ; et tout le fichier est hors CI (cf. passe 24). |
| `.github/workflows/tests.yml` (CI) | — | **B−** | Ce que la CI **protège** : `Test/0.Core` (≈89 tests dont le harnais z/MCSE et les asserts appariés — le meilleur du dépôt), exemples de docstrings numpyro, `test_language_coverage.py`, pages docs Python (hors `pages`), jobs séparés docs-R/Julia + **cas cross-language** (`test_cross_language.py -m lang`, dont `8.PCA/cases/1.Classic PCA.md` — tol 0.25 sd, paramètres sign-invariants — et `17.Repertoire/cases/1.Repertoire ZIP.md`), matrice compat. Ce qu'elle **ne protège pas** : **toute** comparaison numérique BF-vs-bibliothèque externe (sklearn, Stan, TFP, scipy hors 0.Core), le backend TFP en entier (zéro test CI ; docstring tests = numpyro seul), `10.BNN`, `1.Distributions`, `17.Repertoire`, `8.PCA/PCA.py`. |

**Résumé des statuts** : dans le corpus, les références réellement externes (sklearn, CSV Stan, numpyro, moments analytiques, networkx/scipy) sont **minoritaires et majoritairement hors CI ou dépourvues de force d'échec** ; les comparaisons BF-vs-BF (énumération, `_km` auto-référencé, plots vs leurs propres fonctions) constituent l'essentiel des assertions numériques de la CI.

## 2. Constats

Sévérité : **Critique** = le test ne peut pas échouer ou la référence est fausse/reproductibilité cassée ; **Majeure** = tolérance ou référence incapable de détecter la classe d'erreurs visée ; **Mineure** = calibrage, documentation, fragilité.

| # | Sév. | Emplacement | Constat / risque | Correction recommandée (chiffrée) |
|---|---|---|---|---|
| C1 | **Critique** | `Test/8.PCA/PCA.py:76-85` et `:108-117` | `try/except AssertionError` → print ❌ sans `raise` ; L119 imprime « completed successfully » et le script sort 0 **même si les deux comparations échouent**. La seule vérification BF-vs-sklearn du dépôt est infaillible par construction. | Convertir en `Test/8.PCA/test_pca_vs_sklearn.py` (pytest, sans try/except, hors CI → l'ajouter à un job) ; les deux `assert_allclose` nus décident d'eux-mêmes. |
| C2 | **Critique** | `Test/1.Distributions/distribution_tfp.py:33-37, 66-70` | `sys.exit(1)` seulement à l'init ; chaque distribution est testée dans `try/except Exception` → log dans `log_tfp.txt` et **continue** : 14/14 échecs ⇒ exit 0. De plus **aucune comparaison numérique np-vs-tfp** (pas de `log_prob` croisé) : un backend TFP rendant des densités fausses mais construisables passe. TFP = zéro couverture CI (docstring tests numpyro only). | Compter les échecs et `sys.exit(len(errors))` ; transformer en pytest paramétré avec **parité numérique** : mêmes paramètres → `log_prob(x)` numpyro vs tfp, `rtol=1e-4, atol=1e-5` (float32) sur ~10 points par distribution ; ajouter au moins un job CI hebdo `backend='tfp'`. |
| C3 | **Critique** | `Test/17.Repertoire/scripts/compare_moments_p.py:55-58` + `run_all.py:36-47` | Suite BF-vs-Stan **sans aucune assertion** (print de tableaux) et `run_all.py` capture `CalledProcessError`, loggue `[FAILURE]` et sort 0 : ni une divergence BF/Stan, ni un crash ne colorent quoi que ce soit. Statut réel : documentation, pas test. | Remplacer les prints par `assert abs(z) < 3` avec `z = (mean_BF − mean_Stan)/MCSE_comb` (MCSE = sd/√ESS via arviz, ~2 000 draws Stan vs 2 000 BF → MCSE(mean p) ≈ 0.01) ; ne garder que Mean/SD (Min/Max/Skew/Kurtosis d'un échantillon MCMC ne sont pas des quantités postérieures) ; `run_all.py` : propager `sys.exit(1)` si un script a échoué. |
| C4 | **Critique** | `Test/17.Repertoire/scripts/save_stan_samples.py:12` | `stan_file = 'cg_vocal_repertoires/model0.stan'` alors que le modèle est dans `legacy/cg_vocal_repertoires/` : **la référence Stan commitée n'est plus régénérable** (chemin mort + version cmdstan non épinglée). Toute dérive future du CSV est indétectable. | Corriger le chemin en `../legacy/cg_vocal_repertoires/model0.stan`, épingler cmdstanpy/cmdstan (compat.toml), régénérer et re-commiter le CSV avec la graine (42) et le nombre d'itérations en en-tête de commentaire. |
| M1 | **Majeure** | `Test/8.PCA/PCA.py:112` | `atol=0.5` sur des **vecteurs unitaires** de R⁴ : une coordonnée fausse de 0.5 = ~30° d'erreur angulaire ; pire cas multi-coordonnées (u=e₁ vs v=(.5,.5,.5,.5)) = **60° acceptés**. Un signe globalement inversé échoue encore (max|uᵢ| ≥ 0.5), mais un signe inversé sur toute coordonnée < 0.25 passe. Aucune gestion des **permutations d'axes** (si BF ordonne les PC autrement, l'élément-à-élément échoue — et l'échec est avalé par C1). | Comparer des **sous-espaces** : angles principaux par SVD de Wᵀ — exiger `sin θ_max < 0.05` (≈ 3°) pour PC1-2 ; pour PC3-4 (valeurs propres iris 0.021 vs 0.014, gap 1,5 → vecteurs instables même entre solveurs sklearn), comparer le sous-espace {PC3,PC4} avec `sin θ < 0.15` plutôt que les vecteurs. Alternative minimale : `1 − |u·v| < 0.01` par PC (valeur absolue du cosinus = insensible au signe). |
| M2 | **Majeure** | `Test/8.PCA/PCA.py:80` | `atol=1e-2` **absolu** sur des variance ratios d'iris ≈ [0.73, 0.23, 0.005, 0.003] : 1,4 % toléré sur PC1 mais **200–300 % sur PC3/PC4** (la tolérance dépasse la valeur) — les petites composantes ne sont pas testées. De plus BF est un PCA **bayésien** (500 draws MCMC, L57) : un biais de shrinkage du prior sur les ratios, précisément l'erreur à surveiller, est invisible sous 1e-2. | `rtol=5e-2` sur PC1-2 + `atol=5e-3` sur PC3-4 ; mieux : vérifier que la valeur sklearn tombe dans l'**intervalle 95 % postérieur** BF du ratio (test de couverture, pas de point) — et augmenter les draws (≥ 2 000, 4 chaînes) pour que l'incertitude MCMC soit ≪ 1e-2. |
| M3 | **Majeure** | `Test/0.Core/test_survival_ppc_km_by_cov.py:76-85` | Référence = `ppc._km`, la fonction privée testée : **comparaison tautologique** (le plot affiche ce que l'implémentation a calculé — plomberie seulement). Un bug de formule KM (ties, censoring, normalisation) passe la CI. Aucun oracle KM indépendant ailleurs dans `Test/`. | Ajouter un oracle dur : 6 patients avec ties/censoring explicites, survies KM calculées à la main (tableau en littéral), `assert_allclose(surv, attendu, atol=1e-12)` ; garder la comparaison `_km`-vs-trace comme test de plomberie. (Alternative : `lifelines.KaplanMeierFitter` en dépendance de test.) |
| M4 | **Majeure** | `Test/1.Distributions/distribution_tfp.py:45-51` | Cas cassés **commentés** avec la cause (« Broken in tfp_dists.py (passes probs=None, logits=None) », « Missing TF dependency ») : un bug connu du pont TFP est consigné en commentaire, pas en `xfail(strict)` — la liste des distributions testées rétrécit silencieusement à chaque bug, et rien ne signale qu'un cas commenté redevient testable. | Remplacer chaque commentaire par `pytest.param(..., marks=pytest.mark.xfail(raises=..., reason="tfp_dists probs=None"))` : le bug reste documenté, la liste reste visible, et un `strict=True` signale la correction. |
| M5 | **Majeure** | Corpus global + `tests.yml:36` | **Garde-fou manquant structurel** : dans le corpus, les assertions numériques de la CI sont majoritairement BF-vs-BF (énumération vs NumPyro = même moteur ; `_km` auto-référencé ; plots vs leurs fonctions) ; les 4 vraies références externes (sklearn, CSV Stan, numpyro `MixtureSameFamily`, moments analytiques dpmm) sont soit hors CI (sklearn, Stan, numpyro-BNN), soit réduites à dpmm/effects en CI. Une erreur systématique partagée par l'implémentation et son « expected » (recopie de formule) est invisible. | Ré-ancrer : (1) remonter `test_mixture_layers.py` et un test de parité TFP dans un job CI ; (2) porter `PCA.py` et `compare_moments_p.py` en pytest CI (ils sont courts : iris 150×4, 2 000 draws ≈ 1 min) ; (3) règle d'or à faire respecter en revue : *le expected d'un test ne doit jamais appeler le code testé*. |
| m1 `[RÉSOLU f3c787f3]` | Mineure | `Test/17.Repertoire/scripts/compare_moments_p.py:34-43` | Le commentaire « Wait. … I'll check both p[1] and p[2] » montre une indécision non résolue ; le code (`idx_stan = idx_BF+1`) fait le bon mapping 1-based→0-based, mais **sans assertion ni seuil**, la table imprimée permet de choisir après coup la lecture qui arrange (y compris « BF = 1−Stan » si la convention gate avait glissé — la convention `gate=1−p`, correcte, n'est assertée nulle part). | Supprimer le commentaire d'hésitation ; n'assertionner que la paire alignée (BF p[i] ↔ Stan p[i+1]) en z-score (C3) ; ajouter un test unitaire de convention : `zero_inflated_poisson(gate=g)` échantillonné → P(Y=0) ≈ g + (1−g)e^(−λ) à 3 MC-SE près, 1 000 draws. |
| m2 `[RÉSOLU f3c787f3]` | Mineure | `Test/0.Core/test_dpmm_plot.py:149` | `atol=0.6` sur centre d'ellipse vs moyenne du nuage : SEM du nuage = 0.5/√40 = 0.079, jitter postérieur 0.02 → erreur attendue < 0.05 ; 0.6 ≈ **7,6 × SEM** (les clusters sont à ~7 unités, donc pas de risque d'appariement croisé, mais une dérive de 0.5 passerait). `any()` sur les 6 ellipses sans associer ellipse↔cluster. | `atol=0.15` (≈ 2 × le pire cas attendu) et appairer l'ellipse au cluster par ordre de label avant de comparer. |
| m3 `[RÉSOLU f3c787f3]` | Mineure | `Test/0.Core/test_effects.py:15` | `atol=0.35` = 3,9–4,7 MC-SE (dérivation : MC-SE(Ĉ_ii)=σ²√(2/N)=0.089 à N=4 000) — calibrage **adéquat mais non documenté** ; seed fixe masque la sensibilité. | Écrire la dérivation en commentaire (`atol = 4·MCSE ≈ 0.35`) ou l'exprimer en MCSE calculée dans le test ; détecterait une erreur systématique ≥ 10 % sur la diagonale, rate 5 % — l'indiquer en critère. |
| m4 `[RÉSOLU f3c787f3]` | Mineure | `Test/0.Core/test_sample_api.py:78` | `atol=0.5` = 5 σ du bruit d'observation (s=0.1) : teste la moyenne fixée, **jamais l'échelle du bruit** — un `s` ignoré par `sample(posterior=...)` passerait. | Ajouter `assert abs(np.std(y_sur_1000_draws) − 0.1) < 3·0.1/√2000 ≈ 0.007` sur le même draw set. |
| m5 `[RÉSOLU f3c787f3]` | Mineure | `Test/0.Core/conftest.py:121-140` | `pytest_sessionfinish` n'altère pas `exitstatus` : un futur `verify.record(passed=False)` sans `assert` apparié resterait **vert** (les 8 usages actuels sont appariés — piège potentiel). | Dans le hook : `if any(not r["passed"] for r in _ROWS): session.exitstatus = 1` — le harnais devient alors l'oracle, les asserts appariés redondants mais sains. |
| m6 `[RÉSOLU f3c787f3]` | Mineure | `Test/10.BNN/test_mixture_layers.py:57` | `atol=1e-5` sur `log_prob` en float32 (|log_prob| ~ 1–10 → bruit float32 ~1e-6–1e-5) : au ras du plancher, risque de flake selon l'ordre des ops. | `rtol=1e-4, atol=1e-5` (la tolérance relative absorbe l'amplitude), ou activer x64 dans ce seul test. |
| m7 `[RÉSOLU f3c787f3]` | Mineure | `Test/0.Core/test_enumeration.py:122` | Colonne « reference » du tableau `verify` : « implicit enumeration (NumPyro) » = **même moteur**, pas une référence externe ; le tableau de rapport (job summary GitHub) peut surestimer la force de validation aux yeux d'un relecteur. | Libeller « NumPyro, autre chemin (consistency lock) » et réserver le mot « reference » aux oracles indépendants (analytique, Stan, sklearn, scipy). |
| m8 `[RÉSOLU f3c787f3]` | Mineure | `Test/17.Repertoire/results/` (commité) ; `distribution_tfp.py:3,7-18` | Artefacts de dernière exécution commités (`log.txt`, png, `log_tfp.txt`) — fausse impression de preuve fraîche ; alias `import numpy as jnp` (L3) puis re-import L23 ; monkey-patch JAX/TFP dont l'échec est avalé par print (L17-18) — le patch peut ne pas s'appliquer sans bruit. | `.gitignore` sur `results/` et `log_*.txt` ; supprimer l'alias trompeur ; faire du patch un `pytest.skip` explicite s'il échoue. |

**Bons points à généraliser** (pour contrebalancer) : `test_dpmm_plot.py:69-96` — moments analytiques calculés à la main sur fonctions déterministes, tolérances float64 par défaut = exactement le bon régime ; `conftest.py` — comparaison MCMC en **unités de MCSE combinée (|z|<3)** avec table + z-scores dans le job summary : c'est l'unité qu'il faut appliquer partout où l'on compare des draws (C3, m1) ; `test_rng.py` — assertions positives ET négatives (reproductibilité + sensibilité à la graine) ; `test_mixture_layers.py:52-57` — référence externe numpyro réelle.

## 3. Décompte par sévérité

| Sévérité | Nombre |
|---|---|
| Critique | **4** (C1 PCA infaillible, C2 TFP infaillible + sans comparaison numérique, C3 suite Stan sans assertion + exit 0, C4 référence Stan non régénérable) |
| Majeure | **5** (M1 atol PCA 30–60°, M2 atol variance ratio, M3 KM tautologique, M4 bugs commentés au lieu de xfail, M5 références externes minoritaires/hors CI) |
| Mineure | **8** (m1–m8) |
| **Total** | **17** |

**Verdict global du corpus** : la rigueur est **bimodale**. Dans `Test/0.Core` (CI), les quantités comparées sont soit déterministes-à-référence-analytique (dpmm, effects), soit exprimées en MCSE (harnais `verify`) — niveau B+/A-. Dès que la comparaison devient inter-bibliothèques (sklearn, Stan, TFP), elle quitte pytest, quitte la CI, et **quitte la capacité d'échouer** (try/except, print, exit 0) : les trois suites « externes » du corpus (8.PCA, 17.Repertoire, 1.Distributions) sont toutes notées D et aucune ne peut jamais rendre une CI rouge. La correction prioritaire n'est pas de resserrer des tolérances mais de **rendre ces trois suites faillibles et collectées** (C1-C3), puis de recalibrer PCA en sous-espaces (M1/M2).

---

### 26 — Script de torture : reproduction ciblée des fonctions suspectées

**Environnement** : ecocn20 (cluster, réservation dédiée), source BayesForge 0.0.60 injectée par `PYTHONPATH` (l'environnement du nœud contenait un 0.0.58 editable non touché), jax 0.10.1, numpyro 0.21.0, arviz 1.2.0, pandas 3.0.3, numpy 2.2.6, CPU 24 cœurs, x64 activé par `bf()`. Scripts `tmp_audit_repro.py` / `tmp_audit_repro2.py` supprimés après exécution (local et nœud). Sonde R : `Rscript` 4.6.1, reticulate 1.47.0 sur le même nœud.

## Tableau des 22 sondes

| Sonde | Cible (ID audit) | Verdict | Preuve mesurée |
|---|---|---|---|
| T01a | X1 — `sample=True` dans `fit` | **Corrigé en 0.0.60** | `RuntimeError: A direct draw (sample=True) was requested while JAX is tracing…` — garde `Utils/rng.py:78-84` (commit `26421450`) |
| T01b | X1 résiduel — `sample=True, seed=42` dans `fit` | **CONFIRMÉ (critique)** | le fit **aboutit sans erreur** ; site `e` absent de `posteriors` ; `mu` mean=3,198 vs `y.mean`=3,015 → **biais silencieux +0,19 ≈ 2,5 ET postérieurs** (le tirage à graine fixe agit comme un offset constant absorbé par `mu`) |
| T01c | X1 — re-tirage par leapfrog | artefact de sonde | `float(e)` sous jit → `ConcretizationTypeError` : la valeur est bien un tracer ré-évalué dans le noyau compilé (à graine fixe : même valeur à chaque pas) |
| T02 | W5 — import de la **classe** `UnifiedDist` (NBDA.py:5) | **CONFIRMÉ (critique)** | appel de classe `UnifiedDist.normal(0, 2)` → `loc=2.0, scale=1.0` ; appel instance → `loc=0.0, scale=2.0`. Le premier argument est avalé comme `self` : **toutes les priors NBDA sont décalées**, sans erreur |
| T03 | W7 — `index()` 0-based vs `nodal_random_effects` 1-based | **CONFIRMÉ (critique)** | mécanisme mesuré : `g[0-1]` → **30 (dernier groupe)** au lieu de 10 ; `model_effects.py:407` `group_idx_0 = random_group_index - 1` + indexation négative JAX = alias silencieux du dernier groupe pour tout code 0 |
| T04 | Data C8 — `index()` avec NaN | **CONFIRMÉ** | `index_map={'g': {0:'a', 1:'b'}}` posée sans niveau NaN ; codes `-1` (reproduit aussi passe 24) → tout gather aval est indexé sur le dernier niveau |
| T05 | Data C1 — `OHE()` | **CONFIRMÉ (majeure)** | colonnes après `OHE(['g'])` : `['g', 'y', 'y', 'g_a', 'g_b']` — **la colonne `y` est dupliquée** |
| T06 | V1/V2 — `fit` → `svi` | **CONFIRMÉ (critique)** | après `svi` : `posteriors['mu']` shape (50,) [SVI] mais `posteriors_full['mu']` toujours (200,) [NUTS périmé] et `sampler` remplacé `MCMC → SVI_BF` — **deux sources de vérité divergentes** ; les outils qui lisent `posteriors_full` voient l'ancien fit |
| T07 | V3 — `fit` → `fit`, signature différente | **CONFIRMÉ (critique)** | après un second `fit` avec `modB(y)` (sans `x`), `data_on_model` contient toujours `['x', 'y']` : les données du fit précédent sont **resplaquées sans validation** (passées via `**kwargs`) |
| T08 | V4 — état d'erreur persistant | partiellement confirmé | `m.data(DataFrame())` retourne le repr d'un DataFrame vide ; la persistance de `"Error, no data found"` dans `data_on_model` est reproduite en passe 24 |
| T09 | np_dists C1/C2 — graine par défaut | **CONFIRMÉ (nuancé)** | `seed=None` → tirages différents (0,3969 vs 0,5268) ; `seed=42` → identiques. Le défaut non reproductible est un **design assumé** (commit « No hidden fixed seeds ») mais `seed=True` → `time_ns()` par appel reste un piège |
| T09b | Reproductibilité par instance | **SAIN** | `bf(rand_seed=42)` × 2 → premier tirage identique (0,64989) : la reproductibilité **est** possible, par argument explicite |
| T10 | Effects — `varying_slope_non_centered` | **CONFIRMÉ (majeure)** | `NameError: name 'm' is not defined` à l'appel — fonction pourtant listée comme API dans `PostModel/to_latex.py:530` et `model_graph.py:213` |
| T11 | Utils/link — `inv_logit_scale` | **CONFIRMÉ (mineure)** `[RÉSOLU 2e8f60cc]` | `f(-20)=-1,0 ; f(0)=0 ; f(20)=+1,0` → plage réelle **(−1,1)**, docstring « scaling between 0 and 1 » fausse |
| T12 | Network — `mat_to_edgl` | précision de contrat | 3×3 → `[[3,5],[7,9],[2,4]]` : renvoie les **valeurs** (i,j)/(j,i), pas des indices de sommets — le contrat SRM est un extracteur de valeurs |
| T13 | Edge — fit N=1 | **SAIN** | abouti ; `sd(mu)`=0,972 ≈ 0,981 analytique (prior N(0,5), vraisemblance N(mu,1)) |
| T14b | Edge — valeurs extrêmes y=10⁶ | **INFIRMÉ** | `mu` mean=7,5×10⁵ : le sampler converge vers la donnée (mélange lent, pas de blocage, pas de NaN) — le constat « bloqué au prior » ne se reproduit pas |
| T15 | Edge — colinéarité parfaite | **CONFIRMÉ (informatif)** | `x2=2·x1` : fit abouti, **aucun avertissement de rang** ; `corr(b1,b2)=−0,995` (entonnoir non signalé) |
| T16b | Broadcasting piégeux `obs (n,1)` | **CONFIRMÉ (critique)** | `log_prob` de forme **(60,60)** au lieu de (60,) ; `fit` aboutit **sans avertissement** ; `sd(mu)`=0,130 vs 1,285 correct → **fausse précision ×10** par sur-comptage n² des termes |
| T17 | API documentée `scale()`/`scale_var()`/`to_int()` | **CONFIRMÉ (majeure)** | les trois appels nominaux cassent : `scale()` et `scale_var('y')` → `TypeError` (pd_to_jax sur `manip.scale_var`), `to_int(['g'])` → `ValueError` |
| T18 | save/load round-trip | **SAIN** | fichier écrit, rechargé, `posteriors` identiques bit à bit |
| T19 | Axes après fit 2 chaînes | confirmé (design) | `posteriors['mu']` shape **(200,)** — aplati ; `group_by_chain` n'est pas le défaut, à documenter à l'endroit exact |
| R1 | Pont R — sourcer `R warpper.R` | **CONFIRMÉ (nouveau)** | la ligne 5 `py_install('BayesForge')` **s'exécute et installe BayesForge-0.0.59 depuis PyPI** dans l'environnement R ; puis mort ligne 11 : `fromJSON("Doc_R.json")` → « lexical error: invalid char in json text » (fichier inexistant, traité comme texte JSON littéral). Le fossile a un **effet de bord d'installation avant de mourir** |

## Décompte

| Statut | Nombre |
|---|---|
| Confirmés à l'identique (dont 6 critiques : T01b, T02, T03, T06, T07, T16b) | 15 |
| Confirmés-nuancés / informatifs (T04, T09, T15, T19) | 4 |
| Infirmés (T14b) | 1 |
| Sains (T09b, T13, T18) | 3 |
| Reclassement (T01a : X1 chemin principal déjà corrigé) + découverte (R1) | 2 |
| Artefact de sonde (T01c) / partiel (T08) | 2 |

**Lecture pour les correctifs** : les cinq cibles prioritaires de la roadmap (X1 résiduel, W5, W7, V1-V3, pont R) sont **toutes confirmées par reproduction** sur le cluster ; la sonde T16b (broadcasting) et T17 (API documentées cassées) rejoignent les P0 ; T14b est rayé de la liste ; le garde X1 existant (`rng.py:78`) n'attend qu'une extension au chemin `seed=`.

---

