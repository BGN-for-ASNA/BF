# Index des tests BF (`work/BF/Test`)

Périmètre : dossiers numérotés uniquement (`N.Nom`), sans `4. Rethinking` ni dossiers non numérotés (WIP, Imputation, scripts racine). Descriptions de `0.Core` et `1.Distributions` générées par Strata (Qwen3.8-Flash-Next) à partir des docstrings et débuts de corps ; liste des autres dossiers générée par script (fichiers de code + noms de fonctions `test_*`).

# Partie 1 — 0.Core (description détaillée)

### `0.Core/conftest.py`
Ce fichier configure les fixtures et les hooks pytest pour générer un rapport de vérification statistique des tests.
- `_Verify` — Classe utilitaire qui enregistre les résultats de chaque vérification statistique dans une liste globale.
- `verify` — Fixture pytest qui retourne une instance de `_Verify` liée au nœud de test courant.
- `_fmt` — Formate une valeur numérique pour l'affichage dans le rapport (notation scientifique ou décimale).
- `_table_md` — Génère une table Markdown récapitulative des vérifications effectuées.
- `_plots` — Génère des graphiques de densité comparant les échantillons de référence et testés si disponibles.
- `pytest_terminal_summary` — Affiche un résumé des vérifications dans la sortie du terminal à la fin de la session.
- `pytest_sessionfinish` — Écrit le rapport complet (table Markdown et graphiques) dans un fichier à la fin de la session.
- `__init__` — Initialise l'objet `_Verify` avec le nom du test.
- `record` — Enregistre une vérification unique avec ses métadonnées (quantité, critère, valeur, résultat).

### `0.Core/test_audit_build_critical.py`
Ce fichier teste les régressions critiques (T-1..T-3, N-1..N-4) du module Build liées à l'audit de version.
- `_import` — Importe un module en le retirant d'abord de `sys.modules` pour forcer un rechargement propre.
- `test_t1_import_has_no_side_effect` — Vérifie que l'import du générateur TFP ne crée aucun fichier dans le répertoire courant.
- `test_t1_refuses_production_name` — Vérifie que le générateur TFP refuse d'écraser un fichier de production existant.
- `test_t2_t3_generated_tfp_file_compiles_and_conditions_obs` — Vérifie que le fichier TFP généré compile correctement et conditionne les observations.
- `_sig` — Extrait les paramètres de signature d'une fonction pour les tests de génération de code.
- `test_t2_n1_signature_rules` — Vérifie que la construction de signature respecte les règles pour les arguments par défaut et les kwargs.
- `test_n1_n4_np_import_has_no_side_effect_and_no_ollama` — Vérifie que l'import du générateur NumPyro n'a pas d'effets de bord et ne dépend pas d'Ollama au niveau module.
- `test_n1_n2_generated_np_file_compiles` — Vérifie que le fichier NumPyro généré compile correctement avec des docstrings nettoyées.
- `test_n2_clean_llm_docstring` — Vérifie que le nettoyage des docstrings LLM supprime les balises Markdown et produit du code valide.
- `test_n3_half_normal_doc_formula` — Vérifie que la formule de la distribution HalfNormal est correctement formatée dans la documentation.
- `f` — Fonction factice utilisée pour tester la génération de signature.
- `h` — Fonction factice utilisée pour tester la génération de signature.

### `0.Core/test_audit_build_major.py`
Ce fichier teste les régressions majeures (statiques) du module Build sans exécuter les générateurs lourds.
- `_source` — Récupère le code source d'un fichier générateur donné.
- `_tree` — Analyse l'arbre syntaxique (AST) d'un fichier générateur.
- `_functions` — Extrait les définitions de fonctions d'un arbre AST.
- `_body` — Récupère le corps d'une fonction spécifique sous forme de chaîne de caractères.
- `_module_level_imports` — Identifie les modules importés au niveau du module (top-level).
- `_string_constants` — Extrait toutes les constantes de chaîne de caractères d'un nœud AST.
- `_written_paths` — Identifie les chemins de fichiers passés aux appels `open(..., "w")` dans le code source.
- `_raises_value_error` — Vérifie si une fonction lève une `ValueError` avec un message spécifique.
- `_run_isolated` — Exécute des définitions isolées d'un générateur sans importer le module complet.
- `test_tfp_light_collects_classes_not_dir_entries` — Vérifie que le collecteur TFP ne garde que les classes et non les entrées de `dir()`.
- `test_tfp_light_collector_skips_private_and_base_class` — Vérifie que les alias privés et la classe de base TFP sont exclus.
- `test_tfp_light_detects_duplicate_method_names` — Vérifie la détection des noms de méthodes dupliqués.
- `test_tfp_light_refuses_to_write_a_duplicated_name` — Vérifie qu'une erreur est levée avant écriture si un nom est dupliqué.
- `test_tfp_light_checks_duplicates_before_opening_the_output` — Vérifie que la garde contre les doublons s'exécute avant l'ouverture du fichier de sortie.
- `test_wrapper_seed_defaults_to_none` — Vérifie que l'argument `seed` des wrappers par défaut est `None`.
- `test_no_fixed_prng_key_in_generated_source` — Vérifie qu'aucune clé PRNG fixe n'est codée en dur dans le code généré.
- `test_np_light_has_no_top_level_ollama_import` — Vérifie qu'Ollama n'est pas importé au niveau du module NumPyro.
- `test_np_light_imports_ollama_lazily_with_an_actionable_message` — Vérifie qu'Ollama est importé paresseusement avec un message d'erreur clair.
- `test_np_light_reads_the_committed_doc_cache` — Vérifie que le cache de documentation est utilisé par défaut (mode hors ligne).
- `test_np_light_doc_cache_path_is_absolute` — Vérifie que le chemin du cache de documentation est résolu de manière absolue.
- `test_np_light_reports_skipped_classes` — Vérifie que les classes ignorées sont signalées par un print.
- `test_generated_source_is_compile_checked` — Vérifie que le code généré est compilé avant d'être écrit.
- `test_np_light_required_set_comes_from_production` — Vérifie que l'ensemble des noms requis provient de la façade de production.
- `test_np_light_reads_production_without_importing_it` — Vérifie que la façade de production est analysée via AST sans être importée.
- `test_production_method_names_finds_the_facade` — Vérifie que l'helper extrait correctement les méthodes de la façade de production.
- `test_required_names_is_the_intersection_with_production` — Vérifie que les noms requis sont l'intersection avec la production.
- `test_np_light_missing_required_name_raises_value_error` — Vérifie qu'une erreur est levée si un nom de production est manquant.
- `test_np_light_coverage_runs_before_the_output_is_written` — Vérifie que la vérification de couverture s'exécute avant l'écriture du fichier.
- `test_generated_files_name_their_generator` — Vérifie que les fichiers générés mentionnent leur générateur et l'avertissement "do not edit".
- `test_generators_read_installed_versions_from_the_metadata` — Vérifie que les versions des dépendances sont lues via `importlib.metadata`.
- `test_generated_header_records_the_backend_version` — Vérifie que l'en-tête généré enregistre la version du backend.
- `test_np_light_generated_header_names_its_inputs` — Vérifie que l'en-tête NumPyro mentionne le générateur et la date.
- `_header_texts` — Extrait toutes les chaînes de caractères émises par le module, y compris les f-strings.
- `_provenance_line` — Extrait la ligne de provenance (f-string) écrite en premier par le générateur.
- `test_tfp_generated_header_names_its_inputs` — Vérifie que l'en-tête TFP mentionne le générateur et la version de TFP.
- `test_test_file_targets_the_test_tree` — Vérifie que les tests générés sont placés dans l'arborescence de test correcte.
- `test_test_file_sys_path_points_at_the_repo_root` — Vérifie que le `sys.path` généré pointe vers la racine du dépôt.
- `test_test_file_creates_its_output_directory` — Vérifie que le répertoire de sortie est créé avant l'écriture.
- `test_test_file_compiles_before_writing` — Vérifie que le fichier de test généré est compilé avant écriture.
- `test_test_file_declares_what_it_does_not_prove` — Vérifie que le fichier généré indique qu'il s'agit de tests de fumée et non de comportement.
- `test_generated_test_file_asserts_on_every_example` — Vérifie qu'un test est généré pour chaque exemple et qu'il vérifie l'exécution.
- `test_generated_test_file_drops_markdown_fences` — Vérifie que les balises Markdown sont supprimées des exemples générés.
- `test_generated_example_runner_checks_what_the_example_produced` — Vérifie que le runner d'exemple vérifie que l'appel a retourné une valeur.
- `test_generated_example_runner_rejects_a_none_binding` — Vérifie que le runner rejette les liaisons `None`.
- `test_generator_is_valid_python` — Vérifie que chaque générateur est un code Python valide.
- `test_generator_has_a_main_guard` — Vérifie que chaque générateur possède un garde `if __name__ == "__main__":`.
- `test_np_light_cli_options_document_the_offline_path` — Vérifie la présence des options CLI pour le mode hors ligne.
- `test_np_light_generate_accepts_injected_docs` — Vérifie que la fonction `generate` accepte des docstrings injectées.

### `0.Core/test_audit_build_minor.py`
Ce fichier teste les régressions mineures (T-7, N-9, N-10, B-5, B-6) du module Build.
- `_source` — Récupère le code source d'un fichier générateur.
- `_tree` — Analyse l'arbre syntaxique d'un fichier générateur.
- `_functions` — Extrait les définitions de fonctions d'un arbre AST.
- `_body` — Récupère le corps d'une fonction spécifique.
- `_module_level_imports` — Identifie les imports au niveau du module.
- `_exec_defs` — Exécute des définitions isolées d'un générateur sans importer le module.
- `test_tfp_docstring_uses_the_default_formatter` — Vérifie que les lignes "Default =" utilisent `format_default`.
- `test_default_formatter_renders_classes_strings_and_numbers` — Test unitaire de `format_default` sur divers types de valeurs.
- `test_no_open_without_encoding` — Vérifie que tous les appels `open` spécifient l'encodage UTF-8.
- `test_no_notebook_cell_markers` — Vérifie l'absence de marqueurs de cellules de notebook (`# %%`).
- `_stray_space_offenses` — Détecte les espaces superflus dans le code (hors chaînes de caractères).
- `test_no_stray_spaces_in_code` — Vérifie qu'il n'y a pas d'espaces superflus dans le code source.
- `test_negative_binomial_private_submodule_is_documented` — Vérifie que l'import du sous-module privé est documenté.
- `test_test_file_paths_resolve_from_dunder_file` — Vérifie que les chemins sont résolus depuis `__file__`.
- `test_test_file_repo_root_computes_the_repository_root` — Vérifie que la racine du dépôt est correctement calculée.
- `test_generated_template_resolves_root_from_its_own_file` — Vérifie que le modèle généré résout la racine depuis son propre fichier.
- `test_collect_dist_methods_keeps_only_public_callables` — Vérifie que seules les méthodes publiques sont collectées.
- `test_build_examples_skips_methods_without_a_docstring` — Vérifie que les méthodes sans docstring sont ignorées lors de la génération d'exemples.
- `test_bf_instance_is_created_in_main_only` — Vérifie que l'instance BayesForge n'est créée que dans le bloc main.
- `test_importing_build_test_file_does_not_import_jax` — Vérifie que l'import du script ne déclenche pas l'import de JAX.
- `Sample` — Classe factice pour les tests.
- `FakeDist` — Classe factice simulant une distribution pour les tests.
- `fake_llm` — Fonction factice simulant un appel LLM.
- `documented` — Fonction factice avec docstring.
- `undocumented` — Fonction factice sans docstring.
- `normal` — Méthode factice.
- `mask` — Méthode factice.
- `plate` — Méthode factice.
- `normalize_seed` — Méthode factice.
- `_helper` — Fonction factice.
- `Distribution` — Classe factice.

### `0.Core/test_audit_data_critical.py`
Ce fichier teste les régressions critiques (C1-C8) du module Data (manipulation).
- `_m` — Crée une instance de `manip` avec un DataFrame de test standard.
- `test_c1_scale_all_default` — Vérifie que la mise à l'échelle par défaut normalise les colonnes numériques et ignore les catégorielles.
- `test_c1_scale_single_name` — Vérifie que la mise à l'échelle d'une seule colonne ne modifie pas les autres.
- `test_c3_ohe_does_not_duplicate_columns` — Vérifie que l'encodage One-Hot n'ajoute pas de colonnes en double.
- `test_c4_to_int` — Vérifie que la conversion en entier fonctionne correctement.
- `test_c5_to_float_really_casts` — Vérifie que la conversion en flottant change réellement le dtype.
- `test_c6_missing_column_raises` — Vérifie qu'une erreur est levée si une colonne du modèle est manquante dans les données.
- `test_c7_array_and_none_defaults` — Vérifie que les valeurs par défaut (tableaux, None, chaînes) sont correctement converties pour JAX.
- `test_c8_index_rejects_missing_values` — Vérifie que l'indexation rejette les colonnes contenant des valeurs manquantes.
- `test_c8_index_without_missing_unchanged` — Vérifie que l'indexation fonctionne correctement sans valeurs manquantes.
- `model` — Fonction modèle factice pour les tests de conversion.
- `model` — Fonction modèle factice pour les tests de conversion.

### `0.Core/test_audit_data_major.py`
Ce fichier teste les régressions majeures (M1-M6) du module Data.
- `_m` — Crée une instance de `manip` avec un DataFrame de test.
- `test_m1_constant_column_raises` — Vérifie qu'une erreur est levée pour une colonne constante lors de la mise à l'échelle.
- `test_m1_single_row_raises` — Vérifie qu'une erreur est levée pour un DataFrame d'une seule ligne lors de la mise à l'échelle.
- `test_m1_variable_column_still_scales` — Vérifie que les colonnes variables sont correctement mises à l'échelle.
- `test_m2_pandas_and_jax_std_equal` — Vérifie que l'écart-type calculé par Pandas et JAX est identique.
- `test_m3_pd_to_jax_x64_off_no_truncation_warning` — Vérifie qu'aucun avertissement de troncature n'est émis lorsque x64 est désactivé.
- `test_m4_ohe_encodes_category_dtype` — Vérifie que l'encodage One-Hot fonctionne sur les dtypes catégoriels.
- `test_m5_names_are_normalized_once_at_load_and_original_names_still_work` — Vérifie que les noms de colonnes sont normalisés une fois et que les originaux restent utilisables.
- `test_m5_names_colliding_after_normalization_raise` — Vérifie qu'une erreur est levée si les noms normalisés entrent en collision.
- `test_m5_index_sanitizes_only_added_column` — Vérifie que seule la colonne ajoutée par l'indexation est sanitizée.
- `test_m6_datetime_dtype_raises` — Vérifie qu'une erreur est levée pour les dtypes datetime.
- `test_m6_complex_dtype_raises` — Vérifie qu'une erreur est levée pour les dtypes complexes.
- `test_m6_mapped_dtypes_still_convert` — Vérifie que les dtypes mappés sont correctement convertis.
- `test_m6_bool_and_small_int_columns_convert` — Vérifie que les colonnes booléennes et petits entiers sont converties.
- `model` — Fonction modèle factice.
- `model` — Fonction modèle factice.
- `model` — Fonction modèle factice.
- `model` — Fonction modèle factice.
- `model` — Fonction modèle factice.

### `0.Core/test_audit_data_minor.py`
Ce fichier teste les régressions mineures (m1-m12) du module Data.
- `_m` — Crée une instance de `manip` avec un DataFrame de test.
- `_model` — Fonction modèle factice retournant y.
- `test_m1_index_map_survives_later_calls` — Vérifie que la carte d'indexation n'est pas réinitialisée lors des appels suivants.
- `test_m2_data_modification_stores_resolved_columns` — Vérifie que les colonnes résolues sont stockées dans le journal des modifications.
- `test_m3_to_int_returns_self_df_like_to_float` — Vérifie que `to_int` retourne le DataFrame pour permettre le chaînage.
- `test_m4_str_replace_uses_regex_false` — Vérifie que les appels à `.str.replace` utilisent `regex=False`.
- `test_m5_bit_param_docstring_and_robust_values` — Vérifie la documentation et la robustesse des valeurs pour le paramètre de bits.
- `test_m6_data_to_model_duplicate_and_int64` — Vérifie la gestion des noms de colonnes dupliqués et des int64.
- `test_m7_nan_warns_and_scales_non_nan_rows` — Vérifie que les NaN déclenchent un avertissement et sont ignorés lors de la mise à l'échelle.
- `test_m8_ohe_nan_documented_all_zero_dummies` — Vérifie que le comportement des NaN dans l'encodage One-Hot est documenté.
- `test_m9_type_kwarg_public_and_commented` — Vérifie que le mot-clé `type` est public et commenté.
- `test_m10_data_package_has_docstring_only_init` — Vérifie que le package Data a un `__init__.py` avec seulement une docstring.
- `test_m11_no_jax_jit_on_preprocessing` — Vérifie que le prétraitement n'est pas compilé avec `jax.jit`.
- `test_m12_nonstring_column_labels` — Vérifie que les étiquettes de colonnes non-string sont gérées correctement.

### `0.Core/test_audit_effects_vis_cholesky.py`
Ce fichier teste la gestion des facteurs de Cholesky et des matrices de corrélation dans les effets variés.
- `_run` — Exécute `varying_intercept_slope` avec des paramètres donnés.
- `test_mvn_receives_the_scaled_cholesky_factor` — Vérifie que la distribution normale multivariée reçoit le facteur de Cholesky mis à l'échelle.
- `test_correlation_matrix_instead_of_cholesky_factor_raises` — Vérifie qu'une erreur est levée si une matrice de corrélation est passée à la place d'un facteur de Cholesky.
- `test_valid_cholesky_factor_and_traced_factor_are_accepted` — Vérifie que les facteurs de Cholesky valides et tracés sont acceptés.
- `test_message_tells_covariance_from_correlation` — Vérifie que le message d'erreur distingue covariance et corrélation.
- `test_default_lkj_cholesky_factor_is_not_checked` — Vérifie que le facteur de Cholesky par défaut LKJ n'est pas vérifié.
- `test_group_std_must_have_two_entries` — Vérifie que `group_std` doit avoir deux entrées.
- `test_diag_pre_multiply_rejects_wrong_shapes_even_under_jit` — Vérifie que `diag_pre_multiply` rejette les formes incorrectes même sous JIT.
- `test_nested_mu_shape_warning_also_for_traced_mu` — Vérifie que l'avertissement de forme pour `mu` s'applique aussi aux valeurs tracées.
- `spy` — Fonction espion pour capturer les arguments.
- `boom` — Fonction factice levant une erreur.
- `run` — Fonction d'exécution pour les effets variés imbriqués.

### `0.Core/test_audit_integration_v.py`
Ce fichier teste les régressions d'intégration (V5-V11) liées aux traces empiriques.
- `_m` — Crée une instance BayesForge sur CPU avec une graine aléatoire.
- `_two_param_model` — Définit un modèle à deux paramètres latents pour les tests.
- `_fit` — Ajuste le modèle à deux paramètres avec des paramètres de test.
- `test_v5_posteriors_is_a_copy_of_the_sampler_cache` — Vérifie que les postérieurs sont une copie des échantillons du sampler.
- `test_v6_sampler_stats_per_chain_without_num_steps` — Vérifie que les statistiques du sampler fonctionnent par chaîne sans `num_steps`.
- `test_v7_ohe_does_not_duplicate_columns` — Vérifie que l'encodage One-Hot n'ajoute pas de colonnes en double dans l'intégration.
- `test_v8_log_prob_conditions_on_the_data` — Vérifie que la log-probabilité est conditionnée sur les données.
- `test_v9_tfp_posteriors_honour_the_chain_contract` — Vérifie que les postérieurs TFP respectent le contrat de chaîne.
- `test_v10_to_az_tfp_rejects_misaligned_var_names` — Vérifie que la conversion vers ArviZ rejette les noms de variables mal alignés pour TFP.
- `test_v11_summary_include_does_not_filter_persistently` — Vérifie que le filtre `include` dans le résumé n'est pas persistant.
- `model` — Modèle à deux paramètres pour les tests.
- `model` — Modèle TFP pour les tests.
- `_FakeTFPSampler` — Classe factice simulant un sampler TFP.
- `__init__` — Initialise le sampler factice.
- `get_samples` — Retourne des échantillons factices.

### `0.Core/test_audit_integration_v_minor.py`
Ce fichier teste les régressions d'intégration mineures (V16-V18, X10-X11, W16).
- `_m` — Crée une instance BayesForge sur CPU.
- `_model` — Définit un modèle simple à un paramètre.
- `_fit` — Ajuste le modèle simple.
- `fitted` — Fixture retournant une instance ajustée.
- `test_v16_diagnose_follows_current_fit` — Vérifie que le diagnostic suit l'ajustement courant et non un ancien.
- `test_v17_to_az_failure_warns_and_stashes_cause` — Vérifie qu'une erreur de conversion ArviZ émet un avertissement et stocke la cause.
- `test_v18_sampler_stats_names_composite_sampler` — Vérifie que les statistiques du sampler nomment le sampler composite en cas d'erreur.
- `test_x11_summary_docstring_documents_different_sources` — Vérifie que la docstring du résumé documente les sources différentes.
- `test_w16_fit_docstring_documents_model_identification` — Vérifie que la docstring de `fit` documente l'identification du modèle.
- `test_w16_lambda_model_warns` — Vérifie qu'un modèle lambda émet un avertissement.
- `test_x16_setstate_warns_on_device_count_mismatch` — Vérifie qu'une incohérence de nombre de dispositifs lors du `setstate` émet un avertissement.
- `model` — Modèle simple pour les tests.
- `boom` — Fonction factice levant une erreur.

### `0.Core/test_audit_integration_w_a.py`
Ce fichier teste les régressions d'intégration W2, W4-W8.
- `test_w2_bind_library_rng_covers_model_effects2_pca_surv` — Vérifie que le RNG de la bibliothèque est lié aux modules de modèles.
- `test_w2_pca_instances_share_bound_singleton` — Vérifie que les instances PCA partagent le singleton lié.
- `test_w4_bsrm_rejects_upper_triangle_edgelist` — Vérifie que le modèle SRM rejette les listes d'arêtes en triangle supérieur.
- `test_w5_nbda_priors_use_instance_dist` — Vérifie que les priors NBDA utilisent la distribution de l'instance.
- `model_w6` — Modèle pour tester les pentes variées non centrées.
- `test_w6_varying_slope_non_centered_traces_without_name_error` — Vérifie que les pentes variées non centrées se tracent sans erreur de nom.
- `test_w7_nodal_random_effects_rejects_zero_based_codes` — Vérifie que les effets aléatoires nodaux rejettent les codes basés sur zéro.
- `test_w8_inv_logit_scale_documented_on_minus_one_one` — Vérifie que l'échelle inverse logit est documentée sur [-1, 1].

### `0.Core/test_audit_integration_w_b.py`
Ce fichier teste les régressions d'intégration W9-W12.
- `_bf` — Crée une nouvelle instance BayesForge.
- `test_w9_pd_to_jax_keeps_none_and_str_defaults` — Vérifie que `pd_to_jax` conserve les valeurs par défaut None et str.
- `test_w10_pca_class_rejected_with_instance_hint` — Vérifie qu'une classe PCA est rejetée avec un conseil d'instance.
- `test_w10_nbda_class_rejected_with_nbda_model_hint` — Vérifie qu'une classe NBDA est rejetée avec un conseil de modèle.
- `test_w10_survival_instance_rejected` — Vérifie qu'une instance de survie est rejetée si ce n'est pas une fonction modèle.
- `test_w10_none_rejected` — Vérifie que `None` est rejeté comme modèle.
- `_xy` — Génère des données aléatoires 2D.
- `test_w11_gmm_without_K_raises` — Vérifie qu'une erreur est levée si K est manquant pour GMM.
- `test_w11_gmm_with_int_K_runs` — Vérifie que GMM fonctionne avec un K entier.
- `_net` — Génère une matrice d'adjacence aléatoire.
- `_status` — Génère une matrice de statut aléatoire.
- `test_w12_nbda_bound_model_records_nbda_name` — Vérifie que le modèle NBDA lié enregistre le nom "nbda".
- `f` — Fonction factice.

### `0.Core/test_audit_integration_w_x_minor.py`
Ce fichier teste les régressions d'intégration mineures W13, W14, X7, X15.
- `test_w13_tfp_branch_binds_bnn_to_bf_seed` — Vérifie que la branche TFP lie le BNN à la graine BF.
- `test_w13_bnn_docstring_documents_layer_state` — Vérifie que la docstring BNN documente l'état de la couche.
- `test_w14_ppc_warns_when_event_dims_are_flattened` — Vérifie qu'un avertissement est émis si les dimensions d'événement sont aplatis dans PPC.
- `test_x7_worker_env_carries_parent_x64` — Vérifie que l'environnement du worker porte la configuration x64 du parent.
- `test_x15_graph_does_not_advance_the_rng_stream` — Vérifie que la construction du graphe n'avance pas le flux RNG.
- `model` — Modèle multivarié pour les tests.
- `model` — Modèle avec bruit pour les tests.

### `0.Core/test_audit_integration_x13_bunch.py`
Ce fichier teste l'acceptation des objets Bunch par `m.data()`.
- `test_data_accepts_a_frame_false_bunch` — Vérifie que `data` accepte un Bunch avec `frame=False`.
- `test_bunch_without_frame_gives_a_clear_error` — Vérifie qu'un Bunch sans frame lève une erreur claire.

### `0.Core/test_audit_integration_x9_import_order.py`
Ce fichier teste l'ordre d'importation de BayesForge par rapport aux autres bibliothèques.
- `test_import_after_other_libraries` — Vérifie que BayesForge s'importe correctement après d'autres bibliothèques.
- `test_arviz_pin_has_an_upper_bound` — Vérifie que la dépendance ArviZ a une borne supérieure dans `pyproject.toml`.

### `0.Core/test_audit_interface_critical.py`
Ce fichier teste les régressions critiques d'interface (V1-V3, W1).
- `_m` — Crée une instance BayesForge sur CPU.
- `_data` — Génère des données de régression linéaire simple.
- `test_v1_v2_svi_drops_previous_mcmc_state` — Vérifie que SVI efface l'état MCMC précédent.
- `test_v3_data_from_df_is_rebuilt_for_a_new_model` — Vérifie que les données sont reconstruites pour un nouveau modèle.
- `test_v3_stale_user_data_raises_clearly` — Vérifie qu'une erreur claire est levée pour des données utilisateur obsolètes.
- `test_w1_singletons_follow_the_instance_that_reseeds` — Vérifie que les singletons suivent l'instance qui reseed.
- `model_a` — Modèle à deux paramètres.
- `model_xy` — Modèle de régression.
- `model_y` — Modèle simple.
- `model_xy` — Modèle de régression.
- `model_z` — Modèle avec z.

### `0.Core/test_audit_interface_x1.py`
Ce fichier teste la régression critique d'interface X1.
- `test_x1_sample_true_inside_traced_model_warns` — Vérifie qu'un avertissement est émis si `sample=True` est utilisé dans un modèle tracé.
- `test_x1_plain_dgp_draw_does_not_warn` — Vérifie qu'un tirage DGP simple n'émet pas d'avertissement.
- `model` — Modèle avec tirage latent.

### `0.Core/test_audit_main_critical.py`
Ce fichier teste les régressions critiques du module Main (M-01..M-03).
- `test_m01_log_prob_conditions_on_the_data` — Vérifie que la log-probabilité est conditionnée sur les données.
- `test_m02_fit_does_not_alias_the_users_data_dict` — Vérifie que `fit` ne crée pas d'alias sur le dictionnaire de données de l'utilisateur.
- `test_m03_unknown_backend_is_rejected` — Vérifie qu'un backend inconnu est rejeté.
- `test_m03_svi_on_tfp_backend_raises` — Vérifie que SVI sur le backend TFP lève une erreur.
- `model` — Modèle simple.
- `model` — Modèle simple.
- `model` — Modèle TFP.

### `0.Core/test_audit_main_m06_x4.py`
Ce fichier teste les régressions Main M-06 et Intégration X4.
- `_m` — Crée une instance BayesForge sur CPU.
- `_normal_model` — Définit un modèle normal simple.
- `_mean_stat` — Fonction de statistique résumée.
- `_assert_bound_diag_surface` — Vérifie que la surface de diagnostic est liée.
- `_svi_normal` — Effectue une inférence SVI sur le modèle normal.
- `test_m06_svi_sampler_declares_one_chain` — Vérifie que le sampler SVI déclare une chaîne.
- `test_x4_svi_binds_diag_surface` — Vérifie que SVI lie la surface de diagnostic.
- `test_x4_sbi_binds_diag_surface` — Vérifie que SBI lie la surface de diagnostic.
- `model` — Modèle normal.

### `0.Core/test_audit_main_major.py`
Ce fichier teste les régressions majeures du module Main (M-07, M-11, M-13, M-14).
- `_tiny_model` — Définit le plus petit modèle conditionnel.
- `test_m14_prngkey_seed_accepted` — Vérifie qu'une clé PRNG JAX est acceptée comme graine.
- `test_m14_bad_seeds_raise` — Vérifie que les graines booléennes ou flottantes fractionnaires lèvent une erreur.
- `test_m11_user_function_named_pca` — Vérifie qu'une fonction utilisateur nommée 'pca' n'utilise pas la branche PCA intégrée.
- `test_m11_user_function_named_gmm` — Vérifie qu'une fonction utilisateur nommée 'gmm' n'utilise pas le warm start KMEANS.
- `test_m13_data_key_collision_raises` — Vérifie qu'une collision de clé de données lève une erreur claire.
- `test_m07_tfp_float64_kept` — Vérifie que la précision float64 est conservée avec le backend TFP.
- `model` — Modèle simple.
- `pca` — Fonction PCA utilisateur.
- `gmm` — Fonction GMM utilisateur.
- `model` — Modèle avec collision.

### `0.Core/test_audit_main_major_b.py`
Ce fichier teste les régressions majeures du module Main (M-08 et M-16).
- `_fit_tiny` — Ajuste un modèle à deux paramètres sur trois points.
- `test_m08_second_instance_does_not_steal_rng` — Vérifie qu'une deuxième instance ne vole pas le RNG.
- `test_m08_tfp_instances_have_own_rng` — Vérifie que les instances TFP ont leur propre RNG.
- `test_m16_summary_does_not_mutate_posteriors` — Vérifie que le résumé ne modifie pas les postérieurs.
- `test_m16_filter_regex_arviz` — Vérifie que le filtre regex fonctionne avec ArviZ.
- `test_m16_group_by_chain_arviz_warns` — Vérifie qu'un avertissement est émis pour `group_by_chain` avec ArviZ.
- `model` — Modèle à deux paramètres.
- `_tfp_model` — Modèle TFP à deux paramètres.
- `model` — Modèle TFP à deux paramètres.

### `0.Core/test_audit_main_minor_a.py`
Ce fichier teste les régressions mineures du module Main (M-17 à M-31).
- `_new` — Crée une nouvelle instance BayesForge.
- `_tiny_model` — Définit le plus petit modèle conditionnel.
- `_fit_tiny` — Ajuste le petit modèle.
- `test_m17_save_default_name_follows_run_model_name` — Vérifie que le nom de sauvegarde par défaut suit le nom du modèle.
- `test_m18_sampler_stats_per_chain_without_num_steps` — Vérifie que les statistiques du sampler fonctionnent sans `num_steps`.
- `test_m19_extra_fields_string_rejected` — Vérifie qu'une chaîne pour `extra_fields` est rejetée.
- `test_m20_count_and_seed_validation` — Vérifie la validation des compteurs et des graines.
- `test_m21_npe_helpers_before_fit` — Vérifie que les helpers NPE lèvent une erreur avant l'ajustement.
- `test_m22_sample_without_model_or_data` — Vérifie que `sample` lève une erreur claire sans modèle ou données.
- `test_m23_pca_missing_X_raises` — Vérifie qu'une erreur est levée si X manque pour PCA.
- `test_m24_history_snapshots_dicts` — Vérifie que l'historique contient des copies des dictionnaires.
- `test_m25_shard_report_honest_after_fallback` — Vérifie que le rapport de sharding est honnête après un fallback.
- `test_m26_shard_check_fail_closed` — Vérifie que la vérification de sharding échoue en mode fermé.
- `test_m27_plot_before_fit` — Vérifie que `plot` lève une erreur claire avant l'ajustement.
- `test_m28_failed_save_keeps_previous_file` — Vérifie qu'une sauvegarde échouée conserve le fichier précédent.
- `test_m29_get_samples_error_not_masked` — Vérifie que les erreurs de `get_samples` ne sont pas masquées.
- `test_m30_observed_site_matched_by_identity` — Vérifie que le site observé est identifié par identité.
- `test_m31_untraceable_model_raises` — Vérifie qu'un modèle non traçable lève une erreur.
- `tiny` — Modèle simple.
- `fake_auto` — Fonction factice pour le sharding.
- `boom` — Fonction factice levant une erreur.
- `boom` — Fonction factice levant une erreur.
- `model` — Modèle simple.
- `model` — Modèle avec échelle.

### `0.Core/test_audit_main_minor_b.py`
Ce fichier teste les régressions mineures du module Main (M-32 à M-45).
- `_new` — Crée une nouvelle instance BayesForge.
- `_tiny_model` — Définit le plus petit modèle conditionnel.
- `_fit_tiny` — Ajuste le petit modèle.
- `test_m32_sample_count_validated` — Vérifie la validation du nombre d'échantillons.
- `test_m33_setstate_resyncs_n_devices_with_mesh` — Vérifie que `setstate` resynchronise le nombre de dispositifs avec le mesh.
- `test_m33_setstate_single_device_or_failure_clears_state` — Vérifie que `setstate` nettoie l'état en cas d'échec ou de dispositif unique.
- `test_m34_shard_limits_warning_is_the_experimental_one_once` — Vérifie que l'avertissement de limites de sharding est émis une seule fois.
- `test_m35_constructor_leaves_global_rngs_untouched` — Vérifie que le constructeur ne touche pas aux RNG globaux.
- `test_m35_seed_global_rngs_is_opt_in` — Vérifie que le seed des RNG globaux est optionnel.
- `test_m36_stale_fit_data_mismatch_raises_clear_error` — Vérifie qu'une erreur claire est levée pour une incohérence de données d'ajustement obsolètes.
- `test_m37_summary_before_fit_raises_clear_error` — Vérifie que le résumé lève une erreur claire avant l'ajustement.
- `test_m38_empty_posterior_dict_raises_value_error` — Vérifie qu'un dictionnaire de postérieurs vide lève une erreur.
- `test_m39_expand_observed_rank_mismatch_raises` — Vérifie qu'une erreur est levée pour une incohérence de rang dans l'expansion.
- `test_m39_expand_failure_warns_instead_of_passing_silently` — Vérifie qu'un échec d'expansion émet un avertissement.
- `test_m40_load_dual_semantics_documented_and_working` — Vérifie que la sémantique double de `load` est documentée et fonctionne.
- `test_m41_load_docstring_warns_about_code_execution` — Vérifie que la docstring de `load` avertit de l'exécution de code.
- `test_m42_pca_latent_dim_default_warns` — Vérifie qu'un avertissement est émis pour la dimension latente par défaut de PCA.
- `test_m43_nested_partial_model_fits` — Vérifie que les modèles partiels imbriqués s'ajustent correctement.
- `test_m44_pca_posterior_is_pooled` — Vérifie que le postérieur PCA est poolé.
- `test_m45_pooled_ebfmi_is_the_worst_chain` — Vérifie que l'E-BFMI poolé est celui de la pire chaîne.
- `tiny` — Modèle simple.
- `boom` — Fonction factice levant une erreur.
- `other` — Modèle alternatif.
- `_NoExpand` — Classe factice sans expansion.
- `model` — Modèle avec échelle.
- `ModelObj` — Objet modèle factice.
- `expand` — Méthode factice.
- `__call__` — Méthode factice.

### `0.Core/test_audit_mcmc_tfp.py`
Ce fichier teste les régressions critiques du module MCMC TFP (C1-C3).
- `_model_factory` — Fabrique un modèle pour les tests TFP.
- `_bf_tfp` — Crée une instance BayesForge avec le backend TFP.
- `test_c1_dual_averaging_adapts_over_num_burnin_steps` — Vérifie que le dual averaging s'adapte sur les étapes de burn-in.
- `test_c2_custom_target_log_prob_fn_is_used` — Vérifie que la fonction de log-probabilité cible personnalisée est utilisée.
- `test_c3_chain_starts_from_the_constrained_init_state` — Vérifie que la chaîne démarre de l'état initial contraint.
- `test_m1_obs_not_last_raises` — Vérifie qu'une erreur est levée si l'observation n'est pas le dernier yield.
- `test_m2_seed_changes_init` — Vérifie que la graine change l'initialisation.
- `test_m3_posteriors_flat_like_numpyro` — Vérifie que les postérieurs sont aplatis comme dans NumPyro.
- `test_m4_thinning_is_applied` — Vérifie que l'éclaircissement (thinning) est appliqué.
- `test_m5_beta_parameter_gets_sigmoid_bijector` — Vérifie que le paramètre beta reçoit un bijecteur sigmoid.
- `test_m6_param_named_obs_something_is_kept` — Vérifie qu'un paramètre nommé 'obs_something' est conservé.
- `model` — Modèle TFP.
- `spy` — Fonction espion.
- `custom` — Fonction de log-probabilité personnalisée.
- `bad` — Modèle invalide.
- `model` — Modèle avec beta.
- `model` — Modèle avec obs_rate.

### `0.Core/test_audit_ml_critical.py`
Ce fichier teste les régressions critiques du module ML (C1/C2).
- `_X` — Génère des données de test en blobs.
- `test_c1_random_state_changes_initialisation` — Vérifie que l'état aléatoire change l'initialisation.
- `test_c1_same_random_state_is_reproducible` — Vérifie que le même état aléatoire est reproductible.
- `test_c2_fluent_fit_without_arguments` — Vérifie que le fit fluent fonctionne sans arguments.
- `test_c2_kmeans_fit_uses_constructor_settings` — Vérifie que le fit KMeans utilise les paramètres du constructeur.

### `0.Core/test_audit_ml_major.py`
Ce fichier teste les régressions majeures du module ML (M1-M4).
- `_X` — Génère des données de test en blobs.
- `_X_sep` — Génère des données de test avec des centres bien séparés.
- `test_m1_kmeanspp_ninit_stable_across_seeds` — Vérifie la stabilité de KMeans++ à travers les graines.
- `test_m1_kmeanspp_matches_sklearn` — Vérifie que KMeans++ correspond à scikit-learn.
- `test_m1_init_random_legacy_option` — Vérifie l'option d'initialisation aléatoire héritée.
- `test_m2_sq_dist_matches_broadcast_and_nonnegative` — Vérifie que les distances carrées correspondent au broadcast et sont non négatives.
- `test_m3_rejects_1d_input` — Vérifie que l'entrée 1D est rejetée.
- `test_m3_rejects_k_zero_and_k_above_n` — Vérifie que K=0 ou K>N sont rejetés.
- `test_m3_rejects_non_finite_X` — Vérifie que les données non finies sont rejetées.
- `test_m4_fit_override_updates_instance_state` — Vérifie que le fit met à jour l'état de l'instance.
- `test_m4_constructor_still_fits_for_gmm_path` — Vérifie que le constructeur ajuste toujours pour le chemin GMM.
- `test_m4_unfitted_estimator_can_be_constructed` — Vérifie qu'un estimateur non ajusté peut être construit.
- `test_m4_successive_fits_do_not_reuse_the_same_key` — Vérifie que les fits successifs n'utilisent pas la même clé.
- `test_m4_handler_results_not_aliased` — Vérifie que les résultats du handler ne sont pas aliasés.

### `0.Core/test_audit_ml_minor.py`
Ce fichier teste les régressions mineures du module ML (m1–m8).
- `_ml_module` — Importe le module ML pour tester les collisions de noms.
- `_blobs` — Génère trois blobs gaussiens 2D.
- `_fit` — Ajuste un modèle KMeans sur les blobs.
- `_close_figures` — Ferme les figures matplotlib après les tests.
- `test_tol_zero_runs_the_full_iteration_budget` — Vérifie que tol=0 exécute le budget d'itération complet.
- `test_positive_tol_stops_early` — Vérifie qu'un tol positif s'arrête tôt.
- `test_default_tol_is_positive` — Vérifie que le tol par défaut est positif.
- `test_tol_does_not_change_the_result_on_converged_data` — Vérifie que le tol ne change pas le résultat sur des données convergées.
- `test_fit_tol_overrides_the_constructor` — Vérifie que le tol du fit écrase celui du constructeur.
- `test_invalid_tol_raises_value_error` — Vérifie qu'un tol invalide lève une erreur.
- `test_negative_tol_in_fit_raises_value_error` — Vérifie qu'un tol négatif dans le fit lève une erreur.
- `test_exhausted_budget_warns` — Vérifie qu'un budget épuisé émet un avertissement.
- `test_tol_zero_never_warns_about_the_budget` — Vérifie que tol=0 n'émet jamais d'avertissement de budget.
- `test_clustering_matches_sklearn` — Vérifie que le clustering correspond à scikit-learn.
- `test_predictions_is_none_before_predict` — Vérifie que les prédictions sont None avant predict.
- `test_predict_stores_an_array` — Vérifie que predict stocke un tableau.
- `test_fit_accepts_a_plain_list` — Vérifie que fit accepte une liste simple.
- `test_fit_rejects_non_2d_input` — Vérifie que fit rejette les entrées non 2D.
- `test_predict_rejects_non_2d_input` — Vérifie que predict rejette les entrées non 2D.
- `test_plot_grid_size_bounds_the_mesh` — Vérifie que la taille de la grille de tracé borne le maillage.
- `test_plot_rejects_invalid_grid_size` — Vérifie que le tracé rejette une taille de grille invalide.
- `test_plot_warns_when_data_is_not_the_fitted_one` — Vérifie que le tracé avertit si les données ne sont pas celles ajustées.
- `test_plot_style_is_validated_against_matplotlib` — Vérifie que le style de tracé est validé contre matplotlib.
- `test_contour_is_no_longer_labelled_decision_boundaries` — Vérifie que les contours ne sont plus étiquetés "Decision Boundaries".
- `test_contour_grid_uses_a_point_count_not_an_open_ended_step` — Vérifie que la grille de contour utilise un nombre de points.
- `test_downcast_warning_follows_the_x64_flag` — Vérifie que l'avertissement de downcast suit le drapeau x64.
- `test_no_downcast_warning_when_x64_is_enabled` — Vérifie qu'il n'y a pas d'avertissement de downcast quand x64 est activé.
- `test_predict_takes_only_newdata` — Vérifie que predict ne prend que les nouvelles données.
- `test_predict_docstring_no_longer_documents_parameters` — Vérifie que la docstring de predict ne documente plus les paramètres.
- `test_handler_exposes_centroids_and_labels` — Vérifie que le handler expose les centroïdes et les étiquettes.
- `test_handler_drops_the_unused_y_and_model_params` — Vérifie que le handler supprime y et les paramètres de modèle inutilisés.
- `test_handler_records_predictions` — Vérifie que le handler enregistre les prédictions.
- `test_kmeans_has_no_unused_imports` — Vérifie que KMeans n'a pas d'imports inutilisés.
- `test_ml_has_no_unused_imports` — Vérifie que ML n'a pas d'imports inutilisés.
- `test_ml_alias_exists` — Vérifie que l'alias ML existe.
- `test_naming_collision_is_documented` — Vérifie que la collision de noms est documentée.
- `test_handler_forwards_tol` — Vérifie que le handler transmet le tol.
- `fake_contour` — Fonction factice pour les contours.

### `0.Core/test_audit_ml_x64.py`
Ce fichier teste le fonctionnement de JAXKMeans sur des données float32 avec x64 activé.
- `test_kmeans_float32_data_under_x64` — Vérifie que KMeans fonctionne sur des données float32 avec x64 activé.

### `0.Core/test_audit_models_major_bnnc.py`
Ce fichier teste les régressions majeures du module Models pour BNNC (B1-B4).
- `_tiny_data` — Génère des données 2D standardisées en 2 blobs.
- `_tiny_fit` — Ajuste un petit modèle BNNC.
- `test_b1_theta_not_stored_but_predict_works` — Vérifie que theta n'est pas stocké mais que predict fonctionne.
- `test_b2_plot_runs_and_uses_theta` — Vérifie que le tracé fonctionne et utilise theta.
- `test_b3_warns_on_raw_scale` — Vérifie qu'un avertissement est émis sur l'échelle brute.
- `counting` — Fonction factice pour compter les appels.

### `0.Core/test_audit_models_major_gmm_dpmm.py`
Ce fichier teste les régressions majeures du module Models pour GMM et DPMM (G1-G3, D1-D2).
- `_FakeSampler` — Classe factice simulant un sampler.
- `test_g1_relabel_undoes_label_switching` — Vérifie que le relabeling annule le changement d'étiquettes.
- `test_g2_psm_uses_at_most_300_draws` — Vérifie que la matrice de similarité utilise au plus 300 tirages.
- `test_g3_gmm_scale_invariant_prior` — Vérifie que le prior GMM est invariant à l'échelle.
- `test_d2_dpmm_constant_column_does_not_crash` — Vérifie que DPMM ne plante pas sur une colonne constante.
- `__init__` — Initialise le sampler factice.
- `get_samples` — Retourne les échantillons du sampler factice.

### `0.Core/test_audit_models_major_pca_surv.py`
Ce fichier teste les régressions majeures du module Models pour PCA et Survie (P1-P3, V1-V4).
- `_surv_object` — Crée un objet de survie pour les tests.
- `test_v1_observed_method_still_callable` — Vérifie que la méthode observed est toujours appelable.
- `test_v2_no_covariate_model_has_likelihood` — Vérifie que le modèle sans covariable a une vraisemblance.
- `test_v3_bad_times_raise` — Vérifie que des temps invalides lèvent une erreur.
- `test_p1_fit_twice_keeps_user_data` — Vérifie que deux fits conservent les données utilisateur.
- `test_p3_scalar_mappable_branch` — Vérifie que la branche ScalarMappable est utilisée.
- `_base` — Fonction utilitaire pour déballer les distributions.

### `0.Core/test_audit_network_minor_a.py`
Ce fichier teste les régressions mineures du module Network (m1-m5).
- `_asymmetric_4x4` — Génère une matrice d'adjacence asymétrique.
- `_symmetric_4x4` — Génère une matrice d'adjacence symétrique.
- `test_m1_asymmetric_matrix_rejected_when_undirected` — Vérifie qu'une matrice asymétrique est rejetée si non dirigée.
- `test_m1_asymmetric_matrix_accepted_when_directed` — Vérifie qu'une matrice asymétrique est acceptée si dirigée.
- `test_m1_n_nodes_mismatch_raises` — Vérifie qu'une erreur est levée si le nombre de nœuds ne correspond pas.
- `test_m1_non_square_matrix_raises` — Vérifie qu'une erreur est levée pour une matrice non carrée.
- `test_m1_symmetric_undirected_values_stay_finite` — Vérifie que les valeurs restent finies pour une matrice symétrique non dirigée.
- `test_m1_symmetric_undirected_matches_networkx` — Vérifie que les résultats correspondent à NetworkX.
- `test_m1_path_count_dtype_follows_x64_config` — Vérifie que le dtype du comptage de chemins suit la config x64.
- `test_m2_self_loop_no_longer_deflates_a_complete_triangle` — Vérifie qu'une boucle propre ne dégonfle pas un triangle complet.
- `test_m2_plain_triangle_is_one_without_self_loops` — Vérifie que le coefficient de clustering d'un triangle simple est 1.
- `test_m2_negative_weight_raises` — Vérifie qu'un poids négatif lève une erreur.
- `test_m3_init_does_not_warm_up_betweenness` — Vérifie que l'initialisation ne préchauffe pas la centralité d'intermédiarité.
- `test_m4_normalize_works_on_an_instance` — Vérifie que la normalisation fonctionne sur une instance.
- `test_m4_clustering_coefficient_works_on_an_instance` — Vérifie que le coefficient de clustering fonctionne sur une instance.
- `test_m5_dijkstra_variants_document_their_semantics` — Vérifie que les variantes de Dijkstra documentent leur sémantique.

### `0.Core/test_audit_network_minor_b.py`
Ce fichier teste les régressions mineures du module Network (m6-m10, m16, m18).
- `_source` — Récupère le texte source d'un module importé.
- `test_m6_block_model_carries_a_single_staticmethod_decorator` — Vérifie que le modèle de bloc a un seul décorateur staticmethod.
- `test_m6_block_model_is_a_plain_static_function` — Vérifie que le modèle de bloc est une fonction statique simple.
- `test_m7_dyadic_effect_without_predictors_or_shape_raises` — Vérifie qu'une erreur est levée si les prédicteurs ou la forme manquent.
- `test_m8_net_dyadic_effect_documents_2d_contract_drift` — Vérifie que la docstring documente la dérive du contrat 2D.
- `test_m10_dyad_vec_to_mat_still_a_net_method` — Vérifie que dyad_vec_to_mat est toujours une méthode net.
- `test_m8_net_dyadic_effect_still_accepts_3d_edgelist` — Vérifie que l'effet dyadique accepte toujours les listes d'arêtes 3D.
- `test_m8_net2_dyadic_effect_rejects_non_triangular_shape` — Vérifie que l'effet dyadique Net2 rejette les formes non triangulaires.
- `test_m8_net2_dyadic_effect_accepts_triangular_shape` — Vérifie que l'effet dyadique Net2 accepte les formes triangulaires.
- `test_m9_nodal_random_effects_rejects_zero_based_group_index` — Vérifie que les effets aléatoires nodaux rejettent les indices basés sur zéro.
- `test_m10_util_defines_dyad_vec_to_mat_exactly_once` — Vérifie que dyad_vec_to_mat est défini exactement une fois.
- `test_m10_dyad_vec_to_mat_wrong_length_raises` — Vérifie qu'une erreur est levée pour une longueur incorrecte.
- `test_m10_dyad_vec_to_mat_directed_wrong_length_raises` — Vérifie qu'une erreur est levée pour une longueur incorrecte en mode dirigé.
- `test_m10_dyad_vec_to_mat_roundtrips_known_3x3_symmetric` — Vérifie le roundtrip pour une matrice 3x3 symétrique.
- `test_m10_dyad_vec_to_mat_roundtrips_known_3x3_directed` — Vérifie le roundtrip pour une matrice 3x3 dirigée.
- `test_m16_no_dead_cache_attribute_in_model_effects` — Vérifie qu'il n'y a pas d'attribut de cache mort.
- `test_m18_net2_usage_example_masks_diagonal_with_where` — Vérifie que l'exemple d'utilisation masque la diagonale avec `where`.
- `test_m18_zeroing_logits_is_not_an_exclusion` — Vérifie que la mise à zéro des logits n'est pas une exclusion.

### `0.Core/test_audit_network_minor_c.py`
Ce fichier teste les régressions mineures du module Network (m11-m15, m17).
- `test_m11_minmax_norm_constant_column_no_nan` — Vérifie que la normalisation min-max ne produit pas de NaN pour une colonne constante.
- `test_m11_format_att_constant_opacity_column_no_nan` — Vérifie que le formatage des attributs ne produit pas de NaN pour une opacité constante.
- `test_m12_check_node_table_helper` — Vérifie l'helper de vérification de la table de nœuds.
- `test_m12_vis_net_length_mismatch_raises` — Vérifie qu'une erreur est levée pour une incohérence de longueur dans vis_net.
- `test_m13_layouts_empty_and_single_node` — Vérifie que les dispositions fonctionnent pour les graphes vides ou à un nœud.
- `test_m13_spectral_disconnected_warns_and_falls_back` — Vérifie que la disposition spectrale avertit et fait un fallback pour les graphes déconnectés.
- `test_m13_spectral_connected_does_not_warn` — Vérifie que la disposition spectrale n'avertit pas pour les graphes connectés.
- `test_m14_chord_matrix_matches_reference_double_loop` — Vérifie que la matrice d'accords correspond à la référence.
- `test_m14_chord_matrix_string_groups_and_length_guard` — Vérifie la gestion des groupes de chaînes et la garde de longueur dans la matrice d'accords.
- `test_m15_density_fallback_directed_vs_undirected` — Vérifie le fallback de densité pour les graphes dirigés et non dirigés.
- `test_m15_network_stats_ignores_self_loops` — Vérifie que les statistiques du réseau ignorent les boucles propres.
- `test_m15_network_stats_directed_density` — Vérifie la densité des statistiques du réseau dirigé.
- `test_m17_js_json_escapes_script_break` — Vérifie que le JSON JS échappe les ruptures de script.
- `test_m17_js_val_and_tooltip_escape` — Vérifie que les valeurs JS et les infobulles sont échappées.
- `test_m17_label_escaped_in_generated_html` — Vérifie que les étiquettes sont échappées dans le HTML généré.

### `0.Core/test_audit_np_dists_minor.py`
Ce fichier teste les régressions mineures de np_dists.py (m1-m10).
- `_source` — Lit le code source de np_dists.py.
- `_func` — Retourne la définition AST d'une méthode de niveau supérieur.
- `_docstring` — Retourne la docstring d'une méthode.
- `test_m1_lkj_authors_are_lewandowski_kurowicka_joe` — Vérifie que les auteurs LKJ sont correctement cités.
- `test_m2_power_law_alpha_constraint_matches_formula` — Vérifie que la contrainte alpha de la loi de puissance correspond à la formule.
- `test_m3_scale_documented_as_scale_not_sample` — Vérifie que l'échelle est documentée comme échelle et non comme échantillon.
- `test_m4_improper_uniform_docstring_clean_and_complete` — Vérifie que la docstring de la loi uniforme impropre est propre et complète.
- `test_m5_bernoullilogits_returns_only_in_own_method` — Vérifie que BernoulliLogits retourne uniquement dans sa propre méthode.
- `test_m6_gaussian_copula_beta_validate_args_defaults_to_none` — Vérifie que validate_args par défaut est None pour la copule gaussienne.
- `test_m7_no_unused_matplotlib_or_dist_import` — Vérifie qu'il n'y a pas d'imports matplotlib ou dist inutilisés.
- `test_m8_infer_caller_name_has_no_dead_return` — Vérifie qu'il n'y a pas de retour mort dans infer_caller_name.
- `test_m9_seeds_beyond_32_bits_are_not_truncated` — Vérifie que les graines au-delà de 32 bits ne sont pas tronquées.
- `test_m10_empty_infer_dict_is_kept_on_the_site` — Vérifie que le dictionnaire infer vide est conservé sur le site.
- `model` — Modèle pour tester le dictionnaire infer.

### `0.Core/test_audit_postmodel_major.py`
Ce fichier teste les régressions majeures du module PostModel (M1-M5).
- `model_m1` — Modèle avec pente variée non centrée.
- `test_to_latex_expands_varying_slope_non_centered_block` — Vérifie que to_latex développe le bloc de pente variée non centrée.
- `test_graph_varying_slope_non_centered_has_direct_sample_nodes` — Vérifie que le graphe a des nœuds d'échantillonnage directs.
- `model_m2` — Modèle simple avec sigma.
- `model_m2_boom` — Modèle qui lève une erreur lors du tracé.
- `test_m2_tracing_failure_warns_and_marks_graph_inexact` — Vérifie qu'un échec de tracé avertit et marque le graphe comme inexact.
- `test_m2_warning_names_the_failure` — Vérifie que l'avertissement nomme l'échec.
- `test_m2_successful_tracing_keeps_exact_flag_true` — Vérifie que le tracé réussi garde le drapeau exact à vrai.
- `model_m4` — Modèle avec plate.
- `test_m4_with_plate_header_precedes_its_body` — Vérifie que l'en-tête du plate précède son corps.
- `model_m5` — Modèle avec plate et observation.
- `test_m5_trace_latex_marks_observed_sites_and_plates` — Vérifie que le latex de trace marque les sites observés et les plates.
- `model_m3` — Modèle avec deterministic.
- `test_m3_edge_through_bypassed_deterministic_is_kept` — Vérifie que l'arête à travers le deterministic contourné est conservée.

### `0.Core/test_audit_postmodel_minor_a.py`
Ce fichier teste les régressions mineures du module PostModel (to_latex).
- `model_m1_lkj` — Modèle avec LKJ.
- `test_m1_lkj_class_names` — Vérifie que les noms de classes LKJ sont corrects dans le latex.
- `model_m2_truncated` — Modèle avec normale tronquée.
- `test_m2_truncated_normal_support` — Vérifie que le support de la normale tronquée est affiché correctement.
- `model_m3_gp` — Modèle avec processus gaussien.
- `test_m3_gp_kernel_fixed_nugget` — Vérifie que le noyau GP a un nugget fixe.
- `model_m4_edge_direct` — Modèle avec poids d'arête directs.
- `model_m4_edge_pooled` — Modèle avec poids d'arête poolés.
- `model_m5_nested_centered` — Modèle avec effets imbriqués centrés.
- `model_m5_nested_noncentered` — Modèle avec effets imbriqués non centrés.
- `test_m4_m5_centered_parameterizations` — Vérifie les paramétrisations centrées.
- `model_m17_exprs` — Modèle avec expressions mathématiques.
- `test_m17_mathematically_wrong_translations` — Vérifie que les traductions mathématiquement incorrectes sont corrigées.
- `model_m23_loop` — Modèle avec boucle.
- `test_m23_line_by_line_fallbacks_keep_headers_and_bodies` — Vérifie que les fallbacks ligne par ligne conservent les en-têtes et les corps.
- `test_m26_fallback_chain_warns` — Vérifie que la chaîne de fallback avertit.
- `boom` — Fonction factice levant une erreur.
- `boom` — Fonction factice levant une erreur.

### `0.Core/test_audit_postmodel_minor_b.py`
Ce fichier teste les régressions mineures du module PostModel (model_graph, model_trace).
- `model_m6` — Modèle avec effets imbriqués.
- `test_m6_nested_varying_effects_positional_names` — Vérifie que les noms positionnels des effets imbriqués sont corrects.
- `model_m7_default` — Modèle avec effets variés par défaut.
- `model_m7_overridden` — Modèle avec effets variés surchargés.
- `test_m7_supplied_hyperpriors_drop_ghost_sites` — Vérifie que les hyperpriors fournis suppriment les sites fantômes.
- `test_m19_transform_params_traced_fallback_leaves_unnamed` — Vérifie que le fallback de tracé des paramètres de transformation laisse des noms non nommés.
- `_identity_deco` — Décorateur d'identité.
- `_m11_helper` — Helper pour le modèle m11.
- `model_m11` — Modèle avec helper décoré.
- `test_m11_decorated_model_line_mapping` — Vérifie le mapping des lignes pour les modèles décorés.
- `test_m12_draw_closes_figure_and_renders_plates` — Vérifie que draw ferme la figure et rend les plates.
- `model_m27` — Modèle simple pour tester l'avertissement expérimental.
- `test_m27_experimental_notice_is_a_warning_not_print` — Vérifie que l'avis expérimental est un avertissement et non un print.
- `test_m20_malformed_data_gets_guided_errors` — Vérifie que les données mal formées obtiennent des erreurs guidées.
- `model_m21` — Modèle avec param SVI.
- `test_m21_svi_param_sites_ignored_in_ast` — Vérifie que les sites de param SVI sont ignorés dans l'AST.
- `model_m8` — Modèle avec sites latents nus.
- `test_m8_bare_latent_sites_not_labelled_observed` — Vérifie que les sites latents nus ne sont pas étiquetés comme observés.
- `model_m9m10` — Modèle avec assignations multiples et plates.
- `test_m9_m10_multi_target_assignments_and_plate_sizes` — Vérifie les assignations à cible multiple et les tailles de plate.
- `model_m15` — Modèle avec distribution transformée.
- `test_m15_m16_transform_dependency_and_observed_marker` — Vérifie la dépendance de transformation et le marqueur observé.
- `model_api_latex` — Modèle pour tester l'API latex.
- `model_api_det` — Modèle pour tester l'API deterministic.
- `model_api_dist_in_var` — Modèle pour tester la distribution dans une variable.
- `model_api_dist_param` — Modèle pour tester le paramètre de distribution.
- `model_api_plate` — Modèle pour tester l'API plate.
- `test_m13_m14_m24_api_coherence` — Vérifie la cohérence de l'API.
- `_NoArgDist` — Classe de distribution sans arguments.
- `wrapper` — Wrapper de fonction.
- `model_m20` — Modèle pour tester les données mal formées.
- `model_m20_scalar` — Modèle scalaire pour tester les données mal formées.
- `__init__` — Constructeur de la classe de distribution.

### `0.Core/test_audit_r_critical.py`
Ce fichier teste les régressions critiques du module R (R-C1/R-C2).
- `test_rc1_doc_extractor_lists_distributions` — Vérifie que l'extracteur de documentation liste les distributions.
- `test_rc1_distproxy_dir_lists_methods` — Vérifie que dir() sur le proxy de distribution liste les méthodes.
- `test_rc2_no_dev_script_inside_package` — Vérifie qu'il n'y a pas de scripts de développement dans le package.

### `0.Core/test_audit_r_major.py`
Ce fichier teste les régressions majeures du module R (R-M2..R-M4).
- `_load_fill_params` — Charge le module fill_params.
- `test_parse_args_section_google_style` — Vérifie l'analyse de la section Args au style Google.
- `test_parse_args_section_markdown_style` — Vérifie l'analyse de la section Args au style Markdown.
- `test_parse_args_section_markdown_continuation` — Vérifie l'analyse des continuations Markdown.
- `test_fill_drops_and_repairs` — Vérifie que le remplissage supprime et répare les paramètres.
- `test_fill_undocumented_marker` — Vérifie que le marqueur non documenté est ajouté.
- `test_fill_replaces_bare_literal_defaults` — Vérifie que les valeurs par défaut littérales nues sont remplacées.
- `test_fill_literal_default_without_pydoc` — Vérifie le remplissage des valeurs par défaut littérales sans pydoc.
- `test_checked_in_rdoc_is_clean` — Vérifie que le RDoc vérifié est propre.
- `test_r_m2_paths_are_script_relative_and_named_consistently` — Vérifie que les chemins sont relatifs au script et nommés de manière cohérente.
- `test_r_m4_prompt_moved_out_of_bf_doc` — Vérifie que le prompt est déplacé hors de bf_doc.

### `0.Core/test_audit_r_minor.py`
Ce fichier teste les régressions mineures du module R (R-m1..R-m5).
- `_load_tool` — Charge l'outil de documentation R.
- `_FakeDist` — Classe factice pour la distribution.
- `_FakeModel` — Classe factice pour le modèle.
- `test_r_m1_tool_has_no_import_side_effect_and_no_cwd_default` — Vérifie que l'outil n'a pas d'effets de bord d'import et pas de cwd par défaut.
- `test_r_m2_rule_based_filter_not_frozen_blacklist` — Vérifie que le filtre basé sur des règles n'est pas une liste noire figée.
- `test_r_m3_json_is_utf8_readable_and_stable` — Vérifie que le JSON est lisible en UTF-8 et stable.
- `test_r_m4_two_sources_of_truth_documented` — Vérifie que les deux sources de vérité sont documentées.
- `test_r_m5_manifest_has_no_dead_bf_rule` — Vérifie que le manifest n'a pas de règle BF morte.
- `normal` — Méthode de distribution factice.
- `sineskewed` — Méthode de distribution factice.
- `mask` — Helper factice.
- `plate` — Helper factice.
- `biject_to` — Helper factice.

### `0.Core/test_audit_resources_critical.py`
Ce fichier teste les régressions critiques du module Resources (C-1..C-3).
- `test_c1_no_loader_points_to_a_missing_file` — Vérifie qu'aucun chargeur ne pointe vers un fichier manquant.
- `test_c2_c3_default_frame_has_columns` — Vérifie que le frame par défaut a des colonnes.
- `test_c2_c3_frame_and_bunch_share_schema` — Vérifie que le frame et le bunch partagent le même schéma.
### `0.Core/test_audit_resources_major.py`
Ce fichier teste les régressions majeures (M-1 à M-7) du module Resources liées à l'audit de version.
- `test_m1_ucbadmit_explicit_header_and_unchanged_frame` — Vérifie que le chargeur `ucbadmit` parse correctement l'en-tête fixe et retourne un DataFrame inchangé.
- `test_m2_sim_multivariate_normal_honours_frame` — Vérifie que `sim_multivariate_normal` retourne un DataFrame par défaut et un objet `Bunch` si `frame=False`.
- `test_m3_matrix_loaders_return_float64` — Vérifie que les chargeurs de matrices (`phylo_L_simple`, `islands_dist_matrix`) retournent des données de type `float64`.
- `test_m4_bunch_schema_unified` — Vérifie que les objets `Bunch` retournés par les chargeurs possèdent un schéma de clés unifié.
- `test_m5_object_dtype_warns` — Vérifie qu'un avertissement est émis lorsque des données non numériques sont chargées avec `frame=False`.
- `test_m6_load_roles` — Vérifie que la fonction `load` lève les erreurs appropriées (`TypeError`, `FileNotFoundError`) pour des entrées invalides.
- `test_m7_security_documented` — Vérifie que la documentation de `load` mentionne les aspects de sécurité liés à `cloudpickle`.

### `0.Core/test_audit_resources_minor.py`
Ce fichier teste les régressions mineures (m-1 à m-11) du module Resources liées à l'audit de version.
- `test_m1_only_path_returns_str_for_every_loader` — Vérifie que l'option `only_path=True` retourne une chaîne de caractères pour tous les chargeurs.
- `test_m2_islands_only_path_does_not_read_the_csv` — Vérifie que `islands_dist_matrix` avec `only_path=True` ne lit pas le fichier CSV.
- `test_m3_class_docstring_describes_dataframe_default` — Vérifie que la docstring de la classe décrit correctement le retour par défaut d'un DataFrame.
- `test_m4_module_docstring_uses_real_import_path` — Vérifie que la docstring du module utilise le chemin d'importation réel.
- `test_m5_describe_matches_file_contents` — Vérifie que les descriptions (`DESCR`) correspondent au contenu réel des fichiers de données.
- `test_m6_nan_columns_documented` — Vérifie que les colonnes contenant des valeurs manquantes (NaN) sont correctement documentées.
- `test_m7_packaged_data_cleanup` — Vérifie que les jeux de données utilisés dans la documentation sont bien inclus dans le paquet.
- `test_m8_no_unused_imports` — Vérifie l'absence d'imports inutilisés dans le code source du module.
- `test_m9_public_names_unchanged_and_naming_documented` — Vérifie que les noms publics sont inchangés et que la convention de nommage est documentée.
- `test_m10_explicit_sep_frames_identical_to_sniffed` — Vérifie que les DataFrames chargés avec un séparateur explicite sont identiques à ceux chargés avec détection automatique.
- `test_m11_no_trailing_whitespace_single_data_dir_assignment` — Vérifie l'absence d'espaces en fin de ligne et une seule affectation de `data_dir`.
- `boom` — Fonction auxiliaire qui lève une erreur pour vérifier qu'une lecture CSV n'a pas lieu.

### `0.Core/test_audit_samplers_minor_a.py`
Ce fichier teste les régressions mineures (1-14) du module Samplers, couvrant les gestionnaires de modèle, les échantillonneurs NumPyro/TFP et SVI.
- `_no_warnings` — Filtre les avertissements liés aux séquences ou aux périphériques.
- `test_1_bare_yield_and_no_call_lines_do_not_crash` — Vérifie que les lignes de code sans assignation ou sans appel de fonction ne provoquent pas de crash lors de l'analyse AST.
- `test_2_dead_var_model_line_removed` — Vérifie la suppression d'une variable morte dans le code source de `Model_handler.py`.
- `test_3_no_prints_warning_goes_through_warnings` — Vérifie que les messages d'information sont supprimés et que les avertissements passent par le module `warnings`.
- `test_4_init_params_length_raises_value_error` — Vérifie qu'une erreur `ValueError` claire est levée en cas de désaccord de longueur des paramètres d'initialisation.
- `_tiny_numpyro_model` — Définit un modèle NumPyro minimal pour les tests.
- `test_5_manip_import_removed` — Vérifie l'absence de couplage résiduel avec `BayesForge.Data` dans `mcmc_numpyro`.
- `test_6_mcmc_numpyro_validates_run_counts` — Vérifie la validation précoce des compteurs d'échantillonnage (warmup, samples, chains).
- `test_6b_parallel_chains_on_few_devices_warns` — Vérifie qu'un avertissement est émis si le nombre de chaînes dépasse le nombre de périphériques disponibles en mode parallèle.
- `_tfp_sampler` — Crée un échantillonneur TFP minimal pour les tests.
- `_tfp_model` — Définit un modèle TFP minimal pour les tests.
- `test_7_obs_names_initialised_in_constructor` — Vérifie que `obs_names` est initialisé dans le constructeur pour éviter les erreurs d'attribut.
- `test_8_nuts_returns_self` — Vérifie que les méthodes `sampler` et `run` exposent l'échantillonneur lui-même.
- `test_9_nuts_model_obs_documented_as_ignored` — Vérifie que les arguments `model` et `obs` sont documentés comme ignorés/dépréciés.
- `test_10_posterior_and_sample_stats_same_type_both_branches` — Vérifie que les statistiques postérieures et d'échantillonnage sont des listes dans les deux branches de code.
- `test_11_n_chains_validated` — Vérifie que `n_chains` est validé et lève une erreur si invalide.
- `_svi_model` — Définit un modèle SVI minimal pour les tests.
- `_svi_guide` — Définit un guide SVI minimal pour les tests.
- `test_12_svi_run_num_steps_is_keyword_only` — Vérifie que `num_steps` est un argument uniquement par mot-clé dans `SVI_BF.run`.
- `test_13_loss_default_is_not_a_shared_mutable` — Vérifie que la perte par défaut crée une nouvelle instance `Trace_ELBO` par objet.
- `test_14_get_samples_seed_default_documented` — Vérifie que la valeur par défaut du seed est documentée dans `get_samples`.
- `model` — Définit un modèle avec des rendements variés pour tester l'analyse AST.
- `model` — Définit un modèle TFP avec des distributions normales et demi-normales.

### `0.Core/test_audit_setdevice_critical.py`
Ce fichier teste la régression critique C1 du module SetDevice concernant les avertissements lors de la reconfiguration des périphériques.
- `_run` — Exécute du code dans un sous-processus avec la plateforme JAX forcée en CPU.
- `test_c1_second_call_with_other_cores_warns` — Vérifie qu'un avertissement est émis lors d'un second appel avec un nombre de cœurs différent.
- `test_c1_same_settings_do_not_warn` — Vérifie qu'aucun avertissement n'est émis lors d'un second appel avec les mêmes paramètres.
- `test_c1_first_call_does_not_warn` — Vérifie qu'aucun avertissement n'est émis lors du premier appel à `setup_device`.

### `0.Core/test_audit_setdevice_major.py`
Ce fichier teste les régressions majeures (M1-M3) du module SetDevice concernant la validation des plateformes et des cœurs.
- `_run` — Exécute du code dans un sous-processus avec la plateforme JAX forcée en CPU.
- `test_m1a_uppercase_platform_cpu_works` — Vérifie que la plateforme 'CPU' en majuscules est acceptée.
- `test_m1b_invalid_platform_raises_value_error` — Vérifie qu'une plateforme invalide lève une `ValueError`.
- `test_m1c_gpu_without_gpu_raises_runtime_error` — Vérifie qu'une demande de GPU sans GPU disponible lève une `RuntimeError`.
- `test_m2_affinity_limits_cpu_device_count` — Vérifie que l'affinité du processeur limite correctement le nombre de périphériques CPU.
- `test_m3_invalid_cores_raise_value_error` — Vérifie que des valeurs invalides pour `cores` lèvent une `ValueError`.
- `test_m3_invalid_bf_max_cores_raise_value_error` — Vérifie que des valeurs invalides pour les cœurs max lèvent une `ValueError`.
- `test_m3_whole_float_cores_works` — Vérifie que les nombres flottants entiers pour `cores` sont acceptés.

### `0.Core/test_audit_setdevice_minor.py`
Ce fichier teste les régressions mineures (m1-m8) du module SetDevice concernant la validation des GPU, la documentation et les drapeaux XLA.
- `_run` — Exécute du code dans un sous-processus avec la plateforme JAX forcée en CPU.
- `test_m1_invalid_gpu_index_raises_value_error` — Vérifie qu'un index de GPU invalide lève une `ValueError`.
- `test_m1_comma_separated_gpu_index_accepted` — Vérifie que les index de GPU séparés par des virgules sont acceptés par la validation.
- `test_m2_deallocate_docstring_and_frozen_warning` — Vérifie la documentation de la libération mémoire et l'avertissement de paramètres figés.
- `test_m3_docstring_backend_init_contract` — Vérifie que la docstring décrit le contrat d'initialisation du backend.
- `test_m4_existing_xla_flags_value_warns` — Vérifie qu'un avertissement est émis si des drapeaux XLA existants entrent en conflit.
- `test_m4_matching_xla_flags_value_does_not_warn` — Vérifie qu'aucun avertissement n'est émis si les drapeaux XLA correspondent.
- `test_m5_float_precision_docstring_process_global` — Vérifie que la docstring mentionne l'impact global du processus sur la précision flottante.
- `test_m6_docstring_not_thread_safe` — Vérifie que la docstring indique que la configuration n'est pas thread-safe.
- `test_m7_print_names_requested_platform` — Vérifie que le nom de la plateforme demandée est affiché lors de la configuration.
- `test_m8_setup_multi_device_cores_same_error_as_setup_device` — Vérifie que `setup_multi_device` lève la même erreur que `setup_device` pour des cœurs invalides.

### `0.Core/test_audit_setdevice_x64_switch.py`
Ce fichier teste les avertissements lors du changement de la précision flottante globale (x64) entre les instances.
- `restore_x64` — Fixture pour restaurer la configuration x64 de JAX après les tests.
- `_x64_warnings` — Filtre les avertissements liés à la configuration globale de `jax_enable_x64`.
- `test_switching_precision_warns` — Vérifie qu'un avertissement est émis lors du changement de précision entre les instances.
- `test_same_precision_and_first_call_are_quiet` — Vérifie qu'aucun avertissement n'est émis si la précision est la même ou lors du premier appel.

### `0.Core/test_audit_surv_plot.py`
Ce fichier teste la régression critique C5 concernant le tracé des taux de risque dans le module de survie.
- `_object` — Crée un objet de survie minimal avec des données synthétiques.
- `_met_curves` — Extrait les données des courbes de risque et de survie d'une figure.
- `test_c5_covariate_hazard_is_lambda0_times_exp_beta` — Vérifie que le taux de risque avec covariable est calculé comme lambda0 * exp(beta).
- `test_c5_default_beta_is_the_model_site` — Vérifie que le beta par défaut correspond au site du modèle.

### `0.Core/test_audit_tfp_dists_minor.py`
Ce fichier teste les régressions mineures (1-17) du module `tfp_dists.py` concernant les distributions unifiées.
- `_doc` — Récupère la docstring d'une méthode de `UnifiedDist`.
- `test_minor1_delta_defined_once_with_v_signature` — Vérifie que la distribution delta est définie une seule fois avec la signature correcte.
- `test_minor2_log_normal_built_once_and_works` — Vérifie que la distribution log-normale est construite une fois et fonctionne correctement.
- `test_minor3_non_samplable_wrappers` — Vérifie que les wrappers non échantillonnables lèvent une erreur si `sample=True`.
- `test_minor4_header_documents_partial_lkj_rename` — Vérifie que l'en-tête documente le renommage partiel de LKJ.
- `test_minor5_header_documents_wrap_not_warp` — Vérifie que l'en-tête documente l'argument `wrap` et non `warp`.
- `test_minor6_multivariate_normal_diag_requires_args` — Vérifie que la normale multivariée diagonale nécessite des arguments loc ou scale.
- `test_minor7_dtype_follows_x64` — Vérifie que le dtype suit la configuration x64 de JAX.
- `test_minor8_truncated_distribution_note` — Vérifie que la note sur la distribution tronquée est présente dans la docstring.
- `test_minor9_half_cauchy_note` — Vérifie que la note sur la demi-Cauchy est présente dans la docstring.
- `test_minor10_delta_note` — Vérifie que la note sur la distribution delta est présente dans la docstring.
- `test_minor11_beta_binomial_total_count_documented` — Vérifie que `total_count` est documenté pour la binomiale bêta.
- `test_minor12_beta_proportion_rejects_bad_mean` — Vérifie que la proportion bêta rejette une moyenne invalide.
- `test_minor12_beta_proportion_accepts_valid_mean` — Vérifie que la proportion bêta accepte une moyenne valide.
- `test_minor12_beta_proportion_skips_validation_for_tracers` — Vérifie que la validation est ignorée pour les traceurs JAX.
- `test_minor13_zero_inflated_poisson_rate_default` — Vérifie que le taux par défaut de la Poisson zéro-inflée est 1.0.
- `test_minor14_tfb_alias_used` — Vérifie que l'alias `tfb` est utilisé pour les bijecteurs TFP.
- `test_minor15_package_docstring` — Vérifie que la docstring du paquet mentionne les modules de distributions.
- `test_minor16_name_conventions_documented` — Vérifie que les conventions de nommage sont documentées.
- `test_minor17_von_mises_note_and_aligned_defaults` — Vérifie la note sur von Mises et l'alignement des valeurs par défaut.
- `f` — Fonction auxiliaire pour tester la validation avec des traceurs.

### `0.Core/test_audit_to_latex.py`
Ce fichier teste la régression critique C1 concernant la génération LaTeX des modèles à effets variables.
- `model_vis` — Définit un modèle avec intercept et pente variables.
- `model_ve` — Définit un modèle avec effets variables.
- `_text` — Convertit la sortie LaTeX en chaîne de caractères.
- `test_c1_varying_intercept_slope_has_no_invented_global_priors` — Vérifie que les priors globaux inventés sont absents du LaTeX pour les intercepts/pentes variables.
- `test_c1_varying_effects_keeps_its_sampled_global_priors` — Vérifie que les priors globaux échantillonnés sont conservés dans le LaTeX pour les effets variables.

### `0.Core/test_audit_utils_critical.py`
Ce fichier teste les 5 régressions critiques du module Utils concernant les effets, les noyaux et les fonctions de lien.
- `test_varying_slope_non_centered_runs` — Vérifie que la pente variable non centrée s'exécute sans erreur.
- `test_varying_intercept_slope_uses_the_correlation_matrix` — Vérifie que l'intercept et la pente variables utilisent la matrice de corrélation.
- `test_gaussian_process2_periodic_runs` — Vérifie que le processus gaussien périodique s'exécute correctement.
- `test_lkjcor_matches_numpyro_density` — Vérifie que la densité LKJ correspond à celle de NumPyro.
- `test_hdi_2d_is_per_column_and_nan_rejected` — Vérifie que l'intervalle de plus haute densité (HDI) est calculé par colonne et rejette les NaN.
- `test_hdi_1d_still_correct` — Vérifie que le calcul HDI en 1D reste correct.

### `0.Core/test_audit_utils_major.py`
Ce fichier teste les régressions majeures du module Utils concernant les identifiants parents, les noyaux et les fonctions de lien.
- `test_parent_ids_crossed_raises` — Vérifie qu'une hiérarchie croisée lève une erreur.
- `test_parent_ids_missing_child_raises` — Vérifie qu'un enfant non observé lève une erreur.
- `test_parent_ids_ok` — Vérifie que les identifiants parents sont correctement résolus pour une hiérarchie valide.
- `test_kernel_sq_exp_duplicate_points_finite` — Vérifie que le noyau exponentiel carré retourne des valeurs finies pour des points dupliqués.
- `test_kernel_exp_no_truncation_warning` — Vérifie que le noyau exponentiel ne génère pas d'avertissement de troncature de dtype.
- `test_logit_bounds` — Vérifie que la fonction logit gère correctement les bornes 0 et 1.

### `0.Core/test_audit_utils_major_b.py`
Ce fichier teste les régressions majeures du module Utils concernant les cartes de chaleur, les imports paresseux et le seed global.
- `test_bf_does_not_reseed_numpy` — Vérifie que la construction de `bf` ne reseed pas le générateur aléatoire NumPy par défaut.
- `test_corr_heatmap_is_correlation` — Vérifie que la carte de chaleur représente bien une corrélation.
- `test_lazy_importer_no_threads` — Vérifie que l'importateur paresseux fonctionne sans threads.
- `fake_show` — Fonction auxiliaire pour capturer les figures Plotly.

### `0.Core/test_audit_utils_minor_a.py`
Ce fichier teste les régressions mineures du module Utils (Effets, Gaussien, Lien).
- `_module_func` — Récupère la définition AST d'une fonction de module.
- `test_centered_noncentered_docstrings_match_code` — Vérifie que les docstrings des paramétrisations centrées/non centrées correspondent au code.
- `test_effects_dead_code_and_traps` — Vérifie l'absence de code mort et de pièges dans le module des effets.
- `test_gaussian_class_defined_once_and_kernel_periodic_local` — Vérifie que la classe gaussienne est définie une fois et que le noyau périodique est local.
- `test_gaussian_process_contract` — Vérifie que le contrat du processus gaussien est correctement documenté.
- `test_inv_logit_scale_range_documented` — Vérifie que la plage de la fonction inverse logit scale est documentée.
- `test_link_dead_code_and_class_consistency` — Vérifie l'absence de code mort et la cohérence des classes dans le module de lien.

### `0.Core/test_audit_utils_minor_b.py`
Ce fichier teste les régressions mineures du module Utils (SampledData, RNG, inférence discrète, etc.).
- `capture_plotly` — Fixture pour capturer les figures Plotly.
- `quiet_mpl` — Fixture pour supprimer les fenêtres Matplotlib.
- `test_hist_3d_axes_and_bins` — Vérifie que l'histogramme 3D gère correctement les axes et les bins.
- `test_ridgeline_interactive_iterates_categories` — Vérifie que le graphique ridgeline interactif itère sur les catégories.
- `test_ppc_plot_local_rng_and_constant_data` — Vérifie que le PPC plot utilise un RNG local et gère les données constantes.
- `test_autocorr_nan_free_and_single_semantics` — Vérifie que l'autocorrélation est sans NaN et a une sémantique simple.
- `test_getattr_no_recursion_when_data_missing` — Vérifie qu'il n'y a pas de récursion infinie si les données sont manquantes.
- `test_type_contracts_ufunc_hash_and_dead_code` — Vérifie les contrats de type, le hashage et l'absence de code mort.
- `test_public_contract_documented` — Vérifie que le contrat public est correctement documenté.
- `test_rng_stream_order_dependence_documented_and_guard_kept` — Vérifie que la dépendance à l'ordre du flux RNG est documentée.
- `test_set_seed_consistent_with_constructor_globals` — Vérifie que `set_seed` est cohérent avec les globaux du constructeur.
- `test_infer_discrete_validates_inputs_and_smoke` — Vérifie la validation des entrées et un test de fumée pour l'inférence discrète.
- `test_dev_warning_uses_warnings_warn` — Vérifie que les avertissements de développement utilisent `warnings.warn`.
- `test_import_manager_no_executor_clear_errors` — Vérifie que le gestionnaire d'import n'utilise pas d'exécuteur et lève des erreurs claires.
- `fake_show` — Fonction auxiliaire pour capturer les figures.
- `run` — Fonction auxiliaire pour exécuter le PPC plot.
- `model` — Définit un modèle discret pour les tests.

### `0.Core/test_build_drift.py`
Ce fichier teste la détection de dérive entre les générateurs de Build et les wrappers de distribution de production.
- `_load_generator` — Charge un fichier générateur Build dans un espace de noms.
- `_unified_method_names` — Récupère les noms des méthodes de la classe `UnifiedDist`.
- `test_np_generator_reproduces_np_dists_methods` — Compare les noms des méthodes générées pour NumPyro avec celles de production.
- `test_tfp_generator_reproduces_tfp_dists_methods` — Compare les noms des méthodes générées pour TFP avec celles de production.
- `test_generators_refuse_to_overwrite_production_files` — Vérifie que les générateurs refusent d'écraser les fichiers de production.

### `0.Core/test_compat.py`
Ce fichier teste la cohérence des manifestes des trois dépôts (Python, R, Julia) avec la matrice de compatibilité.
- `_write` — Écrit des fichiers de manifeste dans un répertoire temporaire.
- `_repos` — Crée une structure de dépôts simulée avec des manifestes.
- `test_consistent_manifests_pass` — Vérifie que des manifestes cohérents passent la vérification.
- `test_every_disagreement_is_reported` — Vérifie que chaque désaccord entre les manifestes est signalé.
- `test_unreadable_manifest_is_reported` — Vérifie qu'un manifeste illisible est signalé comme erreur.
- `test_this_repository_agrees_with_compat_toml` — Vérifie que le dépôt actuel est conforme au fichier `compat.toml`.

### `0.Core/test_data_input.py`
Ce fichier teste l'acceptation des chemins CSV et des tableaux en mémoire par `m.data()`.
- `m` — Crée une instance `bf` pour les tests.
- `test_csv_path` — Vérifie que `m.data` accepte un chemin CSV.
- `test_dataframe_is_copied` — Vérifie que `m.data` copie le DataFrame fourni sans modifier l'original.
- `test_dict_of_columns` — Vérifie que `m.data` accepte un dictionnaire de colonnes.
- `test_csv_kwargs_rejected_for_tables` — Vérifie que les arguments CSV sont rejetés pour les tableaux en mémoire.
- `test_fit_accepts_whole_number_floats` — Vérifie que `fit` accepte des flottants entiers pour les paramètres.
- `model` — Définit un modèle linéaire simple pour les tests.

### `0.Core/test_diag_api.py`
Ce fichier vérifie que chaque méthode de `m.diag` est documentée et s'exécute sur un modèle ajusté.
- `m` — Crée et ajuste une instance `bf` avec un modèle linéaire.
- `public_methods` — Récupère la liste des méthodes publiques de `diag`.
- `test_every_diag_method_is_documented` — Vérifie que toutes les méthodes publiques sont documentées.
- `test_diag_method_runs` — Vérifie que chaque méthode de diagnostic s'exécute sans erreur.
- `test_ppc_signature_marks_y_yrep_optional` — Vérifie que les arguments `y` et `yrep` sont optionnels dans la signature PPC.
- `test_yrep_is_simulated_not_the_observed_data` — Vérifie que `yrep` est simulé et non les données observées.
- `test_network_cc_nodes_subset` — Vérifie que le coefficient de clustering sur un sous-ensemble de nœuds est correct.
- `test_sample_with_observed_argument_first` — Vérifie que `sample` fonctionne si l'argument observé est en premier.
- `model` — Définit un modèle linéaire avec intercept et pente.
- `model` — Définit un modèle linéaire simple.

### `0.Core/test_diagnostics_ic.py`
Ce fichier teste les critères d'information et la comparaison de modèles sur un modèle ajusté.
- `_fit` — Ajuste un modèle de régression sur les données Howell.
- `slope_model` — Fixture pour un modèle avec pente.
- `intercept_model` — Fixture pour un modèle sans pente.
- `test_loo` — Vérifie que le calcul LOO retourne une valeur finie.
- `test_loo_pointwise` — Vérifie que le calcul LOO point par point retourne le bon nombre de valeurs.
- `test_waic_and_alias` — Vérifie que WAIC et son alias retournent des valeurs cohérentes.
- `test_compare_fitted_models` — Vérifie que la comparaison de modèles ajustés fonctionne.
- `model` — Définit le modèle de régression avec ou sans pente.

### `0.Core/test_discrete_samplers.py`
Ce fichier vérifie l'exposition et le fonctionnement des échantillonneurs discrets (MixedHMC, DiscreteHMCGibbs, etc.).
- `_z_mcse` — Calcule le score z de la différence des moyennes postérieures.
- `make_nmix` — Crée un modèle de mélange N avec abondance de Poisson tronquée.
- `_new_session` — Crée une nouvelle session `bf`.
- `test_nmix_discrete_gibbs_matches_enumeration` — Vérifie que le posterior Gibbs discret correspond à l'énumération exacte.
- `_ari` — Calcule l'indice Rand ajusté.
- `test_dpmm_latent_recovers_clusters` — Vérifie que le DPMM latent avec Gibbs discret récupère les clusters.
- `_mixture_stats` — Calcule les statistiques invariantes par permutation pour un mélange.
- `test_mixed_hmc_matches_enumeration` — Vérifie que MixedHMC correspond à NUTS avec énumération.
- `test_hmc_gibbs_matches_nuts` — Vérifie que HMCGibbs correspond à NUTS.
- `test_sampler_args_validated` — Vérifie la validation des arguments de l'échantillonneur.
- `test_non_numpyro_backend_rejects_other_samplers` — Vérifie que le backend TFP rejette les autres échantillonneurs.
- `model` — Définit le modèle de mélange N.
- `fit` — Ajuste un modèle de mélange.
- `model` — Définit un modèle avec site obs nommé explicitement.
- `gibbs_fn` — Définit la fonction Gibbs pour l'échantillonneur HMCGibbs.
- `model` — Définit un modèle simple avec sigma.
- `model` — Définit un modèle simple avec sigma.
- `model` — Définit un modèle de mélange avec énumération parallèle.

### `0.Core/test_doc_regressions_trace_checks.py`
Ce fichier teste les régressions trouvées en exécutant les pages de documentation, notamment les vérifications de valeurs sous tracing.
- `_in_trace` — Exécute une fonction sous `jit` et `shard_map`.
- `test_block_build_mu_ij_inside_trace` — Vérifie que `block_build_mu_ij` fonctionne sous tracing.
- `test_block_build_mu_ij_still_rejects_empty_group` — Vérifie que `block_build_mu_ij` rejette toujours les groupes vides.
- `test_mat_to_edgl_inside_trace` — Vérifie que `mat_to_edgl` fonctionne sous tracing.
- `test_onnela_inside_trace` — Vérifie que le coefficient de clustering Onnela fonctionne sous tracing.
- `test_betweenness_inside_trace` — Vérifie que la centralité de proximité fonctionne sous tracing.
- `test_nodal_random_effects_inside_trace` — Vérifie que les effets aléatoires nodaux fonctionnent sous tracing.
- `test_tfp_asymmetric_laplace_quantile_inside_trace` — Vérifie que le quantile Laplace asymétrique TFP fonctionne sous tracing.
- `_FakeSampler` — Classe simulante pour les échantillonneurs.
- `test_plot_gmm_with_more_than_300_draws` — Vérifie que le tracé GMM fonctionne avec plus de 300 tirages.
- `body` — Fonction auxiliaire pour `_in_trace`.
- `run` — Exécute les effets aléatoires nodaux sous seed.
- `__init__` — Initialise le faux échantillonneur.
- `get_samples` — Retourne les échantillons du faux échantillonneur.

### `0.Core/test_docstring_examples.py`
Ce fichier exécute les extraits d'exemple de chaque docstring de distribution.
- `example_of` — Extrait l'exemple d'une docstring.
- `test_examples_found` — Vérifie qu'un nombre suffisant d'exemples sont trouvés.
- `test_distribution_example_runs` — Vérifie que chaque exemple de distribution s'exécute sans erreur.

### `0.Core/test_dpmm_plot.py`
Ce fichier teste le tracé postérieur du DPMM avec une région de crédibilité par cluster de consensus.
- `_FakeSampler` — Classe simulante pour les échantillonneurs DPMM.
- `_synthetic_posterior` — Génère un posterior synthétique avec des blobs séparés.
- `test_component_covariances_are_diag_sigma_l_lt_diag_sigma` — Vérifie la relation entre les covariances des composants.
- `test_cluster_moments_match_analytic_two_component_mixture` — Vérifie que les moments des clusters correspondent à un mélange analytique.
- `test_cluster_moments_anchor_uses_within_component_spread_only` — Vérifie que l'ancre des moments utilise la dispersion intra-composant.
- `test_cap_covariance_bounds_ellipse_by_point_spread` — Vérifie que la covariance est bornée par la dispersion des points.
- `test_predict_returns_probs_only_when_asked` — Vérifie que `predict` retourne les probabilités seulement si demandé.
- `test_plot_draws_one_ellipse_per_cluster_and_level` — Vérifie qu'une ellipse est tracée par cluster et niveau.
- `test_density_overlay_uses_positive_relative_levels` — Vérifie que la superposition de densité utilise des niveaux relatifs positifs.
- `_synthetic_gmm_posterior` — Génère un posterior GMM synthétique.
- `test_gmm_predict_returns_probs_only_when_asked` — Vérifie le retour des probabilités pour GMM.
- `test_gmm_plot_draws_one_ellipse_per_cluster_per_level` — Vérifie le tracé des ellipses pour GMM.
- `test_gmm_plot_credible_list_controls_ellipse_count` — Vérifie que la liste de crédibilité contrôle le nombre d'ellipses.
- `__init__` — Initialise le faux échantillonneur.
- `get_samples` — Retourne les échantillons du faux échantillonneur.

### `0.Core/test_effects.py`
Ce fichier teste les blocs de construction d'effets aléatoires (`m.effects`).
- `test_phylo_effect_covariance_and_tree_stack` — Vérifie la covariance et l'empilement d'arbres pour les effets phylogénétiques.

### `0.Core/test_enumeration.py`
Ce fichier vérifie le marquage explicite de l'énumération (`enum=True`).
- `make_model` — Crée un modèle de mélange à deux composants avec différents modes d'énumération.
- `fit_mixture` — Ajuste le modèle de mélange.
- `_means` — Calcule les moyennes des postérieurs.
- `test_enum_matches_default` — Vérifie que le posterior avec `enum=True` correspond à celui par défaut.
- `test_enum_is_exact_enumeration` — Vérifie que `enum=True` correspond exactement à l'énumération parallèle.
- `model` — Définit le modèle de mélange avec plaques.

### `0.Core/test_index_categorical.py`
Ce fichier teste l'indexation des colonnes catégorielles par `m.index()`.
- `test_index_all_includes_categorical_and_keeps_level_order` — Vérifie que l'indexation inclut les colonnes catégorielles et conserve l'ordre des niveaux.

### `0.Core/test_obs_args_tracing.py`
Ce fichier teste que les arguments de données observés sont trouvés par tracing du modèle.
- `_fit` — Ajuste un modèle avec des données.
- `_source_model` — Définit un modèle avec source disponible.
- `_exec_model` — Définit un modèle sans source disponible (via exec).
- `_partial_model` — Définit un modèle partiel.
- `test_obs_args_found_by_tracing` — Vérifie que les arguments observés sont trouvés par tracing pour tous les types de modèles.
- `test_predict_on_new_data_without_the_outcome` — Vérifie la prédiction sur de nouvelles données sans la variable réponse.
- `test_same_predictions_with_and_without_source` — Vérifie que les prédictions sont identiques avec ou sans source.
- `test_predict_with_default_after_required_data` — Vérifie la prédiction avec des valeurs par défaut après les données requises.
- `_computed_obs_model` — Définit un modèle avec une observation calculée.
- `test_computed_obs_warns_and_is_not_replicated` — Vérifie qu'une observation calculée génère un avertissement et n'est pas répliquée.
- `test_data_obs_replicates_without_warning` — Vérifie que les observations de données sont répliquées sans avertissement.
- `model` — Définit un modèle linéaire.
- `model` — Définit un modèle linéaire avec échelle.
- `_default_after_obs` — Définit un modèle avec défaut après obs.
- `_required_after_obs` — Définit un modèle avec requis après obs.
- `model` — Définit un modèle avec obs calculée.
- `model` — Définit un modèle linéaire avec échelle.
- `model` — Définit un modèle linéaire avec z.

### `0.Core/test_obs_names.py`
Ce fichier teste les noms automatiques des sites observés (`obs=y` nomme le site `"y"`).
- `_sites` — Récupère les sites observés d'un modèle tracé.
- `test_obs_variable_names_the_site` — Vérifie que la variable d'observation nomme le site.
- `test_latent_called_x_no_longer_collides` — Vérifie qu'un latent nommé x ne collide plus avec le site obs.
- `test_expression_explicit_name_and_assignment` — Vérifie le nommage des expressions et des affectations.
- `test_posterior_predictive_uses_the_data_name` — Vérifie que la prédiction postérieure utilise le nom des données.
- `test_two_observed_sites_keep_their_names_in_predictions` — Vérifie que deux sites observés gardent leurs noms.
- `test_prior_predictive_before_any_fit` — Vérifie la prédiction a priori avant tout ajustement.
- `test_obs_names_cache_is_per_instance` — Vérifie que le cache des noms obs est par instance.
- `test_multiple_obs_sites_get_yrep_and_ppc` — Vérifie que plusieurs sites obs génèrent yrep et PPC.
- `model` — Définit un modèle linéaire.
- `model` — Définit un modèle avec latent x.
- `model` — Définit un modèle avec expressions et noms explicites.
- `model` — Définit un modèle linéaire simple.
- `model` — Définit un modèle avec deux obs.
- `model` — Définit un modèle linéaire simple.
- `model1` — Définit un modèle avec deux obs pour m1.
- `model` — Définit un modèle avec deux obs.

### `0.Core/test_rng.py`
Ce fichier vérifie que les tirages directs suivent le seed de la session.
- `test_no_distribution_hardcodes_a_seed` — Vérifie qu'aucune distribution n'a de seed codé en dur.
- `_mixture_draw` — Effectue un tirage de mélange.
- `test_consecutive_draws_differ_and_session_seed_reproduces` — Vérifie que les tirages consécutifs diffèrent et que le seed reproduit les résultats.
- `_fitted` — Ajuste un modèle linéaire.
- `test_sample_and_yrep_have_no_hidden_fixed_seed` — Vérifie que sample et yrep n'ont pas de seed fixe caché.
- `test_direct_draw_inside_fitted_model_raises_clear_error` — Vérifie qu'un tirage direct dans un modèle ajusté lève une erreur claire.
- `model` — Définit un modèle linéaire.
- `model` — Définit un modèle avec tirage direct.

### `0.Core/test_run_simulations_languages (conflicted copy 2026-10-23 141016).py`
Ce fichier teste la cohérence de `run_simulations` entre Python, R et Julia (copie en conflit).
- `_reference` — Génère la référence Python en série.
- `_norm` — Normalise les lignes pour la comparaison numérique.
- `_env` — Configure l'environnement pour les sous-processus.
- `_check` — Vérifie les résultats CSV contre la référence.
- `test_r_matches_python` — Vérifie que les résultats R correspondent à Python.
- `_find_julia` — Trouve l'exécutable Julia.
- `test_julia_matches_python` — Vérifie que les résultats Julia correspondent à Python.
- `fn` — Fonction de simulation simple.

### `0.Core/test_run_simulations_languages.py`
Ce fichier teste la cohérence de `run_simulations` entre Python, R et Julia.
- `_reference` — Génère la référence Python en série.
- `_norm` — Normalise les lignes pour la comparaison numérique.
- `_env` — Configure l'environnement pour les sous-processus.
- `_check` — Vérifie les résultats CSV contre la référence.
- `test_r_matches_python` — Vérifie que les résultats R correspondent à Python.
- `_find_julia` — Trouve l'exécutable Julia.
- `test_julia_matches_python` — Vérifie que les résultats Julia correspondent à Python.
- `fn` — Fonction de simulation simple.

### `0.Core/test_sample_api (conflicted copy 2026-10-23 171956).py`
Ce fichier teste la sémantique de `bf.sample()` (copie en conflit).
- `_linear_model` — Définit un modèle linéaire.
- `fitted` — Fixture pour une instance ajustée.
- `test_default_posterior_predictive` — Vérifie la prédiction postérieure par défaut.
- `test_samples_subsample_and_overflow` — Vérifie le sous-échantillonnage et le débordement.
- `test_new_data_predictions` — Vérifie les prédictions sur de nouvelles données.
- `test_prior_predictive` — Vérifie la prédiction a priori.
- `test_posterior_dict_fixed_values` — Vérifie les valeurs fixes du dictionnaire postérieur.
- `test_posterior_draws_uses_each_draw` — Vérifie que chaque tirage est utilisé.
- `test_remove_obs_false_drops_observed_site` — Vérifie que le site obs est supprimé si `remove_obs=False`.
- `test_infer_discrete_requires_posterior` — Vérifie que l'inférence discrète nécessite un posterior.
- `test_unfitted_posterior_raises` — Vérifie qu'une erreur est levée si le modèle n'est pas ajusté.
- `test_default_name_x_warns` — Vérifie qu'un avertissement est émis pour le nom par défaut x.
- `test_explicit_seed_is_reproducible` — Vérifie que le seed explicite est reproductible.
- `model` — Définit un modèle linéaire.
- `model` — Définit un modèle avec nom legacy.

### `0.Core/test_sample_api.py`
Ce fichier teste la sémantique de `bf.sample()`.
- `_linear_model` — Définit un modèle linéaire.
- `fitted` — Fixture pour une instance ajustée.
- `test_default_posterior_predictive` — Vérifie la prédiction postérieure par défaut.
- `test_samples_subsample_and_overflow` — Vérifie le sous-échantillonnage et le débordement.
- `test_new_data_predictions` — Vérifie les prédictions sur de nouvelles données.
- `test_prior_predictive` — Vérifie la prédiction a priori.
- `test_posterior_dict_fixed_values` — Vérifie les valeurs fixes du dictionnaire postérieur.
- `test_remove_obs_false_drops_observed_site` — Vérifie que le site obs est supprimé si `remove_obs=False`.
- `test_infer_discrete_requires_posterior` — Vérifie que l'inférence discrète nécessite un posterior.
- `test_unfitted_posterior_raises` — Vérifie qu'une erreur est levée si le modèle n'est pas ajusté.
- `test_default_name_x_warns` — Vérifie qu'un avertissement est émis pour le nom par défaut x.
- `test_explicit_seed_is_reproducible` — Vérifie que le seed explicite est reproductible.
- `model` — Définit un modèle linéaire.
- `model` — Définit un modèle avec nom legacy.

### `0.Core/test_sample_obs_shape.py`
Ce fichier teste que `m.sample(remove_obs=True)` simule les sites observés à la forme des données.
- `_fit` — Ajuste un modèle.
- `_poisson_model` — Définit un modèle de Poisson.
- `_categorical_model` — Définit un modèle catégoriel.
- `_linear_model` — Définit un modèle linéaire.
- `test_scalar_rate_poisson_expands_to_obs_shape` — Vérifie que le taux scalaire de Poisson s'étend à la forme des obs.
- `test_categorical_probs_vector_expands_to_obs_shape` — Vérifie que les probabilités catégorielles s'étendent à la forme des obs.
- `test_batched_likelihood_unchanged` — Vérifie que la vraisemblance batchée est inchangée.
- `model` — Définit le modèle de Poisson.
- `model` — Définit le modèle catégoriel.
- `model` — Définit le modèle linéaire.

### `0.Core/test_save_load.py`
Ce fichier teste la sauvegarde et le chargement d'un objet `bf` ajusté.
- `fitted` — Fixture pour une instance ajustée.
- `test_round_trip` — Vérifie le cycle sauvegarde/chargement.
- `test_load_missing_file_raises` — Vérifie qu'un fichier manquant lève une erreur.
- `test_load_rejects_corrupted_file` — Vérifie qu'un fichier corrompu est rejeté.
- `test_load_rejects_non_bf_object` — Vérifie qu'un objet non-bf est rejeté.
- `test_load_warns_on_version_mismatch` — Vérifie qu'un avertissement est émis en cas de désaccord de version.
- `model` — Définit un modèle linéaire.

### `0.Core/test_summary_factor_sites.py`
Ce fichier teste que `m.summary()` exclut les sites de facteur.
- `_factor_model` — Définit un modèle avec un site de facteur.
- `test_summary_skips_factor_sites` — Vérifie que le résumé ignore les sites de facteur.
- `model` — Définit le modèle avec facteur.

### `0.Core/test_survival_ppc_km_by_cov.py`
Ce fichier teste `survival.ppc_km_by_cov()` pour les PPC KM groupés.
- `_object` — Crée un objet de survie minimal.
- `_posteriors` — Remplit les postérieurs avec des tirages faits main.
- `_panel_titles` — Récupère les titres des panneaux.
- `test_one_figure_per_covariate_plus_risk` — Vérifie qu'une figure est créée par covariable plus le risque.
- `test_subset_and_unknown_covariate` — Vérifie la sous-sélection et les covariables inconnues.
- `test_panels_hold_the_right_patients` — Vérifie que les panneaux contiennent les bons patients.
- `test_posterior_draws_hazard_rate_fallback` — Vérifie le repli sur le taux de risque pour les tirages postérieurs.

### `0.Core/test_survival_sample_times.py`
Ce fichier teste `survival.sample_times()` pour les temps d'événement prédictifs postérieurs.
- `_object` — Crée un objet de survie minimal.
- `test_constant_hazard_is_exponential` — Vérifie qu'un risque constant est exponentiel.
- `test_covariate_halves_the_mean_time` — Vérifie qu'une covariable divise le temps moyen par deux.
- `test_zero_rate_first_interval_delays_every_event` — Vérifie qu'un taux nul au premier intervalle retarde tous les événements.
- `test_zero_hazard_and_draws_and_seed` — Vérifie le comportement avec un risque nul, des tirages et un seed.

# Partie 2 — 1.Distributions (description détaillée)

### `1.Distributions/distribution_numpyro.py`
Ce fichier ne contient pas de contenu de test exploitable (seulement l'en-tête du fichier).

### `1.Distributions/distribution_tfp.py`
Ce fichier ne contient pas de contenu de test exploitable (seulement l'en-tête du fichier).
- `log_error` — Ajoute le nom de la distribution et le message d'erreur dans `log_tfp.txt`.

### `1.Distributions/test_audit_np_dists_major.py`
Ce fichier teste les régressions majeures (M1-M5) pour le module `np_dists` selon l'audit de release.
- `_m` — Crée une instance BayesForge sur CPU avec une graine aléatoire fixe.
- `test_m1_discrete_masked_draws_warn_and_are_float_nan` — Vérifie que les tirages discrets masqués émettent un avertissement et produisent des NaN.
- `test_m1_continuous_masked_draws_no_warning` — Vérifie que les tirages continus masqués ne produisent pas d'avertissement mais génèrent des NaN.
- `test_m3_sine_bivariate_vonmises_honours_mask_and_to_jax` — Vérifie que la distribution von Mises bivariée sinusoïdale respecte le masque et la conversion JAX.
- `test_m4_improper_uniform_sample_raises_clearly` — Vérifie qu'une tentative d'échantillonnage d'une uniforme impropre lève une erreur claire.
- `test_m5_obs_name_cache_is_bounded` — Vérifie que le cache des noms d'observations est borné et fonctionne correctement.

### `1.Distributions/test_audit_np_dists_seed.py`
Ce fichier teste les corrections critiques (C1, C2) liées à la gestion des graines aléatoires dans `np_dists.py`.
- `_draw` — Effectue un tirage normal standard avec les arguments fournis.
- `test_c1_seed_false_draws_differ_with_rng_manager` — Vérifie que les tirages avec `seed=False` diffèrent lorsqu'un gestionnaire RNG est présent.
- `test_c1_seed_false_draws_differ_without_rng_manager` — Vérifie que les tirages avec `seed=False` diffèrent même sans gestionnaire RNG explicite.
- `test_c1_seed_false_different_distributions_not_same_key` — Vérifie que les tirages successifs avec `seed=False` ne réutilisent pas la même clé PRNG.
- `test_c2_int_rand_seed_without_rng_is_a_reproducible_stream` — Vérifie que la graine entière sans RNG produit un flux reproductible entre instances.
- `test_c2_rand_seed_false_without_rng_is_reproducible` — Vérifie que `rand_seed=False` sans RNG produit des résultats reproductibles.
- `test_explicit_int_seed_still_honoured_verbatim` — Vérifie qu'une graine entière explicite est respectée verbatim pour la reproductibilité.

### `1.Distributions/test_audit_tfp_dists.py`
Ce fichier teste les corrections critiques (C1-C8) pour les wrappers TFP en les comparant aux distributions NumPyro correspondantes.
- `_lp` — Calcule la log-probabilité d'une valeur donnée pour une distribution.
- `test_c1_zero_inflated_negative_binomial_builds_and_matches_mixture` — Vérifie la construction et la correspondance de la binomiale négative à inflation zéro avec un mélange.
- `test_c2_gamma_poisson_matches_numpyro` — Vérifie que la log-probabilité de la Gamma-Poisson correspond à celle de NumPyro.
- `test_c2_gamma_poisson_scale_is_inverse_rate` — Vérifie que le paramètre d'échelle est bien l'inverse du taux dans la Gamma-Poisson.
- `test_c3_discrete_uniform_is_a_pmf` — Vérifie que la PMF de l'uniforme discrète correspond à celle de NumPyro.
- `test_c3_discrete_uniform_zero_mass_off_support` — Vérifie que la masse de probabilité est nulle hors du support de l'uniforme discrète.
- `test_c3_discrete_uniform_samples_are_integers_in_range` — Vérifie que les échantillons de l'uniforme discrète sont des entiers dans la plage définie.
- `test_c4_unit_is_a_log_factor_like_numpyro` — Vérifie que la distribution Unit se comporte comme un facteur de log-probabilité comme dans NumPyro.
- `test_c5_inverse_gamma_rate_matches_numpyro` — Vérifie que la log-probabilité de l'inverse Gamma correspond à NumPyro pour plusieurs valeurs.
- `test_c5_inverse_gamma_positional_second_arg_is_rate` — Vérifie que le deuxième argument positionnel de l'inverse Gamma est bien le taux.
- `test_c6_pareto_positional_order_matches_numpyro` — Vérifie que l'ordre des arguments positionnels de Pareto correspond à NumPyro.
- `test_c6_pareto_without_alpha_raises` — Vérifie qu'une erreur est levée si le paramètre alpha manque pour Pareto.
- `test_c7_asymmetric_laplace_quantile_matches_numpyro` — Vérifie que la log-probabilité de la Laplace asymétrique quantile correspond à NumPyro.
- `test_c7_asymmetric_laplace_quantile_rejects_q_outside_open_interval` — Vérifie que le quantile hors de l'intervalle ouvert (0,1) lève une erreur.
- `test_c8_default_seed_draws_differ` — Vérifie que les tirages par défaut avec des graines différentes produisent des résultats différents.
- `test_c8_seed_none_and_out_of_range_ints_do_not_crash` — Vérifie que les graines `None` ou hors plage n'entraînent pas de crash.
- `test_c8_explicit_seed_reproducible_and_session_stream_reproducible` — Vérifie la reproductibilité avec une graine explicite et le flux de session.

### `1.Distributions/test_audit_tfp_dists_major.py`
Ce fichier teste les régressions majeures (M1-M11) pour le module `tfp_dists.py` selon l'audit de release.
- `test_m1_mask_model_mode_zeroes_masked_log_prob` — Vérifie que la log-probabilité est mise à zéro pour les observations masquées en mode modèle.
- `test_m1_mask_sample_mode_nan` — Vérifie que les échantillons masqués sont des NaN et émettent un avertissement.
- `test_m2_scalar_params_vector_obs` — Vérifie que les paramètres scalaires avec des observations vectorielles produisent une somme de log-probabilités.
- `test_m2_multivariate_obs_keeps_event_dim` — Vérifie que les dimensions d'événement sont conservées pour les observations multivariées.
- `test_m2_sample_with_obs_warns` — Vérifie qu'un avertissement est émis si on tente d'échantillonner avec des observations fournies.
- `test_m3_outer_wrapper_carries_user_name` — Vérifie que le nom utilisateur est correctement propagé dans le wrapper externe.
- `test_m4_create_obj_returns_bare_distribution` — Vérifie que `create_obj=True` retourne l'objet de distribution TFP brut.
- `test_m5_no_silent_clamping` — Vérifie que les paramètres invalides ne sont pas silencieusement corrigés (clamping).
- `test_m6_precision_matrix_and_positional_covariance` — Vérifie la cohérence entre matrice de précision et covariance positionnelle pour la normale multivariée.
- `test_m6_scale_identity_multiplier` — Vérifie le comportement du multiplicateur d'identité pour l'échelle de la normale multivariée.
- `test_m7_positional_probs` — Vérifie la correspondance de la log-probabilité avec les probabilités positionnelles.
- `test_m8_gaussian_random_walk_matches_numpyro` — Vérifie que la marche aléatoire gaussienne correspond à l'implémentation NumPyro.
- `test_m9_zero_inflated_poisson_matches_numpyro` — Vérifie que la Poisson à inflation zéro correspond à l'implémentation NumPyro.
- `test_m10_wishart_numpyro_keywords` — Vérifie que les mots-clés Wishart sont correctement interprétés par rapport à NumPyro.
- `test_m11_gamma_default_rate` — Vérifie que le taux par défaut de la distribution Gamma correspond à NumPyro.

### `1.Distributions/test_zip_gate_convention.py`
Ce fichier teste la convention de paramétrage du taux d'inflation zéro pour la distribution Poisson.
- `test_zip_gate_convention_zero_inflation_probability` — Vérifie que le paramètre `gate` correspond bien à la probabilité d'inflation zéro définie.

### `0.Core/libm_julia/repro.py`
Ce fichier définit un modèle Bayesien simple pour comparer les résultats entre Julia et Python via libm.
- `model` — Définit un modèle avec une normale, une exponentielle et une normale conditionnelle.

### `0.Core/libm_julia/repro.jl`
Ce fichier n'est pas un fichier de test Python (pytest) mais un script Julia.

### `0.Core/_report/summary.md`
Ce fichier n'est pas un fichier de test Python (pytest) mais un rapport de vérification.

# Partie 3 — Autres dossiers numérotés (liste des tests)


## 2.Diagnostic_plots


## 3.Documentation

- `doc_blocks.py`
- `test_language_coverage.py` — 1 tests: `test_every_example_in_three_languages`
- `Julia/test_julia_model_pages.py` — 1 tests: `test_model_page_julia_examples_run`
- `Julia/test_julia_start_pages.py` — 1 tests: `test_page_julia_examples_run`
- `Python/test_model_pages.py` — 1 tests: `test_model_page_python_examples_run`
- `Python/test_python_start_pages.py` — 1 tests: `test_page_python_examples_run`
- `R/test_r_model_pages.py` — 1 tests: `test_model_page_r_examples_run`
- `R/test_r_start_pages.py` — 1 tests: `test_page_r_examples_run`

## 3.Model diagnostic.ipynb

- `3.Model diagnostic.ipynb` (fichier unique)

## 5. Nested effect

- `builtin_function.py`

## 5.NBDA.ipynb

- `5.NBDA.ipynb` (fichier unique)

## 6.Network

- `test_audit_network_critical.py` — 4 tests: `test_c1_weighted_betweenness_on_disconnected_graph_matches_networkx`, `test_c2_endpoints_true_runs_and_matches_networkx`, `test_c3_sender_receiver_with_receiver_only`, `test_c4_single_source_dijkstra_on_binary_adjacency`
- `test_audit_network_major_a.py` — 15 tests: `test_m1_public_dijkstra_wrapper_matches_networkx`, `test_m2_component_aware_true_differs_from_false`, `test_m2_directed_path_matches_networkx_receiver_ordering`, `test_m3_clustering_ignores_self_loop_and_stays_in_unit_interval`, `test_m3_asymmetric_matrix_warns_and_stays_in_unit_interval`, `test_m4_density_single_node_is_zero_not_nan`, `test_m4_density_ignores_self_loop_matches_networkx`, `test_m5_symmetric_degree_with_sym_false_warns_sym_true`, `test_m5_symmetric_strength_with_sym_false_warns_sym_true`, `test_m5_degree_sym_true_matches_networkx_and_does_not_warn`, `test_m5_normalize_single_node_is_zero_not_nan`, `test_m5_normalize_directed_total_divides_by_two_n_minus_two`, `test_m6_diameter_raises_value_error_on_disconnected_graph`, `test_m6_diameter_connected_path_still_works`, `test_m3_clustering_asymmetric_warns`
- `test_audit_network_major_b.py` — 8 tests: `test_m7_block_build_mu_ij_rejects_empty_group`, `test_m7_logit_finite_near_one`, `test_m7_block_prior_mean_finite_from_valid_sizes`, `test_m8_mat_to_edgl_rejects_non_square`, `test_m8_mat_to_edgl_rejects_1d_input`, `test_m8_mat_to_edgl_warns_on_nonzero_diagonal`, `test_m8_mat_to_edgl_output_format_unchanged`, `test_m9_betweenness_skipped_above_limit`
- `test_audit_srm.py` — 10 tests: `test_c1_every_group_pair_is_modelled`, `test_c2_block_predictors2_index1_keeps_intercept`, `test_c3_bsrm_model_supports_three_groups_and_two_block_variables`, `test_c4_invalid_outcome_mode_raises`, `test_c4_valid_outcome_mode_has_likelihood`, `test_s3_bad_network_raises`, `test_s3_bad_exposure_raises`, `test_s2_square_dyadic_matrix_is_accepted`, `test_s2_wrong_dyadic_shape_raises`, `test_s1_block_padding_is_finite`
- `test_audit_srm_minor.py` — 10 tests: `test_s1_dat_B_is_bit_identical_to_the_reference_loop`, `test_s1_import_block_predictors_has_no_triple_loop`, `test_s2_block_prior_does_not_ask_for_float64`, `test_s3_core_model_has_no_dead_if_else`, `test_s4_logit_is_finite_at_the_endpoints`, `test_s4_logit_interior_values_unchanged`, `test_s5_labels_must_be_zero_based_contiguous`, `test_s5_valid_labels_still_import`, `test_s6_signed_column_is_not_read_as_a_column_without_groups`, `test_s7_block_importers_document_each_other`
- `Multiplex (WIP)/SIM.py`
- `Multiplex (WIP)/strand_sim.R`
- `NetExplorer/netexplorer.py`
- `NetExplorer/test.ipynb`
- `NetExplorer/test_netexplorer.py` — 8 tests: `test_build_and_payload`, `test_mat_to_edgl_roundtrip`, `test_df_none_defaults_and_unique_ids`, `test_layouts_emit_coords_or_chord`, `test_feature_layer_payload`, `test_scatter_axes_payload`, `test_colour_numeric_gradient_vs_group_sampling`, `test_stroke_defaults_width_1_when_declared`
- `Network edges/print_stan.R`
- `Network edges/run_all.py`
- `Network edges/run_count_directed_zi_small.R`
- `Network edges/scratch_print_stan.R`
- `Network edges/R_scripts/BF_model_binary.R`
- `Network edges/R_scripts/BF_model_count.R`
- `Network edges/R_scripts/BF_model_duration.R`
- `Network edges/R_scripts/Modified simulate_bison_model.R`
- `Network edges/R_scripts/aggregate_kl.R`
- `Network edges/R_scripts/aggregate_timing.R`
- `Network edges/R_scripts/run_binary_directed_no_zi.R`
- `Network edges/R_scripts/run_binary_directed_zi.R`
- `Network edges/R_scripts/run_binary_undirected_no_zi.R`
- `Network edges/R_scripts/run_binary_undirected_zi.R`
- `Network edges/R_scripts/run_count_directed_no_zi.R`
- `Network edges/R_scripts/run_count_directed_zi.R`
- `Network edges/R_scripts/run_count_undirected_no_zi.R`
- `Network edges/R_scripts/run_count_undirected_zi.R`
- `Network edges/R_scripts/run_duration_directed_no_zi.R`
- `Network edges/R_scripts/run_duration_directed_zi.R`
- `Network edges/R_scripts/run_duration_undirected_no_zi.R`
- `Network edges/R_scripts/run_duration_undirected_zi.R`
- `Network edges/R_scripts/util.R`
- `Network manipulations/16. Utility function.ipynb`
- `Network metrics/10.Network metrics.ipynb`
- `Network metrics/network_metrics.py`
- `SRM/SIM.py`
- `SRM/SRM.stan`
- `SRM/SRM2.stan`
- `SRM/STAN2.stan`
- `SRM/benchmark_stan2.py`
- `SRM/fit_comparison.py`
- `SRM/original_srm.stan`
- `SRM/benchmark_suite/STAN2.stan`
- `SRM/benchmark_suite/benchmark_bf_matrix.py`
- `SRM/benchmark_suite/benchmark_bf_matrix_c16.py`
- `SRM/benchmark_suite/benchmark_bf_sharding.py`
- `SRM/benchmark_suite/benchmark_stan2.py`
- `SRM/benchmark_suite/diag_small.py`
- `SRM/benchmark_suite/opt_experiment.py`
- `SRM/benchmark_suite/original_srm.stan`

## 7.Survival analysis

- `Survival_analysis_verify.py`
- `test_audit_surv_minor.py` — 12 tests: `test_u1_posterior_draws_missing_beta_raises_informative_keyerror`, `test_u2_legacy_hazards_uses_key_arguments`, `test_u3_legacy_plot_censoring_uses_time_column_name`, `test_u4_survival_old_construction_keeps_parent_df`, `test_u5_import_summaries_silent_by_default_and_restored_with_verbose`, `test_u6_survival_construction_does_not_set_model_name`, `test_u7_interval_assignment_exact_multiples`, `test_u8_import_covF_accepts_single_string_name`, `test_u8_import_covV_accepts_single_string_name`, `test_u9_n_cov_attributes_initialised_and_synced`, `test_g1_models_gmm_exposes_predict_and_plot_without_leaking`, `test_g2_models_stores_parent_and_wrapper_keeps_identity`
- `tmp.py`

## 8.PCA

- `6.PCA.ipynb`
- `PCA.py`
- `test_audit_pca_minor.py` — 11 tests: `test_p1_get_attributes_raises_value_error`, `test_p1_individual_correlation_raises_value_error`, `test_p2_epsilon_is_documented_as_scale`, `test_p3_scale_sizes`, `test_p4_confidence_ellipse_finite_for_constant_group`, `test_p5_two_component_guards_in_source`, `test_p5_one_component_raises_value_error`, `test_p6_components_uses_show_plot`, `test_p7_call_default_type_matches_constructor`, `test_p8_unknown_type_rejected`, `test_p9_deterministic_sign_matches_old_loop_and_jits`

## 9.Phylogenic analysis

- `run_all.py`
- `Model_1_Simple/brms_fit_simple.R`
- `Model_1_Simple/compare_simple.R`
- `Model_1_Simple/fit_BF_simple.py`
- `Model_2_Poisson/brms_fit_poisson.R`
- `Model_2_Poisson/compare_pois.R`
- `Model_2_Poisson/fit_BF_pois.py`
- `Model_3_Repeated/compare_repeat.R`
- `Model_3_Repeated/fit_BF_repeat.py`
- `Model_3_Repeated/save_L.R`
- `Model_4_Meta/compare_meta.R`
- `Model_4_Meta/fit_BF_meta.py`
- `Model_5_Slopes/compare_slopes.R`
- `Model_5_Slopes/fit_BF_slopes.py`
- `Model_5_Slopes/get_stan_slopes.R`
- `Model_5_Slopes/save_L_slopes.R`
- `Model_5_Slopes/sim_data_effect.R`

## 10.BNN

- `9.BNN.ipynb`
- `run_all.py`
- `test_audit_bnn_critical.py` — 4 tests: `test_crit1_call_before_attention_raises_clear_error`, `test_crit1_call_after_attention_runs`, `test_crit2_cov_offsets_align_with_rows`, `test_crit2_cov_scalar_offsets_still_work`
- `test_audit_bnn_major.py` — 15 tests: `test_maj1_softplus_finite_and_grad_finite_at_extremes`, `test_maj2_exp_clipped_at_80`, `test_maj3_attention_scores_scale_by_sqrt_d_model`, `test_maj4_compound_symmetry_uses_sigma_squared`, `test_maj5_prior_scale_helper`, `test_maj5_cov_prior_scale_traced_scale`, `test_maj5_signatures_default_to_none`, `test_maj6_get_rho_valid_shapes`, `test_maj6_get_rho_n_lt_2_raises`, `test_maj6_get_rho_zero_variance_raises`, `test_maj6_get_rho_bad_ndim_raises`, `test_maj7_npe_bnn_rebound_by_bf_seed`, `test_maj8_swish_activation_resolves`, `test_maj8_unknown_activation_raises_value_error`, `test_maj8_none_activation_is_identity`
- `test_audit_bnn_minor.py` — 18 tests: `test_min1_elu_stable_for_large_negative_input`, `test_min2_make_pd_escalates_jitter_on_non_pd_matrix`, `test_min3_log_space_comments_match_log_variance_names`, `test_min4_layer_linear_rejects_incompatible_shapes`, `test_min5_cov_has_no_eye_matrix_multiplication`, `test_min6_layer_attention_keeps_batch_dim`, `test_min8_cov_prefix_avoids_plate_name_collisions`, `test_min9_sample_flag_documented_in_layer_docstrings`, `test_min10_layer_linear_rejects_bias_none`, `test_min11_layer_linear_dist_documented_as_weights`, `test_min12_mixture_sample_with_name_raises`, `test_min12_mixture_obs_is_never_silently_ignored`, `test_min12_mixture_guard_runs_before_cholesky`, `test_min13_layer_toeplitz_has_docstring`, `test_min14_get_rho_documented`, `test_min15_package_exposes_bnn_class`, `test_min16_layer_diagonal_does_not_store_block_size`, `test_min2_make_pd_cholesky_under_jit`
- `test_mixture_layers.py` — 5 tests: `test_layer_cholesky_batched_lower_triangular_positive_diagonal`, `test_layer_cholesky_diag_offset_and_size_check`, `test_layer_mixture_normal_matches_manual_mixture`, `test_layer_mixture_normal_sampling_is_reproducible_with_model_seed`, `test_layer_mixture_normal_as_likelihood_trained_with_svi_and_sampled`
- `code/BNN covariance.py`
- `code/BNN regressions.py`

## 11.Figures.ipynb

- `11.Figures.ipynb` (fichier unique)

## 11.MGM and DPMM

- `7.MGM & DPM.ipynb`
- `7.MGM_DPM_Similarity_Test.py`
- `Similarity_Test.py`
- `test_audit_models_minor_mix.py` — 14 tests: `test_g1_gmm_validates_data_and_K`, `test_g2_distance_threshold_exposed`, `test_g3_gmm_docstring_label_switching`, `test_g4_plot_gmm_requires_2d_data`, `test_g5_gmm_no_duplicate_imports`, `test_g6_get_cluster_probs_clips_zero_weights`, `test_d1_credible_levels_documented_as_four`, `test_d2_predict_verbose_keyword`, `test_d4_inconsistent_draws_rejected`, `test_d5_dpmm_plot_requires_2d_data`, `test_d6_dpmm_no_duplicate_imports`, `test_b1_bnnc_model_validation`, `test_b2_single_log_epsilon`, `test_b3_bnnc_plot_requires_2d_data`
- `Ellipse_Validation/run_ellipse_simulations.py`

## 12.NBDA

- `density_plots.R`
- `full_comparison.R`
- `install_deps.R`
- `temp_full_ILV.stan`
- `temp_full_OADA.stan`
- `temp_full_OADA_Asocial.stan`
- `temp_full_cTADA.stan`
- `temp_full_complex_f.stan`
- `temp_full_dynamic_tweights.stan`
- `temp_full_posterior_edges.stan`
- `temp_full_veff.stan`
- `test_audit_nbda_critical.py`
- `test_audit_nbda_major_a.py` — 9 tests: `test_m1_import_covnf_asocial_on_fresh_object`, `test_m2_auto_names_one_per_column_and_unique`, `test_m3_locations_stored_per_column`, `test_m4_import_covdf_scale_false_keeps_binary_values`, `test_m5_constant_column_scaled_to_zero_and_nan_input_rejected`, `test_m6_scale_along_time_standardizes_each_series_separately`, `test_m11_covnv_asocial_names_match_columns`, `test_m12_invalid_where_raises`, `test_m13a_import_covnf_accepts_1d_input`
- `test_audit_nbda_major_b.py` — 12 tests: `test_m7_status_with_two_processes_raises`, `test_m7_status_with_one_process_squeezed_and_traces`, `test_m8_network_with_two_networks_raises`, `test_m8_network_with_one_network_builds`, `test_m9_3d_network_expanded_and_traces`, `test_m10_t0_masked_out_of_likelihood`, `test_m13b_model_calls_get_cov_when_not_called_first`, `test_m13c_non_square_network_raises`, `test_m13c_network_n_mismatch_raises`, `test_m13c_network_t_mismatch_raises`, `test_m13c_status_with_value_2_raises`, `test_m13c_status_with_nan_raises`
- `test_audit_nbda_minor.py` — 15 tests: `test_m1_no_identity_comparison_with_literal`, `test_m2_give_name_removed`, `test_m3_network_dim_error_message_lists_accepted_dims`, `test_m4_empty_constructor_raises_value_error`, `test_m5_status_stored_as_n_t_1_and_documented`, `test_m6_status_i_status_j_removed`, `test_m7_covdf_3d_broadcast_identical_values_and_covdv_cost_documented`, `test_m8_ces_is_jitted_staticmethod_same_values`, `test_m9_import_covnv_scale_false_warns_and_no_print`, `test_m10_print_model_scale_and_replacements`, `test_m11_dead_key_test_removed_and_get_cov_unchanged`, `test_m12_import_covdv_docstring_matches_code_axis_order`, `test_m13_ces_documented_as_linear_mixture`, `test_m14_nan_sentinel_documented`, `test_m15_module_docstring_documents_numpyro_backend`
- `BF_Models/model_ILV.R`
- `BF_Models/model_OADA.R`
- `BF_Models/model_OADA_asocial.R`
- `BF_Models/model_cTADA.R`
- `BF_Models/model_complex_f.R`
- `BF_Models/model_dynamic_tweights.R`
- `BF_Models/model_posterior_edges.R`
- `BF_Models/model_veff.R`

## 13.Parallel

- `conftest.py`
- `test_audit_parallel_major.py` — 11 tests: `test_m1_oversubscription_warns_under_affinity_limit`, `test_m1_plan_caps_on_affinity_not_cpu_count`, `test_m1_pinning_blocks_cut_from_the_affinity_set`, `test_m2_workers_1_warns_and_honours_quiet`, `test_m2_explicit_serial_does_not_warn`, `test_m3_duplicate_sim_raises`, `test_m4_dead_worker_keeps_other_results`, `test_m5_interrupt_message_reports_rows`, `test_m6_manager_failure_message_names_the_guard`, `test_m7_gpu_env_round_robin`, `test_m7_visible_gpus_reads_env`
- `test_audit_parallel_minor.py` — 22 tests: `test_m1_pin_worker_block_overlap_documented`, `test_m2_merge_xla_flags_preserves_other_user_flags`, `test_m2_env_override_keeps_user_xla_flags`, `test_m2_spawn_child_inherits_merged_xla_flags`, `test_m3_devnull_handle_is_closed_at_exit`, `test_m3_worker_stdout_exported_from_simulate`, `test_m4_string_scenario_rejected_with_element_index`, `test_m4_non_int_reps_rejected`, `test_m5_scenario_order_is_deterministic`, `test_m5_run_simulations_rows_follow_sorted_order`, `test_m6_reserved_column_names_recorded_not_raised`, `test_m6_reserved_column_names_raise_under_on_error_raise`, `test_m7_check_int_type_and_range_errors`, `test_m7_plan_errors_are_typed`, `test_m8_parallel_takes_env_and_block_size_as_required_params`, `test_m8_run_simulations_passes_plan_values_to_parallel`, `test_m9_display_closed_on_base_exception_path`, `test_m9_keyboard_interrupt_propagates_and_leaves_no_worker_pools`, `test_m10_manager_failure_falls_back_to_no_progress`, `test_m11_meta_columns_come_first`, `test_m12_multiindex_frame_rejected`, `test_m12_duplicate_columns_rejected`
- `test_audit_parallel_record.py` — 1 tests: `test_c1_unpicklable_result_is_recorded_not_fatal`
- `test_audit_parallel_x.py` — 4 tests: `test_x2_serial_records_unpicklable_result`, `test_x2_process_records_unpicklable_result`, `test_x5_worker_env_disables_prealloc`, `test_x5_worker_env_keeps_user_value`
- `test_run_simulations.py`

## 14.Jax extension (WIP)

- `Plots.ipynb`
- `test_sampleddata.py` — 112 tests: `test_property_delegation`, `test_method_delegation_no_args`, `test_method_delegation_with_args`, `test_method_chaining`, `test_dtype_and_astype`, `test_arithmetic_operators`, `test_reflected_operators`, `test_matmul_operator`, `test_comparison_operators_return_bool_arrays`, `test_unary_operators`, `test_getitem_slicing`, `test_getitem_single_element`, `test_boolean_mask_indexing`, `test_fancy_indexing`, `test_setitem_raises_like_jax`, `test_at_set_returns_new_wrapper`, `test_at_add_returns_new_wrapper`, `test_jnp_reductions_on_object`, `test_jnp_elementwise_on_object`, `test_numpy_ufunc_on_object`, `test_jnp_dot_on_object`, `test_jnp_where_with_object`, `test_jnp_concatenate_and_stack`, `test_broadcasting_with_raw_array`, `test_numpy_conversion`, `test_jnp_asarray_and_to_jax`, `test_tree_flatten_unflatten_roundtrip`, `test_tree_map_preserves_wrapper`, `test_jit_takes_object_directly`, `test_jit_returns_wrapper`, `test_grad_through_object`, `test_vmap_over_object`, `test_lax_scan_with_wrapper_carry`, `test_lax_cond_with_wrapper`, `test_pmap_over_object`, `test_not_an_instance_of_jax_ndarray`, `test_isinstance_self_identity_preserved`, `test_type_keyed_pytree_still_works`, `test_pickle_roundtrip`, `test_cloudpickle_roundtrip`, `test_sampled_data_as_distribution_loc`, `test_sampled_data_as_distribution_log_prob`, `test_chained_sampling_with_wrapper`, `test_repr_shows_values_not_pointer`, `test_str_and_print_show_values`, `test_repr_reports_dtype`, `test_repr_html_shows_values`, `test_repr_pretty_shows_values`, `test_ipython_text_plain_shows_values`, `test_ipython_real_jax_array_unaffected`, `test_distribution_sample_true_returns_printable_sampleddata`, `test_uncovered_arithmetic_operators_stay_wrapped`, `test_bitwise_binary_operators`, `test_invert_operator`, `test_reflected_bitwise_operators`, `test_remaining_comparison_operators`, `test_len_iter_contains`, `test_scalar_casts`, `test_bool_on_multi_element_raises_like_jax`, `test_zero_dim_wrapper`, `test_empty_wrapper`, `test_nested_construction_does_not_double_wrap`, `test_item_and_tolist`, `test_block_until_ready`, `test_device_attributes`, `test_device_put`, `test_internal_jax_attributes`, `test_np_save_load_roundtrip`, `test_value_and_grad_through_object`, `test_jacobian_and_hessian_through_object`, `test_grad_of_wrapper_nested_in_a_pytree`, `test_eval_shape_with_wrapper`, `test_aot_lower_and_compile`, `test_jit_cache_is_reused_for_same_shape`, `test_fori_loop_with_wrapper_carry`, `test_while_loop_with_wrapper_carry`, `test_vmap_with_mixed_in_axes`, `test_tree_map_over_two_wrappers`, `test_tree_map_wrapper_and_raw_array_mismatch`, `test_copy_and_deepcopy`, `test_pickle_inside_a_container`, `test_pickle_preserves_dtype`, `test_hash_raises_like_a_jax_array`, `test_dtype_preserved_in_both_precision_modes`, `test_explicit_dtype_survives_x64`, `test_weak_type_promotion_matches_raw`, `test_wrapped_path_is_bitwise_identical_to_raw_path`, `test_same_key_gives_identical_samples_through_the_wrapper`, `test_computations_bitwise_identical_wrapped_vs_raw`, `test_sampled_data_as_scale_parameter`, `test_sampled_data_as_multivariate_covariance`, `test_sampled_data_as_mcmc_observation`, `test_mcmc_result_identical_wrapped_vs_raw_obs`, `test_setitem_matches_jax_immutability`, `test_at_update_does_not_touch_aliases`, `test_at_set_promotes_dtype_like_jax`, `test_at_update_is_invisible_to_a_cached_jit`, `test_numpy_array_on_the_left_falls_back_to_numpy`, `test_divmod_matches_jax`, `test_format_spec_matches_jax`, `test_array_protocol_accepts_copy_keyword`, `test_no_class_spoof_contract`, `test_hdi_returns_narrowest_interval`, `test_hdi_is_tighter_than_the_full_range`, `test_hdi_on_a_normal_sample_brackets_the_mean`, `test_plot_methods_accept_1d`, `test_plot_methods_accept_2d`, `test_plot_methods_static_backend`, `test_plot_methods_reject_1d_with_a_clear_error`, `test_surface_3d_accepts_2d`, `test_ppc_plot_runs`, `test_ridgeline_interactive_runs`

## 14.SampledData.py

- `14.SampledData.py` (fichier unique)

## 15.Diagnostics

- `diag_arviz.py`
- `test_audit_diag_critical.py` — 3 tests: `test_bug1_tfp_sample_stats_are_named_in_trace_fn_order`, `test_bug1_mismatched_stat_names_raise`, `test_bug2_sbi_pseudo_chains_warn`
- `test_audit_diag_major.py` — 6 tests: `test_bug3_single_chain_rhat_warns`, `test_bug5_ess_raw_guards`, `test_bug6_explicit_has_chains_no_warning`, `test_bug6_heuristic_warns`, `test_bug13_shard_check_fail_closed`, `test_bug24_summary_exclude`
- `test_audit_diag_minor.py` — 46 tests: `test_filter_posterior_dict_include`, `test_filter_posterior_dict_exclude`, `test_filter_posterior_dict_unknown_name_returns_empty`, `test_ess_tail_prob_changes_result`, `test_ess_tail_default_prob_matches_explicit`, `test_ess_prob_ignored_for_other_kinds`, `test_ess_tail_1d_accepts_prob`, `test_loo_default_reff_equals_one`, `test_relative_eff_returns_finite_float`, `test_loo_accepts_relative_eff_output`, `test_dict_posterior_num_chains_from_leading_dim`, `test_dict_posterior_rejects_bad_input`, `test_diagwip_dict_infers_num_chains`, `test_diagwip_rejects_sampler_chain_mismatch`, `test_to_az_dict_posterior_smoke`, `test_to_az_preserves_chain_dimension`, `test_to_az_tfp_smoke`, `test_to_az_tfp_preserves_chain_dimension`, `test_to_az_unknown_backend_raises`, `test_to_az_tfp_rejects_short_sample_stats_name`, `test_to_az_tfp_rejects_var_posterior_mismatch`, `test_to_az_tfp_accepts_matching_stat_names`, `test_acf_constant_input_is_delta`, `test_acf_matches_manual_formula`, `test_ebfmi_per_chain_iid_is_about_two`, `test_ebfmi_per_chain_zero_variance_is_nan`, `test_ebfmi_per_chain_is_per_chain_not_pooled`, `test_diag2_ebfmi_manual_delegates_to_canonical`, `test_patch_diag_autocor_uses_canonical_acf`, `test_diag2_autocor_uses_canonical_acf`, `test_sensitivity_energy_plot_uses_canonical_ebfmi`, `test_pair_seed_is_a_parameter`, `test_plot_pair_same_seed_same_subsample`, `test_plot_pair_different_seed_different_subsample`, `test_pair_smoke_returns_figure`, `test_pair_jax_smoke_returns_figure`, `test_treedepth_reported_for_nuts`, `test_treedepth_saturation_reported_for_nuts`, `test_treedepth_skipped_for_plain_hmc`, `test_treedepth_unavailable_without_extra_fields`, `test_diagnose_var_names_filters_report`, `test_diagnose_without_var_names_reports_all`, `test_diagnose_var_names_unknown_raises`, `test_package_docstring_names_public_api`, `test_legacy_diag_class_marked_deprecated`, `test_diag2_module_marks_patch_deprecated`
- `test_audit_diag_plots.py` — 4 tests: `test_bug22_legacy_plots_run`, `test_bug22_single_parameter_model`, `test_bug22_ignored_option_warns`, `test_bug23_plot_compare_runs`
- `test_diag_az_summary.py` — 8 tests: `test_summary_matches_az_summary_values`, `test_summary_median_column_present_and_correct`, `test_summary_var_names_filtering`, `test_summary_exclude_vars_filtering`, `test_diagnose_single_chain_lists_rhat_not_computable`, `test_diagnose_two_chains_reports_no_rhat_problem`, `test_jax_diagnostics_rhat_single_chain_nan`, `test_diag2_rhat_manual_single_chain_nan`
- `test_summary_quiet_warnings.py` — 7 tests: `test_helper_swallows_invalid_and_zero_divide`, `test_helper_passes_unrelated_warnings_through`, `test_helper_restores_state_afterwards`, `test_summary_constant_chains_quiet_with_nan_kept`, `test_summary_nan_draws_quiet_and_nan_output`, `test_rhat_ess_constant_chains_quiet_and_nan_kept`, `test_summary_round_to_and_kind_still_keywords`
- `API_Routing/test_diag_backend.py` — 15 tests: `test_arviz_is_the_default`, `test_constructing_without_diag_backend_gives_arviz`, `test_resolve_backend`, `test_resolve_backend_rejects_nonsense`, `test_summary_returns_an_arviz_table`, `test_summary_jax_backend_warns_and_works`, `test_summary_jax_helper_warns`, `test_backends_agree_on_summary`, `test_diag_metrics_use_arviz`, `test_diag_metric_values_are_sane`, `test_diag_loo_and_waic_run`, `test_jax_backend_binding_warns_and_switches`, `test_backends_agree_through_the_live_api`, `test_plots_and_ppc_are_backend_independent`, `test_diagnose_still_available`
- `Integration_Tests/diagnose_sampling/__init__.py`
- `Integration_Tests/diagnose_sampling/diagnose.py`
- `Integration_Tests/personality_traits/traits.py`
- `Integration_Tests/ppc/test_ppc.py`
- `Integration_Tests/regression/test_regression.py`
- `Integration_Tests/sensitivity/test_sensitivity.py`
- `Integration_Tests/summary/diag test.py`
- `Integration_Tests/summary/jax_diagnostics.py`
- `Integration_Tests/summary/patch_diag.py`
- `Parity_Tests/test_audit_diag_bug4.py` — 2 tests: `test_short_chain_ess_rhat_finite_and_match_arviz`, `test_three_draws_still_nan_like_arviz`
- `Parity_Tests/test_diag_parity.py` — 15 tests: `test_same_parameter_names`, `test_public_rhat_matches`, `test_public_ess_matches`, `test_metric_methods_are_reusable`, `test_manual_rhat_matches_arviz_rank`, `test_manual_rhat_detects_variance_nonconvergence`, `test_manual_ess_bulk_matches_arviz`, `test_manual_ess_tail_matches_arviz`, `test_manual_ebfmi_matches_arviz_bfmi`, `test_summary_mean_sd_match`, `test_summary_uses_real_hdi_not_percentiles`, `test_loo_matches`, `test_waic_matches`, `test_waic_scale_transforms_elpd_and_se`, `test_loo_runs`
- `Parity_Tests/test_jax_diagnostics_parity.py` — 22 tests: `test_rhat_matches_arviz_rank`, `test_rhat_detects_variance_nonconvergence`, `test_rhat_dict_api`, `test_ess_matches_arviz`, `test_ess_mean_and_sd_match_arviz`, `test_summary_column_matches_arviz`, `test_summary_reports_rhat_for_a_single_chain`, `test_hdi_matches_arviz`, `test_hdi_returns_double_precision`, `test_waic_matches_arviz`, `test_loo_matches_arviz`, `test_loo_pareto_k_matches_arviz_when_reff_is_supplied`, `test_relative_eff_is_a_fraction`, `test_waic_scales`, `test_compare_elpd_diff_sign_matches_arviz`, `test_compare_rejects_mismatched_observation_counts`, `test_filter_posterior_dict_matches_base_names`, `test_filter_posterior_dict_does_not_match_prefixes`, `test_ess_and_mcse_accept_flat_posteriors`, `test_ess_and_mcse_accept_flat_vector_posteriors`, `test_mcse_matches_arviz`, `test_multidim_parameters_are_expanded`

## SBI (WIP)

- `test_audit_samplers_major_sbi.py` — 4 tests: `test_m7_mcmc_numpyro_seed_warns_and_default_is_silent`, `test_m8_abc_kernel_covariance_jitter_survives_float32`, `test_m10_npe_validates_hyperparameters_before_simulating`, `test_m12_svi_get_samples_before_run_raises`
- `test_audit_sbi_minor.py` — 21 tests: `test_15_nan_summary_stats_raise_instead_of_silent_garbage`, `test_15_few_nan_summaries_warn_and_are_excluded`, `test_16_regression_adjust_survives_eps_zero`, `test_17_smc_round0_stays_within_declared_budget`, `test_18_constructor_validates_quantile_epsilon_alpha_rounds`, `test_19_callable_distance_clear_errors`, `test_20_mahalanobis_survives_degenerate_reference_stats`, `test_21_forest_seed_derives_from_run_key_random_state_warns`, `test_22_non_finite_s_obs_rejected`, `test_23_reference_table_and_oob_consistent_constant_column_finite`, `test_24_qrf_weights_processed_in_row_chunks`, `test_25_run_splits_key_once_no_reuse`, `test_26_npe_constructor_validates_lr_batch_prior_scale`, `test_27_log_q_u_shape_errors_are_explicit`, `test_28_sample_u_documents_fold_in_correlation`, `test_29_sanitize_log_density_reports_plus_inf_inversion`, `test_30_mutating_data_invalidates_jit_caches`, `test_31_zero_batch_raises_explicit_value_error`, `test_32_dead_sbi_observed_flag_removed`, `test_33_weighted_resample_rejects_bad_weights`, `test_34_sbi_posterior_pseudo_chains_and_dict_contract`
- `test_sbi.py` — 7 tests: `test_abc_rejection_recovers_analytic_posterior`, `test_abc_smc_recovers_analytic_posterior_and_respects_budget`, `test_abc_with_tracked_output_and_nuisance_latents`, `test_npe_recovers_analytic_posterior`, `test_npe_amortised_and_constrained`, `test_bayesian_npe_is_wider_than_npe_with_small_budget`, `test_abc_rf_recovers_analytic_posterior_and_ignores_noise_stat`
- `g-and-k/1.ABC.py`
- `g-and-k/2.ABC random forest.py`
- `g-and-k/3.NPE.py`
- `g-and-k/4.Bayesian NPE.py`
- `g-and-k/Utils.py`
- `g-and-k/run_all.py`

## 17.Repertoire

- `run_all.py`
- `scripts/BF_only.py`
- `scripts/benchmark.py`
- `scripts/check_stan.py`
- `scripts/compare_moments_p.py`
- `scripts/compare_summary.py`
- `scripts/generate_data.r`
- `scripts/save_stan_samples.py`

## 18.EWA

- `run_comparison.py`
- `BF_backend/model.py`
- `Stan/PN_social_global_age.stan`
- `Stan/temp_PN_social_global_age.stan`
