# Ordre de production des fichiers de sortie

Le pipeline produit les fichiers dans `output_dir/` dans l'ordre suivant :

```
prompts.json → raw_results.json → <model>_summary.json → model_comparison.json → full_results.tsv → errors_long_format.tsv
```

| Ordre | Fichier | Étape du pipeline | Description |
|-------|---------|-------------------|-------------|
| 1 | `prompts.json` | Step 1 — Data Loading | Prompts d'entrée et continuations de référence |
| 2 | `raw_results.json` | Step 4 — Annotation | Toutes les données accumulées : générations, corrections GEC, annotations ERRANT |
| 3 | `<model>_summary.json` | Step 5 — Analysis | Métriques par modèle (PPL, taux d'erreur, types d'erreurs) |
| 4 | `model_comparison.json` | Step 5 — Analysis | Comparaison inter-modèles et tests statistiques |
| 5 | `full_results.tsv` | Step 5 — Analysis | CSV principal — 1 ligne par phrase, tous modèles côte à côte |
| 6 | `errors_long_format.tsv` | Step 5 — Analysis | CSV des erreurs — 1 ligne par erreur (pour tableaux croisés / R) |

## Étapes du pipeline

```
Step 1: Data Loading       → prompts.json
Step 2: Generation         → (résultats en mémoire)
Step 3: GEC Correction     → (résultats en mémoire)
Step 4: ERRANT Annotation  → raw_results.json
Step 5: Analysis & Export  → *_summary.json, model_comparison.json, full_results.tsv, errors_long_format.tsv
```
