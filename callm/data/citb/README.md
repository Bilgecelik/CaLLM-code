# CITB Data Utilities

This directory contains scripts for preparing CITB-style continual-learning streams and optional synthetic data. Raw task JSONs and generated data are not bundled in this public release.

Expected local layout:

```text
callm/data/citb/tasks/<task_name>.json
callm/data/citb/generated_data/CITB/<version>/<task_name>.json
```

Generate synthetic data:

```bash
python -m callm.data.citb.gen_synthetic_data_citb --n_shots 2 --permutations 80
```

Remove duplicates:

```bash
python -m callm.data.citb.postprocess_synthetic_data_citb
```

Select high-quality synthetic examples:

```bash
python -m callm.data.citb.text2emb
python -m callm.data.citb.kmeans_for_postprocess
```

Synthetic data version notes:

- `v0.4_kmeans_postprocess`: Llama 3.1 8B, 50 permutations, 2 shots, 600 examples per task
- `v0.5_kmeans_postprocess`: Llama 3.1 8B, 80 permutations, 2 shots, 1600 examples per task
- `v0.6_kmeans_postprocess`: Gemma 3 12B, 80 permutations, 2 shots, 3000 examples per task
- `v0.6_kmeans_postprocess_1600`: Gemma 3 12B, 80 permutations, 2 shots, 1600 examples per task
