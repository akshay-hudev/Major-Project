# GTAN on Colab GPU — quick handoff

Train the production DL model (**Gated Temporal Attention Network**, a
Temporal-Fusion-Transformer-style architecture) as a 5-seed deep ensemble on a
free Colab **T4 GPU**. Wall-clock ~8–12 min vs ~80–90 min on this CPU.

## Files
- `gtan_bundle.zip` (9 MB) — the preprocessed arrays + `feature_names.pkl` +
  `train_gtan.py` (the exact same model source as the local repo, so there's no
  code divergence between local and Colab).
- `GTAN_Colab_GPU.ipynb` — the notebook to run.

## Steps
1. Open [colab.research.google.com](https://colab.research.google.com) →
   **File → Upload notebook** → pick `GTAN_Colab_GPU.ipynb`.
2. **Runtime → Change runtime type → GPU (T4)**.
3. **Cell 1**: run it, click *Choose Files*, upload `gtan_bundle.zip`.
4. Run **cells 2 → 5** in order. Cell 4 does the training.
5. Cell 5 downloads `gtan_outputs.zip`.

## Bring results back
Unzip `gtan_outputs.zip` into the repo:
- `improved_probs/gtan_*.npy` → `repo/data/preprocessed/improved_probs/`
- `results_improved.json` → merge/replace `repo/results_improved.json`

Then locally run the combined evaluator to fold GTAN into the ensemble and print
the final baseline-vs-improved comparison:
```bash
python src/model_training/evaluate_improved.py
```

## Notes
- The notebook calls `train_gtan.main()` directly after repointing its path
  globals to `/content/proj`, so the model definition and training recipe are
  identical to local — GPU only changes speed, not results (up to seed/backend).
- To scale up on GPU (it's cheap there): raise `--seeds` or `--epochs` in Cell 4.
