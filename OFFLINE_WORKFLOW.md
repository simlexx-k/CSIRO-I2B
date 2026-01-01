## Offline Training & Inference Workflow

Kaggle scoring now takes several hours end-to-end, so the recommended workflow is:

1. **Train each pillar locally or on a fast GPU box.**
   - `SigLIP` (stacked regressors in `dino_vit_solution.ipynb`)
   - `DINO` checkpoints (`submission3.csv`)
   - `MVP` checkpoints (`submission2.csv`)
   - `Dinov2-L` embeddings + regressors (new pillar)
2. **Export the pillar predictions as Kaggle-style CSVs** (`sample_id`, `target`).
3. **Upload the CSVs to Kaggle Datasets** (or copy them into the notebook input folder).
4. **Run the Kaggle notebook in inference mode only**: load the pillar submissions, blend with the desired weights (default SigLIP-heavy 0.60/0.20/0.10/0.10), and skip retraining.

### Producing Pillar Submissions

| Pillar | Output | Notes |
| --- | --- | --- |
| SigLIP stack | `submission_siglip.csv` | Run the stacked regressors offline; store both OOF metrics and final submission. |
| DINO | `submission3.csv` | Already produced by `run_dino_inference()`; reuse the best checkpoint run. |
| MVP | `submission2.csv` | Ensemble already configured; reuse exported CSVs. |
| Dinov2 | `submission_dinov2.csv` | Run the Dinov2 embedding/regressor notebook locally, save the predictions. |

Upload the four CSVs to Kaggle (public or private dataset) and attach them to the competition notebook.

### Blending Offline

Use the new blender script to combine the four submissions:

```bash
python scripts/inference_blend.py \
  --siglip /kaggle/input/pillar-submissions/submission_siglip.csv \
  --dino /kaggle/input/pillar-submissions/submission3.csv \
  --mvp /kaggle/input/pillar-submissions/submission2.csv \
  --dinov2 /kaggle/input/pillar-submissions/submission_dinov2.csv \
  --weights 0.6 0.2 0.1 0.1 \
  --clip-min 0 \
  --output submission.csv
```

The script validates the column structure, normalizes the weights, and writes the final `submission.csv`. Adjust weights as needed when experimenting (e.g., more SigLIP-heavy mixes).

### Notebook Tips

- Add a lightweight cell that calls the blender script (or re-implements the same logic) so the Kaggle notebook does not retrain anything—just loads the uploaded CSVs and blends them.
- Keep logging each Kaggle run in `run_logs/` for traceability (follow the existing `scores_YYYY-MM-DD_label.csv` pattern).
- If you must re-train online, gate those cells behind a flag so the default execution path remains inference-only.
