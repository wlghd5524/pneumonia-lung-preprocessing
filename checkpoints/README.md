# Checkpoint placement

Large weight files are intentionally not committed to Git.

Place the MedSAM3 LoRA weights at:

`checkpoints/best_lora_weights.pt`

Expected SHA256:

`499e638bb7c51dbe0dcc3bfb9dbfada74fc2d725e953fbb5bdb2dd1b72106f91`

Place the EVA-X weights at:

`eva-x/eva_x_base_patch16_merged520k_mim.pt`

Expected SHA256:

`35dc7012c1a49cba01f5a31a8d777db39258078c85cfad3f3c9a41deb172e1b4`

Fine-tuned classifier checkpoints retain the training output layout:

`<results_root>/<run_name>/fold_<1-5>/best_model.pth`

`trained_checkpoint_manifest.csv` contains the file size and SHA256 for all
350 paper checkpoints (70 canonical runs × 5 folds). Its own SHA256 is:

`1997eb174087c69df2d70db32055f3f3dc5962c100bc4928b3dd38c177a2cd6b`

The fold-specific trained checkpoints are not currently redistributed through
this public repository. `trained_checkpoint_manifest.csv` records sizes and
checksums only. Regenerate that list from a local results folder with:

```bash
python tools/export_checkpoint_manifest.py \
  --results-root /path/to/results_pneumonia \
  --runs-file results/primary/canonical_70runs.txt \
  --output checkpoints/trained_checkpoint_manifest.csv
```
