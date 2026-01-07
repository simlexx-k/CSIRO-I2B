  ### New Pillar/Checkpoint Sources in cisro-baseline-train-infer-21-12-dinov3-siglip.ipynb

  | Path | Purpose (in that notebook) | Do we already use it? |
  | --- | --- | --- |
  | /kaggle/input/5-folds-dinov3-840m/other/fold-0-1-2/1 and /2 | Dinov3-trained fold checkpoints (840M “huge+” backbone) – they ensemble multiple fold groups. | No. We currently only load /kaggle/input/csiro/pytorch/default/12
  (their Dinov2 CrossPVT) and our own Dinov2-L pillar. |
  | /kaggle/input/dinov2/pytorch/giant/1 | Dinov2-G (bigger than our L). | No – our notebook mounts /kaggle/input/dinov2/pytorch/large/1. |
  | /kaggle/input/vit-huge-plus-patch16-dinov3-lvd1689m/vit_huge_plus_patch16_dinov3.lvd1689m_backbone.pth | Raw Dinov3 “huge+” backbone weights (used to initialize their pillar). | No. |
  | /kaggle/input/vit-large-patch16-dinov3-lvd1689m-backbone-pth/vit_large_patch16_dinov3.lvd1689m_backbone.pth | Raw Dinov3 “large” weights. | No. |
  | (You mentioned) /kaggle/input/dinov3/keras/dinov3_vit_huge_plus_lvd1689m/1 | Another packaging of the same Dinov3 huge+ weights (Keras export). | Not in our notebook; baseline didn’t reference it directly but we can use it
  if preferred. |

  ### Shared Inputs (both notebooks already use)

  - /kaggle/input/csiro-biomass and /kaggle/input/csiro-datasplit/csiro_data_split.csv
  - /kaggle/input/google-siglip-so400m-patch14-384/transformers/default/1
  - MVP checkpoints (/kaggle/input/csiro-mvp-models) – we already have our MVP pillar.

  ### Summary

  The extra assets we don’t currently consume are:

  1. Dinov3 fold checkpoints: /kaggle/input/5-folds-dinov3-840m/other/fold-0-1-2/1 (and /2). These are ready-to-use models; we could integrate them as an additional pillar instead of training Dinov2-L ourselves.
  2. Dinov2-G weights: /kaggle/input/dinov2/pytorch/giant/1. Upgrading from “large” to “giant” should give stronger representations if we keep the rest of our pipeline.
  3. Dinov3 backbone weights: /kaggle/input/vit-huge-plus-patch16-dinov3-lvd1689m/... and /kaggle/input/vit-large-patch16-dinov3-lvd1689m-backbone-pth/.... These are required if we want to re-create their Dinov3 pillar from
     scratch (or fine‑tune a custom one). The Keras-format package you mentioned (/kaggle/input/dinov3/keras/dinov3_vit_huge_plus_lvd1689m/1) is yet another way to load the same core weights.