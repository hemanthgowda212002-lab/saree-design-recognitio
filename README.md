# ColorID — Colour-Invariant Saree Design Recognition

Identify a saree by the **design on its surface**, independent of the colour palette it is rendered in ("face recognition for textiles"). PyTorch, metric learning, runs on a free Kaggle GPU.

> **Approach note (<=500 chars)** — see [`APPROACH_NOTE.md`](APPROACH_NOTE.md). Evaluation protocol — see [`docs/EVALUATION.md`](docs/EVALUATION.md).

## Idea in one picture

```
RGB image ─► colour-agnostic stem ─► MobileNetV3-L ─► GeM ─► Linear ─► BNNeck ─► L2-norm ─► 256-d embedding
             (luminance + local        (ImageNet-       pooling                         cosine similarity
              contrast norm, fixed)     pretrained)                                     = same design?
```

Colour invariance comes from **three independent mechanisms**, so no single one has to be perfect:

1. **Architecture/input** — a fixed, parameter-free stem converts to luminance and applies local contrast normalisation, removing palette brightness/contrast while keeping motif structure (`--stem lcn`; `lcn_edge` adds gradient magnitude and |LCN| for light-on-dark polarity invariance).
2. **Data** — every view gets random palette augmentation: hue shift, channel permutation, random per-channel tone-curve *gradient maps* of the luminance, grayscale, inversion. Two views of one image are therefore a synthetic "same design, new colourway" pair, so designs with a single real colourway still train.
3. **Loss/sampling** — batches contain P designs x K colourways; ArcFace (global identity margin) + supervised contrastive loss (in-batch positives across colourways, negatives incl. identical-palette different designs).

Different motifs in the same palette are separated because the stem keeps structure and the losses push different design ids apart.

## Repo layout

```
colorid/          manifest.py (scan/split/sampler)  transforms.py (palette augs)  model.py  losses.py
                  data.py  train.py  evaluate.py  metrics.py  efficiency.py  infer.py  synthetic.py
scripts/          build_manifest.py   run_ablation.sh
tests/            unit + end-to-end smoke tests (CPU, ~10 s)
notebooks/        kaggle_runner.ipynb
docs/             EVALUATION.md
```

## Quick start (no data needed)

```bash
pip install -r requirements.txt && pip install -e .
pytest -q                                              # unit + end-to-end smoke test
python -m colorid.synthetic --out data/synthetic       # procedural designs x palettes
python -m colorid.train --manifest data/synthetic/manifest.csv --out runs/demo \
       --backbone mobilenetv3_small_050 --no_pretrained --img_size 64 --epochs 20 -P 8 -K 4 --workers 0
python -m colorid.evaluate --ckpt runs/demo/best.pt --manifest data/synthetic/manifest.csv --split test
```

## Real data

```bash
# 1. scan each source (design id = first sub-folder; use --mode regex if encoded in file names)
python scripts/build_manifest.py scan --root /path/deeplure --source deeplure --mode parent --out m_deeplure.csv
# optional extra data without design ids: every image its own design, train split only
python scripts/build_manifest.py scan --root /kaggle/input/indian-saree-patterns --source kaggle \
       --mode instance --train-only --out m_kaggle.csv
# 2. merge + split BY DESIGN
python scripts/build_manifest.py merge --inputs m_deeplure.csv m_kaggle.csv --out manifest.csv

# 3. train / evaluate / profile
python -m colorid.train      --manifest manifest.csv --out runs/main --epochs 30
python -m colorid.evaluate   --ckpt runs/main/best.pt --manifest manifest.csv --split test --out results/test.json
python -m colorid.efficiency --ckpt runs/main/best.pt --out results/efficiency.json
bash scripts/run_ablation.sh manifest.csv --epochs 15      # stem / loss / pretraining ablations
```

Inference:

```bash
python -m colorid.infer build-gallery --ckpt runs/main/best.pt --manifest manifest.csv --split test --out gallery.npz
python -m colorid.infer search --ckpt runs/main/best.pt --gallery gallery.npz --query q.jpg --topk 5
python -m colorid.infer verify --ckpt runs/main/best.pt --a a.jpg --b b.jpg     # uses the val-calibrated threshold
```

## Pipeline

| Stage | Details |
|---|---|
| Pre-processing | resize shorter side to 256, centre crop 256, RGB in [0,1]; stem (inside the model): luminance -> local contrast norm |
| Train augmentations | RandomResizedCrop(0.35-1), h/v flip, palette augs (above), brightness/contrast/saturation jitter, random erasing |
| Training | AdamW (backbone 3e-4, head x10), cosine + warm-up, AMP, ArcFace(s=30, m=0.3) + SupCon(T=0.1), P=16 x K=4 x 2 views |
| Post-processing | flip-TTA average, L2-norm, cosine similarity, gallery images collapsed to designs by best colourway; verification threshold = EER point on validation |

## Results

> Fill in after running on the real data (numbers below are intentionally empty — nothing here is fabricated).

| Split | R@1 (random) | R@1 (hard) | mAP | Verif. AUC (hard) | EER (hard) | Synthetic-recolour R@1 |
|---|---|---|---|---|---|---|
| test | — | — | — | — | — | — |

Ablations (`scripts/run_ablation.sh`): — / Efficiency (`results/efficiency.md`): —

## Data, disclosure and compliance

* **Pretrained weights**: `mobilenetv3_large_100` ImageNet weights via [timm](https://github.com/huggingface/pytorch-image-models). No other external checkpoints or data.
* **Datasets**: DeepLure saree corpus (proprietary, *not included, not redistributed, local copy deleted after the exercise*); Indian Saree Patterns (Kaggle, public). How labels/splits were derived is printed by `build_manifest.py` — document any cleaning you did here.
* **AI assistance**: an AI assistant (Claude) was used to help scaffold the code and docs; I reviewed, ran, and take responsibility for every decision. *(Edit to reflect what is true for you.)*
* The code was smoke-tested on synthetic data only; all real-data numbers must come from your own runs.

## Limitations / next steps

Palette-hard negatives use a coarse RGB histogram; zari/gold motifs whose contrast is purely chromatic are weakened by the luminance stem (try `--stem rgb` + stronger augmentation or add an isoluminant channel); extend to other garments by swapping the manifest.
