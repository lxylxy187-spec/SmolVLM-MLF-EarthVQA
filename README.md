# SmolVLM MLF EarthVQA

Research code for the manuscript “基于跨层视觉特征残差融合的轻量级遥感视觉问答方法”. This release contains the actual MLF, baseline, single-layer ablation and adapter implementations and evaluation routines used in the experiments. Run configurations were copied from the corresponding training records, with the original server root removed from paths. The recorded seed stability and ablation summaries are in `results/`.

## Data and availability

EarthVQA images and QA annotations are not included. Obtain them through the [official EarthVQA repository](https://github.com/Junjue-Wang/EarthVQA) and follow the [rights holder’s terms](https://rsidea.whu.edu.cn/EarthVQA.htm). The upstream [EarthVLSet download](https://huggingface.co/datasets/Kingdrone-Junjue/EarthVLSet) is gated and requires the researcher’s own Hugging Face account and acceptance of the conditions. Google Earth imagery may have additional terms. The release does not claim to grant redistribution rights.

The recorded fixed Train/Dev image split and QA-index manifests are included under `data/processed/`; they contain image identifiers, QA indices, categories and counts, not images or QA text. Use these recorded files instead of generating a new random split. The original preparation scripts are included for inspection. The official Val evaluation covers the complete 1,669 images and 57,202 QA pairs, without resampling. Raw data, trained checkpoints and prediction files are not included. Users must comply with EarthVQA terms when obtaining or using the underlying data.

Expected layout after obtaining the official data separately (the two fixed manifests are included):

```text
data/raw/EarthVQA/Train_QA.json
data/raw/EarthVQA/Val_QA.json
data/raw/EarthVQA/Train/images_png/
data/raw/EarthVQA/Val/images_png/
data/processed/splits/earthvqa_train_dev_seed42.json
data/processed/manifests/earthvqa_baseline_30k_seed42.json
pretrained/SmolVLM-500M-Instruct/
```

The original source annotation SHA-256 is `9ce587927bfffd27f846d046f886d1c34530b85e305596b69537095969e14907`. The original QA manifest SHA-256 is `3f464c2ce3c6ffee4b72c325ff58ce313e12919a04f02dcc25c6fd21a33df937`; the image split SHA-256 is `03745b66c01341fff2b55a399e0eebcc38da904b6eaa5068b327c8ac1292afc2`. The manifests contain absolute metadata paths from the original machine; their byte hashes refer to the original files. The actual loader reads annotation and image paths from the project directory.

## Environment

The recorded environment was Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8, Transformers 4.51.3. On a compatible Linux GPU system, for example:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Obtain [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) separately, following its license, and place the model and processor files in the directory shown above. Installation on a fresh system has not yet been verified.

## Training

All commands assume the included fixed manifests, separately obtained raw data and pretrained model are present. The paper uses 30,000 training QA pairs, an internal Dev set of 8,803 QA pairs from 252 held-out training images, and training seeds 17, 42 and 73. Full MLF seed 42:

```bash
python trainers/train_smolvlm_mlf.py --seed 42 --model-dir pretrained/SmolVLM-500M-Instruct --manifest data/processed/manifests/earthvqa_baseline_30k_seed42.json --output-dir outputs/c_mlf_earthvqa_seed42 --train-scope text_connector --epochs 1 --batch-size 1 --gradient-accumulation 8 --learning-rate 2e-5 --weight-decay 0.01 --warmup-ratio 0.03 --max-grad-norm 1.0 --gradient-checkpointing
```

Additional seed 42 runs (the original defaults contain the same training budget; see `configs/`):

```bash
python trainers/train_smolvlm_baseline.py --seed 42 --output-dir outputs/baseline_seed42
python trainers/train_smolvlm_adapter.py --seed 42 --output-dir outputs/adapter_seed42
python trainers/train_smolvlm_mlf_ablation.py --seed 42 --selected-layer 4 --output-dir outputs/h4_seed42
python trainers/train_smolvlm_mlf_ablation.py --seed 42 --selected-layer 8 --output-dir outputs/h8_seed42
python trainers/train_smolvlm_baseline_stability.py --seed 17 --output-dir outputs/baseline_seed17
python trainers/train_smolvlm_mlf_stability.py --seed 17 --output-dir outputs/mlf_seed17
# Repeat the last two commands with --seed 73 and new output directories.
```

These commands require a suitable GPU; choose new output directories for every run. The training data and model default to the paths shown above.

## Evaluation

The evaluator normalizes answers by lowercasing, replacing punctuation, removing English articles and normalizing whitespace. EM compares normalized answers; Token-F1 uses token multiset overlap. Evaluation requires the trained checkpoint; MLF evaluation loads its separate fusion sidecar.

```bash
python evaluation/evaluate_earthvqa_mlf.py --model-dir outputs/c_mlf_earthvqa_seed42/final --manifest data/processed/manifests/earthvqa_baseline_30k_seed42.json --split dev --output-dir outputs/eval_c_mlf_seed42_dev --max-new-tokens 48
python evaluation/evaluate_earthvqa_mlf.py --model-dir outputs/c_mlf_earthvqa_seed42/final --manifest data/processed/manifests/earthvqa_baseline_30k_seed42.json --split official_val --confirm-final-test --output-dir outputs/eval_c_mlf_seed42_val --max-new-tokens 48
```

Baseline and adapter evaluators are named `evaluate_earthvqa.py` and `evaluate_earthvqa_adapter.py`; H4/H8 use `evaluate_earthvqa_mlf_ablation.py`. See each program’s `--help`. Use `--max-samples` for small inference checks and a new output directory. The official Val numbers reported in the paper are Baseline 69.44% EM / 77.14% Token-F1 and full MLF 73.17% / 80.18%, both seed 42.

## Scope and provenance

The released source files are copied from the experiment directory; the core MLF and the principal training files matched the training-time SHA-256 records. This package does not include raw data, checkpoints, prediction records, tokens, environments, or caches. The scripts may require additional packages and GPU memory. Evaluating the trained models requires the corresponding weights and the correct source data version; trained checkpoints are not included.
