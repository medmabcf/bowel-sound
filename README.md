# Bowel Sound Detection

Automatic detection and classification of bowel sounds using self-supervised learning (SSL) models. Two experiments are provided:

- **exp13** — DistilHuBERT (`ntu-spml/distilhubert`, 23.7 M params)
- **exp14** — WavLM-Base (`microsoft/wavlm-base`, 94.4 M params)

Both models are trained on `AS_1.wav` and evaluated against the held-out OOD recording `23M74M.wav`.

---

## Run on Google Colab (recommended)

Open either notebook directly in Colab — no setup needed, everything installs automatically.

> **For training: use a T4 GPU runtime.**
> In Colab go to `Runtime → Change runtime type → T4 GPU` before running.
> Training on CPU will work but is extremely slow.

| Experiment | Model | Open from GitHub | Open direct Colab link |
|---|---|---|---|
| exp00 | Data Analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medmabcf/bowel-sound-detection/blob/main/exp00_data_analysis.ipynb) | [▶ Open exp00 notebook](https://colab.research.google.com/drive/1SxlBMoc_vqbtcFXc_hzxHi7WWK9HwvLh?usp=sharing) |
| exp13 | DistilHuBERT ⭐ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medmabcf/bowel-sound-detection/blob/main/exp13_distilhubert_as1val.ipynb) | [▶ Open exp13 notebook](https://colab.research.google.com/drive/1RYGhCcknr_prM0Q3z0c7JLgmDsARnOce?usp=sharing) |
| exp14 | WavLM-Base | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medmabcf/bowel-sound-detection/blob/main/exp14_wavlm_as1val.ipynb) | [▶ Open exp14 notebook](https://colab.research.google.com/drive/1z8_e_6bwt-ADbiMmI2ArsmmQl2s8NSb4?usp=sharing) |

> ⭐ exp13 (DistilHuBERT) is the recommended notebook — it achieves the best results with 4× fewer parameters than exp14.

The first cell will automatically:
1. Clone this repo into `/content/bowel-sound-detection`
2. Install all required packages
3. Optionally mount Google Drive to save checkpoints across sessions

---

## Run Locally

> **Note:** Running locally can cause dependency mismatches between operating systems (Windows vs Linux), especially with audio libraries like `librosa` and `soundfile`. If you face issues, **Kaggle Notebooks** (Linux environment, free GPU) are a more reliable alternative to local Windows setups for anything other than quick testing.

### Requirements

- Python **3.10** (other versions may cause dependency conflicts)
- The `bowel-data/` folder must be present in the repo root

### Setup

**1. Clone the repo**
```bash
git clone https://github.com/medmabcf/bowel-sound-detection.git
cd bowel-sound-detection
```

**2. Create a Python 3.10 virtual environment**
```bash
# Linux / macOS
virtualenv -p python3.10 venv
source venv/bin/activate

# Windows (PowerShell)
virtualenv -p python3.10 venv
venv\Scripts\Activate.ps1
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set your HuggingFace token**

Both models (`distilhubert` and `wavlm-base`) are public, but HuggingFace may rate-limit anonymous downloads. A free token is recommended and only takes a minute to create.

**Get a token:** go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → *New token* → select **Read** access → copy the token.

You have two options:

**Option A — `.env` file (recommended)**

Copy the example file and fill in your token:
```bash
cp .env.example .env
```
Then open `.env` and replace the placeholder:
```
HF_TOKEN=hf_your_actual_token_here
```
The notebook reads `HF_TOKEN` automatically from the environment. Load it before launching:
```bash
# Linux / macOS
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2)

# Windows (PowerShell)
$env:HF_TOKEN = (Get-Content .env | Select-String "HF_TOKEN").ToString().Split("=")[1]
```

**Option B — directly in the notebook**

In **cell 2** of the notebook, find this line:
```python
HF_TOKEN = os.environ.get('HF_TOKEN', '')
```
and replace it with your token directly:
```python
HF_TOKEN = 'hf_your_actual_token_here'
```
> ⚠️ If you do this, make sure you **do not share or commit** the notebook with your token inside it.

> **Never commit your real token to GitHub.** `.env` is already listed in `.gitignore`. `.env.example` (committed to the repo) contains only a placeholder.

**5. Launch the notebook**
```bash
jupyter notebook exp13_distilhubert_as1val.ipynb
# or
jupyter notebook exp14_wavlm_as1val.ipynb
```

The notebook will detect that it is not running on Colab and automatically use `./bowel-data/` for data and `./checkpoints_v13.../` for saving checkpoints.

---

## Repository Structure

```
bowel-sound-detection/
├── exp13_distilhubert_as1val.ipynb   # DistilHuBERT training & evaluation
├── exp14_wavlm_as1val.ipynb          # WavLM-Base training & evaluation
├── exp00_data_analysis.ipynb         # Dataset exploration & annotation stats
├── bowel_sound_final_report.docx     # Full technical report
├── requirements.txt                  # Python dependencies
└── bowel-data/
    ├── AS_1.wav                      # Training + validation audio
    ├── AS_1.txt                      # Annotations (Audacity label format)
    ├── 23M74M.wav                    # OOD test audio (never seen during training)
    └── 23M74M.txt                    # OOD annotations
```

---

## Results

The key metric is performance on **23M74M (OOD test set)** — audio the model never saw during training. AS-1 val is the same-domain validation split used during training, so OOD results reflect true generalisation.

### OOD Test Set — 23M74M (never seen during training)

| Metric | exp13 DistilHuBERT | exp14 WavLM-Base |
|---|---|---|
| Best Val F1 (AS-1) | 0.5764 | 0.5137 |
| OOD macro F1 @ IoU ≥ 0.3 | **0.424 ★** | 0.257 |
| OOD macro F1 @ IoU ≥ 0.5 | **0.321 ★** | 0.163 |
| OOD macro F1 @ IoU ≥ 0.8 | **0.093 ★** | 0.019 |
| OOD mAP (avg IoU 0.3–0.8) | **0.279 ★** | 0.146 |
| Domain gap (mAP) | **+0.043 ★** | +0.171 |
| Parameters | 23.7 M | 94.4 M |
| Training stopped at epoch | 44 | 43 |

### Per-class OOD breakdown — exp13 DistilHuBERT (IoU ≥ 0.3)

| Class | Precision | Recall | F1 | FAR/min |
|---|---|---|---|---|
| h (harmonic) | 0.383 | 0.621 | 0.474 | 5.8 |
| mb (mid bowel) | 0.463 | 0.352 | 0.400 | 18.8 |
| sb (short bowel) | 0.375 | 0.427 | 0.399 | 36.3 |

### Key findings

- **exp13 (DistilHuBERT) wins on every OOD metric** despite being 4× smaller than exp14.
- **Domain gap**: exp13 generalises well (mAP drop of only 0.043 from val to OOD). exp14 drops 0.171 — nearly half its val performance disappears on unseen audio.
- **Why the smaller model wins**: the dataset is small (two recordings). WavLM-Base (94M params, 12 layers) has too much capacity to fine-tune reliably on limited data. DistilHuBERT (23.7M params, 2 layers) fits the data regime better.
- **IoU ≥ 0.8 is very strict** (80% temporal overlap required) — scores drop sharply for both models, which is expected for short events (median sb duration ~70 ms).
