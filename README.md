# Foley à Deux

Ghassan El Bounni & Joshua Ford

Much Foley. So much sound. Such wow 

Video-to-foley pipeline that generates any desired sound synchronised to on-screen motion.

## How it works

1. **Video preprocessing** — videos are segmented, optical flow is extracted (RAFT), and BN-Inception RGB/Flow features are computed.
2. **Video2RMS** — predicts an RMS envelope (timing + dynamics) from the visual features.
3. **AudioLDM (free mode)** — the Greatest-Hits ControlNet branch is bypassed so the base AudioLDM + CLAP text/audio embedding can produce *any* sound. The Video2RMS envelope is then applied as amplitude modulation to impose the video's rhythm on the output.

## Setup

`setup.sh` creates the `foley` conda environment, installs all dependencies, and downloads the model checkpoints automatically.

**Linux / macOS / Windows (WSL2 recommended) / Windows (Git Bash + Conda)**

```bash
bash setup.sh
conda activate foley
```

```bash
source ./setup.sh
```

## Usage

```bash
# text prompt
python inference.py -i examples/ -p "a dog barking"
python inference.py -i examples/ -p "orchestral strings"

# reference audio (timbre transfer)
python inference.py -i examples/ -a path/to/reference.wav

# custom output directory
python inference.py -i examples/ -p "glass shattering" -o ./my_output
```

### Arguments

| Flag | Description |
|------|-------------|
| `-i / --input-dir` | Folder containing input `.mp4` / `.avi` files (required) |
| `-p / --prompt` | Text prompt describing the desired sound |
| `-a / --audio-prompt` | Path to reference audio file (timbre transfer) |
| `-o / --output-dir` | Output folder (default: `./output`) |
| `--epoch` | Video2RMS checkpoint epoch (default: `500`) |
| `--num-workers` | Parallel workers for preprocessing (default: `2`) |
| `--batch-size` | RAFT optical-flow batch size (default: `1`) |

Exactly one of `-p` / `-a` is required.

## Output

```
output/
  audio/   — generated .wav files (one per video segment)
  video/   — .mp4 files with generated audio muxed in
```

## Checkpoints

Place the following files in `ckpt/`:

| File | Description |
|------|-------------|
| `checkpoint_000500_Video2RMS.pt` | Video2RMS model (epoch 500) |
| `opts.yml` | Training / data configuration |

