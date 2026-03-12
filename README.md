# Foley à Deux

Ghassan El Bounni & Joshua Ford

Much Foley. So much sound. Such wow 

Video-to-foley pipeline that generates any desired sound synchronised to on-screen motion.

## How it works

1. **Video preprocessing** — videos are segmented into 10-second clips; optical flow is extracted with RAFT, and BN-Inception RGB/Flow features are computed per segment.
2. **Video2RMS** — predicts an RMS envelope (timing + dynamics) from the visual features.
3. **AudioLDM** — the base AudioLDM + CLAP text embedding generates *any* sound you describe.
4. **Amplitude modulation** — the Video2RMS envelope is applied to the generated audio so the output's rhythm and intensity follow the on-screen motion.
5. **Merge** — all output segments are concatenated into a single result video.

## Setup

`setup.sh` creates the `foley` conda environment, installs all dependencies, downloads the model checkpoints, and writes the inference config automatically.

```bash
bash setup.sh
conda activate foley
```

## Usage

### Web UI

```bash
python -m app                    # http://localhost:7860
python -m app --port 8080
python -m app --share            # public Gradio tunnel URL
```

**UI features**
- **Upload** any `.mp4` / `.avi` video, or pick one from the built-in **example gallery**
- **Sound prompt** — describe the sound you want in free text
- **Theme** — prefix your prompt with a style preset (Cinematic, Cartoon, Funny, Horror, Nature, Sci-Fi, Fantasy, Rock, Jazz)
- **Auto-caption** button — powered by **SmolVLM2-2.2B-Instruct**; analyses the video and suggests a sound prompt automatically (first run downloads the model ~4 GB)
- **Output video** — all segments merged into one, with generated foley audio muxed in
- **Audio visualisations** — waveform, mel spectrogram, and RMS envelope plot

### Command line

```bash
# Explicit prompt
python infer.py --video examples/hitting_a_plastic_bag.mp4 --prompt "hitting a plastic bag"

# Auto-caption (SmolVLM2 generates the prompt from the video)
python infer.py --video examples/hitting_a_plastic_bag.mp4 --auto-caption

# Combine both — auto-caption is appended to the manual prompt
python infer.py --video examples/typing_on_a_keyboard.mp4 --prompt "keyboard" --auto-caption

# With theme and explicit output directory
python infer.py \
  --video  examples/typing.mp4 \
  --prompt "typing on a keyboard" \
  --theme  cinematic \
  --output output/my_run
```

| Flag | Short | Description |
|------|-------|-------------|
| `--video` | `-v` | Path to input video (`.mp4` / `.avi`) |
| `--prompt` | `-p` | Sound description (optional if `--auto-caption` is set) |
| `--auto-caption` | `-a` | Use SmolVLM2 to generate the prompt from the video |
| `--theme` | `-t` | Style preset (see below) |
| `--output` | `-o` | Output directory (defaults to `gradio_output/run_<id>/`) |

Available themes: `none`, `cinematic`, `cartoon`, `funny`, `horror`, `nature`, `sci-fi`, `fantasy`, `rock`, `jazz`

## Output

```
gradio_output/          (UI)   or the path passed to --output (CLI)
  run_<id>/
    input/              — copy of the input video
    output/
      audio/            — generated .wav files (one per 10-s segment)
      video/            — .mp4 clips with audio muxed in (one per segment)
    result.mp4          — all segments merged into the final output
```

## Checkpoints

Downloaded automatically by `setup.sh` into `ckpt/`:

| File | Description |
|------|-------------|
| `checkpoint_000500_Video2RMS.pt` | Video2RMS model (epoch 500) |
| `ControlNetstep300000.pth` | AudioLDM ControlNet weights |
| `opts.yml` | Inference configuration (written by setup.sh) |

## Project structure

```
infer.py         — CLI entry point (no Gradio required)
app/
  config.py      — paths, theme definitions, colour palette, create_device()
  preprocess.py  — video preprocessing pipeline (RAFT, BN-Inception)
  pipeline.py    — shared core: generate() + Gradio run_inference() wrapper
  plots.py       — waveform, spectrogram, RMS envelope figures
  caption.py     — auto-caption via SmolVLM2-2.2B-Instruct (lazy-loaded on first use)
  ui.py          — Gradio layout and event wiring
  __main__.py    — python -m app entry point
video2rms/       — Video2RMS model and data utilities
libs/            — AudioLDM, TorchJaekwon
ckpt/            — model checkpoints and inference config (gitignored)
examples/        — sample videos for the UI gallery
```


