# Foley à Deux

Ghassan El Bounni & Joshua Ford

Much Foley. So much sound. Such wow 

Video-to-foley pipeline that generates any desired sound synchronised to on-screen motion.

## How it works

1. **Video preprocessing** — videos are segmented, optical flow is extracted (RAFT), and BN-Inception RGB/Flow features are computed.
2. **Video2RMS** — predicts an RMS envelope (timing + dynamics) from the visual features.
3. **AudioLDM** — the base AudioLDM + CLAP text embedding generates *any* sound you describe.
4. **Amplitude modulation** — the Video2RMS envelope is applied to the generated audio so the output's rhythm and intensity follow the on-screen motion.

## Setup

`setup.sh` creates the `foley` conda environment, installs all dependencies, downloads the model checkpoints, and writes the inference config automatically.

```bash
bash setup.sh
conda activate foley
```

## Running the UI

```bash
python -m app                    # default: http://localhost:7860
python -m app --port 8080
python -m app --share            # create a public Gradio tunnel URL
```

### UI features

- **Upload** any `.mp4` / `.avi` video
- **Sound prompt** — describe the sound you want in free text
- **Theme** — prefix your prompt with a style preset (Cinematic, Cartoon, Funny, Horror, Nature, Sci-Fi, Fantasy, Rock, Jazz)
- **Auto-caption** button *(coming soon)* — a vision-language model will analyse the video and suggest a default prompt
- **Output video** with generated foley audio muxed in
- **Audio visualisations** — waveform, mel spectrogram, and a two-panel RMS envelope plot showing the Video2RMS prediction vs. the realised audio RMS

## Output

```
gradio_output/
  run_<id>/
    input/    — copy of the uploaded video
    output/
      audio/  — generated .wav files (one per segment)
      video/  — .mp4 files with generated audio muxed in
    result.mp4 — final output served to the UI
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
app/
  config.py      — paths, theme definitions, colour palette
  preprocess.py  — video preprocessing pipeline (RAFT, BN-Inception)
  pipeline.py    — Video2RMS + AudioLDM inference loop
  plots.py       — waveform, spectrogram, RMS envelope figures
  caption.py     — auto-caption stub (future VLM integration)
  ui.py          — Gradio layout and event wiring
  __main__.py    — python -m app entry point
video2rms/       — Video2RMS model and data utilities
libs/            — AudioLDM, TorchJaekwon, taming-transformers
ckpt/            — model checkpoints and inference config (gitignored)
```


