# kotacanvas
Sprint in PyConHK 2024 for generating art

---

## video_extractor

A command-line script that extracts speech from an MP4 video and writes the transcription to a plain-text file.  
Speech in **English** and **Mandarin Chinese** is supported (language is detected automatically).

Transcription is performed by a [Whisper](https://huggingface.co/openai/whisper-large-v3) model that is downloaded from Hugging Face on first run and cached locally, so all subsequent runs are fully offline.

### Requirements

Python 3.9 or later.

```bash
pip install -r requirements.txt
```

> **GPU (optional):** If a CUDA-capable GPU is available, `torch` will use it automatically, which significantly speeds up transcription.

### Usage

```
python video_extractor.py <video.mp4> [-o output.txt] [-m model_name_or_path]
```

| Argument | Description |
|---|---|
| `video` | Path to the input MP4 file (**required**) |
| `-o / --output` | Path for the output `.txt` file (default: same name as the video) |
| `-m / --model` | Hugging Face model ID or local directory (default: `openai/whisper-large-v3`) |

### Examples

```bash
# Basic usage – creates lecture.txt next to the video
python video_extractor.py lecture.mp4

# Custom output path
python video_extractor.py lecture.mp4 -o transcript.txt

# Use a smaller / faster model
python video_extractor.py lecture.mp4 -m openai/whisper-medium

# Use a model that has already been downloaded to a local directory
python video_extractor.py lecture.mp4 -m ./models/whisper-large-v3
```

### How it works

1. **Extract audio** – `moviepy` reads the MP4 and writes the audio track to a temporary WAV file.
2. **Transcribe** – The Hugging Face `transformers` ASR pipeline loads the Whisper model locally and transcribes the audio in 30-second chunks so that videos of any length are handled correctly.
3. **Save** – The transcribed text is written to the output file in UTF-8 encoding.
