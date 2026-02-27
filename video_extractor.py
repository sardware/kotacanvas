#!/usr/bin/env python3
"""
video_extractor.py – Extract speech from an MP4 video and transcribe it to a
text file using a locally deployed Hugging Face Whisper model.

Supported speech languages: English and Mandarin Chinese (auto-detected).

Usage:
    python video_extractor.py <video.mp4> [-o output.txt] [-m model_name_or_path]

The Whisper model is downloaded from Hugging Face on first run and cached
locally so that subsequent runs are fully offline.
"""

import argparse
import os
import sys
import tempfile

import moviepy.editor as mp
from transformers import pipeline


def extract_audio(video_path: str, audio_path: str) -> None:
    """Extract the audio track from *video_path* and write it to *audio_path* (WAV)."""
    clip = mp.VideoFileClip(video_path)
    if clip.audio is None:
        clip.close()
        raise ValueError(f"No audio stream found in '{video_path}'.")
    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
    clip.close()


def transcribe_audio(audio_path: str, model_name: str) -> str:
    """Transcribe *audio_path* with the given Hugging Face Whisper model.

    Language detection is automatic so both English and Mandarin Chinese are
    handled without any extra configuration.
    """
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        # chunk_length_s splits long audio into overlapping segments so the
        # model can handle videos of any duration.
        chunk_length_s=30,
        stride_length_s=5,
    )
    result = transcriber(audio_path, return_timestamps=False)
    return result["text"]


def save_transcript(text: str, output_path: str) -> None:
    """Write *text* to *output_path* encoded as UTF-8."""
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(text)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract speech from an MP4 video and save the transcription as a "
            "text file.  The speech may be in English or Mandarin Chinese."
        )
    )
    parser.add_argument("video", help="Path to the input MP4 video file.")
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Path for the output text file.  "
            "Defaults to <video_name>.txt in the same directory."
        ),
    )
    parser.add_argument(
        "-m",
        "--model",
        default="openai/whisper-large-v3",
        help=(
            "Hugging Face model name or local path.  "
            "Defaults to 'openai/whisper-large-v3'."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: video file '{args.video}' not found.", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or os.path.splitext(args.video)[0] + ".txt"

    # Extract audio to a temporary WAV file so that the transcriber always
    # receives a clean, uncompressed audio stream.
    audio_path = None
    try:
        fd, audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        print(f"[1/3] Extracting audio from '{args.video}' …")
        extract_audio(args.video, audio_path)

        print(f"[2/3] Transcribing with model '{args.model}' …")
        transcript = transcribe_audio(audio_path, args.model)

        print(f"[3/3] Saving transcript to '{output_path}' …")
        save_transcript(transcript, output_path)

        print("Done.")
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


if __name__ == "__main__":
    main()
