#video_to_audio_extraction.py to extarct audio from video and preprocess for STT models

import subprocess
import librosa
import soundfile as sf
from moviepy.editor import VideoFileClip

# Configuring variables and parameters

YOUTUBE_URL  = "https://www.youtube.com/watch?v=gYYtJQsLIuo"
VIDEO_FILE   = "downloaded_video.mp4"
RAW_AUDIO    = "audio_raw.wav"
FINAL_AUDIO  = "audio_processed.wav"
SAMPLE_RATE  = 16000   # Standard sample rate for Speech-to-Text models


# Downloads YouTube video using yt-dlp and saves it as an MP4 file
def download_video(url, output_path):
    print(f"Downloading video from: {url}")
    subprocess.run([
        "yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "--merge-output-format", "mp4", "-o", output_path, url
    ], check=True)
    print(f"  Video saved to: {output_path}")

 # Extracts mono audio from video using MoviePy and saves it as a WAV file
def extract_audio(video_path, audio_path, sr):
    print(f"Extracting raw audio from {video_path}")
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)
    audio.close()
    video.close()
    print(f"Raw audio saved to: {audio_path}")

# Applies resampling, mono conversion, and loudness normalization using ffmpeg for efficient processing
def preprocess_audio(raw_path, output_path, sr):
    print(f"Applying preprocessing to {raw_path}")

    # Use ffmpeg for efficient combined processing
    subprocess.run([
        "ffmpeg", "-y", "-i", raw_path,
        "-af", "aresample=16000,aformat=sample_fmts=s16:channel_layouts=mono,loudnorm",
        "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", output_path
    ], check=True, capture_output=True)

    print(f"  Processed audio saved to: {output_path}")


def main():
    try:
        download_video(YOUTUBE_URL, VIDEO_FILE)
        extract_audio(VIDEO_FILE, RAW_AUDIO, SAMPLE_RATE)
        preprocess_audio(RAW_AUDIO, FINAL_AUDIO, SAMPLE_RATE)

        # Confirm processed audio details
        signal, sr = librosa.load(FINAL_AUDIO, sr=None, mono=True)

        print(f"\nProcessing complete! Processed audio: {FINAL_AUDIO}")
        print(f"  Duration   : {len(signal)/sr:.2f} seconds")
        print(f"  Sample rate: {sr} Hz | Channels: Mono")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()