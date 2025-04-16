import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import requests
import moviepy.editor as mp
import math
import re
from moviepy.editor import concatenate_videoclips
from typing import Any, List, Dict
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

SEGMENT_LENGTH = 30

FILLER_WORDS = {
    'um', 'uh', 'er', 'ah', 'like', 'hmm', 'you know', 'i mean', 'actually',
    'basically', 'literally', 'so', 'right', 'well', 'kind of', 'sort of',
    'just', 'anyway', 'whatever', 'yeah', 'okay', 'uhm', 'mmm', 'huh', 'eh',
    'ya know', 'mm-hmm', 'mmm-hmm', 'erm', 'umm', 'uhh', 'err'
}

TRAILING_WORDS = {
    "and", "or", "but", "because", "so", "then", "than", "although", "though", "yet",
    "if", "when", "while", "as", "unless", "until", "where", "whether", "even", "since"
}

def load_video(video_path: str) -> mp.VideoFileClip:
    return mp.VideoFileClip(video_path)

def extract_audio(video: mp.VideoFileClip, audio_path: str) -> mp.AudioFileClip:
    video.audio.write_audiofile(audio_path, verbose=False)
    return mp.AudioFileClip(audio_path)

def split_audio(audio_clip: mp.AudioFileClip, segment_length: int) -> List[Dict[str, Any]]:
    duration = audio_clip.duration
    segments = []
    num_segments = math.ceil(duration / segment_length)

    for i in range(num_segments):
        start_time = i * segment_length
        end_time = min((i + 1) * segment_length, duration)
        segment = audio_clip.subclip(start_time, end_time)
        segment_audio_path = f"segment_{i}.wav"
        segment.write_audiofile(segment_audio_path, verbose=False)
        segments.append({"path": segment_audio_path, "start": start_time})

    return segments

def transcribe_segment(segment_audio_path: str, start_time: float) -> List[Dict[str, Any]]:
    url = "http://34.46.113.13:9001/asr"
    params = {"task": "transcribe", "language": "en", "output": "vtt", "encode": "true"}

    with open(segment_audio_path, 'rb') as f:
        response = requests.post(url, params=params, files={'audio_file': f}, timeout=60)

    if response.status_code != 200:
        raise Exception(f"Transcription failed: {response.status_code} - {response.text}")

    vtt_lines = [line.strip() for line in response.text.strip().splitlines() if line.strip() and not line.startswith("WEBVTT")]
    segments = []
    i = 0
    while i < len(vtt_lines):
        if '-->' in vtt_lines[i]:
            times = vtt_lines[i].split('-->')
            start = to_seconds(times[0].strip()) + start_time
            end = to_seconds(times[1].strip()) + start_time
            i += 1
            if i < len(vtt_lines):
                text = vtt_lines[i].strip()
                segments.append({'start': start, 'end': end, 'text': text})
        i += 1
    return segments

def to_seconds(timestamp: str) -> float:
    parts = timestamp.replace(',', '.').split(':')
    while len(parts) < 3:
        parts.insert(0, '0')
    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])

def normalize_text(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9 ]+', '', text.lower()).strip()

def semantic_similarity(a: str, b: str) -> float:
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(emb1, emb2))

def detect_issues(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    removals = set()
    for i, seg1 in enumerate(segments):
        for j, seg2 in enumerate(segments):
            if i >= j:
                continue
            if semantic_similarity(seg1['text'], seg2['text']) > 0.87:
                removals.add(min(i, j))
    removals.update(i for i, seg in enumerate(segments) if any(filler in normalize_text(seg['text']) for filler in FILLER_WORDS))
    removals.update(i for i, seg in enumerate(segments) if not seg['text'].strip().endswith(('.', '!', '?')) and seg['text'].split()[-1].lower() in TRAILING_WORDS)
    return [segments[i] for i in removals]

def remove_segments(video: mp.VideoFileClip, segments: List[Dict[str, Any]]) -> mp.VideoFileClip:
    remove_ranges = sorted([(seg['start'], seg['end']) for seg in segments], key=lambda x: x[0])
    keep_ranges, prev_end = [], 0
    for start, end in remove_ranges:
        if start > prev_end:
            keep_ranges.append((prev_end, start))
        prev_end = end
    if prev_end < video.duration:
        keep_ranges.append((prev_end, video.duration))

    return concatenate_videoclips([video.subclip(s, e) for s, e in keep_ranges])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video file path.")
    args = parser.parse_args()

    video = load_video(args.video)
    audio_clip = extract_audio(video, "temp_audio.wav")

    segments = []
    for seg in split_audio(audio_clip, SEGMENT_LENGTH):
        segments.extend(transcribe_segment(seg['path'], seg['start']))

    to_remove = detect_issues(segments)
    final_video = remove_segments(video, to_remove)

    final_video.write_videofile("final_trimmed_video.mp4", codec="libx264", audio_codec="aac")

    with open("final_report.txt", "w") as f:
        for seg in segments:
            f.write(f"{seg['start']:.2f}s-{seg['end']:.2f}s: {seg['text']}\n")

if __name__ == "__main__":
    main()
