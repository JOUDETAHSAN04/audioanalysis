import json
import re
import os
import requests
from typing import Any, List, Dict
import moviepy.editor as mp
from moviepy.editor import TextClip, CompositeVideoClip
from sentence_transformers import SentenceTransformer, util

VIDEO_PATH = "Video1.mov"
AUDIO_PATH = "Video_audio.wav"
TRANSCRIPTION_JSON = "transcription.json"
WHISPER_URL = "http://34.46.113.13:9001/asr"

FILLER_WORDS = {
    'um','uh','er','ah','like','hmm','you know','i mean','actually','basically','literally','so','right','well',
    'kind of','sort of','just','anyway','whatever','yeah','okay','uhm','mmm','huh','eh','ya know','mm-hmm','mmm-hmm','erm','umm','uhh','err'
}

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_video(video_path: str) -> mp.VideoFileClip:
    return mp.VideoFileClip(video_path)

def extract_audio(video: mp.VideoFileClip, audio_path: str) -> mp.AudioFileClip:
    video.audio.write_audiofile(audio_path, verbose=False)
    return mp.AudioFileClip(audio_path)

def transcribe_audio(audio_path: str, output_path: str) -> List[Dict[str, Any]]:
    with open(audio_path, 'rb') as f:
        files = {'audio_file': f}
        params = {
            "task": "transcribe",
            "language": "en",
            "output": "json",
            "encode": "true",
            "word_timestamps": "true"
        }
        response = requests.post(WHISPER_URL, params=params, files=files, timeout=90)

    if response.status_code != 200:
        raise Exception(f"Transcription failed: {response.status_code} - {response.text}")

    data = response.json()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data.get("segments", [])

def parse_transcription_json(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("segments", [])

def is_filler_word(word: str) -> bool:
    w = re.sub(r'[^a-zA-Z0-9 ]+', '', word.lower()).strip()
    return w in FILLER_WORDS

def build_filler_intervals(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    intervals = []
    for seg in segments:
        for w in seg.get("words", []):
            if is_filler_word(w["word"]):
                intervals.append({
                    "start": w["start"],
                    "end": w["end"],
                    "text": w["word"]
                })
    return intervals

def normalize_text(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9 ]+', '', text.lower()).strip()

def semantic_similarity(a: str, b: str) -> float:
    e1 = model.encode(a, convert_to_tensor=True)
    e2 = model.encode(b, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(e1, e2))

def detect_repeated_segments(segments: List[Dict[str, Any]], threshold=0.85) -> List[Dict[str, Any]]:
    seen, repeats = set(), []
    for i in range(len(segments)):
        if i in seen:
            continue
        for j in range(i + 1, len(segments)):
            if semantic_similarity(segments[i]["text"], segments[j]["text"]) > threshold:
                seen.add(i)
                repeats.append(segments[i])
                break
    return repeats

def add_overlays(video: mp.VideoFileClip, segments: List[Dict[str, Any]]) -> mp.VideoFileClip:
    overlays = []
    for seg in segments:
        txt = seg["text"][:80]
        start, end = seg['start'], seg['end']
        duration = max(end - start, 0.1)
        txt_clip = TextClip(txt, fontsize=24, color='white', bg_color='black')
        txt_clip = txt_clip.set_position(('center', 'bottom')).set_duration(duration).set_start(start)
        overlays.append(txt_clip)
    return CompositeVideoClip([video] + overlays)

def remove_intervals(video: mp.VideoFileClip, intervals: List[Dict[str, Any]]) -> mp.VideoFileClip:
    intervals.sort(key=lambda x: x["start"])
    merged, keep, prev_end = [], [], 0

    for iv in intervals:
        st, en = round(iv["start"], 2), round(iv["end"], 2)
        if not merged or st > merged[-1][1] + 0.05:
            merged.append([st, en])
        else:
            merged[-1][1] = max(merged[-1][1], en)

    for (st, en) in merged:
        if st > prev_end:
            keep.append((prev_end, st))
        prev_end = en
    if prev_end < video.duration:
        keep.append((prev_end, video.duration))

    clips = [video.subclip(max(start, 0.001), min(end, video.duration)) for start, end in keep if end - start > 0.25]
    return mp.concatenate_videoclips(clips)

def main():
    video = load_video(VIDEO_PATH)

    if not os.path.exists(TRANSCRIPTION_JSON):
        extract_audio(video, AUDIO_PATH)
        transcribe_audio(AUDIO_PATH, TRANSCRIPTION_JSON)

    segments = parse_transcription_json(TRANSCRIPTION_JSON)
    filler_intervals = build_filler_intervals(segments)
    repeated_intervals = detect_repeated_segments(segments)

    to_remove = repeated_intervals
    kept_segments = [s for s in segments if s not in repeated_intervals]

    cleaned_video = remove_intervals(video, to_remove)
    final_video_with_overlays = add_overlays(cleaned_video, kept_segments)

    with open("transcript_detailed_log.txt", "w", encoding="utf-8") as f:
        f.write("===== ORIGINAL SEGMENTS =====\n\n")
        for seg in segments:
            f.write(f"{seg['start']:.2f}-{seg['end']:.2f}: {seg['text']}\n")

        f.write("\n===== FILLERS FOUND (not removed from video) =====\n\n")
        for seg in filler_intervals:
            f.write(f"{seg['start']:.2f}-{seg['end']:.2f}: {seg['text']}\n")

        f.write("\n===== REPEATED SEGMENTS REMOVED =====\n\n")
        for seg in repeated_intervals:
            f.write(f"{seg['start']:.2f}-{seg['end']:.2f}: {seg['text']}\n")

        f.write("\n===== FINAL SEGMENTS KEPT =====\n\n")
        for seg in kept_segments:
            f.write(f"{seg['start']:.2f}-{seg['end']:.2f}: {seg['text']}\n")

    final_video_with_overlays.write_videofile("final_cleaned_video.mp4", codec="libx264", audio_codec="aac")

if __name__ == "__main__":
    main()
