import json
import re
import os
import time
import requests
from typing import Any, List, Dict
import moviepy.editor as mp
import numpy as np
from sentence_transformers import SentenceTransformer, util

VIDEO_PATH = "Video3.mov"
AUDIO_PATH = "Video_audio.wav"
SEGMENT_LENGTH = 30
WHISPER_URL = "http://34.46.113.13:9001/asr"
PADDING = 0.5
FILLER_WORDS = {
    'um','uh','er','ah','like','hmm','you know','i mean','actually','basically','literally','so','right','well',
    'kind of','sort sort','just','anyway','whatever','yeah','okay','uhm','mmm','huh','eh','ya know','mm-hmm','mmm-hmm','erm','umm','uhh','err'
}

model = SentenceTransformer('all-MiniLM-L6-v2')

def to_seconds(timestamp: str) -> float:
    parts = timestamp.replace(',', '.').split(':')
    while len(parts) < 3:
        parts.insert(0, '0')
    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])

def load_video(video_path: str) -> mp.VideoFileClip:
    return mp.VideoFileClip(video_path)

def extract_audio(video: mp.VideoFileClip, audio_path: str) -> mp.AudioFileClip:
    video.audio.write_audiofile(audio_path, verbose=False)
    return mp.AudioFileClip(audio_path)

def transcribe_segment(segment_audio_path: str, start_time: float, retries: int = 3, timeout: int = 120) -> Dict[str, Any]:
    params = {
        "task": "transcribe",
        "language": "en",
        "output": "json",
        "encode": "true",
        "word_timestamps": "true"
    }
    attempt = 0
    while attempt < retries:
        try:
            with open(segment_audio_path, 'rb') as f:
                files = {'audio_file': f}
                response = requests.post(WHISPER_URL, params=params, files=files, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                words = []
                phrases = []
                for seg in data.get("segments", []):
                    start = to_seconds(f"0:0:{seg['start']:.2f}") + start_time
                    end = to_seconds(f"0:0:{seg['end']:.2f}") + start_time
                    text = seg["text"].strip()
                    phrases.append({"start": start, "end": end, "text": text})
                    for word_info in seg.get("words", []):
                        words.append({
                            "start": to_seconds(f"0:0:{word_info['start']:.2f}") + start_time,
                            "end": to_seconds(f"0:0:{word_info['end']:.2f}") + start_time,
                            "word": word_info['word'].strip()
                        })
                return {"words": words, "phrases": phrases}
            else:
                raise Exception(f"Transcription failed: {response.status_code} - {response.text}")
        except requests.exceptions.ReadTimeout as e:
            attempt += 1
            print(f"Attempt {attempt}/{retries}: Timeout error for segment '{segment_audio_path}'. Retrying in 1 second...")
            time.sleep(1)
    # If all attempts fail, raise an exception.
    raise Exception(f"Transcription failed for segment '{segment_audio_path}' after {retries} attempts.")

def split_audio(audio_clip: mp.AudioFileClip, segment_length: int) -> List[Dict[str, Any]]:
    duration = audio_clip.duration
    segments = []
    num_segments = int(np.ceil(duration / segment_length))
    for i in range(num_segments):
        start_time = i * segment_length
        end_time = min((i + 1) * segment_length, duration)
        segment = audio_clip.subclip(start_time, end_time)
        segment_audio_path = f"segment_{i}.wav"
        segment.write_audiofile(segment_audio_path, verbose=False)
        segments.append({"path": segment_audio_path, "start": start_time})
    return segments

def normalize_text(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9 ]+', '', text.lower()).strip()

def semantic_similarity(a: str, b: str) -> float:
    e1 = model.encode(a, convert_to_tensor=True)
    e2 = model.encode(b, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(e1, e2))

def detect_repeated_phrases(phrases: List[Dict[str, Any]], threshold=0.85) -> List[Dict[str, Any]]:
    # Remove the earlier occurrence so that the last occurrence (best take) is preserved.
    repeated = []
    for i in range(len(phrases) - 1):
        for j in range(i + 1, len(phrases)):
            if semantic_similarity(phrases[i]["text"], phrases[j]["text"]) > threshold:
                if phrases[i] not in repeated:
                    repeated.append(phrases[i])
                break  # Once a duplicate is found, move to the next phrase.
    return repeated

def is_filler_word(word: str) -> bool:
    cleaned = re.sub(r'[^a-zA-Z0-9 ]+', '', word.lower()).strip()
    return cleaned in FILLER_WORDS

def build_filler_intervals(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    intervals = []
    for w in words:
        if is_filler_word(w["word"]):
            intervals.append({"start": w["start"], "end": w["end"], "text": w["word"]})
    return intervals

def build_best_take_video(video: mp.VideoFileClip, kept_phrases: List[Dict[str, Any]], padding: float = PADDING) -> mp.VideoFileClip:
    kept_phrases.sort(key=lambda x: x["start"])
    clips = []
    for phrase in kept_phrases:
        start = phrase["start"]
        end = min(phrase["end"] + padding, video.duration)
        clips.append(video.subclip(start, end))
    return mp.concatenate_videoclips(clips)

def main():
    video = load_video(VIDEO_PATH)
    audio_clip = extract_audio(video, AUDIO_PATH)
    audio_segments = split_audio(audio_clip, SEGMENT_LENGTH)

    all_words = []
    all_phrases = []
    for seg in audio_segments:
        # If transcription fails, the error will now be more informative.
        result = transcribe_segment(seg["path"], seg["start"])
        all_words.extend(result["words"])
        all_phrases.extend(result["phrases"])

    filler_intervals = build_filler_intervals(all_words)
    repeated_phrases = detect_repeated_phrases(all_phrases)
    
    # For the final best-take transcript, exclude the repeated (earlier) phrases.
    kept_phrases = [s for s in all_phrases if s not in repeated_phrases]
    
    # Write a detailed log file.
    with open("transcript_detailed_log.txt", "w", encoding="utf-8") as f:
        f.write("===== WORDS =====\n\n")
        for w in all_words:
            f.write(f"{w['start']:.2f}-{w['end']:.2f}: {w['word']}\n")
        f.write("\n===== PHRASES =====\n\n")
        for ph in all_phrases:
            f.write(f"{ph['start']:.2f}-{ph['end']:.2f}: {ph['text']}\n")
        f.write("\n===== FILLERS REMOVED =====\n\n")
        for fi in filler_intervals:
            f.write(f"{fi['start']:.2f}-{fi['end']:.2f}: {fi['text']}\n")
        f.write("\n===== REPEATED PHRASES REMOVED =====\n\n")
        for rp in repeated_phrases:
            f.write(f"{rp['start']:.2f}-{rp['end']:.2f}: {rp['text']}\n")
        f.write("\n===== FINAL TRANSCRIPT KEPT (BEST Takes) =====\n\n")
        for seg in kept_phrases:
            f.write(f"{seg['start']:.2f}-{seg['end']:.2f}: {seg['text']}\n")

    # Build final best-take video only from the kept phrases.
    best_take_video = build_best_take_video(video, kept_phrases)
    best_take_video.write_videofile("final_cleaned_video.mp4", codec="libx264", audio_codec="aac")

if __name__ == "__main__":
    main()
