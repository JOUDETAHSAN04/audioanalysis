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
    'um','uh','er','ah','like','hmm','you know','i mean','actually','basically',
    'literally','so','right','well','kind of','sort sort','just','anyway','whatever',
    'yeah','okay','uhm','mmm','huh','eh','ya know','mm-hmm','mmm-hmm','erm','umm','uhh','err'
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

# Updated semantic similarity: includes normalization.
def semantic_similarity(a: str, b: str) -> float:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    e1 = model.encode(a_norm, convert_to_tensor=True)
    e2 = model.encode(b_norm, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(e1, e2))

# New duplicate detection that also checks if one phrase is contained in a later phrase.
def detect_repeated_phrase_indices(phrases: List[Dict[str, Any]], threshold=0.85) -> set:
    repeated_indices = set()
    n = len(phrases)
    for i in range(n - 1):
        if i in repeated_indices:
            continue
        text_i = normalize_text(phrases[i]["text"])
        for j in range(i + 1, n):
            text_j = normalize_text(phrases[j]["text"])
            # If the earlier phrase is a substring of the later phrase, mark it as duplicate.
            if text_i and text_i in text_j:
                repeated_indices.add(i)
                break
            # Otherwise, check semantic similarity.
            if semantic_similarity(phrases[i]["text"], phrases[j]["text"]) > threshold:
                repeated_indices.add(i)
                break
    return repeated_indices

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

# Merge consecutive phrases if the gap between them is small.
def merge_consecutive_phrases(phrases: List[Dict[str, Any]], merge_gap_threshold: float = 0.3) -> List[Dict[str, Any]]:
    if not phrases:
        return phrases
    phrases = sorted(phrases, key=lambda x: x["start"])
    merged = []
    current_phrase = phrases[0]
    for phrase in phrases[1:]:
        # If the next phrase starts within the threshold gap, merge it.
        if phrase["start"] - current_phrase["end"] <= merge_gap_threshold:
            current_phrase["text"] += " " + phrase["text"]
            current_phrase["end"] = phrase["end"]
        else:
            merged.append(current_phrase)
            current_phrase = phrase
    merged.append(current_phrase)
    return merged

# New helper: Remove duplicate sentences from final transcript phrases.
def remove_duplicate_sentences(phrases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_sentences = set()
    cleaned_phrases = []
    for phrase in phrases:
        text = phrase["text"]
        # Split on period; you can adjust this regex to cover other punctuation if desired.
        sentences = [s.strip() for s in re.split(r'\.\s*', text) if s.strip() != '']
        new_sentences = []
        for sentence in sentences:
            norm_sentence = normalize_text(sentence)
            if norm_sentence not in seen_sentences:
                seen_sentences.add(norm_sentence)
                new_sentences.append(sentence)
        # Re-join the remaining sentences if any.
        if new_sentences:
            # Add a period at the end if desired.
            phrase["text"] = '. '.join(new_sentences).strip() + '.'
            cleaned_phrases.append(phrase)
    return cleaned_phrases

def main():
    video = load_video(VIDEO_PATH)
    audio_clip = extract_audio(video, AUDIO_PATH)
    audio_segments = split_audio(audio_clip, SEGMENT_LENGTH)

    all_words = []
    all_phrases = []
    for seg in audio_segments:
        result = transcribe_segment(seg["path"], seg["start"])
        all_words.extend(result["words"])
        all_phrases.extend(result["phrases"])

    # Merge consecutive phrases to avoid splits of the same sentence.
    all_phrases = merge_consecutive_phrases(all_phrases)

    filler_intervals = build_filler_intervals(all_words)
    # Detect duplicate phrases by index.
    duplicate_indices = detect_repeated_phrase_indices(all_phrases)
    kept_phrases = [p for i, p in enumerate(all_phrases) if i not in duplicate_indices]

    # Remove duplicate sentences that might be merged within longer phrases.
    kept_phrases = remove_duplicate_sentences(kept_phrases)
    
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
        for i in duplicate_indices:
            ph = all_phrases[i]
            f.write(f"{ph['start']:.2f}-{ph['end']:.2f}: {ph['text']}\n")
        f.write("\n===== FINAL TRANSCRIPT KEPT (BEST Takes) =====\n\n")
        for seg in kept_phrases:
            f.write(f"{seg['start']:.2f}-{seg['end']:.2f}: {seg['text']}\n")

    # Build final best-take video only from the kept phrases.
    best_take_video = build_best_take_video(video, kept_phrases)
    best_take_video.write_videofile("final_cleaned_video.mp4", codec="libx264", audio_codec="aac")

if __name__ == "__main__":
    main()
