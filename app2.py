import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests
import moviepy.editor as mp
import math
import re
import os
from moviepy.editor import concatenate_videoclips, CompositeVideoClip, TextClip
from typing import Any, List, Dict
import numpy as np
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util  # For semantic similarity

model = SentenceTransformer('all-MiniLM-L6-v2')


VIDEO_PATH = 'Video1.mov'
AUDIO_PATH = 'Video_audio.wav'
SEGMENT_LENGTH = 30


# Common filler words in English
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

def transcribe_segment(segment_audio_path: str, start_time: float) -> Any:
    url = "http://34.46.113.13:9001/asr"
    params = {"task": "transcribe", "language": "en", "output": "json", "encode": "true","word_timestamps":"true"}

    with open(segment_audio_path, 'rb') as f:
        files = {'audio_file': f}
        response = requests.post(url, params=params, files=files, timeout=60)
        print(f"response: {response.text}")
    if response.status_code == 200:
        transcription = response.json()
        words = []
        for segment in transcription.get('segments', []):
            words.append({
                'start': segment['start'] + start_time,
                'end': segment['end'] + start_time,
                'word': segment['text'].strip()
            })
        return {'words': words}
    else:
        raise Exception(f"Transcription failed: {response.status_code} - {response.text}")

def merge_transcriptions(transcriptions: List[Any]) -> Any:
    merged = {"words": []}
    for trans in transcriptions:
        merged["words"].extend(trans["words"])
    merged["words"].sort(key=lambda x: x["start"])
    return merged

def group_words_to_segments(words: List[Any], gap_threshold: float = 0.5) -> List[Dict[str, Any]]:
    segments = []
    sentence_endings = re.compile(r'[.!?]')
    if not words:
        return segments

    current_segment = {"start": words[0]['start'], "end": words[0]['end'], "text": words[0]['word']}
    for word in words[1:]:
        if word['start'] - current_segment['end'] <= gap_threshold and not sentence_endings.search(current_segment['text']):
            current_segment['end'] = word['end']
            current_segment['text'] += f" {word['word']}"
        else:
            segments.append(current_segment)
            current_segment = {"start": word['start'], "end": word['end'], "text": word['word']}
    segments.append(current_segment)
    return segments


def normalize_text(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9 ]+', '', text.lower()).strip()

def detect_repeated_content(segments: List[Dict[str, Any]], threshold: float = 0.85) -> List[Dict[str, Any]]:
    repeated_indices = set()
    n = len(segments)

    for i in range(n):
        for j in range(i + 1, n):
            sim = semantic_similarity(segments[i]['text'], segments[j]['text'])
            if sim > threshold:
                repeated_indices.add(i)  # earlier one is marked for removal
                break  # we only need to match once

    repeated_segments = [segments[i] for i in repeated_indices]
    return repeated_segments


def semantic_similarity(a: str, b: str) -> float:
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(emb1, emb2))
# def text_similarity(a: str, b: str) -> float:
#     return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()

def detect_fillers(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fillers = []
    for seg in segments:
        if any(filler in normalize_text(seg['text']) for filler in FILLER_WORDS):
            fillers.append(seg)
    return fillers

def remove_segments(video: mp.VideoFileClip, segments: List[Dict[str, Any]]) -> mp.VideoFileClip:
    segments.sort(key=lambda x: x['start'])
    merged_segs = []
    for seg in segments:
        if not merged_segs or seg['start'] > merged_segs[-1][1]:
            merged_segs.append([seg['start'], seg['end']])
        else:
            merged_segs[-1][1] = max(merged_segs[-1][1], seg['end'])

    keep_ranges = []
    prev_end = 0
    for start, end in merged_segs:
        if start > prev_end:
            keep_ranges.append((prev_end, start))
        prev_end = end
    if prev_end < video.duration:
        keep_ranges.append((prev_end, video.duration))

    return concatenate_videoclips([video.subclip(s, e) for s, e in keep_ranges])


def detect_incomplete_sentences(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    incomplete = []
    for i, seg in enumerate(segments):
        text = seg['text'].strip()
        words = normalize_text(text).split()
        if (
            not re.search(r'[.!?]$', text) and (
                len(words) <= 3 or
                words[-1] in TRAILING_WORDS or
                text[-1] in {",", "-", ":", ";"}
            )
        ):
            if i + 1 < len(segments):
                joined = text + " " + segments[i+1]['text']
                sim = semantic_similarity(text, joined)
                if sim > 0.7:
                    continue
            incomplete.append(seg)
    return incomplete 

def main():
    video = load_video(VIDEO_PATH)
    audio_clip = extract_audio(video, AUDIO_PATH)

    audio_segments = split_audio(audio_clip, SEGMENT_LENGTH)
    transcriptions = [transcribe_segment(seg['path'], seg['start']) for seg in audio_segments]

    merged_transcription = merge_transcriptions(transcriptions)
    segments = group_words_to_segments(merged_transcription['words'])

    repeated_segments = detect_repeated_content(segments)
    filler_segments = detect_fillers(segments)
    incomplete_segments = detect_incomplete_sentences(segments)
    
    all_removals = repeated_segments + filler_segments
    final_video = remove_segments(video, all_removals)

    with open("transcript_detailed_log.txt", "w", encoding="utf-8") as f:
        f.write("===== ORIGINAL TRANSCRIPT =====\n\n")
        for seg in segments:
            f.write(f"{seg['start']:.2f}s - {seg['end']:.2f}s: {seg['text']}\n")

        f.write("\n===== SEGMENTS REMOVED (Repetitions & Fillers) =====\n\n")
        for seg in sorted(all_removals, key=lambda x: x['start']):
            reason = "Filler" if seg in filler_segments else "Repetition"
            f.write(f"[{reason}] {seg['start']:.2f}s - {seg['end']:.2f}s: {seg['text']}\n")

        cleaned_segments = [seg for seg in segments if seg not in all_removals]

        f.write("\n===== CLEANED TRANSCRIPT (Final Video) =====\n\n")
        for seg in cleaned_segments:
            f.write(f"{seg['start']:.2f}s - {seg['end']:.2f}s: {seg['text']}\n")

    final_video.write_videofile("final_cleaned_video.mp4", codec="libx264", audio_codec="aac")


if __name__ == "__main__":
    main()
