import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests
import moviepy.editor as mp
import math
import re
from moviepy.editor import concatenate_videoclips, CompositeVideoClip, TextClip
from typing import Any, List, Dict
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

VIDEO_PATH = 'Video1.mov'
AUDIO_PATH = 'Video_audio.wav'
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

NEW_SENTENCE_CUES = {'so', 'but', 'and', 'however', 'moreover', 'meanwhile', 'now', 'then'}

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
    url = "http://localhost:9000/asr"
    params = {"task": "transcribe", "language": "en", "output": "json", "encode": "true"}

    with open(segment_audio_path, 'rb') as f:
        files = {'audio_file': f}
        response = requests.post(url, params=params, files=files, timeout=60)

    if response.status_code == 200:
        transcription = response.json()
        segments = []
        for seg in transcription.get("segments", []):
            segments.append({
                'start': seg['start'] + start_time,
                'end': seg['end'] + start_time,
                'text': seg['text'].strip()
            })
        return segments
    else:
        raise Exception(f"Transcription failed: {response.status_code} - {response.text}")

def normalize_text(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9 ]+', '', text.lower()).strip()

def semantic_similarity(a: str, b: str) -> float:
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(emb1, emb2))

def merge_segments_contextually(segments: List[Dict[str, Any]], gap_threshold: float = 1.0) -> List[Dict[str, Any]]:
    if not segments:
        return []

    merged = []
    buffer = segments[0].copy()

    for seg in segments[1:]:
        gap = seg['start'] - buffer['end']
        no_punctuation = not re.search(r'[.!?]$', buffer['text'])
        first_word = seg['text'].strip().split()[0].lower() if seg['text'].strip() else ""

        if gap <= gap_threshold and no_punctuation and first_word not in NEW_SENTENCE_CUES:
            buffer['end'] = seg['end']
            buffer['text'] += " " + seg['text']
        else:
            merged.append(buffer)
            buffer = seg.copy()

    merged.append(buffer)
    return merged

def detect_repeated_content(segments: List[Dict[str, Any]], threshold: float = 0.85) -> List[Dict[str, Any]]:
    repeated_indices = set()
    n = len(segments)

    for i in range(n):
        for j in range(i + 1, n):
            sim = semantic_similarity(segments[i]['text'], segments[j]['text'])
            if sim > threshold:
                repeated_indices.add(i)  # Mark earlier one, keep latest
                break

    return [segments[i] for i in repeated_indices]

def detect_fillers(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [seg for seg in segments if any(filler in normalize_text(seg['text']) for filler in FILLER_WORDS)]

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
            # check if next segment completes it
            if i + 1 < len(segments):
                joined = text + " " + segments[i+1]['text']
                sim = semantic_similarity(text, joined)
                if sim > 0.7:
                    continue  # it's continued by next segment
            incomplete.append(seg)
    return incomplete

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

def main():
    video = load_video(VIDEO_PATH)
    audio_clip = extract_audio(video, AUDIO_PATH)

    audio_segments = split_audio(audio_clip, SEGMENT_LENGTH)
    raw_segments = []
    for seg in audio_segments:
        raw_segments.extend(transcribe_segment(seg['path'], seg['start']))

    segments = merge_segments_contextually(raw_segments)

    repeated_segments = detect_repeated_content(segments)
    filler_segments = detect_fillers(segments)
    incomplete_segments = detect_incomplete_sentences(segments)

    all_removals = list({id(seg): seg for seg in repeated_segments + filler_segments + incomplete_segments}.values())
    final_video = remove_segments(video, all_removals)

    with open("transcript_detailed_log.txt", "w", encoding="utf-8") as f:
        f.write("===== ORIGINAL TRANSCRIPT =====\n\n")
        for seg in segments:
            f.write(f"{seg['start']:.2f}s - {seg['end']:.2f}s: {seg['text']}\n")

        f.write("\n===== SEGMENTS REMOVED (Repetitions, Fillers, Incomplete) =====\n\n")
        for seg in sorted(all_removals, key=lambda x: x['start']):
            reason = []
            if seg in repeated_segments: reason.append("Repetition")
            if seg in filler_segments: reason.append("Filler")
            if seg in incomplete_segments: reason.append("Incomplete")
            f.write(f"[{', '.join(reason)}] {seg['start']:.2f}s - {seg['end']:.2f}s: {seg['text']}\n")

        cleaned_segments = [seg for seg in segments if seg not in all_removals]
        f.write("\n===== CLEANED TRANSCRIPT (Final Video) =====\n\n")
        for seg in cleaned_segments:
            f.write(f"{seg['start']:.2f}s - {seg['end']:.2f}s: {seg['text']}\n")

        final_video.write_videofile("final_cleaned_video.mp4", codec="libx264", audio_codec="aac")

if __name__ == "__main__":
    main()
