import os
import re
import string
import time
import requests
import numpy as np
import moviepy.editor as mp

# -------------------- CONSTANTS --------------------
# Filler words and regex patterns
FILLER_WORDS = {
    'um', 'uh', 'er', 'ah', 'like', 'hmm', 'you know', 'i mean', 'actually',
    'basically', 'literally', 'so', 'right', 'well', 'kind of', 'sort of',
    'just', 'anyway', 'whatever', 'yeah', 'okay', 'uhm', 'mmm', 'huh', 'eh',
    'ya know', 'mm-hmm', 'mmm-hmm', 'erm', 'umm', 'uhh', 'err'
}

FILLER_PATTERNS = [
    r'\b(?:um|uh|er|ah|uhm|hmm|mmm)\b',
    r'\b(?:i mean|you know|ya know|kind of|sort of)\b',
    r'\b(?:like|basically|actually|literally|so)\b(?:\s+(?:um|uh|like))?',
    r'\b(?:right|okay|anyway|whatever)\b(?:\s+(?:um|uh|like))?',
    r'\bso\b\s+(?:um|uh|like|yeah)?',
    r'\bwell\b\s+(?:um|uh|like|yeah)?',
    r'\buhh+\b|\bumm+\b|\buhm+\b|\bah+\b|\ber+\b|\bmm+\b',
    r'\boh\b\s+(?:um|uh|like|yeah)?',
    r'\b(?:mm-hmm|mmm-hmm|mm-mm)\b'
]

# Whisper API file size and duration limits (example constants; adjust as needed)
WHISPER_MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024   # 25 MB file size limit
WHISPER_MAX_DURATION_SECONDS = 300                # 5 minutes duration limit

# Other global settings
VIDEO_PATH = "Video3.mov"
AUDIO_PATH = "Video_audio.wav"
SEGMENT_LENGTH = 30         # seconds per segment
WHISPER_URL = "http://34.46.113.13:9001/asr"
PADDING = 0.5               # seconds (if needed for segment adjustments)

# -------------------- UTILITY FUNCTIONS --------------------
def get_audio_info(audio_path: str):
    """Get duration and file size of an audio file."""
    audio_clip = mp.AudioFileClip(audio_path)
    duration = audio_clip.duration
    file_size = os.path.getsize(audio_path)
    audio_clip.close()
    return duration, file_size

def to_seconds(timestamp: str) -> float:
    """
    Convert a timestamp (e.g., "0:0:12.34") to seconds.
    Supports comma or dot as decimal separator.
    """
    parts = timestamp.replace(',', '.').split(':')
    while len(parts) < 3:
        parts.insert(0, '0')
    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])

# -------------------- CORE FUNCTIONS --------------------
def calculate_optimal_chunk_duration(audio_path: str) -> float:
    """
    Calculate the optimal chunk duration for splitting the audio
    based on the audio file's size-to-duration ratio to stay within Whisper's limits.
    """
    try:
        duration, file_size = get_audio_info(audio_path)
        bytes_per_second = file_size / duration if duration > 0 else 0
        size_limited_duration = WHISPER_MAX_FILE_SIZE_BYTES / bytes_per_second if bytes_per_second > 0 else float('inf')
        optimal_duration = min(WHISPER_MAX_DURATION_SECONDS, size_limited_duration)
        safe_duration = optimal_duration * 0.9  # safety margin
        print(f"Calculated optimal chunk duration: {safe_duration:.2f}s")
        return max(30, safe_duration)  # ensure at least 30 seconds per chunk
    except Exception as e:
        print(f"Error calculating chunk duration: {e}")
        return 300  # default to 5 minutes if errors arise

def adjust_timestamps(transcription: dict, chunk_offset: float) -> dict:
    """
    Adjusts the word timestamps in a transcription dict based on the chunk offset.
    """
    for word in transcription.get("words", []):
        word["start"] += chunk_offset
        word["end"] += chunk_offset
    return transcription

def merge_transcriptions(transcriptions: list) -> dict:
    """
    Merge multiple transcriptions (from individual audio segments) into one.
    Merges the 'words' lists and sorts them by start time.
    """
    if not transcriptions:
        return None
    merged = {"words": [], "phrases": []}
    for transcription in transcriptions:
        merged["words"].extend(transcription.get("words", []))
        merged["phrases"].extend(transcription.get("phrases", []))
    merged["words"].sort(key=lambda x: x["start"])
    merged["phrases"].sort(key=lambda x: x["start"])
    print(f"Merged transcription contains {len(merged['words'])} words from {len(transcriptions)} chunks")
    return merged

def group_words_to_segments(words: list, gap_threshold: float = 0.5) -> list:
    """
    Group a list of word objects (with start, end, and word keys) into segments.
    A new segment is started if the gap between words exceeds the gap_threshold.
    """
    segments = []
    if not words:
        return segments

    current_start = words[0]["start"]
    current_end = words[0]["end"]
    current_text = words[0]["word"]
    current_words = [words[0]]
    
    for word in words[1:]:
        if word["start"] - current_end <= gap_threshold:
            current_text += " " + word["word"]
            current_end = word["end"]
            current_words.append(word)
        else:
            segments.append({
                "start": current_start,
                "end": current_end,
                "text": current_text,
                "words": current_words
            })
            current_start = word["start"]
            current_end = word["end"]
            current_text = word["word"]
            current_words = [word]
    
    segments.append({
        "start": current_start,
        "end": current_end,
        "text": current_text,
        "words": current_words
    })
    print(f"Created {len(segments)} caption segments from {len(words)} words")
    return segments

def detect_filler_words(segments: list) -> list:
    """
    Detect segments that contain filler words using both exact word matching and regex patterns.
    Returns a list of segments flagged as filler.
    """
    print("Analyzing transcript for filler words...")
    filler_segments = []
    compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in FILLER_PATTERNS]
    
    for segment in segments:
        text = segment["text"].lower()
        words = text.split()
        # Check for exact filler word matches
        for word in words:
            clean_word = word.strip(string.punctuation)
            if clean_word in FILLER_WORDS:
                filler_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "filler": clean_word,
                    "type": "filler_word"
                })
                print(f"Found filler word '{clean_word}' in: {text}")
                break

        # If not already flagged, check regex patterns
        if not any(fs["start"] == segment["start"] and fs["end"] == segment["end"] for fs in filler_segments):
            for pattern in compiled_patterns:
                match = pattern.search(text)
                if match:
                    filler_segments.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"],
                        "filler": match.group(0),
                        "type": "filler_pattern"
                    })
                    print(f"Found filler pattern '{match.group(0)}' in: {text}")
                    break
    print(f"Found {len(filler_segments)} segments with filler words/patterns")
    return filler_segments

def detect_repeated_content(segments: list, similarity_threshold: float = 0.8, max_gap: float = 30.0) -> list:
    """
    Detect repeated words or phrases within segments based on token overlap.
    Marks segments as repeated if similarity exceeds the threshold.
    """
    print("Analyzing transcript for repeated content...")
    
    def normalize_text(text):
        return re.sub(r'[^\w\s]', '', text.lower())
    
    def calculate_similarity(text1, text2):
        set1 = set(normalize_text(text1).split())
        set2 = set(normalize_text(text2).split())
        if not set1 or not set2:
            return 0.0
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union)
    
    repeated_segments = []
    sorted_segments = sorted(segments, key=lambda x: x["start"])
    
    for i in range(len(sorted_segments) - 1):
        for j in range(i + 1, len(sorted_segments)):
            time_gap = sorted_segments[j]["start"] - sorted_segments[i]["end"]
            if time_gap > max_gap:
                break
            similarity = calculate_similarity(sorted_segments[i]["text"], sorted_segments[j]["text"])
            if similarity >= similarity_threshold:
                print("Found repeated content:")
                print(f"  First: {sorted_segments[i]['text']}")
                print(f"  Second: {sorted_segments[j]['text']}")
                print(f"  Similarity: {similarity:.2f}")
                repeated_segments.append({
                    "start": sorted_segments[i]["start"],
                    "end": sorted_segments[i]["end"],
                    "text": sorted_segments[i]["text"],
                    "repeated_at": sorted_segments[j]["start"],
                    "similarity": similarity,
                    "type": "repetition"
                })
                break
    print(f"Found {len(repeated_segments)} repeated segments")
    return repeated_segments
def combine_segments_to_remove(filler_segments: list, repeated_segments: list,
                               min_segment_duration: float = 0.8,
                               merge_threshold: float = 0.3) -> list:
    """
    Combine only filler and repeated content segments that should be removed,
    merging overlapping or closely spaced segments.

    Args:
        filler_segments: List of segments identified as containing filler words.
        repeated_segments: List of segments identified as repeated content.
        min_segment_duration: Minimum duration (in seconds) for a segment to be removed.
        merge_threshold: Maximum gap (in seconds) between segments to merge them.

    Returns:
        List of merged segments to remove.
    """
    # Combine only filler and repeated segments.
    all_segments = []
    all_segments.extend(filler_segments)
    all_segments.extend(repeated_segments)
    
    # Sort the combined segments by start time.
    all_segments.sort(key=lambda x: x["start"])
    print(f"Combined {len(all_segments)} segments for potential removal")
    
    merged_segments = []
    if not all_segments:
        return merged_segments

    current_segment = all_segments[0].copy()
    for segment in all_segments[1:]:
        # If the segment overlaps or is close enough, merge it.
        if segment["start"] <= current_segment["end"] + merge_threshold:
            current_segment["end"] = max(current_segment["end"], segment["end"])
            # Concatenate types if needed.
            if current_segment["type"] != segment["type"]:
                current_segment["type"] = f"{current_segment['type']}+{segment['type']}"
        else:
            # Only keep segments that are long enough.
            if current_segment["end"] - current_segment["start"] >= min_segment_duration:
                merged_segments.append(current_segment)
            current_segment = segment.copy()
    
    # Append the final segment if it's long enough.
    if current_segment["end"] - current_segment["start"] >= min_segment_duration:
        merged_segments.append(current_segment)
    
    print(f"After merging, {len(merged_segments)} segments will be removed")
    total_duration = sum(seg["end"] - seg["start"] for seg in merged_segments)
    print(f"Total duration to remove: {total_duration:.2f} seconds")
    return merged_segments

def calculate_segments_to_keep(video_duration: float, segments_to_remove: list, max_removal_percentage: float = 0.25) -> list:
    """
    Given the total video duration and segments identified for removal,
    compute the segments to keep ensuring that no more than max_removal_percentage of the video is removed.
    """
    total_removal_duration = sum(seg["end"] - seg["start"] for seg in segments_to_remove)
    removal_percentage = total_removal_duration / video_duration
    if removal_percentage > max_removal_percentage:
        print(f"Warning: Removal would exceed threshold ({removal_percentage:.1%} > {max_removal_percentage:.1%})")
        print("Prioritizing segments to stay under threshold...")
        def get_priority(segment_type):
            priorities = {
                "silence": 1,          # highest removal priority
                "filler_word": 2,
                "repetition": 3,
                "low_clarity": 4       # lowest removal priority
            }
            if "+" in segment_type:
                types = segment_type.split("+")
                return min(priorities.get(t, 10) for t in types)
            return priorities.get(segment_type, 10)
        sorted_segments = sorted(
            segments_to_remove,
            key=lambda x: (get_priority(x["type"]), -1 * (x["end"] - x["start"]))
        )
        selected_segments = []
        current_duration = 0
        max_duration = video_duration * max_removal_percentage
        for segment in sorted_segments:
            segment_duration = segment["end"] - segment["start"]
            if current_duration + segment_duration <= max_duration:
                selected_segments.append(segment)
                current_duration += segment_duration
            if current_duration >= max_duration:
                break
        segments_to_remove = selected_segments
        print(f"Limited removal to {len(segments_to_remove)} segments totaling {current_duration:.2f}s")
    
    segments_to_remove.sort(key=lambda x: x["start"])
    segments_to_keep = []
    current_time = 0
    for segment in segments_to_remove:
        if segment["start"] > current_time + 0.1:
            segments_to_keep.append((current_time, segment["start"]))
        current_time = max(current_time, segment["end"])
    if current_time < video_duration - 0.1:
        segments_to_keep.append((current_time, video_duration))
    
    kept_duration = sum(end - start for start, end in segments_to_keep)
    kept_percentage = kept_duration / video_duration
    print(f"Created {len(segments_to_keep)} segments to keep")
    print(f"Kept duration: {kept_duration:.2f}s ({kept_percentage:.1%} of video)")
    return segments_to_keep

# -------------------- VIDEO & AUDIO PROCESSING --------------------
def load_video(video_path: str) -> mp.VideoFileClip:
    """Load a video using moviepy."""
    return mp.VideoFileClip(video_path)

def extract_audio(video: mp.VideoFileClip, audio_path: str) -> mp.AudioFileClip:
    """Extract audio from a video and write it to file."""
    video.audio.write_audiofile(audio_path, verbose=False)
    return mp.AudioFileClip(audio_path)

def split_audio(audio_clip: mp.AudioFileClip, segment_length: int) -> list:
    """
    Splits an audio clip into segments of a given length (in seconds)
    and writes each segment to a separate file.
    """
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

def transcribe_segment(segment_audio_path: str, start_time: float, retries: int = 3, timeout: int = 120) -> dict:
    """
    Transcribe a given audio segment via the Whisper API.
    Adjusts segment timestamps using the provided chunk start time.
    """
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
                    # Convert segment start and end to seconds and add the chunk start_time offset
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
        except requests.exceptions.ReadTimeout:
            attempt += 1
            print(f"Attempt {attempt}/{retries}: Timeout error for segment '{segment_audio_path}'. Retrying in 1 second...")
            time.sleep(1)
    raise Exception(f"Transcription failed for segment '{segment_audio_path}' after {retries} attempts.")

def create_final_video(video: mp.VideoFileClip, segments_to_keep: list, output_path: str):
    """
    Create a new video file containing only the segments to keep.
    Concatenates the subclips extracted from the original video.
    """
    kept_clips = [video.subclip(start, end) for (start, end) in segments_to_keep]
    final_video = mp.concatenate_videoclips(kept_clips)
    final_video.write_videofile(output_path, codec="libx264")
    print(f"Final video saved to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to the input video file")

    args = parser.parse_args()

    video_path = args.video
    output_audio = "temp_audio.wav"
    final_video_path = "final_trimmed_video.mp4"
    report_path = "final_report.txt"

    print("Loading video...")
    video = load_video(video_path)
    print("Extracting audio from video...")
    audio_clip = extract_audio(video, output_audio)

    print("Splitting audio into segments...")
    segments = split_audio(audio_clip, SEGMENT_LENGTH)

    transcriptions = []
    for seg in segments:
        print(f"Transcribing segment starting at {seg['start']:.2f}s...")
        transcription = transcribe_segment(seg["path"], seg["start"])
        transcriptions.append(transcription)

    merged = merge_transcriptions(transcriptions)
    all_words = merged.get("words", [])
    all_phrases = merged.get("phrases", [])

    caption_segments = group_words_to_segments(all_words)
    filler_segments = detect_filler_words(caption_segments)
    repeated_segments = detect_repeated_content(caption_segments)

    segments_to_remove = combine_segments_to_remove(filler_segments, repeated_segments)
    segments_to_keep = calculate_segments_to_keep(video.duration, segments_to_remove)

    kept_phrases = []
    for (start, end) in segments_to_keep:
        texts = [ph["text"] for ph in all_phrases if ph["start"] >= start and ph["end"] <= end]
        if texts:
            kept_phrases.append({"start": start, "end": end, "text": " ".join(texts)})

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("===== WORDS =====\n\n")
        for w in all_words:
            f.write(f"{w['start']:.2f}-{w['end']:.2f}: {w['word']}\n")
        f.write("\n===== PHRASES =====\n\n")
        for ph in all_phrases:
            f.write(f"{ph['start']:.2f}-{ph['end']:.2f}: {ph['text']}\n")
        f.write("\n===== FILLERS REMOVED =====\n\n")
        for fi in filler_segments:
            f.write(f"{fi['start']:.2f}-{fi['end']:.2f}: {fi['text']}\n")
        f.write("\n===== REPEATED PHRASES REMOVED =====\n\n")
        for rp in repeated_segments:
            f.write(f"{rp['start']:.2f}-{rp['end']:.2f}: {rp['text']}\n")
        f.write("\n===== FINAL TRANSCRIPT KEPT (BEST Takes) =====\n\n")
        for seg in kept_phrases:
            f.write(f"{seg['start']:.2f}-{seg['end']:.2f}: {seg['text']}\n")

    print("Detailed transcript log written to", report_path)
    create_final_video(video, segments_to_keep, final_video_path)


if __name__ == '__main__':
    main()
