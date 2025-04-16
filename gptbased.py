import argparse
import time
import json
from typing import Dict, Any, List, Tuple
import os
import re
import requests
import numpy as np
import moviepy.editor as mp
from openai import OpenAI  # Using the new client style

# Set your Whisper API endpoint.
WHISPER_URL = "http://34.46.113.13:9001/asr"  # Replace with your actual Whisper API endpoint URL.

def to_seconds(time_str: str) -> float:
    """
    Convert a timestamp string in the format "H:M:S" into seconds.
    Example: "0:0:12.34" becomes 12.34 seconds.
    """
    parts = time_str.split(":")
    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds

def extract_audio(video: mp.VideoFileClip, audio_path: str) -> mp.AudioFileClip:
    """
    Extracts audio from the video and saves it to the specified file.
    """
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    return mp.AudioFileClip(audio_path)

def transcribe_segment(segment_audio_path: str, start_time: float, retries: int = 3, timeout: int = 1200) -> Dict[str, Any]:
    """
    Transcribes an audio segment using the Whisper API.
    
    :param segment_audio_path: Path to the audio file.
    :param start_time: Offset (in seconds) added to each timestamp.
    :param retries: Number of attempts.
    :param timeout: Request timeout in seconds.
    :return: Dictionary with keys "words" and "phrases".
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
                    seg_start = to_seconds(f"0:0:{seg['start']:.2f}") + start_time
                    seg_end = to_seconds(f"0:0:{seg['end']:.2f}") + start_time
                    text = seg["text"].strip()
                    phrases.append({"start": seg_start, "end": seg_end, "text": text})
                    for word_info in seg.get("words", []):
                        word_start = to_seconds(f"0:0:{word_info['start']:.2f}") + start_time
                        word_end = to_seconds(f"0:0:{word_info['end']:.2f}") + start_time
                        words.append({
                            "start": word_start,
                            "end": word_end,
                            "word": word_info['word'].strip()
                        })
                return {"words": words, "phrases": phrases}
            else:
                raise Exception(f"Transcription failed: {response.status_code} - {response.text}")
        except requests.exceptions.ReadTimeout:
            attempt += 1
            print(f"Attempt {attempt}/{retries}: Timeout error for '{segment_audio_path}'. Retrying in 1 second...")
            time.sleep(1)
    raise Exception(f"Transcription failed for '{segment_audio_path}' after {retries} attempts.")

def split_audio(audio_clip: mp.AudioFileClip, segment_length: int) -> List[Dict[str, Any]]:
    """
    Splits the audio clip into segments of segment_length seconds.
    
    :param audio_clip: The MoviePy AudioFileClip object.
    :param segment_length: Length (in seconds) of each segment.
    :return: A list of dictionaries, each with keys "path" (to the segment file) and "start" (time offset).
    """
    duration = audio_clip.duration
    segments = []
    num_segments = int(np.ceil(duration / segment_length))
    for i in range(num_segments):
        start_time = i * segment_length
        end_time = min((i + 1) * segment_length, duration)
        segment = audio_clip.subclip(start_time, end_time)
        segment_audio_path = f"segment_{i}.wav"
        segment.write_audiofile(segment_audio_path, verbose=False, logger=None)
        segments.append({"path": segment_audio_path, "start": start_time})
    return segments

def save_whisper_output(output: Dict[str, Any], filename: str):
    """
    Saves the word- and phrase-level transcription data to a text file.
    
    The file contains two sections:
      ===== WORDS =====
      (Each line: start-end: word)
      
      ===== PHRASES =====
      (Each line: start-end: phrase)
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write("===== WORDS =====\n")
        for word in output.get("words", []):
            f.write(f"{word['start']:.2f}-{word['end']:.2f}: {word['word']}\n")
        f.write("\n===== PHRASES =====\n")
        for phrase in output.get("phrases", []):
            f.write(f"{phrase['start']:.2f}-{phrase['end']:.2f}: {phrase['text']}\n")
    print(f"Whisper output saved to {filename}")

def build_transcript_text(phrases: List[Dict[str, Any]]) -> str:
    """
    Constructs a transcript text from the phrase list.
    
    Each line: "start-end: phrase".
    """
    lines = []
    for p in phrases:
        line = f"{p['start']:.2f}-{p['end']:.2f}: {p['text']}"
        lines.append(line)
    return "\n".join(lines)

def prompt_chatgpt_for_deduplication(transcript_text: str) -> str:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    improved_prompt = (
        "You are an expert video transcription editor. "
        "I will provide you with a full transcription that contains both word-level and phrase-level timestamps. "
        "Your task is to generate a final transcript that contains only unique phrases—do not include any duplicate or near-duplicate phrases. "
        "If two or more phrases have the same or very similar meaning, only include one version. "
        "Also, if any sentence or part of a sentence repeats within a phrase, trim the repeated parts. "
        "Output your result as plain text with one section header:\n"
        "===== FINAL TRANSCRIPT KEPT (BEST Takes) =====\n"
        "Below that header, list each final segment in the format:\n"
        "start-end: final transcript text\n"
        "Ensure that no two segments are duplicates or very similar."
    )
    response = client.responses.create(
        model="gpt-4o",  # Replace with "gpt-4" or "gpt-3.5-turbo" if needed
        instructions=improved_prompt,
        input=transcript_text,
    )
    output_text = response.output_text.strip()
    # Remove markdown formatting if present.
    if output_text.startswith("```"):
        output_text = re.sub(r"^```(?:json)?", "", output_text).strip()
    if output_text.endswith("```"):
        output_text = output_text[:-3].strip()
    return output_text

def parse_final_transcript(report_text: str) -> List[Tuple[float, float, str]]:
    final_segments = []
    section_pattern = r"===== FINAL TRANSCRIPT KEPT \(BEST Takes\) ====="
    match = re.search(section_pattern, report_text)
    if not match:
        raise Exception("Final transcript section not found in the report.")
    post_section = report_text[match.end():].strip()
    lines = post_section.splitlines()
    for line in lines:
        if line.startswith("====="):
            break
        line = line.strip()
        if not line:
            continue
        parts = line.split(":", 1)
        if len(parts) != 2:
            continue
        times, text = parts[0].strip(), parts[1].strip()
        time_parts = times.split("-")
        if len(time_parts) != 2:
            continue
        try:
            start = float(time_parts[0])
            end = float(time_parts[1])
            final_segments.append((start, end, text))
        except ValueError:
            continue
    return final_segments

def save_report_to_txt(report_text: str, report_filename: str):
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Report saved to {report_filename}")

def trim_and_concatenate(input_video: str, output_video: str, segments: List[Tuple[float, float, str]]):

    print("Loading video...")
    video = mp.VideoFileClip(input_video)
    clips = []
    for start, end, _ in segments:
        print(f"Extracting clip from {start:.2f} to {end:.2f} seconds...")
        clip = video.subclip(start, end)
        clips.append(clip)
    print("Concatenating clips...")
    final_clip = mp.concatenate_videoclips(clips)
    print(f"Writing final video to {output_video} ...")
    final_clip.write_videofile(output_video, codec="libx264", audio_codec="aac")
    try:
        video.close() 
    except Exception:
        pass
def main():
    parser = argparse.ArgumentParser(description="Takes 1 video argument and auto-generates audio, report, and output video paths.")
    parser.add_argument("video", type=str, help="Path to the input video file (e.g., video1.mov)")
    args = parser.parse_args()

    input_video = args.video
    base_name = os.path.splitext(os.path.basename(input_video))[0]
    audio_path = f"{base_name}_audio.wav"
    report_path = f"{base_name}_report.txt"
    output_video = f"{base_name}_final.mp4"

    print("Extracting audio from video...")
    video_clip = mp.VideoFileClip(input_video)
    audio_clip = extract_audio(video_clip, audio_path)

    print("Splitting audio into 30-second segments...")
    segments_list = split_audio(audio_clip, segment_length=30)

    combined_words = []
    combined_phrases = []
    for seg in segments_list:
        print(f"Transcribing segment '{seg['path']}' starting at {seg['start']:.2f} seconds...")
        result = transcribe_segment(seg["path"], start_time=seg["start"])
        combined_words.extend(result.get("words", []))
        combined_phrases.extend(result.get("phrases", []))

    combined_output = {"words": combined_words, "phrases": combined_phrases}
    save_whisper_output(combined_output, f"{base_name}_whisper_output.txt")

    transcript_text = build_transcript_text(combined_phrases)
    with open(f"{base_name}_whisper_transcript.txt", "w", encoding="utf-8") as f:
        f.write(transcript_text)

    print("Sending transcription to ChatGPT for deduplication and trimming...")
    report_text = prompt_chatgpt_for_deduplication(transcript_text)
    save_report_to_txt(report_text, report_path)

    final_segments = parse_final_transcript(report_text)
    if not final_segments:
        raise Exception("No final transcript segments found in the report.")

    print("Processing final video...")
    trim_and_concatenate(input_video, output_video, final_segments)


if __name__ == "__main__":
    main()
