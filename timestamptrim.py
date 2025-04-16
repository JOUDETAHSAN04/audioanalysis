import moviepy.editor as mp

def trim_and_concatenate(input_video, output_video, segments):
    """
    Trims the input video to the specified segments and concatenates them into a final video.
    
    :param input_video: str, path to the input video file.
    :param output_video: str, path for the resulting output video file.
    :param segments: list of tuples, each tuple is (start, end) time in seconds.
    """
    # Load the original video.
    video = mp.VideoFileClip(input_video)
    
    # Extract the specified segments.
    clips = []
    for start, end in segments:
        clip = video.subclip(start, end)
        clips.append(clip)
    
    # Concatenate all extracted clips.
    final_clip = mp.concatenate_videoclips(clips)
    
    # Write the final video to a file.
    final_clip.write_videofile(output_video, codec="libx264", audio_codec="aac")

if __name__ == "__main__":
    # Example input:
    input_video = "video1.mov"  # Replace with your video filename.
    output_video = "final_trimmed_video.mp4"
    
    # List of segments as (start, end) in seconds.
    # segments = [
    #     (2.64, 9.34),
    #     (9.74, 14.96)
    # ]
    segments = [
    (3.80, 7.44),    # "You will lose your job to AI unless you do this now."
    (23.88, 26.74),  # "Share this with a friend who needs to take this advice."
    (46.46, 52.66),  # "Subscribe to learn how to do this and to hear the latest news on AI."
    (61.34, 64.70),  # "The economy is on the brink of massive change and..."
    (92.48, 92.90)   # "Thank you."
    ]

    
    trim_and_concatenate(input_video, output_video, segments)
