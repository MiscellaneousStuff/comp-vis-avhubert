import os
from moviepy.editor import VideoFileClip

def split_video_into_clips(input_file, output_directory, clip_duration=10):
    video = VideoFileClip(input_file)
    duration = video.duration
    
    # Calculate the number of full 10-second clips
    num_full_clips = int(duration / clip_duration)
    
    # Calculate the remaining duration
    remaining_duration = duration % clip_duration
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Extract full 10-second clips
    for i in range(num_full_clips):
        start_time = i * clip_duration
        end_time = (i + 1) * clip_duration
        clip = video.subclip(start_time, end_time)
        clip_filename = os.path.join(output_directory, f"clip_{i + 1}.mp4")
        clip.write_videofile(clip_filename, codec="libx264")
    
    # Extract the final clip with remaining duration if any
    if remaining_duration > 0:
        start_time = num_full_clips * clip_duration
        end_time = duration
        clip = video.subclip(start_time, end_time)
        clip_filename = os.path.join(output_directory, f"clip_{num_full_clips + 1}.mp4")
        clip.write_videofile(clip_filename, codec="libx264")
    
    # Close the original video file
    video.close()

# # Usage example
# input_file = "dataset/NX2ep5fCJZ8/Jordan Peterson on the meaning of life for men MUST WATCH.mp4"
# output_directory = "clips/"
# clip_duration = 10  # Duration of each clip in seconds

# # Usage example
input_file = "dataset/DC0faZiBcG0/Lecture 2015 Personality Lecture 06 Depth Psychology Carl Jung (Part 01).mp4"
output_directory = "clips/"
clip_duration = 10  # Duration of each clip in seconds

split_video_into_clips(input_file, output_directory, clip_duration)