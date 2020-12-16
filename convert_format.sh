video_folder="data/DDPG-0-shaping-rand"
ffmpeg -hwaccel cuvid -i ${video_folder}/output.mp4 -vcodec libx264 ${video_folder}/output_h264.mp4
# ffmpeg -hwaccel cuvid -c:v h264_cuvid -i ${video_folder}/output.mp4 -c:a copy -c:v h264_nvenc ${video_folder}/output_h264.mp4