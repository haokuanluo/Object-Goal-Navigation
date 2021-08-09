import os
#os.system("ffmpeg -r 1 -i ~/habitat-train/habitat-challenge-data/eval_no_sliding_pointnav/dump/exp1/episodes/1/1/*.png -vcodec mpeg4 -y movie.mp4")
os.system("ffmpeg -f image2 -r 10 -i ~/habitat-train/habitat-challenge-data/eval34/dump/exp1/episodes/1/bed1/0-4-Vis-%d.png -vcodec mpeg4 -y findsofa2.mp4")


