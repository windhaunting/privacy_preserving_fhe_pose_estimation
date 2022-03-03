export PYTHONUNBUFFERED="True"
LOG="log/train-`date +'%Y-%m-%d-%H-%M-%S'`.log"

# data still in the TSM_video_analytics directory, because the generated coco image paths use this directory.
python3 train_pose_up_to_models22.py --gpu 0 1 --train_dir /var/fubao/TSM_video_analytics/TSM_Pose_Estimation/input_data/filelist/train2017_file.txt /var/fubao/TSM_video_analytics/TSM_Pose_Estimation/input_data/masklist/train2017_mask.txt /var/fubao/TSM_video_analytics/TSM_Pose_Estimation/input_data/json/train2017.json --val_dir /var/fubao/TSM_video_analytics/TSM_Pose_Estimation/input_data/filelist/val2017_file.txt /var/fubao/TSM_video_analytics/TSM_Pose_Estimation/input_data/masklist/val2017_mask.txt /var/fubao/TSM_video_analytics/TSM_Pose_Estimation/input_data/json/val2017.json --config config.yml 
