#!tsn model testing
#num_segments=5
#haidong kang
#0523.2019
#your_model_resnet50__rgb_model_best.pth.tar
nohup python3 -u main.py 6 RGB /workspace/mnt/group/algorithm/kanghaidong/video_project/Video1.0_list_file/bk-video-list-V1.0-a-train-databalance.txt /workspace/mnt/group/algorithm/kanghaidong/video_project/Video1.0_list_file/bk-video-list-V1.0-a-val.txt \
   --input_type Video --video_handler cv2 \
   --arch resnet50 --num_segments 3 \
   --gd 20 --lr 0.01 --lr_steps 30 60 --epochs 200 \
   --resume /workspace/mnt/group/algorithm/kanghaidong/video_project/Alg-VideoAlgorithm-dev/video-classification/tsn-pytorch/resnet50_80e_69.98p.pth.tar \
   -b 128 -j 8 --dropout 0.8 \
   --snapshot_pref bk-V1.0_2.1.4_resnet50_ > V1.0_2.1.4_out.out 2>&1 &