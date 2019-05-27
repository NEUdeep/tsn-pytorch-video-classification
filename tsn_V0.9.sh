#!tsn_new video classfication model script
#python3
#haidong kang
#20190521
#consensus_type == 'video'
#train_list ==''
#val_list == ''
#no mixup data augmentation 
#num_class==5
#out of memory cuda module
#pytorch1.0.1 gpu cuda 9.0
nohup python3 -u main.py 6 RGB /workspace/mnt/group/algorithm/kanghaidong/video_project/Video0.9_list_file/bk-video-list-V0.9-a-train-databalance.txt /workspace/mnt/group/algorithm/kanghaidong/video_project/Video0.9_list_file/bk-video-list-V0.9-a-val.txt \
   --input_type Video --video_handler cv2 \
   --arch inceptionv4 --num_segments 3 \
   --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 160 \
   -b 128 -j 8 --dropout 0.8 \
   --snapshot_pref your_model_inceptionv4_ > V0.9_pars3_out.out 2>&1 &

