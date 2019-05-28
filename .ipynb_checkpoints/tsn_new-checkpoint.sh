#!tsn_new video classfication model script
#python3
#haidong kang
#20190517
#consensus_type == 'video'
#train_list ==''
#val_list == ''
#no mixup data augmentation 
#no tensorboardX,because it will get error
#out of memory cuda module
#pytorch1.0.1 gpu cuda 9.0
nohup python3 -u main.py 101 RGB /workspace/mnt/group/video/kanghaidong/dataset/ucf_list_file/trainlist01.txt /workspace/mnt/group/video/kanghaidong/dataset/ucf_list_file/testlist01.txt \
   --input_type Video --video_handler cv2 \
   --arch resnet50 --num_segments 3 \
   --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 10 \
   -b 128 -j 8 --dropout 0.8 \
   --snapshot_pref your_model_resnet50_ > out.out 2>&1 &
