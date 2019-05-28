#!tsn_new video classfication model script
#python3
#haidong kang
#20190523
#consensus_type == 'video'
#val_list == ''
#no mixup data augmentation 
#num_class==6
#pretrained model in kinetics:resnet50_80e_69.98p.pth.tar
nohup python3 -u test_models.py 6 RGB /workspace/mnt/group/algorithm/kanghaidong/video_project/Video1.0_list_file/bk-video-list-V1.0-a-val.txt /workspace/mnt/group/algorithm/kanghaidong/video_project/Alg-VideoAlgorithm-dev/video-classification/tsn-pytorch/bk-V1.0_2.1.4_resnet50__rgb_model_best.pth.tar \
   --input_type Video --video_handler cv2 \
   --arch resnet50 --num_segments 25 \
   --crop_fusion_type avg > V1.0_2.1.4_test_out.out 2>&1 &


