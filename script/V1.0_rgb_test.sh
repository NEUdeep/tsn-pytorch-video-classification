#!tsn model testing
#python3
#Kang Haidong
#20190521
#consensus_type == 'video'
#test_list == ''
#mixup data augmentation 
#num_class==6
#out of memory cuda module
#pytorch1.0.1, cuda 9.0
nohup python3 -u train.py --cfg ./configs/V1.0_rgb_test.yml > V1.0_2.1.5_test_out.out 2>&1 &