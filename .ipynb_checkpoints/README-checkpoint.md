# tsn

TSN pytorch training code 
=======
# TSN-Pytorch

Pytorch版TSN，模型比较丰富，代码改动比较灵活。

### Env

所需依赖：pytorch + opencv / nvvl

Pytorch版TSN共支持四种数据输入格式。**图片，帧，视频采用opencv进行读取，视频采用nvvl进行读取**。其中前三种数据输入格式在AVA深度学习平台上进行训练时，选择公开镜像中pytorch的相关镜像即可。若需要使用nvvl读取视频进行训练，则参考[nvvl_ava.Dockerfile](nvvl_ava.Dockerfile)编译镜像进行训练。 

### Data Preparation

准备你的训练数据。根据你的数据属于上述四种数据输入格式中的哪一种，选择需要准备的数据列表格式。数据列表格式可以分为两种，帧格式和图片/视频格式。

(1) 帧格式：如使用帧数据进行训练，则需要准备该形式的数据列表。数据列表的每一行包含每个视频的帧存储位置，视频帧数，视频的groudtruth类别。例如，一个file list长这样：

```
/workspace/data/UCF-frames/v_HorseRace_g11_c02 279 40
/workspace/data/UCF-frames/v_Rowing_g10_c01 481 75
/workspace/data/UCF-frames/v_PlayingTabla_g12_c03 256 65
/workspace/data/UCF-frames/v_BandMarching_g21_c01 311 5
...
```

(2) 图片/视频格式：如使用图片或者视频数据进行训练，则需要准备该形式的数据列表。数据列表的每一行包含图片/存储的位置，图片/视频的groudtruth类别。例如，一个file list长这样：

```
/workspace/data/UCF-videos/v_HorseRace_g11_c02.mp4 40
/workspace/data/UCF-videos/v_Rowing_g10_c01.mp4 75
/workspace/data/UCF-videos/v_PlayingTabla_g12_c03.mp4 65
/workspace/data/UCF-videos/v_BandMarching_g21_c01.mp4 5
...
```

### Training

1. 使用RGB数据进行训练：其中input_data_type可选Image，Frames，Video；当input_data_type为Video时，video_handler可选nvvl，cv2。参见config。

```python
python train.py <num_of_class> RGB <rgb_train_list> <rgb_val_list> \
   --input_type <input_data_type> --video_handler <video_handler> \
   --arch resnet50 --num_segments 3 \
   --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 \
   -b 128 -j 8 --dropout 0.8 \
   --snapshot_pref your_model_resnet50_ 
```

2. 使用Flow数据进行训练：目前Flow的训练仅支持Frames。

```python
python train.py <num_of_class> Flow <rgb_train_list> <rgb_val_list> \
   --input_type Frames
   --arch resnet50 --num_segments 3 \
   --gd 20 --lr 0.001 --lr_steps 190 300 --epochs 340 \
   -b 128 -j 8 --dropout 0.7 \
   --snapshot_pref your_flow_model_resnet50_ --flow_pref flow_  
```

3. 使用RGB-diff数据进行训练：目前RGB-diff的训练仅支持Frames。

```python
python train.py <num_of_class> RGBDiff <rgb_train_list> <rgb_val_list> \
   --input_type Frames
   --arch resnet50 --num_segments 7 \
   --gd 40 --lr 0.001 --lr_steps 80 160 --epochs 180 \
   -b 128 -j 8 --dropout 0.8 \
   --snapshot_pref your_rgb-diff_model_resnet50_ 
```

### Testing

```python
python test_models_without_gt.py <num_of_class> <modality> <rgb_test_list> \
                      <trained_model_weights_file> \
    				  --input_type <input_data_type> --video_handler <video_handler> \
                      --arch resnet50
```


### Reference

[1] https://github.com/yjxiong/tsn-pytorch