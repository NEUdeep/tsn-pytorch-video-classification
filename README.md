# tsn

TSN pytorch training code 
=======
# TSN-Pytorch

Pytorch版TSN，模型比较丰富，代码改动比较灵活。

### Env

所需依赖：pytorch + opencv / nvvl

Pytorch版TSN共支持四种数据输入格式。**图片，帧，视频采用opencv进行读取，视频采用nvvl进行读取**。其中前三种数据输入格式在AVA深度学习平台上进行训练时，选择公开镜像中pytorch的相关镜像即可。若需要使用nvvl读取视频进行训练，则参考[nvvl_ava.Dockerfile](nvvl_ava.Dockerfile)编译镜像进行训练。 

### Action Proposal
TSN本身提供了一种frame2video的视频理解方式，虽然双流目前效果比较好，但是提flow速度太慢。而单独使用I3D等3D模型准确率上不去，而且网络的FLOPs太大，计算量吃不消。本项目我们除了提升原模型的性能(比原模型高3个点左右单RGB，在ucf101上)，以及扩展视频理解模型功能以外，我们还致力于视频理解的范式研究。后期会不断更新，让视频理解模型更加准确、高效、组件化。
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

1. 使用RGB数据进行训练：其中input_data_type可选Image，Frames，Video；当input_data_type为Video时，video_handler可选nvvl，cv2。模型参数配置见configs和script。

```python
bash ./script/V1.0_rgb_train.sh
```

2. 使用Flow数据进行训练：目前Flow的训练仅支持Frames。

```python
bash ./script/V1.0_flow_train.sh
```

3. 使用RGB-diff数据进行训练：目前RGB-diff的训练仅支持Frames。

```python
bash ./script/V1.0_rgb_diff_train.sh
```

### Testing

```python
bash ./script/V1.0_rgb_test.sh
```

### New
增加了专门针对监控视频的异常行为识别的模块；
本模块对输入的transfrom逻辑进行了仔细分析，然后提出两种新的专门针对视频识别的tranform，在不改变模型结构的条件下，将实际监控视频的acc从1%提升至90%。
本方法也适合于一般的2D图像变换。方法还在不断优化当中。
增加了新的逐帧测试的模块；


### Reference

[1] https://github.com/yjxiong/tsn-pytorch
