import torch.utils.data as data
from PIL import Image
import numpy as np
import cv2
from numpy.random import randint
from dataset import TSNDataSet, VideoRecord


class TSNDataSetVideoCV2(TSNDataSet):
    def __init__(self, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):
        super(TSNDataSetVideoCV2, self).__init__()
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        if self.modality != 'RGB':
            raise ValueError("Unsupported modality mode!")

        self._parse_list()


    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]#video_list is a list, incude object of class VideoRecord; it has one list and two attribute;
        print('the length of list_file',len(self.list_file))# 9537
        #print(self.video_list)
        print('the length of video_list',len(self.video_list))# the len of traindaset.txt ; 3783;but it should len(train datasets)=9537

    def _sample_indices(self, num_frames):
        """
        :param record: VideoRecord
        :return: list
        """
        average_duration = (num_frames - self.new_length + 1) // self.num_segments

        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                              size=self.num_segments)
        elif num_frames > self.num_segments:
            offsets = np.sort(randint(num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_val_indices(self, num_frames):
        if num_frames > self.num_segments + self.new_length - 1:
            tick = (num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets
    

    def _get_test_indices(self, num_frames):

        tick = (num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets
    
    def _get_per_frame_test(self,num_frames): # 生成一个list，含一个video的所有帧。
        offsets = np.array([int(x) for x in range(num_frames-1)])
        return offsets
    

    def __getitem__(self, index):# get one video and label of index; but it have some problem with random police in training 
        record = self.video_list[index]
        videoObj = cv2.VideoCapture(record.path)
        try:
            num_frames = int(videoObj.get(cv2.CAP_PROP_FRAME_COUNT))
            print('num of frame is :',num_frames)
        except:
            num_frames = 0
        if num_frames == 0:
            process_data = None

        if num_frames > 0:
            if not self.test_mode:
                segment_indices = self._sample_indices(num_frames) if self.random_shift \
                    else self._get_val_indices(num_frames)
            else:
                segment_indices = self._get_per_frame_test(num_frames) #产生一个video的所有的frame的index的一个list。
                print('test_mode')
                print(segment_indices)

            process_data = self.cv2_get(videoObj, segment_indices)
            label = record.label#label of video from video_list
            #print (label) 

            videoObj.release()

        while process_data is None:
            index = randint(0, len(self.video_list))
            #you should know index is random,but con't over len(self.video_list);if over,will get error:out of range 
            process_data, label = self.__getitem__(index)
        #print (label)#len(label)=42653>len(video_list)


        return process_data, label

    def cv2_get(self, videoObj, indices):

        images = list()
        for frame_index in indices:
            print('frame_index',frame_index)
            videoObj.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            _, cv2_im = videoObj.read()
            if cv2_im is None:
                return None
            cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im)
            images.append(pil_im)
        print(images)

        process_data = self.transform(images)
        print(process_data)
        return process_data


    def __len__(self):
        return len(self.video_list)




