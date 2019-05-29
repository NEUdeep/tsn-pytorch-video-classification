import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
from dataset import TSNDataSet, FramesRecord


class TSNDataSetFrames(TSNDataSet):
    def __init__(self, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):
        super(TSNDataSetFrames,self).__init__()
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            img_path = os.path.join(directory, self.image_tmpl.format(idx))
            try:
                return [Image.open(img_path).convert('RGB')]
            except:
                print("Couldn't load image:{}".format(img_path))
                return None
        elif self.modality == 'Flow':
            x_img_path = os.path.join(directory, self.image_tmpl.format('x', idx))
            y_img_path = os.path.join(directory, self.image_tmpl.format('y', idx))
            try:
                x_img = Image.open(x_img_path).convert('L')
                y_img = Image.open(y_img_path).convert('L')
                return [x_img, y_img]
            except:
                print("Couldn't load flow image:{} or {}".format(x_img_path, y_img_path))
                return None

    def _parse_list(self):
        self.video_list = [FramesRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments and record.num_frames >= self.new_length:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        #shuffle is not affect
        #np.random.shuffle(offsets)
        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        process_data, label = self.get(record, segment_indices)
        while process_data is None:
            index = randint(0, len(self.video_list) - 1)
            process_data, label = self.__getitem__(index)

        return record.path, process_data, label

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                if seg_imgs is None:
                    return None,None
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
