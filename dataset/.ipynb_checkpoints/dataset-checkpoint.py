
import torch.utils.data as data

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]
    @property
    def label(self):
        #print(int(self._data[1]))# print file_list each label
        return int(self._data[1])

class FramesRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

class ImageRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def label(self):
        return int(self._data[1])

class TSNDataSet(data.Dataset):
    def __init__(self):
        pass
    def __getitem__(self, index):
        pass