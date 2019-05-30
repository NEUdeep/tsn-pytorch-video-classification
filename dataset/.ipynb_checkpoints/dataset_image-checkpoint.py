from PIL import Image
from numpy.random import randint
from dataset.dataset import TSNDataSet, ImageRecord


class TSNDataSetImage(TSNDataSet):
    def __init__(self, list_file, transform=None, test_mode=False):
        super(TSNDataSetImage, self).__init__()
        self.list_file = list_file
        self.transform = transform
        self.test_mode = test_mode

        self._parse_list()

    def _load_image(self, image_path):
        try:
            return [Image.open(image_path).convert('RGB')]
        except:
            print("Couldn't load image:{}".format(image_path))
            return None

    def _parse_list(self):
        self.image_list = [ImageRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def __getitem__(self, index):
        record = self.image_list[index]

        process_data, label = self.get(record)
        while process_data is None:
            index = randint(0, len(self.image_list) - 1)
            _, process_data, label = self.__getitem__(index)
        return record.path, process_data, label

    def get(self, record):

        images = list()
        seg_imgs = self._load_image(record.path)
        if seg_imgs is None:
            return None,None
        images.extend(seg_imgs)

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.image_list)