import torch.utils.data as data
import time
import torch.multiprocessing
from PIL import Image
import numpy as np
import nvvl
from numpy.random import randint
from dataset import TSNDataSet, VideoRecord

class TSNDataSetVideoNVVL(TSNDataSet):
	def __init__(self, list_file,
	             num_segments=3, new_length=1, modality='RGB',
	             transform=None, random_shift=True, test_mode=False):
		super(TSNDataSetVideoNVVL, self).__init__()
		self.list_file = list_file
		self.num_segments = num_segments
		self.new_length = new_length
		self.modality = modality
		self.transform = transform
		self.random_shift = random_shift
		self.test_mode = test_mode
		self.worker_id = None

		if self.modality != 'RGB':
			raise ValueError("Unsupported modality mode!")

		self._parse_list()

	def set_video_reader(self, worker_id):
		pid = torch.multiprocessing.current_process().pid
		print("create video reader pid:{}".format(pid))
		self.video_reader_0 = nvvl.VideoReader(device_id=0, log_level="warn")
		self.video_reader_1 = nvvl.VideoReader(device_id=0, log_level="warn")

	def _parse_list(self):
		self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

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

	def __getitem__(self, index):
		record = self.video_list[index]
		try:
			num_frames = nvvl.video_frame_count_from_file(record.path)
		except:
			num_frames = 0
			process_data = None

		if num_frames > 3:
			num_frames = num_frames - 3
		if num_frames > 0:
			if not self.test_mode:
				segment_indices = self._sample_indices(num_frames) if self.random_shift else self._get_val_indices(
					num_frames)
			else:
				segment_indices = self._get_test_indices(num_frames)

			process_data, label = self.nvvl_get(record, segment_indices)

		while process_data is None:
			index = randint(0, len(self.video_list) - 1)
			process_data, label = self.__getitem__(index)

		return process_data, label

	def nvvl_get(self, record, indices):
		image_shape = nvvl.video_size_from_file(record.path)
		start = time.time()

		destroy = False
		if image_shape.width == 640:
			video_reader = self.video_reader_0
		elif image_shape.width == 480:
			video_reader = self.video_reader_1
		else:
			video_reader = nvvl.VideoReader(device_id=0, log_level="warn")
			destroy = True
		try:
			tensor_imgs = video_reader.get_samples(record.path, indices)
		except:
			return None,None
		end = time.time()

		if destroy:
			video_reader.destroy()

		images = list()
		# print(record.path)
		for tensor in tensor_imgs:
			tensor_img = tensor[0].numpy().astype(np.uint8)
			img = Image.fromarray(tensor_img)
			# print("tensor_image:",tensor_img[0][0])
			images.append(img)

		process_data = self.transform(images)
		return process_data, record.label


	def __len__(self):
		return len(self.video_list)


