import os
import argparse
import random
import fnmatch

def parse_video_number(folder):
	video_num_dict = dict()
	for label in os.listdir(folder):
		if label[0] == '.': continue
		label_dir = os.path.join(folder,label)
		video_num = len(os.listdir(label_dir))
		video_num_dict[label] = video_num

	return video_num_dict

def parse_label_file(label_file):
	label2idx = dict()
	with open(label_file,'r') as f:
		for line in f:
			items = line.strip().split(' ')
			if len(items)==2:
				label,index = items
				repeat_time = 1
			elif len(items)==3:
				label,index,repeat_time = items
			index = int(index)
			repeat_time = int(repeat_time)
			label2idx[label]=(index,repeat_time)

	return label2idx

def write_to_list(save_list, traindata, valdata):
	random.shuffle(traindata)
	random.shuffle(valdata)

	train_list_hdl = open(save_list + 'train.txt', 'w')
	val_list_hdl = open(save_list + 'val.txt', 'w')

	for data in traindata:
		train_list_hdl.write(data)
	for data in valdata:
		val_list_hdl.write(data)

def gen_video_list(folder, val_ratio, label2idx, prefix='image_'):
	train_data = []
	val_data = []
	for label, info in label2idx.items():
		label_idx, repeat_time = info
		label_dir = os.path.join(folder,label)
		video_names = []
		for m in os.listdir(label_dir):
			video_names.append(m)

		val_num = int(len(video_names) * val_ratio)
		train_num = len(video_names) - val_num
		print('{},{},{},{}'.format(label_idx,label,train_num,val_num))
		random.shuffle(video_names)

		# generate val list
		for video_name in video_names[:val_num]:
			video_path = os.path.join(label_dir,video_name)
			video_frame_num = len(fnmatch.filter(os.listdir(video_path), prefix + '*'))
			val_data.append('{} {} {}\n'.format(video_path,video_frame_num,label_idx))

		# generate train list
		for video_name in video_names[val_num:]:
			video_path = os.path.join(label_dir, video_name)
			video_frame_num = len(fnmatch.filter(os.listdir(video_path), prefix + '*'))
			for i in range(repeat_time):
				train_data.append('{} {} {}\n'.format(video_path, video_frame_num, label_idx))

	return train_data, val_data

def parse_args():
	parser = argparse.ArgumentParser(description='generate bk frame list',
	                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('root', help='folder saving video frames', type=str)
	parser.add_argument('gen_list',help='generate list name', type=str)
	parser.add_argument('--label_file', help='label2idx file, with repeat times for each label, split by space',
	                    type=str,default='label.txt')
	parser.add_argument('--val_num_ratio', help='split train val datalist at a ratio', type=float, default=0.2)
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()

	ROOT = args.root
	LIST_NAME = args.gen_list
	VAL_RATIO = args.val_num_ratio
	LABEL_FILE = args.label_file

	# step 1: get video num of each label
	video_num_dict = parse_video_number(ROOT)
	print(video_num_dict)

	# step 2: parse label file to get label2idx dict
	label2idx = parse_label_file(LABEL_FILE)
	# step 3: generate train / val list
	traindata,valdata = gen_video_list(ROOT, VAL_RATIO, label2idx)
	write_to_list(LIST_NAME, traindata, valdata)
