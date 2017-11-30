import os, numpy as np
import json, scipy
from os.path import dirname,  exists,  join,  splitext


class Dataset(object):
	def __init__(self,  info_path):
		with open(info_path,  'r') as fp:
			info = json.load(fp)
		self.palette = np.array(info['palette'],  dtype=np.uint8)


def get_semantic_map(palette, path):
	semantic = scipy.misc.imread(path)
	print semantic.shape
	tmp = np.zeros((semantic.shape[0], semantic.shape[1], palette.shape[0]), dtype=np.float32)

	for k in range(palette.shape[0]):
		a = (semantic[:, :, 0] == palette[k, 0])
		b = (semantic[:, :, 1] == palette[k, 1])
		c = (semantic[:, :, 2] == palette[k, 2])
		tmp[:,:,k] = np.float32(a & b & c)

	return tmp.reshape((1, )+tmp.shape)


def upscale_semantic(semantic):
	upscale = np.expand_dims(1 - np.sum(semantic, axis=3), axis=3)
	return np.concatenate((semantic, upscale), axis=3)


def print_semantic_map(semantic, path):
	dataset=Dataset('cityscapes')
	semantic=semantic.transpose([1, 2, 3, 0])
	prediction=np.argmax(semantic, axis=2)
	color_image=dataset.palette[prediction.ravel()].reshape((prediction.shape[0], prediction.shape[1], 3))
	row, col, dump=np.where(np.sum(semantic, axis=2)==0)
	color_image[row, col, :]=0
	scipy.misc.imsave(path, color_image)
