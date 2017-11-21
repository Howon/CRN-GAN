from __future__ import division
import os, os.path
from helper import Dataset, upscale_semantic, get_semantic_map
import time
import scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from datetime import datetime

from enum import Enum

TRAIN_SEMANTICS = "data/cityscapes/semantics/%d.png"
TRAIN_IMAGES = "data/cityscapes/images/%d.png"
VALIDATIONS = "data/cityscapes/validations"
RESULT = "result_{}".format(datetime.now().strftime("%Y_%m_%d_%H"))

NUM_EPOCHS = 600
BATCH_SIZE = 256
GENERATOR_RESOLUTION = 256

vgg_layers = scipy.io.loadmat('vgg_model.mat')['layers'][0]
palette = Dataset("cityscapes.json").palette

vgg_weights = [tf.constant(vgg_layers[x][0][0][2][0][0]) for x in range(1, 35)]
vgg_biases = [vgg_layers[x][0][0][2][0][1] for x in range(1, 35)]
vgg_biases = [tf.constant(np.reshape(b, (b.size))) for b in vgg_biases]


class NType(Enum):
	C = 0
	P = 1


def build_net(ntype, nin, idx=-1, name=None):
	if ntype == NType.C:
		conv = tf.nn.conv2d(input=nin, filter=vgg_weights[idx],
					  strides=[1, 1, 1, 1],
					  padding='SAME', name=name)
		return tf.nn.relu(conv + vgg_biases[idx])

	return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1],
					strides=[1, 2, 2, 1], padding='SAME')


def build_vgg19(inputs, reuse=False):
	if reuse:
		tf.get_variable_scope().reuse_variables()

	net = {}

	net['input'] = inputs - \
		np.array([123.6800, 116.7790, 103.9390]).reshape((1, 1, 1, 3))
	net['conv1_1'] = build_net(NType.C, net['input'], 0, name='vgg_conv1_1')
	net['conv1_2'] = build_net(NType.C, net['conv1_1'], 2, name='vgg_conv1_2')
	net['pool1'] = build_net(NType.P, net['conv1_2'])
	net['conv2_1'] = build_net(NType.C, net['pool1'], 5, name='vgg_conv2_1')
	net['conv2_2'] = build_net(NType.C, net['conv2_1'], 7, name='vgg_conv2_2')
	net['pool2'] = build_net(NType.P, net['conv2_2'])
	net['conv3_1'] = build_net(NType.C, net['pool2'], 10, name='vgg_conv3_1')
	net['conv3_2'] = build_net(NType.C, net['conv3_1'], 12, name='vgg_conv3_2')
	net['conv3_3'] = build_net(NType.C, net['conv3_2'], 14, name='vgg_conv3_3')
	net['conv3_4'] = build_net(NType.C, net['conv3_3'], 16, name='vgg_conv3_4')
	net['pool3'] = build_net(NType.P, net['conv3_4'])
	net['conv4_1'] = build_net(NType.C, net['pool3'], 19, name='vgg_conv4_1')
	net['conv4_2'] = build_net(NType.C, net['conv4_1'], 21, name='vgg_conv4_2')
	net['conv4_3'] = build_net(NType.C, net['conv4_2'], 23, name='vgg_conv4_3')
	net['conv4_4'] = build_net(NType.C, net['conv4_3'], 25, name='vgg_conv4_4')
	net['pool4'] = build_net(NType.P, net['conv4_4'])
	net['conv5_1'] = build_net(NType.C, net['pool4'], 28, name='vgg_conv5_1')
	net['conv5_2'] = build_net(NType.C, net['conv5_1'], 30, name='vgg_conv5_2')
	net['conv5_3'] = build_net(NType.C, net['conv5_2'], 32, name='vgg_conv5_3')
	net['conv5_4'] = build_net(NType.C, net['conv5_3'], 34, name='vgg_conv5_4')
	net['pool5'] = build_net(NType.P, net['conv5_4'])

	return net


def recursive_generator(semantics, res):
	dim = 512 if res >= 128 else 1024

	if res == 4:
		inputs = semantics
	else:
		downsampled = tf.image.resize_area(
			semantics, (res // 2, res), align_corners=False)
		resz = tf.image.resize_bilinear(
			recursive_generator(downsampled, res // 2),
			(res, res * 2), align_corners=True)
		inputs = tf.concat([resz, semantics], 3)

	net = slim.conv2d(inputs, dim, [3, 3], rate=1,
					  normalizer_fn=slim.layer_norm,
					  activation_fn=lrelu,
					  scope='g_{}_conv1'.format(res))
	net = slim.conv2d(net, dim, [3, 3], rate=1,
					  normalizer_fn=slim.layer_norm,
					  activation_fn=lrelu,
					  scope='g_{}_conv2'.format(res))
	if res == 256:
		net = slim.conv2d(net, 27, [1, 1], rate=1, activation_fn=None,
						  scope='g_{}_conv100'.format(res))
		net = (net + 1.0) / 2.0 * 255.0
		split0, split1, split2 = tf.split(tf.transpose(
			net, perm=[3, 1, 2, 0]), num_or_size_splits=3, axis=0)
		net = tf.concat([split0, split1, split2], 3)
	return net


def error(real, fake, semantics):
	# diversity loss
	mean = tf.reduce_mean(tf.abs(fake - real), reduction_indices=[3])
	loss = tf.expand_dims(mean, -1)
	return tf.reduce_mean(semantics * loss, reduction_indices=[1, 2])


sess = tf.Session()

is_training = True
with tf.variable_scope(tf.get_variable_scope()):
	semantics = tf.placeholder(tf.float32, [None, None, None, 20])
	real_image = tf.placeholder(tf.float32, [None, None, None, 3])

	generator = recursive_generator(semantics, GENERATOR_RESOLUTION)

	vgg_fake = build_vgg19(generator, reuse=True)
	vgg_real = build_vgg19(real_image)

	p0 = error(vgg_real['input'], vgg_fake['input'], semantics)
	p1 = error(vgg_real['conv1_2'], vgg_fake['conv1_2'], semantics) / 1.6

	resized = tf.image.resize_area(semantics, (res // 2, res))
	p2 = error(vgg_real['conv2_2'], vgg_fake['conv2_2'], resized) / 2.3

	resized = tf.image.resize_area(semantics, (res // 4, res // 2))
	p3 = error(vgg_real['conv3_2'], vgg_fake['conv3_2'], resized) / 1.8

	resized = tf.image.resize_area(semantics, (res // 8, res // 4))
	p4 = error(vgg_real['conv4_2'], vgg_fake['conv4_2'], resized) / 2.8

	resized = tf.image.resize_area(semantics, (res // 16, res // 8))
	p5 = error(vgg_real['conv5_2'], vgg_fake['conv5_2'], resized) * 10 / 0.8

	content_loss = p0 + p1 + p2 + p3 + p4 + p5
	loss_min = tf.reduce_min(content_loss, reduction_indices=0)
	g_loss = tf.reduce_sum(loss_min)

lr = tf.placeholder(tf.float32)
var_list = [v for v in tf.trainable_variables() if v.name.startswith('g_')]

optimizer = tf.train.AdamOptimizer(learning_rate=lr)
g_opt = optimizer.minimize(g_loss, var_list=var_list)
saver = tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

ckpt = tf.train.get_checkpoint_state(RESULT)
if ckpt:
	print('loaded ' + ckpt.model_checkpoint_path)
	saver.restore(sess, ckpt.model_checkpoint_path)

def stitch_variations(img):
	upper = np.concatenate(
		(img[0, :, :, :], img[1, :, :, :], img[2, :, :, :]), axis=1)
	middle = np.concatenate(
		(img[3, :, :, :], img[4, :, :, :], img[5, :, :, :]), axis=1)
	bottom = np.concatenate(
		(img[6, :, :, :], img[7, :, :, :], img[8, :, :, :]), axis=1)

	sum_img = np.concatenate((upper, middle, bottom), axis=0)
	return scipy.misc.toimage(sum_img, cmin=0, cmax=255)


vals = [get_semantic_map(palette, p) for p in os.listdir(VALIDATIONS)]
val_images = [upscale_semantic(v) for r in vals]

if is_training:
	input_size = len(os.listdir(TRAIN_IMAGES))
	g_loss = np.zeros(input_size, dtype=float)

	for epoch in range(1, NUM_EPOCHS):
		print "===================="
		print "Epoch: {} Starting".format(epoch)
		e_start = time.time()

		epoch_res_dir = path.join(RESULT, "04d" % epoch)
		if not path.isdir(epoch_res_dir):
			os.mkdir(epoch_res_dir)

		cnt = 0
		for ind in np.random.permutation(input_size)[:BATCH_SIZE]:
			st = time.time()
			cnt += 1
			path = TRAIN_SEMANTICS % (ind + 1)

			if not os.path.isfile(path):
				continue
			semantics = helper.upscale_semantic(get_semantic_map(palette, path))

			# training image
			real_img = scipy.misc.imread(TRAIN_IMAGES % (ind + 1))

			sess_arr = [g_loss]
			feed_dict = {
				semantics: semantics,
				real_image: np.expand_dims(np.float32(real_img), axis=0),
				lr: 1e-3}
			run_result = sess.run(sess_arr,	feed_dict=feed_dict)

			# _, G_current, l0, l1, l2, l3, l4, l5
			g_loss[ind] = run_result[0]

			print("Epoch: %d Step: %03d Loss: %.4f %.2f" % (epoch, cnt,
								   np.mean(g_loss[np.where(g_loss)]),
								   time.time() - st))

		saver.save(sess, path.join(epoch_res_dir, "model.ckpt"))

		target = open(path.join(epoch_res_dir, "score.txt"), 'w')
		target.write("%f" % np.mean(g_loss[np.where(g_loss)]))
		target.close()

		for ind, v in enumerate(val_images):
			output = sess.run(generator, feed_dict={semantics: v})
			output = np.minimum(np.maximum(output, 0.0), 255.0)
			stitched = stitch_variations(output)
			stitched.save(path.join(epoch_res_dir, "%06d_output.jpg" % ind))

		print "Epoch {} Took {} Seconds".format(epoch, time.time() - e_start)
		print "====================\n"

final_dir = path.join(RESULT, "/final")
if not path.isdir(final_dir):
	os.makedirs(final_dir)

for ind, v in enumerate(val_images):
	output = sess.run(generator, feed_dict={semantics: v})
	output = np.minimum(np.maximum(output, 0.0), 255.0)
	stitched = stitch_variations(output)
	stitched.save(path.join(final_dir, "%06d_output.jpg" % ind))
