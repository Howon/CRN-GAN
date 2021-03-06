from __future__ import division
import os
import time
import scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from datetime import datetime
from enum import Enum
from os.path import join
from helper import Dataset, upscale_semantic, get_semantic_map


TRAIN_SEMANTICS = "data/cityscapes/semantics"
TRAIN_IMAGES = "data/cityscapes/images"
VALIDATIONS = "data/cityscapes/validations"
RESULT = "result_{}".format(datetime.now().strftime("%Y_%m_%d"))

NUM_EPOCHS = 600
BATCH_SIZE = 256
GEN_RES = 256

vgg_layers = scipy.io.loadmat('vgg_model.mat')['layers'][0]
palette = Dataset("cityscapes.json").palette

vgg_weights = [None] * 35
vgg_biases = [None] * 35
weights = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
for w in weights:
	vgg_weights[w] = vgg_layers[w][0][0][2][0][0]
	bias = vgg_layers[w][0][0][2][0][1]
	vgg_biases[w] = np.reshape(bias, (bias.size))

vgg_weights = np.asarray(vgg_weights)
vgg_biases = np.asarray(vgg_biases)

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

	with tf.variable_scope("casecade_" + str(res)):
		# net = conv(inputs, out_channels=dim, kernel_size=4, scope="conv1")
		# net = conv(net, out_channels=dim, kernel_size=4, scope="conv2")
        #
		net = slim.conv2d(inputs, dim, [3, 3], rate=1,
					  normalizer_fn=slim.layer_norm,
					  activation_fn=tf.nn.leaky_relu,
					  scope='conv1_{}'.format(res))
		net = slim.conv2d(net, dim, [3, 3], rate=1,
					  normalizer_fn=slim.layer_norm,
					  activation_fn=tf.nn.leaky_relu,
					  scope='conv2_{}'.format(res))

	if res == 256:
		with tf.variable_scope("final_gen"):
			# net = tf.tanh(conv(net, out_channels=3, kernel_size=1))
			net = slim.conv2d(net, 3, [1, 1], rate=1, activation_fn=None,
							  scope='conv_final')
			return (net + 1.0) / 2.0 * 255.0
    #
	# net = slim.conv2d(inputs, dim, [3, 3], rate=1,
	# 				  normalizer_fn=slim.layer_norm,
	# 				  activation_fn=tf.nn.leaky_relu,
	# 				  scope='g_{}_conv1'.format(res))
	# net = slim.conv2d(net, dim, [3, 3], rate=1,
	# 				  normalizer_fn=slim.layer_norm,
	# 				  activation_fn=tf.nn.leaky_relu,
	# 				  scope='g_{}_conv2'.format(res))
	# if res == 256:
	# 	net = slim.conv2d(net, 3, [1, 1], rate=1, activation_fn=None,
	# 					  scope='g_{}_conv100'.format(res))
    #
	# 	# net = slim.conv2d(net, 27, [1, 1], rate=1, activation_fn=None,
	# 	# 				  scope='g_{}_conv100'.format(res))
	# 	net = (net + 1.0) / 2.0 * 255.0
	# 	# split0, split1, split2 = tf.split(tf.transpose(
	# 	# 	net, perm=[3, 1, 2, 0]), num_or_size_splits=3, axis=0)
	# 	# net = tf.concat([split0, split1, split2], 3)
	return tf.tanh(net)


def error(real, fake, semantics):
	return tf.reduce_mean(tf.abs(fake-real))
	# # diversity loss
	# mean = tf.reduce_mean(tf.abs(fake - real), reduction_indices=[3])
	# loss = tf.expand_dims(mean, -1)
	# return tf.reduce_mean(semantics * loss, reduction_indices=[1, 2])
    #

def stitch_variations(img):
	# upper = np.concatenate(
	# 	(img[0, :, :, :], img[1, :, :, :], img[2, :, :, :]), axis=1)
	# middle = np.concatenate(
	# 	(img[3, :, :, :], img[4, :, :, :], img[5, :, :, :]), axis=1)
	# bottom = np.concatenate(
	# 	(img[6, :, :, :], img[7, :, :, :], img[8, :, :, :]), axis=1)
    #
	# sum_img = np.concatenate((upper, middle, bottom), axis=0)
	# return scipy.misc.toimage(sum_img, cmin=0, cmax=255)
	return scipy.misc.toimage(img[0,:,:,:], cmin=0, cmax=255)
sess = tf.Session()

is_training = True
with tf.variable_scope(tf.get_variable_scope()):
	semantics = tf.placeholder(tf.float32, [None, None, None, 3])
	real_image = tf.placeholder(tf.float32, [None, None, None, 3])

	generator = recursive_generator(semantics, GEN_RES)

	tf.summary.image(name="Generated", tensor=generator)
	tf.summary.image(name="Real", tensor=real_image)

	vgg_real = build_vgg19(real_image)
	vgg_fake = build_vgg19(generator, reuse=True)

	p0 = error(vgg_real['input'], vgg_fake['input'], semantics)
	p1 = error(vgg_real['conv1_2'], vgg_fake['conv1_2'], semantics) / 1.6

	resized = tf.image.resize_area(semantics, (GEN_RES // 2, GEN_RES))
	p2 = error(vgg_real['conv2_2'], vgg_fake['conv2_2'], resized) / 2.3

	resized = tf.image.resize_area(semantics, (GEN_RES // 4, GEN_RES // 2))
	p3 = error(vgg_real['conv3_2'], vgg_fake['conv3_2'], resized) / 1.8

	resized = tf.image.resize_area(semantics, (GEN_RES // 8, GEN_RES // 4))
	p4 = error(vgg_real['conv4_2'], vgg_fake['conv4_2'], resized) / 2.8

	resized = tf.image.resize_area(semantics, (GEN_RES // 16, GEN_RES // 8))
	p5 = error(vgg_real['conv5_2'], vgg_fake['conv5_2'], resized) * 10 / 0.8

	# content_loss = p0 + p1 + p2 + p3 + p4 + p5
	# loss_min = tf.reduce_min(content_loss, reduction_indices=0)
	# loss_mean = tf.reduce_mean(content_loss, reduction_indices=0)
	# g_loss = tf.reduce_sum(loss_min) * 0.999 + tf.reduce_sum(loss_mean) * 0.001
	g_loss = p0 + p1 + p2 + p3 + p4 + p5

	tf.summary.scalar("P0", tf.reduce_mean(p0))
	tf.summary.scalar("P1", tf.reduce_mean(p1))
	tf.summary.scalar("P2", tf.reduce_mean(p2))
	tf.summary.scalar("P3", tf.reduce_mean(p3))
	tf.summary.scalar("P4", tf.reduce_mean(p4))
	tf.summary.scalar("P5", tf.reduce_mean(p5))

	tf.summary.scalar("Loss", g_loss)

lr = tf.placeholder(tf.float32)

optimizer = tf.train.AdamOptimizer(learning_rate=lr)
g_opt = optimizer.minimize(g_loss,
						   global_step=tf.train.get_global_step())
saver = tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())
summary = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(join(RESULT, "tensorboard"), sess.graph)

if not os.path.isdir(RESULT):
	os.makedirs(RESULT)

ckpt = tf.train.get_checkpoint_state(RESULT)
if ckpt:
	print('loaded ' + ckpt.model_checkpoint_path)
	saver.restore(sess, ckpt.model_checkpoint_path)

vals = [join(VALIDATIONS, p) for p in os.listdir(VALIDATIONS)]
val_images = [np.expand_dims(np.float32(scipy.misc.imread(v)), axis=0) for v in vals]

if is_training:
	input_size = len(os.listdir(TRAIN_IMAGES))
	g_losses = np.zeros(input_size, dtype=float)

	for epoch in range(1, NUM_EPOCHS):
		print "===================="
		print "Epoch: {} Starting".format(epoch)
		e_start = time.time()

		epoch_res_dir = join(RESULT, ("%04d" % epoch))
		if not os.path.isdir(epoch_res_dir):
			os.mkdir(epoch_res_dir)

		cnt = 0
		for ind in np.random.permutation(input_size)[:BATCH_SIZE]:
			st = time.time()
			cnt += 1
			semantics_path = join(TRAIN_SEMANTICS, "%d.png" % (ind + 1))

			if not os.path.isfile(semantics_path):
				continue
			# semantic_labels = upscale_semantic(
			# 	get_semantic_map(palette, semantics_path))
            #
			# training image
			image_path = join(TRAIN_IMAGES, "%d.png" % (ind + 1))
			semantics_img = scipy.misc.imread(semantics_path)
			real_img = scipy.misc.imread(image_path)
			# print real_img.shape
			sess_arr = [g_opt, g_loss, summary]
			feed_dict = {
				semantics: np.expand_dims(np.float32(semantics_img), axis=0),
				real_image: np.expand_dims(np.float32(real_img), axis=0),
				lr: 5e-4}
			run_result = sess.run(sess_arr,	feed_dict=feed_dict)

			g_losses[ind] = run_result[1]
			summary_writer.add_summary(run_result[2])
			print("Epoch: %d Step: %03d Loss: %.4f %.2f" % (epoch, cnt,
								   np.mean(g_losses[np.where(g_losses)]),
								   time.time() - st))

		print "Saving model to {}".format(join(epoch_res_dir, "model.ckpt"))
		saver.save(sess, join(RESULT, "model.ckpt"))

		target = open(join(epoch_res_dir, "score.txt"), 'w')
		target.write("%f" % np.mean(g_losses[np.where(g_losses)]))
		target.close()

		for ind, v in enumerate(val_images):
			output = sess.run(generator, feed_dict={semantics: v})
			output = np.minimum(np.maximum(output, 0.0), 255.0)
			stitched = stitch_variations(output)
			stitched.save(join(epoch_res_dir, "%06d_output.jpg" % ind))

		print "Epoch {} Took {} Seconds".format(epoch, time.time() - e_start)
		print "====================\n"

final_dir = join(RESULT, "/final")
if not os.path.isdir(final_dir):
	os.makedirs(final_dir)

for ind, v in enumerate(val_images):
	output = sess.run(generator, feed_dict={semantics: v})
	output = np.minimum(np.maximum(output, 0.0), 255.0)
	stitched = stitch_variations(output)
	stitched.save(join(final_dir, "%06d_output.jpg" % ind))
