# This generate some sample MNIST image

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

##############################################################################
# Download the MNIST dataset
##############################################################################
from tensorflow.examples.tutorials.mnist import input_data

##############################################################################
# Import TensorFlow
##############################################################################
import tensorflow as tf
import numpy as np
import array
import random
from tensorflow.python.platform import flags
from PIL import Image
import scipy.misc


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                     help='Directory for storing input data')
parser.add_argument(
     '--log_dir',
     type=str,
     default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                          'tensorflow/mnist/logs/mnist_softmax'),
     help='Summaries log directory')
FLAGS, unparsed = parser.parse_known_args()

mnist = input_data.read_data_sets(FLAGS.data_dir)
n_images = 10
for i in range(n_images):
	index_array = random.randint(0,9999)
	test_x = np.zeros((1,784))
	test_x = mnist.test.images[index_array].reshape(28,28)
	test_y = np.zeros((1),dtype=int)
	test_y[0] = mnist.test.labels[index_array]

	test_x = test_x*255
	width = 28
	height = 28

	bin_img_path = "MNIST_images/image" + str(i) + ".bin"
	
	output_file = open(bin_img_path,"wb");

	newFileByteArray = bytearray(test_x)
	output_file.write(newFileByteArray)

	jpg_img_path = "MNIST_images/input" + str(i) + ".jpg"

	# im = Image.fromarray(test_x)
	# im.save(jpg_img_path)
	scipy.misc.imsave(jpg_img_path, test_x)

	# fout = open(jpg_img_path, 'wb')

	# pgmHeader = 'P5' + '\n' + str(width) + '  ' + str(height) + '  ' + str(255) + '\n'

	# fout.write(pgmHeader)

	# test_x.tofile(fout)
	print(test_x)
	# fout.close()
	# print(test_x)
	# im = Image.fromarray(test_x*255)
	# if im.mode != 'RGB':
	#	im = im.convert('RGB')
	# filename = "MNIST_images/input" + str(i) + ".jpg"
	# im.save(filename)

# print(mnist.test.images[99].shape)