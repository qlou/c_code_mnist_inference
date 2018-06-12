from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python import pywrap_tensorflow
import os


tf.reset_default_graph()
reader = pywrap_tensorflow.NewCheckpointReader("/Users/qlou/Documents/tensorflow_to_c/models/model.ckpt")
var_to_shape_map = reader.get_variable_to_shape_map()

for key in sorted(var_to_shape_map):
	# print("tensor_name", key)
	# print(reader.get_tensor(key))
	if key == "conv2d/bias":
		output_file = open("weight_data/bias1.bin","wb");
		newFileByteArray = bytearray(reader.get_tensor(key))
		output_file.write(newFileByteArray)
		np.savetxt('weight_data/bias1.txt',reader.get_tensor(key))
		print(reader.get_tensor(key).shape)
		# print(np.dtype(reader.get_tensor(key)[0]))
	elif key == "conv2d/kernel":
		output_file = open("weight_data/weight1.bin","wb");
		newFileByteArray = bytearray(reader.get_tensor(key))
		output_file.write(newFileByteArray)
		for i in range(0,5):
			for j in range(0,5):
				print("%f,",reader.get_tensor(key)[i,j,0,0])
			print("\n")
		x = reader.get_tensor(key).reshape((5*5,1*32))
		np.savetxt('weight_data/weight1.txt',x)
		print(reader.get_tensor(key).shape)
	elif key == "conv2d_1/bias":
		output_file = open("weight_data/bias2.bin","wb");
		newFileByteArray = bytearray(reader.get_tensor(key))
		output_file.write(newFileByteArray)	
		np.savetxt('weight_data/bias2.txt',reader.get_tensor(key))
		print(reader.get_tensor(key).shape)
	elif key == "conv2d_1/kernel":
		output_file = open("weight_data/weight2.bin","wb");
		newFileByteArray = bytearray(reader.get_tensor(key))
		output_file.write(newFileByteArray)	
		x = reader.get_tensor(key).reshape((5*5,32*64))
		np.savetxt('weight_data/weight2.txt',x)
		print(reader.get_tensor(key).shape)
	elif key == "dense/bias":
		output_file = open("weight_data/bias3.bin","wb");
		newFileByteArray = bytearray(reader.get_tensor(key))
		output_file.write(newFileByteArray)	
		np.savetxt('weight_data/bias3.txt',reader.get_tensor(key))
		print(reader.get_tensor(key).shape)
	elif key == "dense/kernel":
		output_file = open("weight_data/weight3.bin","wb");
		newFileByteArray = bytearray(reader.get_tensor(key))
		output_file.write(newFileByteArray)	
		x = reader.get_tensor(key).reshape
		np.savetxt('weight_data/weight3.txt',reader.get_tensor(key))
		print(reader.get_tensor(key).shape)
	elif key == "dense_1/bias":
		output_file = open("weight_data/bias4.bin","wb");
		newFileByteArray = bytearray(reader.get_tensor(key))
		output_file.write(newFileByteArray)	
		np.savetxt('weight_data/bias4.txt',reader.get_tensor(key))
		print(reader.get_tensor(key).shape)
	elif key == "dense_1/kernel":
		output_file = open("weight_data/weight4.bin","wb");
		newFileByteArray = bytearray(reader.get_tensor(key))
		output_file.write(newFileByteArray)	
		np.savetxt('weight_data/weight4.txt',reader.get_tensor(key))
		print(reader.get_tensor(key).shape)