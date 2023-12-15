import tflite_runtime.interpreter as tflite
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load the TFLite model in TFLite Interpreter
interpreter = tflite.Interpreter(model_path="/home/bimbraw/Downloads/model.tflite")
print(interpreter)
labels = np.load('/home/bimbraw/Downloads/Jack_BSN_8r_4s/labels.npy')

for k in range(1, 8):
	for j in range(0, 3):
		for i in range(10, 99):
			im = plt.imread('/home/bimbraw/Downloads/Jack_BSN_8r_4s/image_'+str(k)+'_'+str(j)+str(i)+'.png')
			im_val = im.reshape((1, 640, 640, 1))
			im_val_re = im_val.astype(np.float32, copy=False)
			interpreter.allocate_tensors()

			# Get input and output tensors.
			input_details = interpreter.get_input_details()
			interpreter.set_tensor(input_details[0]['index'], im_val_re)
			interpreter.invoke()
			output_details = interpreter.get_output_details()
			output_data = interpreter.get_tensor(output_details[0]['index'])
			print(k, j, i, labels[((k-1)*300)+(j*100)+i], np.argmax(output_data), output_data)
