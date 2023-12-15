import tflite_runtime.interpreter as tflite
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import imageio

# Load the TFLite model in TFLite Interpreter
interpreter = tflite.Interpreter(model_path="/home/bimbraw/Downloads/model_50e.tflite")
print(interpreter)

image_tensor = []
image_paths = []
for im_path in glob.glob("/home/bimbraw/Downloads/Jack_BSN_8r_4s/image_*.png"):
     print(im_path)
     im = imageio.imread(im_path)
     image_tensor.append(im)
     image_paths.append(im_path)

image_tensor = np.asarray(image_tensor)

print(im.shape)
print(type(image_tensor))
print(image_tensor.shape)

label = np.load('/home/bimbraw/Downloads/Jack_BSN_8r_4s/labels.npy')
print(label.shape)

image_flatten = image_tensor.reshape((image_tensor.shape[0], 640, 640, 1))
print(image_flatten.shape)

total_samples = 2400

accuracy = 0
time_recorder = []
total_start = time.time()
for i in range(0, total_samples):
	inf_start = time.time()
	im = image_flatten[i]
	im_val = im.reshape((1, 640, 640, 1))
	im_val_re = im_val.astype(np.float32, copy=False)
	interpreter.allocate_tensors()

	# Get input and output tensors.
	input_details = interpreter.get_input_details()
	interpreter.set_tensor(input_details[0]['index'], im_val_re)
	interpreter.invoke()
	output_details = interpreter.get_output_details()
	output_data = interpreter.get_tensor(output_details[0]['index'])
	true_val = label[(int(image_paths[i][len(image_paths[i])-9])-1)*300 + int(image_paths[i][len(image_paths[i])-7:len(image_paths[i])-4])]
	predicted_val = np.argmax(output_data)
	print(image_paths[i][len(image_paths[i])-9], (int(image_paths[i][len(image_paths[i])-9])-1)*300)
	print(image_paths[i], image_paths[i][len(image_paths[i])-7:len(image_paths[i])-4])
	if true_val == predicted_val:
		accuracy += 1
	print(i, true_val, predicted_val, output_data)
	inf_end = time.time()
	time_iter = inf_end - inf_start
	time_recorder.append(time_iter)
	print('For #', str((int(image_paths[i][len(image_paths[i])-9])-1)*300 + int(image_paths[i][len(image_paths[i])-7:len(image_paths[i])-4])), 'it took:', str(time_iter), 's') 

total_end = time.time()		
print('Total time:', str(total_end-total_start), 's, per sample: ', str((total_end-total_start)/total_samples))
print('Accuracy percentage:', str((accuracy/total_samples)*100), '%')
time_recorder = np.array(time_recorder)
np.save('/home/bimbraw/Downloads/Jack_BSN_8r_4s/total_time_50e.npy', time_recorder)
