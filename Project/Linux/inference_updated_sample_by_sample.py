import tflite_runtime.interpreter as tflite
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import imageio

time_start = time.time()
# Load the TFLite model in TFLite Interpreter
interpreter = tflite.Interpreter(model_path="/home/bimbraw/Downloads/model.tflite")
for im_path in glob.glob("/home/bimbraw/Images/image_1_*.png"):
     print(im_path)
     im = imageio.imread(im_path)

im = np.asarray(im)
labels = np.load('/home/bimbraw/Downloads/Jack_BSN_8r_4s/labels.npy')
label_val = im_path[len(im_path)-7:len(im_path)-4]
label = labels[int(label_val)]
image_re = im.reshape((1, 640, 640, 1))
im_val = image_re
im_val_re = im_val.astype(np.float32, copy=False)
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
interpreter.set_tensor(input_details[0]['index'], im_val_re)
interpreter.invoke()
output_details = interpreter.get_output_details()
output_data = interpreter.get_tensor(output_details[0]['index'])
time_end = time.time()
print('True:', label, ', Predicted:', np.argmax(output_data), 'Inference: ', round(time_end-time_start, 2),'ms, Probabilities:', output_data)
