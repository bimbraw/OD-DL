import socket
import pickle
import numpy as np
from PIL import Image
import time
import tflite_runtime.interpreter as tflite

# Load the TFLite model in TFLite Interpreter
interpreter = tflite.Interpreter(model_path="/home/bimbraw/Downloads/model_20e_dse.tflite")
print(interpreter)

udp_host = "192.168.0.144"		# Host IP
udp_port = 12345			        # specified port to connect
sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)      # For UDP
sock.bind((udp_host, udp_port))
buffer_size = 250000
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_size)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)

label = np.load('/home/bimbraw/Test/labels.npy')
print(label.shape)
time_recorder = []

num_values = 300

accuracy = 0
time_main = time.time()
for i in range(0, num_values):
	inf_start = time.time()
	data = sock.recv(250000)
	data_array = pickle.loads(data)
	print(data_array.shape)
	im_val = data_array.reshape((1, 80, 80, 1))
	#print(data_array)
	im_val_re = im_val.astype(np.float32, copy=False)
	interpreter.allocate_tensors()

	# Get input and output tensors.
	input_details = interpreter.get_input_details()
	interpreter.set_tensor(input_details[0]['index'], im_val_re)
	interpreter.invoke()
	output_details = interpreter.get_output_details()
	output_data = interpreter.get_tensor(output_details[0]['index'])
	true_val = label[i+1800]
	predicted_val = np.argmax(output_data)
	if true_val == predicted_val:
		accuracy += 1
	print(i, true_val, predicted_val, output_data)
	inf_end = time.time()
	time_iter = inf_end - inf_start
	time_recorder.append(time_iter)
	print('For #', i, 'it took:', str(time_iter*1000), 'ms') 
time_main_end = time.time()

print('Accuracy:', (accuracy/num_values)*100)
print('Total time:', time_main_end-time_main, ', per frame:', (time_main_end-time_main)/num_values) 
#img = Image.fromarray(data_array)
#img.show()
