# import pip
#
# def install(package):
#     if hasattr(pip, 'main'):
#         pip.main(['install', package])
#     else:
#         pip._internal.main(['install', package])
#
# # Example
# if __name__ == '__main__':
#     install('imageio')

import socket
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import imageio
import glob

image_tensor = []
for im_path in glob.glob("C:/Users/Keshav Bimbraw/Downloads/data_4s/image_7_*.png"):
     print(im_path)
     im = imageio.imread(im_path)
     image_tensor.append(im)
# for im_path in glob.glob("C:/Users/Keshav Bimbraw/Downloads/data_4s/image_8_*.png"):
#      print(im_path)
#      im = imageio.imread(im_path)
#      image_tensor.append(im)

image_tensor = np.asarray(image_tensor)

# print(im.shape)
print(type(image_tensor))
print(image_tensor.shape)

label = np.load("C:/Users/Keshav Bimbraw/Downloads/data_4s/labels.npy")
print(label.shape)

# image_flatten = image_tensor.reshape((image_tensor.shape[0], 640, 640, 1))
# print(image_flatten.shape)
# split_val = 0.2
# limit_val = int((1 - split_val) * image_tensor.shape[0])
# n_epochs = 25

UDP_IP = "192.168.0.144"  # Replace with the actual IP address of computer B
UDP_PORT = 12345

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.connect((UDP_IP, UDP_PORT))
buffer_size = 250000  # Adjust the size as needed
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_size)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)

time_s = time.time()
for i in range(0, 300):
    # message = str(i)

    # array_size = (80, 80)
    # random_array = np.ones(array_size)*i
    array_val = image_tensor[i]
    array_val = array_val[::8, ::8]
    serialized_data = pickle.dumps(array_val)
    # print(serialized_data)
    sock.sendall(serialized_data)


    # sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
    print(i, array_val.shape)
    time.sleep(0.1)

time_e = time.time()

print((time_e-time_s))

plt.imshow(array_val, cmap='hot')
plt.show()

print(array_val)