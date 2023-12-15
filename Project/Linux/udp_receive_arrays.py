import socket
import pickle
import numpy as np
from PIL import Image
import time

udp_host = "192.168.0.144"		# Host IP
udp_port = 12345			        # specified port to connect
sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)      # For UDP
sock.bind((udp_host, udp_port))
buffer_size = 250000
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_size)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)

for i in range(0, 1000):
	data = sock.recv(250000)
	data_array = pickle.loads(data)
	print(data_array)

img = Image.fromarray(data_array)
img.show()
