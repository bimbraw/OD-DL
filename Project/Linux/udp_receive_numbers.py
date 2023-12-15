import socket

udp_host = "192.168.0.144"		# Host IP
udp_port = 12345			        # specified port to connect
sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)      # For UDP
sock.bind((udp_host, udp_port))

while True:
	data, addr = sock.recvfrom(1024)
	print(f"Received message: {data.decode()} from {addr}")
