import socket
import time

UDP_IP = "192.168.0.144"  # Replace with the actual IP address of computer B
UDP_PORT = 12345

message = "Hello"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
time_s = time.time()
for i in range(0, 1000):
    message = str(i)
    sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
    print(i, message)
time_e = time.time()

print((time_e-time_s))