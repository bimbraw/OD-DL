import socket

UDP_IP = "192.168.0.144"  # Replace with the actual IP address of computer B
UDP_PORT = 12345

message = "Hello"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(message.encode(), (UDP_IP, UDP_PORT))