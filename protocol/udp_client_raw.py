import socket
import time

SERVER_IP = "192.168.X.X"
PORT = 9998

def send_file_udp(file_path):
    with open(file_path, "rb") as f:
        data = f.read()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    start = time.time()
    sock.sendto(data, (SERVER_IP, PORT))
    end = time.time()
    sock.close()

    print(f"UDP sin control: {len(data)} bytes en {end - start:.2f} segundos.")
