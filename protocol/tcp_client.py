import socket
import time

SERVER_IP = "192.168.X.X"  # ← Reemplaza con tu IP servidor
PORT = 8888

def send_file_tcp(file_path):
    with open(file_path, "rb") as f:
        data = f.read()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_IP, PORT))

    start = time.time()
    sock.sendall(data)
    sock.close()
    end = time.time()

    print(f"TCP enviado: {len(data)} bytes en {end - start:.2f} segundos.")
