import socket
import os
import time

IP = "0.0.0.0"
PORT = 8888

base_dir = os.path.join(os.path.dirname(__file__), "received_models", "tcp")
os.makedirs(base_dir, exist_ok=True)

print(f"📡 Servidor TCP escuchando en el puerto {PORT}...")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((IP, PORT))
    s.listen()

    while True:
        conn, addr = s.accept()
        with conn:
            print(f"[+] Conexión TCP desde {addr}")

            # Leer nombre del archivo (hasta \n)
            filename = b""
            while not filename.endswith(b"\n"):
                chunk = conn.recv(1)
                if not chunk:
                    break
                filename += chunk

            filename = filename.decode().strip()

            # Obtener modelo desde nombre
            parts = filename.split("_")
            model = parts[2] if len(parts) > 2 else "unknown"

            model_dir = os.path.join(base_dir, model)
            os.makedirs(model_dir, exist_ok=True)

            save_path = os.path.join(model_dir, filename)
            with open(save_path, "wb") as f:
                while True:
                    data = conn.recv(4096)
                    if not data:
                        break
                    f.write(data)

            print(f"✅ Modelo guardado: {save_path} ({os.path.getsize(save_path)} bytes)\n")
