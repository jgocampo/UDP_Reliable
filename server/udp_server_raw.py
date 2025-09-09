import socket
import os
import time

IP = "0.0.0.0"
PORT = 9998
BUFFER_SIZE = 2048
TIMEOUT = 3  # segundos de espera sin paquetes para cerrar y guardar

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((IP, PORT))
sock.setblocking(False)

print(f"UDP RAW server listening on port {PORT}...")

buffers = {}     # filename -> {"chunks": {}, "total": N, "timestamp": T, "addr": A}
timestamps = {}  # filename -> último timestamp

while True:
    try:
        data, addr = sock.recvfrom(BUFFER_SIZE)

        if data.startswith(b"FILENAME:"):
            filename = data.decode(errors='ignore').strip().split("FILENAME:")[1]
            buffers[filename] = {
                "chunks": {},
                "total": None,
                "timestamp": time.time(),
                "addr": addr
            }
            timestamps[filename] = time.time()
            print(f"📩 Receiving file: {filename} from {addr}")

        elif b"SEQ:" in data:
            header_end = data.find(b"|")
            if header_end == -1:
                continue

            header = data[:header_end].decode(errors='ignore')
            payload = data[header_end + 1:]  # ⬅️ Aquí se queda en binario puro (¡correcto!)

            if not header.startswith("SEQ:"):
                continue

            try:
                seq_info = header.replace("SEQ:", "")
                seq_num, total_chunks = map(int, seq_info.split("/"))
            except:
                continue

            filename = next((f for f in buffers if buffers[f]["addr"] == addr), None)
            if not filename:
                continue

            if buffers[filename]["total"] is None:
                buffers[filename]["total"] = total_chunks

            buffers[filename]["chunks"][seq_num] = payload  # ⬅️ Guardamos payload binario
            timestamps[filename] = time.time()

    except BlockingIOError:
        now = time.time()
        expired = [f for f, t in timestamps.items() if now - t > TIMEOUT]
        for filename in expired:
            entry = buffers[filename]
            chunks = entry["chunks"]
            total_chunks = entry["total"] or 0
            received_count = len(chunks)

            # Ordenar y reconstruir
            reconstructed = b''.join(chunks[i] for i in range(total_chunks) if i in chunks)

            parts = filename.split("_")
            model = parts[2] if len(parts) > 2 else "unknown"
            protocol = parts[3] if len(parts) > 3 else "udp_raw"

            save_dir = os.path.join(os.path.dirname(__file__), "received_models", "udp_raw", model)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, filename)

            with open(save_path, "wb") as f:
                f.write(reconstructed)

            print(f"✅ File saved to {save_path} ({len(reconstructed)} bytes)")
            if total_chunks > 0:
                reliability = 100 * received_count / total_chunks
                print(f"Reliability: {reliability:.2f}%")

            del buffers[filename]
            del timestamps[filename]

        time.sleep(0.1)
