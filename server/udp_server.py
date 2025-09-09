import socket
import struct
import zlib
import os
import time
from collections import defaultdict

IP = "0.0.0.0"
PORT = 9999
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((IP, PORT))

buffers = defaultdict(dict)  # addr -> { seq: data }
filenames = {}
total_expected = {}  # addr -> max seq seen + 1
last_flags = {}  # addr -> bandera de último paquete

base_dir = os.path.join(os.path.dirname(__file__), "received_models", "udp_reliable")
os.makedirs(base_dir, exist_ok=True)

def verify_checksum(data, expected_crc):
    return zlib.crc32(data) & 0xFFFF == expected_crc

print(f"UDP reliable listening on port {PORT}...")

while True:
    packet, addr = sock.recvfrom(4096)

    if addr not in filenames and packet.startswith(b"FILENAME:"):
        try:
            filename = packet[len("FILENAME:"):].decode().strip()
            filenames[addr] = filename
            print(f"Filename received {addr}: {filename}")
        except Exception as e:
            print(f"[!] Error decoding filename: {e}")
        continue

    if len(packet) < 7:
        continue

    try:
        seq_num, flags, payload_len = struct.unpack('!HBH', packet[:5])
        payload = packet[5:-2]
        recv_crc = struct.unpack('!H', packet[-2:])[0]
    except struct.error:
        continue

    if not verify_checksum(packet[:-2], recv_crc):
        continue

    buffers[addr][seq_num] = payload
    total_expected[addr] = max(total_expected.get(addr, 0), seq_num + 1)
    if flags & 2:
        last_flags[addr] = True

    # ACK
    ack_packet = struct.pack('!HBBH', seq_num, 1, 0, 0)
    sock.sendto(ack_packet, addr)

    
    if last_flags.get(addr) and len(buffers[addr]) == total_expected[addr]:
        reconstructed = b''.join(buffers[addr][i] for i in range(total_expected[addr]))

        filename = filenames.pop(addr, f"model_update_unknown_{int(time.time()*1000)}.bin")
        model = "unknown"
        protocol = "udp_reliable"
        try:
            parts = filename.split("_")
            if len(parts) >= 4:
                model = parts[2]  # e.g. vgg5
                protocol = parts[3]  # e.g. udp_reliable
        except:
            pass

        save_dir = os.path.join(base_dir, model)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        with open(save_path, 'wb') as f:
            f.write(reconstructed)

        print(f"Model saved: {save_path} ({len(reconstructed)} bytes)")

        buffers.pop(addr, None)
        total_expected.pop(addr, None)
        last_flags.pop(addr, None)
