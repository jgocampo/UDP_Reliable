import socket, struct, zlib, time, sys

if len(sys.argv) != 2:
    print("Uso: python udp_client.py archivo_update.bin")
    sys.exit(1)

archivo = sys.argv[1]
with open(archivo, "rb") as f:
    PAYLOAD = f.read()

SERVER = ("127.0.0.1", 9999)
TIMEOUT = 0.2
MAX_RETRIES = 1

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(TIMEOUT)

def build_packet(seq_num, payload, request_ack=True, is_last=False):
    flags = 1 if request_ack else 0
    if is_last:
        flags |= 2
    header = struct.pack('!HBB', seq_num, flags, len(payload))
    crc = zlib.crc32(header + payload) & 0xFFFF
    return header + payload + struct.pack('!H', crc)

chunks = [PAYLOAD[i:i+100] for i in range(0, len(PAYLOAD), 100)]
acknowledged = 0

for i, chunk in enumerate(chunks):
    is_last = (i == len(chunks) - 1)
    packet = build_packet(i, chunk, True, is_last)
    retries = 0
    while retries <= MAX_RETRIES:
        sock.sendto(packet, SERVER)
        try:
            data, _ = sock.recvfrom(1024)
            ack_seq, flags, _, _ = struct.unpack('!HBBH', data)
            if ack_seq == i:
                print(f"[✓] ACK recibido por #{i}")
                acknowledged += 1
                break
        except socket.timeout:
            retries += 1
            print(f"[!] Timeout en #{i}, reintentando ({retries})...")

print(f"\nTotal enviados: {len(chunks)}, ACKs: {acknowledged}, Confiabilidad: {acknowledged / len(chunks) * 100:.2f}%")
