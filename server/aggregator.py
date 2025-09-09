import argparse
import os
import torch
import numpy as np
from shared.vgg_models import VGG5, VGG8
from paramiko import SSHClient, AutoAddPolicy
from scp import SCPClient

#RASPBERRY_IP = "10.227.65.36"
RASPBERRY_USER = "pi"
RASPBERRY_PASSWORD = "raspberry"
RASPBERRY_DEST_PATH = "/home/pi/fl_client/models/"

def load_model_update(path):
    try:
        if path.endswith(".pt"):
            return torch.load(path)
        elif path.endswith(".npz"):
            data = np.load(path)
            state_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}
            return state_dict
    except Exception as e:
        print(f"⚠️ Could not load {path}: {e}")
        return None

def load_sample_count(meta_path):
    try:
        with open(meta_path, "r") as f:
            return int(f.read().strip())
    except:
        return None

def weighted_average(models, sample_counts):
    avg_model = {}
    total = sum(sample_counts)
    for k in models[0].keys():
        avg_model[k] = sum(m[k] * (n / total) for m, n in zip(models, sample_counts))
    return avg_model

def send_model_with_password(local_path, remote_path, ip, username, password):
    try:
        ssh = SSHClient()
        ssh.set_missing_host_key_policy(AutoAddPolicy())
        ssh.connect(ip, username=username, password=password)
        with SCPClient(ssh.get_transport()) as scp:
            scp.put(local_path, remote_path=remote_path)
        print("✅ Global model sent successfully via SCP.")
    except Exception as e:
        print(f"❌ Failed to send model via SCP: {e}")

def main(protocol, model_name):
    base_path = f"server/received_models/{protocol}/{model_name}"
    #files = [f for f in os.listdir(base_path) if f.endswith(".npz")]
    files = [f for f in os.listdir(base_path) if f.endswith(".pt")]

    models = []
    sample_counts = []
    used_files = []

    print(f"🔍 Found {len(files)} model updates.")

    for fname in files:
        model_path = os.path.join(base_path, fname)
        meta_path = model_path.replace(".pt", ".meta")

        model = load_model_update(model_path)
        n_samples = load_sample_count(meta_path)

        if model and n_samples:
            models.append(model)
            sample_counts.append(n_samples)
            used_files.append(model_path)
            used_files.append(meta_path)
        else:
            print(f"⚠️ Skipped invalid model or missing metadata: {fname}")

    if not models:
        print("❌ No valid models loaded.")
        return

    avg_state_dict = weighted_average(models, sample_counts)
    model = VGG5() if model_name == 'vgg5' else VGG8()
    model.load_state_dict(avg_state_dict, strict=False)

    os.makedirs("server/models", exist_ok=True)
    save_path = f"server/models/global_model_{model_name}_{protocol}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Global model saved to {save_path}")

    # 🗑️ Delete used files
    for f in used_files:
        try:
            os.remove(f)
            #print(f"🗑️ Deleted: {f}")
        except Exception as e:
            print(f"⚠️ Failed to delete {f}: {e}")

    # 📤 Send to Raspberry Pi (original IP)
    print(f"🚀 Sending global model to Raspberry Pi at 10.227.65.36...")
    send_model_with_password(
        local_path=save_path,
        remote_path=RASPBERRY_DEST_PATH,
        ip="10.227.65.36",
        username=RASPBERRY_USER,
        password=RASPBERRY_PASSWORD
    )

    # 📤 Send to second Raspberry Pi
    print(f"🚀 Sending global model to second Raspberry Pi at 10.227.69.193...")
    send_model_with_password(
        local_path=save_path,
        remote_path=RASPBERRY_DEST_PATH,
        ip="10.227.69.193",
        username=RASPBERRY_USER,
        password=RASPBERRY_PASSWORD
    )

    # 📤 Send to second Raspberry Pi
    print(f"🚀 Sending global model to second Raspberry Pi at 10.227.69.193...")
    send_model_with_password(
        local_path=save_path,
        remote_path=RASPBERRY_DEST_PATH,
        ip="10.227.68.157",
        username=RASPBERRY_USER,
        password=RASPBERRY_PASSWORD
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--model", required=True, choices=["vgg5", "vgg8"])
    args = parser.parse_args()
    main(args.protocol, args.model)
