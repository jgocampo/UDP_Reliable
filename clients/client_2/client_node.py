import sys
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import subprocess

# ────────────────────────────────
# 1. Ajuste para importar el modelo compartido
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from shared.model_def import SimpleSensorModel
from sklearn.preprocessing import StandardScaler

# ────────────────────────────────
# 2. Cargar datos locales simulados
data_path = os.path.join(current_dir, "local_data.csv")
df = pd.read_csv(data_path)

scaler = StandardScaler()
X_np = scaler.fit_transform(df.iloc[:, :-1].values)
y_np = df.iloc[:, -1].values.reshape(-1, 1)

X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)

# ────────────────────────────────
# 3. Entrenamiento local ligero
model = SimpleSensorModel()
opt = optim.SGD(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for _ in range(10):  # rondas locales simuladas
    pred = model(X)
    loss = loss_fn(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

# ────────────────────────────────
# 4. Serializar pesos y guardarlos como binario
weights = torch.cat([p.data.view(-1) for p in model.parameters()]).numpy()
update_path = os.path.join(current_dir, "model_update.bin")
with open(update_path, "wb") as f:
    f.write(weights.tobytes())

# ────────────────────────────────
# 5. Enviar modelo al servidor usando tu protocolo
protocol_path = os.path.join(project_root, "protocol", "udp_client.py")
subprocess.run(["python", protocol_path, update_path])
