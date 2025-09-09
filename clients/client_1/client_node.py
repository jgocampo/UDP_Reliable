import sys
import os

# Detectar automáticamente la raíz del proyecto
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Agregar raíz del proyecto a sys.path si no está
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("PROJECT ROOT:", project_root)
print("sys.path:", sys.path)
print("shared exists?", os.path.exists(os.path.join(project_root, "shared")))
print("model_def exists?", os.path.exists(os.path.join(project_root, "shared", "model_def.py")))


# Importar modelo compartido
from shared.model_def import SimpleSensorModel
from sklearn.preprocessing import StandardScaler

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import subprocess, os
import sys
import os

# Leer datos simulados del nodo (sensores)
df = pd.read_csv("local_data.csv")

scaler = StandardScaler()
X_np = scaler.fit_transform(df.iloc[:, :-1].values)
y_np = df.iloc[:, -1].values.reshape(-1, 1)

X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)


# Inicializar y entrenar modelo local
model = SimpleSensorModel()
opt = optim.SGD(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for _ in range(10):  # mini entrenamientos
    pred = model(X)
    loss = loss_fn(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

# Serializar pesos
weights = torch.cat([p.data.view(-1) for p in model.parameters()]).numpy()
with open("model_update.bin", "wb") as f:
    f.write(weights.tobytes())

# Enviar al servidor
subprocess.run(["python", "../../protocol/udp_client.py", "model_update.bin"])
