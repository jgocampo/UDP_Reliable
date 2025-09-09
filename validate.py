import torch
import os
import sys
# Detectar raíz del proyecto
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from shared.model_def import SimpleSensorModel


model = SimpleSensorModel()
model.load_state_dict(torch.load("models/global_model.pt"))

# Verifica los parámetros promedio
for name, param in model.named_parameters():
    print(name, param.data.view(-1)[:5])  # Muestra los primeros 5 valores de cada capa
