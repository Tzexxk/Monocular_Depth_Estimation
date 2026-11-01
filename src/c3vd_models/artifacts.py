from pathlib import Path
import json
import numpy as np
import torch
from .models import MODELS

def load_checkpoint(model_name, path, device='cpu'):
    model=MODELS[model_name]().to(device)
    obj=torch.load(path,map_location=device,weights_only=True)
    state=obj.get('model_state_dict',obj.get('model_state',obj.get('state_dict',obj))) if isinstance(obj,dict) else obj
    model.load_state_dict(state,strict=True)
    return model.eval()

def save_prediction(path, depth_mm):
    arr=np.asarray(depth_mm,dtype=np.float32)
    if arr.shape!=(1080,1350):raise ValueError(f'Unexpected shape {arr.shape}')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream,arr,allow_pickle=False)

def load_calibration(path):
    data=json.loads(Path(path).read_text());return float(data['a']),float(data['b'])
