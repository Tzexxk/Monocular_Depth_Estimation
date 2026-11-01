from pathlib import Path
import sys
import cv2
import numpy as np
import torch

def load_model(repo, checkpoint, device='cuda'):
    repo=str(Path(repo).expanduser().resolve())
    if repo not in sys.path: sys.path.insert(0,repo)
    from depth_anything_v2.dpt import DepthAnythingV2
    model=DepthAnythingV2(encoder='vits',features=64,out_channels=[48,96,192,384])
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    return model.to(device).eval()

def infer_sequence(model,dataset,out_dir,input_size=518):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=False)
    sample_ids = []
    for i in range(len(dataset)):
        sample=dataset[i]; frame=sample['sample_id'].split('/')[-1]
        sample_ids.append(sample['sample_id'])
        path=out_dir/f'{frame}.npy'
        rgb=sample['image'].numpy().transpose(1,2,0)
        bgr=cv2.cvtColor(np.clip(np.rint(rgb*255),0,255).astype(np.uint8),cv2.COLOR_RGB2BGR)
        with torch.inference_mode(): pred=model.infer_image(bgr,input_size=input_size)
        assert pred.shape==(1080,1350)
        if not np.isfinite(pred).all():
            raise ValueError(f"Non-finite relative prediction: {frame}")
        with path.open("xb") as stream:
            np.save(stream,pred.astype(np.float32),allow_pickle=False)
    return sample_ids
