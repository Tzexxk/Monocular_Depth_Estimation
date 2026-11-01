import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pathlib import Path
from .models import MODELS

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def masked_l1(pred, gt, mask):
    valid = mask.bool()

    if not valid.any():
        raise ValueError("No valid pixels in batch")

    return torch.abs(pred[valid] - gt[valid]).mean()


def get_highlight(batch, model_name, device):
    depth = batch["pred"].to(device)

    if model_name == "highlight_c1":
        return batch["highlight"].to(device)

    return torch.zeros_like(depth)


@torch.inference_mode()
def validate_refinement(model, loader, model_name, device):
    model.eval()

    total_error = 0.0
    total_pixels = 0

    for batch in loader:
        rgb = batch["rgb"].to(device)
        depth = batch["pred"].to(device)
        gt = batch["gt"].to(device)
        mask = batch["mask"].to(device)

        highlight = get_highlight(
            batch, model_name, device
        )

        output = model(rgb, depth, highlight)

        total_error += (
            torch.abs(output - gt)[mask].sum().item()
        )
        total_pixels += mask.sum().item()

    return total_error / total_pixels


def train_model(model_name, train_dataset, val_dataset, output_root, seed=42, max_epochs=30, patience=6, lr=1e-4, weight_decay=1e-4, device='cuda'):
    set_seed(seed)
    device=torch.device(device)
    model=MODELS[model_name]().to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator=torch.Generator().manual_seed(seed)
    train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True,num_workers=0,generator=generator)
    val_loader=DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=0)
    best=float('inf'); bad=0; history=[]
    path=Path(output_root)/'checkpoints'/model_name/f'seed_{seed}_best.pth'
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite checkpoint: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,max_epochs+1):
        model.train(); error=0.; pixels=0
        for batch in tqdm(train_loader,desc=f'{model_name} epoch {epoch}',leave=False):
            rgb=batch['rgb'].to(device); depth=batch['pred'].to(device); gt=batch['gt'].to(device); mask=batch['mask'].to(device)
            highlight=get_highlight(batch,model_name,device)
            opt.zero_grad(set_to_none=True)
            out=model(rgb,depth,highlight); loss=masked_l1(out,gt,mask); loss.backward(); opt.step()
            error+=torch.abs(out.detach()-gt)[mask].sum().item();pixels+=mask.sum().item()
        val=validate_refinement(model,val_loader,model_name,device)
        history.append({'epoch':epoch,'train_l1':error/pixels,'val_l1':val})
        if val < best - 1e-5:
            best=val;bad=0
            torch.save({'model_state_dict':model.state_dict(),'model_name':model_name,'seed':seed,'epoch':epoch,'best_val_loss':best},path)
        else:
            bad+=1
            if bad>=patience:break
    return path,history
