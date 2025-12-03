import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
from dataset import RetinaDataset

from model import *
from utils import setup_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_device_batch_size', type=int, default=5)
    parser.add_argument('--base_learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--vessel_weight', type=float, default=4.0)
    parser.add_argument('--mask_ratio', type=float, default=0.6)
    parser.add_argument('--total_epoch', type=int, default=3000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--model_path', type=str, default='bestmodel.pt')

    args = parser.parse_args()

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0

    train_dataset = RetinaDataset('train')
    val_dataset = RetinaDataset('test')
    val_dataloader = torch.utils.data.DataLoader(val_dataset,load_batch_size, num_workers=2)
    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=0)
    writer = SummaryWriter(os.path.join('logs', 'Retina Dataset', 'mae-pretrain'))
    device = torch.device('mps')

    model = MAE_ViT()
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate, weight_decay=0.05)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.total_epoch)

    step_count = 0
    optim.zero_grad()
    best_loss = float('inf')
    for e in range(args.total_epoch):
        model.train()
        losses = []
        for k, (img,_) in enumerate(tqdm(dataloader,desc=f"Epoch {e+1}/{args.total_epoch}")):
            step_count += 1
            img = img.to(device)
            
            if img.shape[1] >= 4:
                vessel_map = img[:, 3:4, : , :]
                rgb = img[:, :3, : , :]
            else:
                vessel_map = None 
                rgb = img

            with torch.autocast(device_type='mps', dtype=torch.float16):
                
                predicted_img, mask = model(rgb)

                recon_error = (predicted_img - rgb) ** 2

                if vessel_map is not None:
                    v_min = vessel_map.amin(dim=(2,3), keepdim=True)
                    v_max = vessel_map.amax(dim=(2,3), keepdim=True)
                    vessel_norm = (vessel_map - v_min) / (v_max - v_min + 1e-6)

                    weight = 1.0 + args.vessel_weight * vessel_norm
                else:
                    weight = 1.0

                if isinstance(weight, torch.Tensor) and weight.ndim == 4 and weight.shape[1] == 1:
                    weight = weight.expand_as(recon_error)

                loss = torch.mean(recon_error * mask * weight) / args.mask_ratio
                
            loss.backward()
            optim.step()
            optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('mae_loss', avg_loss, global_step=e)
        print(f'In epoch {e+1}, average traning loss is {avg_loss}.')

        ''' visualize the first 16 predicted images on val dataset'''
        model.eval()
        with torch.no_grad():
            num_val = min(16, len(val_dataset))
            val_img = torch.stack([val_dataset[i][0] for i in range(num_val)])
            val_img = val_img.to(device)

            if val_img.shape[1] >= 4:
                val_vessel = val_img[:, 3:4, : , :]
                val_rgb = val_img[:, :3, :, :]
            else:
                val_vessel = None
                val_rgb = val_img 

            predicted_val_img, mask = model(val_img)
            blended_pred = predicted_val_img * mask + val_rgb * (1 - mask)

            if val_vessel is not None:
                v_min = val_vessel.amin(dim=(2,3), keepdim=True)
                v_max = val_vessel.amax(dim=(2,3), keepdim=True)
                vessel_norm = (val_vessel - v_min) / ( v_max - v_min + 1e-6)
                weight = 1.0 + args.vessel_weight * vessel_norm
                weight = weight.expand_as(blended_pred)
            else:
                weight = 1.0


            val_recon_error = (blended_pred - val_rgb) ** 2
            val_loss = torch.mean(val_recon_error * mask * weight) / args.mask_ratio
            print('Validation loss: {:.4f}'.format(val_loss))
            img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
            img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            writer.add_image('mae_image', (img + 1) / 2, global_step=e)
        
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            print('### Best loss: {:.4f}, saving model.'.format(best_loss))
            ''' save model '''
            torch.save(model, args.model_path)
