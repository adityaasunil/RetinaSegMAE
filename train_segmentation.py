import os,sys 
import argparse
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import RetinaDataset
from tqdm import tqdm
from utils import setup_seed
from torch.utils.data import DataLoader
import torch 
import torch.nn as n
import torch.nn.functional as f 
from torchvision.utils import save_image
from model import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--patch_size',type=int,default=16)
    parser.add_argument('--base_learning_rate',type=float,default=1e-4)
    parser.add_argument('--max_epochs',type=int,default=300)
    parser.add_argument('--model_path',type=str,default='bestmodel.pt')

    args = parser.parse_args()
    device = torch.device('mps')

    setup_seed(args.seed)

    training_dataset = RetinaDataset('train')
    training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = RetinaDataset('test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    state = torch.load(args.model_path,map_location=device,weights_only=False,)
    model = state.to(device)
    
    # Freezing the encoder weights
    for p in model.encoder.parameters():
        p.requires_grad = False

        
    def dice_loss(pred,truth,eps=1e-6):
        i=(pred*truth).sum(dim=(2,3))
        u = pred.sum(dim=(2,3)) + truth.sum(dim=(2,3))
        dice = (2 * i + eps)/(u + eps)
        return 1 - dice.mean()
    
    def dice_score(pred,truth,eps=1e-6):
        i = (pred*truth).sum(dim=(2,3))
        u = pred.sum(dim=(2,3)) + truth.sum(dim=(2,3))
        dice = (2 * i + eps) / (u + eps)
        return dice.mean().item()
    
    def iou_score(pred,target,eps=1e-6):
        i = (pred*target).sum(dim=(2,3))
        u = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
        iou = (i + eps) / (u + eps)
        return iou.mean().item()

    # Training Loop

    seg_model = MAESegmenation(model)
    seg_model.to(device)

    pos_weight = torch.tensor([5.0], device=device)
    criterion = n.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW([p for p in seg_model.parameters() if p.requires_grad],lr=args.base_learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.1)

    best_loss = float('inf')
    best_model_path = 'segmentation_best.pt'

    for epoch in range(args.max_epochs):
        seg_model.train()
        running_loss = 0.0
        losses=[]
        for i,(img,mask) in enumerate(tqdm(training_dataloader, desc='Epoch {}/{}'.format(epoch+1,args.max_epochs))):
            imgs = img.to(device)
            masks = (mask>0).float()
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            masks = masks.to(device)

            logits = seg_model(imgs)
            bce = criterion(logits,masks)

            probs = torch.sigmoid(logits)
            dl = dice_loss(probs,masks)

            loss = 0.3 * bce + 0.7 * dl 
            
            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.item())
            running_loss += loss.item() 
        print(f"Epoch {epoch+1} , avg loss={running_loss/len(training_dataloader):.6f}")

        avg_loss = running_loss/len(training_dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(seg_model.state_dict(), best_model_path)
            print(f"> New best model saved at epoch {epoch+1} with loss {best_loss:.6f}")

        scheduler.step()


    with torch.no_grad():
        state_dict = torch.load(best_model_path,map_location=device)
        seg_model.load_state_dict(state_dict=state_dict)
        seg_model.to(device)
        seg_model.eval()

        os.makedirs('segmentation_vis',exist_ok=True)
        dices = []
        ious = []
        for i, (img,mask) in enumerate(test_dataloader):
            imgs = img.to(device)
            masks = (mask>0).float().to(device)

            logits = seg_model(imgs)

            probs = torch.sigmoid(logits)

            print(
            f"Batch {i} probs stats -> min: {probs.min().item():.4f}, "
            f"max: {probs.max().item():.4f}, mean: {probs.mean().item():.4f}"
            )

                        # Visualize first few test samples
            if i < 5:
                # Take first item in batch
                img_vis = imgs[0].detach().cpu()
                gt_mask = masks[0].detach().cpu()
                pred_prob = probs[0].detach().cpu()
                pred_bin = (pred_prob > 0.15).float()

                # Unnormalize image from [-1,1] to [0,1] if using mean=0.5,std=0.5
                img_vis = (img_vis * 0.5 + 0.5).clamp(0, 1)

                # Make masks 3-channel for visualization
                gt_3 = gt_mask.repeat(3, 1, 1)
                pred_3 = pred_bin.repeat(3, 1, 1)

                # Stack: [input | GT | prediction]
                grid = torch.stack([img_vis, gt_3, pred_3], dim=0)

                save_image(grid, f"segmentation_vis/sample_{i}.png", nrow=3)

            d = dice_score(probs,masks)
            j = iou_score(probs,masks)

            dices.append(d)
            ious.append(j)

        mean_dice = sum(dices)/len(dices)
        mean_ious = sum(ious)/len(ious)

        print("Logits stats -> min:", logits.min().item(), "max:", logits.max().item())

        print(f"[BEST] Test Dice: {mean_dice:.4f}, Test IoU: {mean_ious:.4f}")
