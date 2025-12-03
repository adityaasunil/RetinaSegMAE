import torch
from torchvision.utils import save_image
from einops import rearrange
from dataset import RetinaDataset
from model import MAE_ViT
from transforms import get_transforms

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 384
PATCH_SIZE = 16
NUM_SHOW = 8
MODEL_WEIGHTS = "bestmodel.pt"   
SPLIT = "test"                     


def denorm(x):
    """Undo normalization (mean=0.5 std=0.5)."""
    return x * 0.5 + 0.5


print("Loading model...")
model = torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=False)
model.to(DEVICE)
model.eval()


print("Loading dataset...")
dataset = RetinaDataset(SPLIT)
indices = list(range(min(NUM_SHOW, len(dataset))))   # get first few samples

imgs = torch.stack([dataset[i][0][:3, :, :] for i in indices]).to(DEVICE)

print("Running inference...")
with torch.no_grad():
    pred, mask = model(imgs)

    # reconstruct only masked regions
    recon = pred * mask + imgs * (1 - mask)

    # masked input (zeroed regions)
    masked_input = imgs * (1 - mask)

    # stack vertically:
    # Row 1 → masked inputs
    # Row 2 → reconstructed outputs
    # Row 3 → original images
    viz = torch.cat([masked_input, recon, imgs], dim=0)

    # clamp + denormalize
    viz = viz.clamp(-1, 1)
    viz = denorm(viz)

    # store grid output
    save_image(viz, "Test_results.png", nrow=NUM_SHOW)

print("Saved reconstruction image as mae_test_recon.png")