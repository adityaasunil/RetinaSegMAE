import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

def random_indexes(T,B,device):
    forward = torch.stack([torch.randperm(T,device=device) for _ in range(B)], dim=1)
    backward = torch.argsort(forward, dim=0)
    return forward, backward

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        device = patches.device
        forward_indexes, backward_indexes = random_indexes(T,B,device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=384,
                 patch_size=16,
                 emb_dim=384,
                 num_layer=8,
                 num_head=8,
                 mask_ratio=0.6,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes

    def forward_all_tokens(self,img):
        patches = self.patchify(img)                           # (B, C, H', W')
        patches = rearrange(patches, 'b c h w -> (h w) b c')    # (T_FULL, B, C)
        patches = patches + self.pos_embedding                  # full pos emb

        patches = torch.cat(
           [self.cls_token.expand(-1, patches.shape[1], -1), patches],
           dim=0
        )

        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        return features

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=384,
                 patch_size=16,
                 emb_dim=384,
                 num_layer=4,
                 num_head=8,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask

class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=384,
                 patch_size=16,
                 emb_dim=384,
                 encoder_layer=8,
                 encoder_head=8,
                 decoder_layer=4,
                 decoder_head=8,
                 mask_ratio=0.6,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features,  backward_indexes)
        return predicted_img, mask

# making the conv block

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels,num_groups: int = 8):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.gn1 = torch.nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.gn2 = torch.nn.GroupNorm(num_groups=min(num_groups,out_channels), num_channels=out_channels)

        self.res_conv = None 
        if in_channels != out_channels:
            self.res_conv = torch.nn.Conv2d(in_channels,out_channels,kernel_size=1)
        
    
    def forward(self, x):
        indentity = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = torch.nn.functional.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.gn2(out)
        if self.res_conv is not None:
            identity = self.res_conv(identity)
        out = out + identity
        out = torch.nn.functional.relu(out, inplace=True)
        return out


class MAESegmenation(torch.nn.Module):
    def __init__(self,
                 mae_model,
                 embed_dim=384):
        super().__init__()

        self.encoder = mae_model.encoder

        for p in self.encoder.parameters():
            p.requires_grad = False 

        self.context = ConvBlock(embed_dim,embed_dim)

        self.dec1 = ConvBlock(embed_dim,256) # 24x24, 384 -> 256
        self.up1 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) # 24x24->48x48
        self.dc2 = ConvBlock(256,192) # 48x48 , 256 -> 128
        self.up2 = torch.nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False) # 96x96
        self.dc3 = ConvBlock(192,128) # 96x96, 128 -> 64
        self.up3 = torch.nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False) # 192x192
        self.dc4 = ConvBlock(128,64) # 192x 192, 64->32
        self.up4 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) # 192 -> 384
        self.dc5 = ConvBlock(64,32)
        self.finalConv = torch.nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, img):

        features = self.encoder.forward_all_tokens(img)
        _,B,C = features.shape
        tokens = features[1:] # (576, B, C)

        T_full = tokens.shape[0]
        H=W= int(T_full ** 0.5)

        tokens = tokens.permute(1,0,2).contiguous() # (B,T,C)
        tokens = tokens.view(B,H,W,C) # (B,24,24,C)
        feat_map = tokens.permute(0,3,1,2).contiguous()
        feat_map = self.context(feat_map) # (B,C,H,W)

        x = self.dec1(feat_map)
        x = self.up1(x)

        x = self.dc2(x)
        x = self.up2(x)

        x = self.dc3(x)
        x = self.up3(x)

        x = self.dc4(x)
        x = self.up4(x)

        x = self.dc5(x)
        logits = self.finalConv(x)
        return logits


if __name__ == '__main__':
    shuffle = PatchShuffle(0.75)
    a = torch.rand(16, 2, 10)
    b, forward_indexes, backward_indexes = shuffle(a)
    print(b.shape)

    img = torch.rand(2, 3, 384, 384)
    encoder = MAE_Encoder()
    decoder = MAE_Decoder()
    features, backward_indexes = encoder(img)
    print(forward_indexes.shape)
    predicted_img, mask = decoder(features, backward_indexes)
    print(predicted_img.shape)
    loss = torch.mean((predicted_img - img) ** 2 * mask / 0.75)
    print(loss.item())

    vit_model = MAE_ViT()
    model = MAESegmenation(vit_model)
    logits = model(predicted_img)
    print(logits.shape)
