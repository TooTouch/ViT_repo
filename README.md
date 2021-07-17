# ViT_repo
Pytorch version of the pre-trained VIT models released by google research 

# Environments

```
ml-collections
einops=0.3.0
torch=1.7.0
torchvision=0.8.0
```
# Load a Pretrained Model

**Available Pretrained Models**

I will update other models ASAP.

Pretrained models in [here](https://console.cloud.google.com/storage/browser/vit_models/imagenet21k?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false).

- ViT-B_16
- ViT-B_32
- ViT-L_16
- ViT-L_32
- ViT-H_14
- R50-ViT-B_16
- R50-ViT-L_16

```python
from ViT.model import VisionTransformer, CONFIGS

class Config:
    img_size         = 224
    vit_patches_size = 16
    vit_name         = 'ViT-L_16'
args = Config()

# model config
config_vit = CONFIGS[args.vit_name]
print('pretrained path: ',config_vit.pretrained_path)

if args.vit_name.find('R50') != -1:
    config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

# build model
model = VisionTransformer(config_vit, vis=True)
model.load_from(config_vit)
```

# Code Reference

- https://github.com/Beckschen/TransUNet/tree/d68a53a2da73ecb496bb7585340eb660ecda1d59