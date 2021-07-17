import ml_collections

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = './model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.download_path = 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16

    return config

def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.pretrained_path = './model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.download_path = 'https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz'
    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    config.pretrained_path = './model/vit_checkpoint/imagenet21k/ViT-B_32.npz'
    config.download_path = 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_32.npz'
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.representation_size = None

    # custom
    config.resnet_pretrained_path = None
    config.pretrained_path = './model/vit_checkpoint/imagenet21k/ViT-L_16.npz'
    config.download_path = 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz'
    return config


def get_r50_l16_config():
    """Returns the Resnet50 + ViT-L/16 configuration """
    config = get_l16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.resnet_pretrained_path = './model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.download_path = 'https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz'
    
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config


def get_h14_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None

    return config


