# Unified model manager for initializing VA_VAE and LightningDiT from same yaml config

import torch
from omegaconf import OmegaConf
from ldit.vavae import VA_VAE
from ldit.lightningdit import LightningDiT_models
from ldit.autoencoder import AutoencoderKL


class ModelManager:
    """Unified model manager for VAE and DiT models from single yaml config"""
    
    def __init__(self, config_path):
        """Initialize model manager from config file"""
        self.config = OmegaConf.load(config_path)
        self.vae = None
        self.dit = None
        
    def load_vae(self):
        """Load VA_VAE model"""
        self.vae = VA_VAE_FromConfig(self.config)
        return self.vae
    
    def load_dit(self, num_classes=None):
        """Load LightningDiT model"""
        dit_config = self.config.lightningdit
        
        model_params = {
            'input_size': dit_config.input_size,
            'in_channels': dit_config.in_channels,
            'mlp_ratio': dit_config.get('mlp_ratio', 4.0),
            'class_dropout_prob': dit_config.get('class_dropout_prob', 0.1),
            'num_classes': num_classes if num_classes is not None else dit_config.num_classes,
            'learn_sigma': dit_config.get('learn_sigma', False),
            'use_qknorm': dit_config.get('use_qknorm', True),
            'use_swiglu': dit_config.get('use_swiglu', True),
            'use_rope': dit_config.get('use_rope', True),
            'use_rmsnorm': dit_config.get('use_rmsnorm', True),
            'wo_shift': dit_config.get('wo_shift', False),
            'use_checkpoint': dit_config.get('use_checkpoint', False),
            'auxiliary_time_cond': dit_config.get('auxiliary_time_cond', False),
            'disable_label_dropout': dit_config.get('disable_label_dropout', False)
        }
        
        model_type = dit_config.get('model_type', 'LightningDiT-XL/1')
        
        if model_type in LightningDiT_models:
            self.dit = LightningDiT_models[model_type](**model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        return self.dit
    
    def load_both(self, num_classes=None):
        """Load both VAE and DiT models"""
        vae = self.load_vae()
        dit = self.load_dit(num_classes)
        return vae, dit


class VA_VAE_FromConfig:
    """VA_VAE initialized from config object"""
    
    def __init__(self, config, img_size=256, horizon_flip=0.5, fp16=True):
        """Initialize VA_VAE from config object"""
        self.config = config
        self.embed_dim = self.config.model.params.embed_dim
        self.ckpt_path = self.config.ckpt_path
        self.img_size = img_size
        self.horizon_flip = horizon_flip
        self.load()

    def load(self):
        """Load and initialize VAE model"""
        self.model = AutoencoderKL(
            embed_dim=self.embed_dim,
            ch_mult=(1, 1, 2, 2, 4),
            ckpt_path=self.ckpt_path
        ).cuda().eval()
        return self
    
    def img_transform(self, p_hflip=0, img_size=None):
        """Image preprocessing transforms"""
        from ldit.vavae import VA_VAE
        temp_vae = VA_VAE.__new__(VA_VAE)
        temp_vae.img_size = self.img_size
        return temp_vae.img_transform(p_hflip, img_size)

    def encode_images(self, images):
        """Encode images to latent representations"""
        with torch.no_grad():
            posterior = self.model.encode(images.cuda())
            return posterior.sample()

    def decode_to_images(self, z):
        """Decode latent representations to images"""
        with torch.no_grad():
            images = self.model.decode(z.cuda())
        return images

    def decode_to_images2(self, z):
        """Decode latent representations to images (alternative method)"""
        images = self.model.decode(z.cuda())
        return images


# Convenience functions
def create_models_from_config(config_path, num_classes=None):
    """Create both models from config file"""
    manager = ModelManager(config_path)
    return manager.load_both(num_classes)


def create_vae_from_config(config_path):
    """Create VAE model from config file"""
    manager = ModelManager(config_path)
    return manager.load_vae()


def create_dit_from_config(config_path, num_classes=None):
    """Create DiT model from config file"""
    manager = ModelManager(config_path)
    return manager.load_dit(num_classes)
