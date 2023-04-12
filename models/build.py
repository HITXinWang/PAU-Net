# --------------------------------------------------------
# --------------------------------------------------------
from .PAU_Net import PAUNet

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'pau-net':
        model = PAUNet(embed_dim=config.MODEL.NET.EMBED_DIM,
                        patch_size=config.DATA.LPATCH_SIZE,
                        sr_scale=config.MODEL.SR_SCALE,
                        num_blocks=config.MODEL.NUM_BLOCKS)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
