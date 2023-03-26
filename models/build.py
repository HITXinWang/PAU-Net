# --------------------------------------------------------
# --------------------------------------------------------
from .PDP_Net import PDPNet

#建模的基本参数
def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'pdp-net':
        model = PDPNet(embed_dim=config.MODEL.NET.EMBED_DIM,
                        patch_size=config.DATA.LPATCH_SIZE,
                        sr_scale=config.MODEL.SR_SCALE,
                        num_blocks=config.MODEL.NUM_BLOCKS)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
