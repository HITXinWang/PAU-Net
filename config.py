# --------------------------------------------------------
# 360SR
# Written by Xin Wang
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 8
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = '360SR'
# Input lR image patch size
_C.DATA.LPATCH_SIZE = (128, 32)  #ERP image's height, it means  the resolution of input ERP image is 128*256
# Input hr image patch size
_C.DATA.HPATCH_SIZE = (1024, 256)
_C.DATA.INTERPOLATION = 'bicubic'
_C.DATA.NUM_WORKERS = 2

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'pdp-net'
# SR scale
_C.MODEL.SR_SCALE = 8
# Model name
_C.MODEL.NAME = '360sr'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
_C.MODEL.NUM_BLOCKS = 9
# NET parameters
_C.MODEL.NET = CN()
_C.MODEL.NET.EMBED_DIM = 180

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 200
_C.TRAIN.START_ITER = 0
_C.TRAIN.ITERS = 500000
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0
_C.TRAIN.BASE_LR = 1e-4
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'torch_multiStep'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 20
# Epoch interval to decay LR, used in Torch StepLR
_C.TRAIN.LR_SCHEDULER.Torch_StepLR_EPOCHS = 100
# Epoch interval to decay LR, used in Torch MultiStepLR
_C.TRAIN.LR_SCHEDULER.Torch_MultiStepLR_EPOCHS = (100, 150, 175, 190)
_C.TRAIN.LR_SCHEDULER.Torch_MultiStepLR_iteration = (250000, 400000, 450000, 475000, 500000)
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adam'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.SAVE_FREQ = 10000  #for epoch or iter
# Frequency to logging info
_C.PRINT_FREQ = 100
# Frequency to eval model
_C.EVAL_FREQ = 10000
# Fixed random seed
_C.SEED = 100

def _update_config_from_file(config, cfg_file):
	config.defrost()
	with open(cfg_file, 'r') as f:
		yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

	for cfg in yaml_cfg.setdefault('BASE', ['']):
		if cfg:
			_update_config_from_file(
				config, os.path.join(os.path.dirname(cfg_file), cfg)
			)
	print('=> merge config from {}'.format(cfg_file))
	config.merge_from_file(cfg_file)
	config.freeze()


def update_config(config, args):
	_update_config_from_file(config, args.cfg)

	config.defrost()
	if args.opts:
		config.merge_from_list(args.opts)
	# merge from specific arguments
	if args.batch_size:
		config.DATA.BATCH_SIZE = args.batch_size
	if args.dataset_name:
		config.DATA.DATASET = args.dataset_name
	if args.data_path:
		config.DATA.DATA_PATH = args.data_path
	if args.sr_scale:
		config.MODEL.SR_SCALE = args.sr_scale
	if args.n_channels:
		config.MODEL.SWIN.IN_CHANS = args.n_channels
		config.MODEL.SWIN_MLP.IN_CHANS = args.n_channels
	if args.resume:
		config.MODEL.RESUME = args.resume
	if args.hpatch_size:
		config.DATA.HPATCH_SIZE = (args.hpatch_size[0], args.hpatch_size[1])
		lpatch_h = args.hpatch_size[0] // args.sr_scale
		lpatch_w = args.hpatch_size[1] // args.sr_scale
		config.DATA.LPATCH_SIZE = (lpatch_h, lpatch_w)
	if args.amp_opt_level:
		config.AMP_OPT_LEVEL = args.amp_opt_level
	if args.output:
		config.OUTPUT = args.output
	# if args.local_rank:
	# 	config.LOCAL_RANK = args.local_rank

	# output folder
	config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.MODEL.TYPE, config.DATA.DATASET, f'scale_{config.MODEL.SR_SCALE}')
	config.freeze()

def get_config(args):
	"""Get a yacs CfgNode object with default values."""
	config = _C.clone()
	update_config(config, args)
	return config
