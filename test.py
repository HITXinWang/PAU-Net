import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
import os
import torch
from util.ws_util import getWeightRec
from models.PAU_Net import PAUNet as net
from util import util_calculate_psnr as util

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--scale', type=int, default=4, help='scale factor: 2, 4, 8')
	parser.add_argument('--limage-size', type=tuple, default=[128, 256], help="LR image size, default as [128, 256]")
	parser.add_argument('--model_path', type=str,
						default='./')
	parser.add_argument('--folder_lq', type=str, default='', help='input low-quality test image folder')
	parser.add_argument('--folder_gt', type=str, default='', help='input ground-truth test image folder')
	parser.add_argument('--tile', type=int, default=None, help='Tile size')
	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# set up model
	if os.path.exists(args.model_path):
		print(f'loading model from {args.model_path}')
	limage_size = args.limage_size
	model = torch.load(args.model_path)["model"]  # loading model
	# model = define_model(args,limage_size) # or define model and load params
	if isinstance(model, torch.nn.DataParallel):
		model = model.module
	model.cuda()
	if args.tile is None:
		weightRec= np.sqrt(np.sqrt(getWeightRec(0, limage_size[0], limage_size[1])))
		model.conv1.weightRec = weightRec
		model.conv2.weightRec = weightRec

	model = model.to(device)
	model.eval()
	# setup folder and path
	folder, save_dir, border = setup(args)
	os.makedirs(save_dir, exist_ok=True)
	test_results = OrderedDict()
	test_results['wpsnr'] = []
	wpsnr = 0

	for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
		# read image
		imgname, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32
		img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HWC-BGR to CHW-RGB
		img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to BCHW-RGB

		# inference
		with torch.no_grad():
			output = test(img_lq, model, args)

		# save image
		output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
		if output.ndim == 3:
			output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
		output = (output * 255.0).round().astype(np.uint8)
		cv2.imwrite(f'{save_dir}/{imgname}.png', output)

		# evaluate images
		if img_gt is not None:
			img_gt = (img_gt * 255.0).round().astype(np.uint8)
			img_gt = np.squeeze(img_gt)
			if img_gt.ndim == 3:
				wpsnr = util.calculate_wspsnr(output, img_gt, crop_border=border, input_order='HWC',
												test_y_channel=True)
				test_results['wpsnr'].append(wpsnr)
			print('Testing {:d} {:20s} - WPSNR: {:.2f} dB; '.format(idx, imgname, wpsnr))
		else:
			print('Testing {:d} {:20s}'.format(idx, imgname))

	# summarize wpsnr
	if img_gt is not None:
		ave_wpsnr = sum(test_results['wpsnr']) / len(test_results['wpsnr'])
		print('-- Average WPSNR: {:.2f} dB'.format(ave_wpsnr))

def define_model(args, img_size):
	model = net(embed_dim=180, patch_size=img_size, sr_scale=8, num_blocks=9)
	param_key_g = 'params'
	pretrained_model = torch.load(args.model_path)  #["model"]
	if isinstance(pretrained_model, torch.nn.DataParallel):
		pretrained_model = pretrained_model.module
	pretrained_model = pretrained_model.state_dict()
	model.load_state_dict(
		pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
	return model

def setup(args):
	save_dir = f'results/x{args.scale}'
	folder = args.folder_gt
	border = 0
	return folder, save_dir, border


def get_image_pair(args, path):
	imageFullname=os.path.basename(path)
	(imgname, imgext) = os.path.splitext(imageFullname)
	img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.  #[0,1]
	img_lq = cv2.imread(f'{args.folder_lq}/{imageFullname}', cv2.IMREAD_COLOR).astype(
        np.float32) / 255.

	return imgname, img_lq, img_gt


def test(img_lq, model, args):
	if args.tile is None:
		# test the image as a whole
		output = model(img_lq)
	else:
		# test the image tile by tile
		b, c, h, w = img_lq.size()
		tile = min(args.tile, h, w)
		sf = args.scale
		stride = tile//2
		w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
		E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
		W = torch.zeros_like(E)

		for w_idx in w_idx_list:
			in_patch = img_lq[..., w_idx:w_idx+tile]
			out_patch = model(in_patch)
			out_patch_mask = torch.ones_like(out_patch)
			E[..., :, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
			W[..., :, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
		output = E.div_(W)

	return output

if __name__ == '__main__':
	main()
