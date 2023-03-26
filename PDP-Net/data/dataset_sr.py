import random
import torch.utils.data as data
import util.utils_image as util


class DatasetSR(data.Dataset):
    def __init__(self, opt):
        super(DatasetSR, self).__init__()
        self.opt = opt
        self.n_channels = opt.n_channels if opt.n_channels else 3
        self.sf = opt.sr_scale if opt.sr_scale else 4
        self.patch_size = self.opt.hpatch_size if self.opt.hpatch_size else [512, 128]
        self.L_size = [i // self.sf for i in self.patch_size]

        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt.dataroot_H)
        self.paths_L = util.get_image_paths(opt.dataroot_L)

        assert self.paths_H, 'Error: H path is empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

    def __getitem__(self, index):

        L_path = None
        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        img_H = util.uint2single(img_H)

        # ------------------------------------
        # modcrop
        # ------------------------------------
        img_H = util.modcrop(img_H, self.sf)

        # ------------------------------------
        # get L image
        # ------------------------------------
        if self.paths_L:
            # --------------------------------
            # directly load L image
            # --------------------------------
            L_path = self.paths_L[index]
            img_L = util.imread_uint(L_path, self.n_channels)
            img_L = util.uint2single(img_L)

        else:
            # --------------------------------
            # sythesize L image via matlab's bicubic
            # --------------------------------
            H, W = img_H.shape[:2]
            img_L = util.imresize_np(img_H, 1 / self.sf, True)

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        #if self.opt['phase'] == 'train':

        H, W, C = img_L.shape

        # --------------------------------
        # randomly crop the L patch
        # --------------------------------
        rnd_h = random.randint(0, max(0, H - self.L_size[0]))
        rnd_w = random.randint(0, max(0, W - self.L_size[1]))
        img_L = img_L[rnd_h:rnd_h + self.L_size[0], rnd_w:rnd_w + self.L_size[1], :]

        # --------------------------------
        # crop corresponding H patch
        # --------------------------------
        rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
        img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size[0], rnd_w_H:rnd_w_H + self.patch_size[1], :]

        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L) #CHW-RGB

        if L_path is None:
            L_path = H_path

        #return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}
        return img_L, img_H

    def __len__(self):
        return len(self.paths_H)
