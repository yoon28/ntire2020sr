import os.path as osp
import time
import argparse
from collections import OrderedDict

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
from torchvision.transforms.functional import rotate, to_pil_image, to_tensor

import torch
torch.backends.cudnn.enabled = False

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    test_loaders.append(test_loader)

model = create_model(opt)
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    for data in test_loader:
        need_GT = False
        data2 = {'LQ':torch.flip(data['LQ'], [3]), 'LQ_path':data['LQ_path']}
        imgs = torch.zeros([1, 3, data['LQ'].shape[2]*16, data['LQ'].shape[3]*16], dtype=torch.float32, device='cpu')

        model.feed_data(data, need_GT=need_GT)
        model.test()
        imgs += model.get_current_visuals(need_GT=need_GT)['SR']

        model.feed_data(data2, need_GT=need_GT)
        model.test()
        imgs += model.get_current_visuals(need_GT=need_GT)['SR']
        for rot in range(1, 4):
            rotim = torch.rot90(data['LQ'], rot, (2, 3))
            augdata = {'LQ':rotim, 'LQ_path':data['LQ_path']}

            model.feed_data(augdata, need_GT=need_GT)
            model.test()
            visuals = model.get_current_visuals(need_GT=need_GT)

            imgs += torch.rot90(visuals['SR'], 4 - rot, (1, 2))

        for rot in range(1, 4):
            rotim = torch.rot90(data2['LQ'], rot, (2, 3))
            augdata = {'LQ':rotim, 'LQ_path':data2['LQ_path']}

            model.feed_data(augdata, need_GT=need_GT)
            model.test()
            visuals = model.get_current_visuals(need_GT=need_GT)

            imgs += torch.flip(torch.rot90(visuals['SR'], 4 - rot, (1, 2)), [2])

        imgs /= 8
        sr_img = util.tensor2img(imgs)  # uint8

        img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
        img_name = osp.splitext(osp.basename(img_path))[0]
        # save images
        suffix = opt['suffix']
        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + '.png')
        else:
            save_img_path = osp.join(dataset_dir, img_name + '.png')
        util.save_img(sr_img, save_img_path)
        print('Saved: {}'.format(save_img_path))
