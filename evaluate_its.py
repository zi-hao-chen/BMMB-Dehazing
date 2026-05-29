import argparse
import os
import time
import torch
import torch.nn as nn

from importlib import import_module
from utils import utils_logger
from utils import utils_test as utils
from torch.utils.data import DataLoader
from math import log10
# from model.fsdgn import FSDGN as Net
from data.test_data import TestData

parser = argparse.ArgumentParser(description="Testing for Dehazing Models")
parser.add_argument("--model", type=str, default='sffn', help="Mode name")
parser.add_argument("--n_level", type=int, default=4, help="level of siu")
parser.add_argument("--order", type=float, default=0.5, help="order of fau")
parser.add_argument('--pre_train', type=str, default='../experiment/sffn_light_its/model/model_best.pt', help="Path to pretrained model") 
parser.add_argument("--dataset", type=str, default='indoor', help='indoor, outdoor, nh, dense')
parser.add_argument("--downsample", type=int, default=16, help='maxmium downsample factor')
parser.add_argument("--batch_size", type=int, default=4, help='batch size')
parser.add_argument("--save_path", type=str, default='../experiment/test_results', help='Save restoration results')
parser.add_argument("--save_image", type=bool, default=False)
parser.add_argument("--multi_supervised", type=bool, default=True)

opt = parser.parse_args()

save_path = os.path.join(opt.save_path, opt.model + '/' + opt.dataset + '_results') 

module = import_module('model.' + opt.model.lower()) 
Net = module.make_model(opt)

if not os.path.exists(save_path):
        utils.create_dir(save_path)

if opt.dataset == 'indoor':
    test_data_dir = '/home/ubuntu/Project/Dehazing/Data/ITS/val_indoor'

elif opt.dataset == 'outdoor':
    test_data_dir = '/home/ubuntu/Project/Dehazing/Data/OTS/val_outdoor'

else:
    raise ValueError(f'Unknown dataset: {opt.dataset}. Supported datasets: indoor, outdoor')


lg = utils_logger.logger_info('efficient image dehazing', log_path=os.path.join(opt.save_path, opt.model + '_' + opt.dataset + '.log'))
lg.info("============Begin Evaluation============")
lg.info('Model: %s || dataset_name: %s|| pre_train: %s' % (opt.model, opt.dataset, opt.pre_train))

def main():
    # --- Gpu device --- #
    device_ids = [Id for Id in range(torch.cuda.device_count())]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Validation data loader --- #
    lg.info('Loading data: %s' % (opt.dataset))
    val_data_loader = DataLoader(TestData(opt.dataset, test_data_dir, opt.downsample), \
        batch_size=opt.batch_size, shuffle=False, num_workers=4)

    # --- Define the network --- #
    lg.info('Loading model: %s' % (opt.pre_train))

    # --- Multi-GPU --- #
    net = Net.to(device)
    net = nn.DataParallel(net, device_ids=device_ids).module

    # --- Load the network weight --- #
    net.load_state_dict(
                torch.load(
                    os.path.join(opt.pre_train),
                    weights_only=True, **{}
                ),
                strict=False
    )

    # --- Use the evaluation model in testing --- #
    net.eval()
    lg.info('--- Testing starts! ---')
    start_time = time.time()

    val_psnr, val_ssim, median_time = utils.validation(net, val_data_loader, device, opt.save_image, save_path, opt.multi_supervised)
    end_time = time.time() - start_time
    lg.info('val_psnr: {0:.3f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))
    lg.info('validation time is {0:.4f}'.format(end_time))
    lg.info('validation average time is {0:.4f}'.format(end_time / len(val_data_loader)))
    lg.info('validation median_time time is {0:.4f}'.format(median_time))
    lg.info('--- Testing end! ---')
    lg.info("============End Evaluation============\n")


if __name__ == "__main__":
    main()
