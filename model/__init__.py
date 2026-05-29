import os
from importlib import import_module

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')

        self.args = args
        self.scale = args.scale
        self.idx_scale = 0
        self.precision = args.precision # FP precision for test (single | half)
        self.cpu = args.cpu 
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs 
        self.save_models = args.save_models  # save all intermediate models

        module = import_module('model.' + args.model.lower()) # 导包
        self.model = module.make_model(args).to(self.device) # model.cuda()
        if args.precision == 'half': self.model.half()

        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs)) # 多个GPU 并行
 
        self.load(
            ckp.dir,
            pre_train=args.pre_train, # pre-trained model directory default = '.'
            resume=args.resume, # resume from specific checkpoint default = 0
            cpu=args.cpu
        )
        if args.print_model: print(self.model)

    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale
        target = self.get_model()
        if hasattr(target, 'set_scale'):
            target.set_scale(idx_scale)

        return self.model(x)

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(), 
            os.path.join(apath, 'model', 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_best.pt')
            )
        
        if self.save_models and epoch % self.args.save_models_freq==0:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_{}.pt'.format(epoch))
            )

    def load(self, apath, pre_train='.', resume=-1, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        # 训练的时候想重新恢复断点
        if resume == -1:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_latest.pt'),
                    weights_only=True, **kwargs
                ),
                strict=False
            )
        elif resume == 0:
            if pre_train != '.':
                print('Loading model from {}'.format(pre_train))
                self.get_model().load_state_dict(
                    torch.load(pre_train, weights_only=True, **kwargs),
                    strict=False
                )
        else:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_{}.pt'.format(resume)),
                    weights_only=True, **kwargs
                ),
                strict=False
            )