import os
from decimal import Decimal
import numpy as np
import utility
import torch.nn.functional as F
import torch
from tqdm import tqdm
import math

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'), weights_only=True)
            )
            self.scheduler.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'scheduler.pt'), weights_only=True)
            )

        self.error_last = 1e8

        # self.downgt1 = torch.nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False)
        # self.downgt2 = torch.nn.AvgPool2d(4, ceil_mode=True, count_include_pad=False)
        # self.downgt3 = torch.nn.AvgPool2d(8, ceil_mode=True, count_include_pad=False)

    def train(self):
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_last_lr()[0]

        print("lr", self.scheduler.get_last_lr()[0], self.optimizer.param_groups[0]['lr'])

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (hazy, gt, _, _) in enumerate(self.loader_train):
            hazy, gt = self.prepare([hazy, gt])

            timer_data.hold()
            timer_model.tic()

            # Zero the parameter gradients 
            self.optimizer.zero_grad()

            out, spa_outs, fre_outs = self.model(hazy, 0)
            # restore = self.model(hazy, 0)
            # out_1 = restore['stage1_out']
            # out_2 = restore['stage2_out']

           
            loss_t = 0
            for i in range(len(spa_outs)):
                t_gt = torch.nn.functional.interpolate(gt, spa_outs[i].shape[-2:], mode='bicubic', align_corners=True).detach()
                tmp = self.loss(spa_outs[i], t_gt) + self.loss(fre_outs[i], t_gt)
                loss_t += 0.01 * tmp

            # if isinstance(out, list):
            #
            #     # step1
            #     if self.args.stage == 'step1':
            #         s = torch.exp(-out[1])
            #         sr_ = torch.mul(out[0], s)
            #         hr_ = torch.mul(out, s)
            #         loss = self.loss(sr_, hr_) + 2 * torch.mean(out[1])
            #
            #     # step2
            #     elif self.args.stage == 'step2':
            #         b, c, h, w = out[1].shape
            #         s1 = out[1].view(b, c, -1)
            #         pmin = torch.min(s1, dim=-1)
            #         pmin = pmin[0].unsqueeze(dim=-1).unsqueeze(dim=-1)
            #         s = out[1]
            #         s = s - pmin + 1
            #         sr_ = torch.mul(out[0], s)
            #         hr_ = torch.mul(out, s)
            #         loss = self.loss(sr_, hr_)
            #
            #     else:
            #         loss = self.loss(out[0], gt)
            # else:
            loss = self.loss(out, gt)

            loss = loss + loss_t

            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print("loss:", loss.item(), "skip_threshold", self.args.skip_threshold, "error_last:", self.error_last)
                raise RuntimeError('Loss exceeds skip threshold, training aborted.')

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    # loss.item(),
                    # pix_loss.item(),
                    # fft_loss.item(),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()
            # for debugging
            # if batch == 10:
            #     break

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.scheduler.step()

    def test(self):
        epoch = self.scheduler.last_epoch

        self.ckp.write_log('\nBegin Evaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                self.ckp.write_log('#####################[dataset={}]---[model={}]#####################'.format(self.args.data_train, self.args.model))
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                psnr_list = []
                # ssim_list = []
                for batch, (hazy, gt, hazy_filename, _) in enumerate(tqdm_test):
                    hazy, gt = self.prepare([hazy, gt])

                    # restore = self.model(hazy, 0)
                    # out_1 = restore['stage1_out']
                    # out_2 = restore['stage2_out']
                    out, spa_outs, fre_outs = self.model(hazy, 0)

                    # --- Calculate the average PSNR --- 
                    # --- 需要保证restore和gt图像的尺寸大小一致， 这部分我们将在数据处理部分就提前处理好---
                    psnr_list.extend(utility.to_psnr(out, gt))
                    # --- Calculate the average PSNR ---
                    # --- 该过程比较耗时，所以在验证时候，并未计算---
                    # ssim_list.extend(utility.to_ssim_skimage(restore, gt))

                    # 不保存结果也会大大减少时间，所以默认也不保存时间
                    if self.args.save_results: 
                        self.ckp.save_image(out, hazy_filename)
            
            self.ckp.log[-1, idx_scale] = sum(psnr_list) / len(psnr_list) # 只保存 psnr 的值
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                    self.args.data_train,
                    scale,
                    self.ckp.log[-1, idx_scale],
                    best[0][idx_scale],
                    best[1][idx_scale] + 1
                )
            )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.args.epochs + 1

    def adjust(self, init, fin, step, fin_step):
        if fin_step == 0:
            return  fin
        deta = fin - init
        adj = min(init + deta * step / fin_step, fin)
        return adj
