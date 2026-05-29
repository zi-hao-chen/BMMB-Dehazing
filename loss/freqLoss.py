import torch
import torch.nn as nn
import torch.nn.functional as F


# def l1_loss(pred, target):
#     return F.l1_loss(pred, target, reduction='none')

class AmplitudeLoss(nn.Module):
    def __init__(self):
        super(AmplitudeLoss, self).__init__()

        self.l1 = nn.L1Loss()

    def forward(self, img, img1):
        fre = torch.fft.rfft2(img, norm='backward')
        amp = torch.abs(fre)
        fre1 = torch.fft.rfft2(img1, norm='backward')
        amp1 = torch.abs(fre1)
        return self.l1(amp, amp1)


class PhaseLoss(nn.Module):
    def __init__(self):
        super(PhaseLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, img, img1):
        fre = torch.fft.rfft2(img, norm='backward')
        pha = torch.angle(fre)
        fre1 = torch.fft.rfft2(img1, norm='backward')
        pha1 = torch.angle(fre1)
        return self.l1(pha, pha1)

class RealLoss(nn.Module):
    def __init__(self):
        super(RealLoss, self).__init__()

        self.l1 = nn.L1Loss()

    def forward(self, img, img1):
        fre = torch.fft.rfft2(img, norm='backward')
        amp = torch.real(fre)
        fre1 = torch.fft.rfft2(img1, norm='backward')
        amp1 = torch.real(fre1)
        return self.l1(amp, amp1)


class ImaginaryLoss(nn.Module):
    def __init__(self):
        super(ImaginaryLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, img, img1):
        fre = torch.fft.rfft2(img, norm='backward')
        pha = torch.imag(fre)
        fre1 = torch.fft.rfft2(img1, norm='backward')
        pha1 = torch.imag(fre1)
        return self.l1(pha, pha1)