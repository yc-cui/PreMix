import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F


class SequentialMultiInput(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, padding=padding,
                               kernel_size=kernel_size, out_channels=out_channels, bias=False)
        self.relu = nn.Sequential(nn.ReLU())
        self.conv2 = nn.Conv2d(in_channels=out_channels, padding=padding,
                               kernel_size=kernel_size, out_channels=out_channels, bias=False)
        if in_channels != out_channels:
            self.conv1x1 = nn.Conv2d(in_channels=in_channels, kernel_size=1, out_channels=out_channels)
        else:
            self.conv1x1 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        if self.conv1x1 is not None:
            x = self.conv1x1(x)

        return x + y


class Encoder(nn.Module):
    def __init__(self, bands, embed_dim, kernel_size):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        pad = (kernel_size - 1) // 2
        self.encoder = nn.Sequential(
            nn.Conv2d(bands, embed_dim, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            ResidualBlock(embed_dim, embed_dim, kernel_size, pad),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class EWFM(nn.Module):
    def __init__(self, embed_dim, kernel_size, beta=0.1, act="tanh+relu", h=256, w=256):
        super().__init__()
        self.kernel_size = kernel_size
        self.beta = beta
        self.act = act
        self.pad = (self.kernel_size - 1) // 2
        self.conv = ResidualBlock(embed_dim, embed_dim, kernel_size, self.pad)
        self.trainable_beta = False
        if self.beta is not None and self.beta > 1:
            print("enable trainable beta.")
            self.beta = nn.Parameter(torch.ones(1, ), requires_grad=True)
            nn.init.constant_(self.beta, 0.5)
            self.trainable_beta = True

        assert act in ["sigmoid", "tanh+relu", "tanh+elu", "softsign+elu", "softsign+relu"]

    def forward(self, ms_up, pan):
        fused = self.conv(ms_up + pan)
        # Soft Mask => Mixing
        if self.act == "tanh+relu":
            mask = F.tanh(F.relu6(fused))
        elif self.act == "sigmoid":
            mask = F.sigmoid(fused)
        elif self.act == "tanh+elu":
            mask = F.tanh(F.elu(fused) + 1)
        elif self.act == "softsign+elu":
            mask = F.softsign(F.elu(fused) + 1)
        elif self.act == "softsign+relu":
            mask = F.softsign(F.relu6(fused))
        # Hard Mask => Swapping
        if self.beta is not None and self.trainable_beta == False:
            mask = torch.where(mask > self.beta, 1., 0.)
        if self.beta is not None and self.trainable_beta == True:
            mask = torch.sigmoid((mask - self.beta) * 20.)

        mixed_ms = (1. - mask) * ms_up + mask * pan
        mixed_pan = (1. - mask) * pan + mask * ms_up
        return mixed_ms, mixed_pan


class EMixBlock(nn.Module):
    def __init__(self, in_ms, in_pan, embed_dim, kernel_size,
                 enable_EWFM=True, beta=0.1, act="tanh+relu"):
        super().__init__()
        self.encoder_ms = Encoder(in_ms, embed_dim, kernel_size)
        self.encoder_pan = Encoder(in_pan, embed_dim, kernel_size)
        self.ewfm = None
        if enable_EWFM:
            self.ewfm = EWFM(embed_dim, kernel_size, beta, act)

    def forward(self, ms_up, pan, last_ms_f=None, last_pan_f=None):
        e_ms = self.encoder_ms(ms_up)
        e_pan = self.encoder_pan(pan)

        if self.ewfm is not None:
            fused_ms, fused_pan = self.ewfm(e_ms, e_pan)
        else:
            fused_ms = e_ms
            fused_pan = e_pan
        return fused_ms, fused_pan


class PF(nn.Module):
    def __init__(self, bands, embed_dim, pf_kernel):
        super().__init__()
        self.bands = bands
        self.pf_kernel = pf_kernel
        self.pad = (self.pf_kernel - 1) // 2
        self.conv = nn.Conv2d(embed_dim, bands * pf_kernel ** 2, kernel_size=self.pf_kernel, padding=self.pad)

    def forward(self, x, o):
        B, _, H, W = x.shape
        C = self.bands
        K = self.pf_kernel
        P = self.pad
        weight = self.conv(x).view(B, C, K**2, H * W).permute(0, 1, 3, 2)
        o_pad = F.pad(o, [P for _ in range(4)], mode='reflect')
        unfolded_o = o_pad.unfold(2, K, 1).unfold(3, K, 1)
        unfolded_o = unfolded_o.contiguous().view(B, C, H * W, K * K)
        weighted_o = torch.sum(unfolded_o * weight, dim=-1).view(B, C, H, W)
        return weighted_o  # B bands H W


class PreMixModel(nn.Module):
    def __init__(self, num_trans, bands, embed_dim, kernel_size=3, pf_kernel=3,
                 enable_EWFM=True, num_layers=3, beta=0.1, act="tanh+relu"):
        super(PreMixModel, self).__init__()
        self.preconv = EMixBlock(bands, 1, embed_dim, kernel_size, enable_EWFM, beta, act)
        self.emix = SequentialMultiInput(*[EMixBlock(embed_dim, embed_dim, embed_dim, kernel_size, enable_EWFM, beta, act) for i in range(num_layers - 1)
                                           ])
        self.pf_ms = PF(bands, embed_dim, pf_kernel)
        self.pf_pan = PF(bands, embed_dim, pf_kernel)
        self.num_trans = num_trans
        if num_trans != 0:
            self.transconv_ms = nn.Sequential(*[nn.ConvTranspose2d(in_channels=embed_dim, out_channels=embed_dim,
                                                                   kernel_size=(3, 3), stride=2, padding=1, output_padding=1) for _ in range(num_trans)])
            self.transconv_pan = nn.Sequential(*[nn.ConvTranspose2d(in_channels=embed_dim, out_channels=embed_dim,
                                                                    kernel_size=(3, 3), stride=2, padding=1, output_padding=1) for _ in range(num_trans)])

    def forward(self, ms_up, pan, filtered_ms=None, filtered_pan=None, last_ms_f=None, last_pan_f=None):

        if filtered_ms is None:
            filtered_ms = ms_up
        if filtered_pan is None:
            filtered_pan = pan
        fused_ms, fused_pan = self.preconv(ms_up, pan)
        if last_ms_f is not None:
            last_ms_f = F.interpolate(last_ms_f, (fused_ms.shape[2], fused_ms.shape[3]))
            fused_ms = fused_ms + last_ms_f
        if last_pan_f is not None:
            last_pan_f = F.interpolate(last_pan_f, (fused_pan.shape[2], fused_pan.shape[3]))
            fused_pan = fused_pan + last_pan_f
        fused_ms, fused_pan = self.emix(fused_ms, fused_pan)
        if self.num_trans != 0:
            fused_ms = self.transconv_ms(fused_ms)
            fused_pan = self.transconv_pan(fused_pan)
        spec_phase = self.pf_ms(fused_ms, filtered_ms)
        spatial_phase = self.pf_pan(fused_pan, filtered_pan.repeat(1, filtered_ms.shape[1], 1, 1))
        pred = spec_phase + spatial_phase
        return pred, fused_ms, fused_pan


def _high_pass_filter(img, ksize=(3, 3), sigma=(1.5, 1.5)):
    blur = kornia.filters.gaussian_blur2d(img, ksize, sigma)
    high_pass_filtered = img - blur
    return high_pass_filtered


def _equalize_clahe(img):
    return kornia.enhance.equalize_clahe(torch.clip(img, 0, 1))


class PreMixHugeModel(nn.Module):
    def __init__(self, bands, embed_dim, kernel_size=3, pf_kernel=3, enable_EWFM=True, num_layers=3, beta=0.1, act="tanh+relu"):
        super(PreMixHugeModel, self).__init__()
        self.model1 = nn.ModuleList([PreMixModel(i, bands, embed_dim, kernel_size, pf_kernel,
                                                 enable_EWFM, num_layers, beta, act,) for i in range(3)])
        self.model2 = nn.ModuleList([PreMixModel(i, bands, embed_dim, kernel_size, pf_kernel,
                                                 enable_EWFM, num_layers, beta, act,) for i in range(3)])
        self.model3 = nn.ModuleList([PreMixModel(i, bands, embed_dim, kernel_size, pf_kernel,
                                                 enable_EWFM, num_layers, beta, act,) for i in range(3)])

        self.postconv = nn.Sequential(ResidualBlock(bands * 3, bands, kernel_size=3, padding=1))

    def forward(self, ms_up, pan):  # all
        hp_ms_up = _high_pass_filter(ms_up)
        hp_pan = _high_pass_filter(pan)
        eq_ms_up = _equalize_clahe(ms_up)
        eq_pan = _equalize_clahe(pan)

        f_hp_ms, last_ms_f, last_pan_f = self._forward_model(self.model3, hp_ms_up, hp_pan, ms_up, pan)
        f_grad_ms, last_ms_f, last_pan_f = self._forward_model(
            self.model2, eq_ms_up, eq_pan, f_hp_ms, pan, last_ms_f, last_pan_f)
        f_ms, last_ms_f, last_pan_f = self._forward_model(self.model1, ms_up, pan, f_grad_ms, pan, last_ms_f, last_pan_f)
        pred = f_ms + ms_up

        return pred

    def _forward_model(self, model, ms, pan, filtered_ms, filtered_pan, last_ms_f=None, last_pan_f=None):  # 4 2
        ms2x = F.interpolate(ms, scale_factor=0.5)
        ms4x = F.interpolate(ms, scale_factor=0.25)
        pan2x = F.interpolate(pan, scale_factor=0.5)
        pan4x = F.interpolate(pan, scale_factor=0.25)

        f4x, last_ms_f, last_pan_f = model[2](ms4x, pan4x, filtered_ms, filtered_pan, last_ms_f, last_pan_f)
        f2x, last_ms_f, last_pan_f = model[1](ms2x, pan2x, f4x, filtered_pan, last_ms_f, last_pan_f)
        f, last_ms_f, last_pan_f = model[0](ms, pan, f2x, filtered_pan, last_ms_f, last_pan_f)
        return f, last_ms_f, last_pan_f
