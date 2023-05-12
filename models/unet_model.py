# Copyright (c) 2023,Semin Kim, AI R&D Center, lululab

from .unet_parts import *

class UNet_texture_front_ds(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_texture_front_ds, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear=bilinear)

        self.outc_4 = OutConv(256, n_classes, kernel_size=1, padding=0)
        self.outc_3 = OutConv(128, n_classes, kernel_size=1, padding=0)
        self.outc_2 = OutConv(64, n_classes, kernel_size=1, padding=0)
        self.outc_1 = OutConv(64, n_classes, kernel_size=1, padding=0)

    def forward(self, x, t):
        x = torch.cat([x, t], dim=1)
        _, _, img_shape, _ = x.size()

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        dx5 = self.down4(x4)

        dx4 = self.up1(dx5, x4)
        dx3 = self.up2(dx4, x3)
        dx2 = self.up3(dx3, x2)
        dx1 = self.up4(dx2, x1)

        dx4 = F.upsample(dx4, size=(img_shape, img_shape), mode='bilinear')
        dx3 = F.upsample(dx3, size=(img_shape, img_shape), mode='bilinear')
        dx2 = F.upsample(dx2, size=(img_shape, img_shape), mode='bilinear')

        out_4 = self.outc_4(dx4)
        out_3 = self.outc_3(dx3)
        out_2 = self.outc_2(dx2)
        out_1 = self.outc_1(dx1)

        return out_1, out_2, out_3, out_4

if __name__ == '__main__':

    model = UNet_texture_front_ds(4, 11)

    model.train()

    shape_1 = (1, 3, 320, 320)
    shape_2 = (1, 1, 320, 320)

    rand_tensor_1 = torch.rand(shape_1)
    rand_tensor_2 = torch.rand(shape_2)

    output = model(rand_tensor_1, rand_tensor_2)
    print('ok')
