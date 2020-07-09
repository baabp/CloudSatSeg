import torch
import torch.nn.functional as F


class SegNet(torch.nn.Module):
    def __init__(self, out_channels, in_channels=3):
        super(SegNet, self).__init__()

        # VGG block
        class VGG_block(torch.nn.Module):
            def __init__(self, dim1, dim2, layer_N):
                super(VGG_block, self).__init__()

                _module = []

                for i in range(layer_N):
                    dim = dim1 if i == 0 else dim2
                    _module.append(torch.nn.Conv2d(dim, dim2, kernel_size=3, padding=1, stride=1))
                    _module.append(torch.nn.BatchNorm2d(dim2))
                    _module.append(torch.nn.ReLU())

                self.module = torch.nn.Sequential(*_module)

            def forward(self, x):
                x = self.module(x)
                return x

        # VGG Decoder block
        class VGG_block_decoder(torch.nn.Module):
            def __init__(self, dim1, dim2, layer_N):
                super(VGG_block_decoder, self).__init__()

                _module = []

                for i in range(layer_N):
                    dim = dim1 if i < (layer_N - 1) else dim2
                    _module.append(torch.nn.Conv2d(dim1, dim, kernel_size=3, padding=1, stride=1))
                    _module.append(torch.nn.BatchNorm2d(dim2))
                    _module.append(torch.nn.ReLU())

                self.module = torch.nn.Sequential(*_module)

            def forward(self, x):
                x = self.module(x)
                return x

        self.enc1 = VGG_block(in_channels, 64, 2)
        self.enc2 = VGG_block(64, 128, 2)
        self.enc3 = VGG_block(128, 256, 3)
        self.enc4 = VGG_block(256, 512, 3)
        self.enc5 = VGG_block(512, 512, 3)

        self.dec5 = VGG_block(512, 512, 3)
        self.dec4 = VGG_block(512, 256, 3)
        self.dec3 = VGG_block(256, 128, 3)
        self.dec2 = VGG_block(128, 64, 2)
        self.dec1 = VGG_block(64, 64, 2)

        self.out = torch.nn.Conv2d(64, out_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        # Encoder block 1
        x_enc1 = self.enc1(x)
        x, pool1_ind = F.max_pool2d(x_enc1, 2, stride=2, padding=0, return_indices=True)

        # Encoder block 2
        x_enc2 = self.enc2(x)
        x, pool2_ind = F.max_pool2d(x_enc2, 2, stride=2, padding=0, return_indices=True)

        # Encoder block 3
        x_enc3 = self.enc3(x)
        x, pool3_ind = F.max_pool2d(x_enc3, 2, stride=2, padding=0, return_indices=True)

        # Encoder block 4
        x_enc4 = self.enc4(x)
        x, pool4_ind = F.max_pool2d(x_enc4, 2, stride=2, padding=0, return_indices=True)

        # Encoder block 5
        x_enc5 = self.enc5(x)
        x, pool5_ind = F.max_pool2d(x_enc5, 2, stride=2, padding=0, return_indices=True)

        # Decoder block 5
        x = F.max_unpool2d(x, pool5_ind, kernel_size=2, stride=2, padding=0)
        x = self.dec5(x)

        # Decoder block 4
        x = F.max_unpool2d(x, pool4_ind, kernel_size=2, stride=2, padding=0)
        x = self.dec4(x)

        # Decoder block 3
        x = F.max_unpool2d(x, pool3_ind, kernel_size=2, stride=2, padding=0)
        x = self.dec3(x)

        # Decoder block 2
        x = F.max_unpool2d(x, pool2_ind, kernel_size=2, stride=2, padding=0)
        x = self.dec2(x)

        # Decoder block 1
        x = F.max_unpool2d(x, pool1_ind, kernel_size=2, stride=2, padding=0)
        x = self.dec1(x)

        # output
        x = self.out(x)
        x = F.softmax(x, dim=1)

        return x
