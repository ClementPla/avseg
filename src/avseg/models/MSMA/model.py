import torch.nn as nn
import torch
import torch.nn.functional as F
from avseg.models.MSMA.blocks import SpatialAttentionBlock, ResidualPath, MSSE


class MSMA(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.encoder = nn.ModuleList(
            [
                MSSE(in_channels=in_channels, out_channels=32),
                MSSE(in_channels=32, out_channels=64),
                MSSE(in_channels=64, out_channels=128),
                MSSE(in_channels=128, out_channels=256),
                MSSE(in_channels=256, out_channels=512),
            ]
        )

        self.decoder = nn.ModuleList(
            [
                MSSE(in_channels=256 + 256, out_channels=256),
                MSSE(in_channels=128 + 128, out_channels=128),
                MSSE(in_channels=64 + 64, out_channels=64),
                MSSE(in_channels=32 + 32, out_channels=32),
            ]
        )

        self.respath = nn.ModuleList(
            [
                ResidualPath(32, 32, respath_length=4),
                ResidualPath(64, 64, respath_length=3),
                ResidualPath(128, 128, respath_length=2),
                ResidualPath(256, 256, respath_length=1),
            ]
        )

        self.upsampler = nn.ModuleList(
            [
                nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            ]
        )

        self.spatial_attention_blocks = nn.ModuleList(
            [
                SpatialAttentionBlock(
                    in_channels=256, gating_channels=256, reduction=8
                ),
                SpatialAttentionBlock(
                    in_channels=128, gating_channels=128, reduction=8
                ),
                SpatialAttentionBlock(in_channels=64, gating_channels=64, reduction=8),
                SpatialAttentionBlock(in_channels=32, gating_channels=32, reduction=8),
            ]
        )

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.encoder[0](x)
        x1 = self.encoder[1](F.max_pool2d(x0, kernel_size=2, stride=2))
        x2 = self.encoder[2](F.max_pool2d(x1, kernel_size=2, stride=2))
        x3 = self.encoder[3](F.max_pool2d(x2, kernel_size=2, stride=2))
        x4 = self.encoder[4](F.max_pool2d(x3, kernel_size=2, stride=2))

        x0mid = self.respath[0](x0)
        x1mid = self.respath[1](x1)
        x2mid = self.respath[2](x2)
        x3mid = self.respath[3](x3)

        y = self.upsampler[0](x4)
        x3gat = self.spatial_attention_blocks[0](y, x3mid)
        x3cat = torch.cat((x3gat, y), dim=1)
        x3dec = self.decoder[0](x3cat)

        y = self.upsampler[1](x3dec)
        x2gat = self.spatial_attention_blocks[1](y, x2mid)
        x2cat = torch.cat((x2gat, y), dim=1)
        x2dec = self.decoder[1](x2cat)

        y = self.upsampler[2](x2dec)
        x1gat = self.spatial_attention_blocks[2](y, x1mid)
        x1cat = torch.cat((x1gat, y), dim=1)
        x1dec = self.decoder[2](x1cat)

        y = self.upsampler[3](x1dec)
        x0gat = self.spatial_attention_blocks[3](y, x0mid)
        x0cat = torch.cat((x0gat, y), dim=1)
        x0dec = self.decoder[3](x0cat)

        return self.final_conv(x0dec)


if __name__ == "__main__":
    model = MSMA(in_channels=3, num_classes=8).cuda()
    x = torch.randn(1, 3, 512, 512).cuda()

    output = model(x)
    print(output.shape)
