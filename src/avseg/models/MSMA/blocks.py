import torch
import torch.nn as nn


class Conv2DBatchNorm(torch.nn.Module):
    """
    2D Convolutional layers

    Arguments:
            num_in_filters {int} -- number of input filters
            num_out_filters {int} -- number of output filters
            kernel_size {tuple} -- size of the convolving kernel
            stride {tuple} -- stride of the convolution (default: {(1, 1)})
            activation {str} -- activation function (default: {'relu'})

    """

    def __init__(
        self,
        num_in_filters,
        num_out_filters,
        kernel_size,
        stride=(1, 1),
        activation="relu",
        bias=True,
    ):
        super().__init__()
        self.activation = activation
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
            bias=bias,
        )
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)

        if self.activation == "relu":
            return torch.nn.functional.relu(x)
        elif self.activation == "sigmoid":
            return torch.nn.functional.sigmoid(x)
        else:
            return x


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        gap = x.mean((2, 3))
        attention = self.fc(gap)
        attention = attention.unsqueeze(2).unsqueeze(3)

        return x * attention


class MSSE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSSE, self).__init__()
        self.conv1 = Conv2DBatchNorm(
            in_channels,
            out_channels // 6,
            kernel_size=3,
        )
        self.conv2 = Conv2DBatchNorm(
            out_channels // 6,
            out_channels // 3,
            kernel_size=3,
        )
        last_chans = out_channels - (out_channels // 6 + out_channels // 3)
        self.conv3 = Conv2DBatchNorm(
            out_channels // 3,
            last_chans,
            kernel_size=3,
        )

        self.shortcut = Conv2DBatchNorm(
            in_channels, out_channels, kernel_size=1, stride=1, activation=None
        )

        self.final_layer = nn.Sequential(
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.SE = SEBlock(out_channels, reduction=8)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        shortcut = self.shortcut(x)

        cat = torch.cat((x1, x2, x3), dim=1)
        se = self.SE(cat) + cat + shortcut
        return self.final_layer(se)


class ResidualPath(torch.nn.Module):
    """
    ResPath

    Arguments:
            num_in_filters {int} -- Number of filters going in the respath
            num_out_filters {int} -- Number of filters going out the respath
            respath_length {int} -- length of ResPath

    """

    def __init__(self, num_in_filters, num_out_filters, respath_length):
        super().__init__()

        self.respath_length = respath_length
        self.shortcuts = torch.nn.ModuleList([])
        self.convs = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])

        for i in range(self.respath_length):
            if i == 0:
                self.shortcuts.append(
                    Conv2DBatchNorm(
                        num_in_filters,
                        num_out_filters,
                        kernel_size=(1, 1),
                        activation="None",
                    )
                )
                self.convs.append(
                    Conv2DBatchNorm(
                        num_in_filters,
                        num_out_filters,
                        kernel_size=(3, 3),
                        activation="relu",
                    )
                )

            else:
                self.shortcuts.append(
                    Conv2DBatchNorm(
                        num_out_filters,
                        num_out_filters,
                        kernel_size=(1, 1),
                        activation="None",
                    )
                )
                self.convs.append(
                    Conv2DBatchNorm(
                        num_out_filters,
                        num_out_filters,
                        kernel_size=(3, 3),
                        activation="relu",
                    )
                )

            self.bns.append(torch.nn.BatchNorm2d(num_out_filters))

    def forward(self, x):
        for i in range(self.respath_length):
            shortcut = self.shortcuts[i](x)

            x = self.convs[i](x)
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)

            x = x + shortcut
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)

        return x


class SpatialAttentionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        gating_channels,
        reduction,
    ):
        super(SpatialAttentionBlock, self).__init__()
        mid_channels = in_channels // reduction

        self.conv1 = Conv2DBatchNorm(
            in_channels, mid_channels, kernel_size=1, activation=None, bias=False
        )
        self.conv2 = Conv2DBatchNorm(
            gating_channels, mid_channels, kernel_size=1, activation=None, bias=True
        )

        self.squeeze_conv = Conv2DBatchNorm(
            mid_channels, 1, kernel_size=1, activation="sigmoid"
        )

    def forward(self, x, gating):
        """
        Forward pass of the spatial attention block.
        :param x: Input tensor.
        :param gating: Gating tensor.
        :return: Output tensor after applying spatial attention.
        """
        x0 = self.conv1(x)
        gating = self.conv2(gating)

        combined = x0 + gating
        combined = torch.nn.functional.relu(combined)
        attention = self.squeeze_conv(combined)
        return x * attention.expand_as(x)
