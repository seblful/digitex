import torch
import torch.nn as nn


class SingleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pad: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=pad,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.single_conv(x)


class SimpleUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_cls: int,
        ks: int,
        dilation: int,
        stage_channels: list[int],
        num_blocks: list[int],
        short_rate: float,
        adw: bool,
    ) -> None:
        """
        Args:

            in_channels: the number of the input channels
            num_cls: the expected number of segmentation classes
            ks: kernel size
            dilation: the dilation rate of convolution
            stage_channels: the number of encoding stages
            num_blocks: number of blocks with each encoding and decoding stage
            short_rate: the shortcut rate of the shortcut features
            adw: whether to apply learnable attention weights to the shortcut features and the deep features for feature fusion
        """

        super(SimpleUNet, self).__init__()
        assert short_rate > 0, "short_rate must be greater than 0!"
        assert len(stage_channels) == len(num_blocks), (
            "The length of stage_channels and num_blocks must match!"
        )
        self.in_channels = in_channels
        self.num_cls = num_cls
        self.kernel_size = ks
        self.dilation = dilation
        self.stage_channels = stage_channels
        self.depth = len(stage_channels)
        self.num_blocks = num_blocks
        self.short_rate = short_rate
        self.adw = adw
        self.pad = self.dilation * (self.kernel_size - 1) // 2
        self.seg_head = SingleConv(
            int(short_rate * stage_channels[0]), self.num_cls, 1, 0, 1
        )

        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.en_layer0 = SingleConv(
            self.in_channels,
            stage_channels[0],
            kernel_size=self.kernel_size,
            pad=self.pad,
            dilation=self.dilation,
        )

        for i in range(1, self.depth):
            # layers = nn.ModuleList()
            layers = []
            for _ in range(num_blocks[i] - 1):
                layers.append(
                    SingleConv(
                        stage_channels[i - 1],
                        stage_channels[i - 1],
                        kernel_size=self.kernel_size,
                        pad=self.pad,
                        dilation=self.dilation,
                    )
                )
            layers.append(
                SingleConv(
                    stage_channels[i - 1],
                    stage_channels[i],
                    kernel_size=self.kernel_size,
                    pad=self.pad,
                    dilation=self.dilation,
                )
            )
            setattr(self, f"en_layer{i}", nn.Sequential(*layers))

        # layers = nn.ModuleList()
        for i in range(0, self.depth - 1):
            layers = []
            layers.append(
                SingleConv(
                    stage_channels[i],
                    int(self.short_rate * stage_channels[i]),
                    kernel_size=1,
                    pad=0,
                    dilation=1,
                )
            )
            setattr(self, f"short_layer{i}", nn.Sequential(*layers))

        re_stage_channels = stage_channels[::-1]
        re_num_blocks = num_blocks[::-1]

        # layers = nn.ModuleList()
        layers = []
        layers.append(
            SingleConv(
                re_stage_channels[0],
                int(self.short_rate * re_stage_channels[0]),
                kernel_size=self.kernel_size,
                pad=self.pad,
                dilation=self.dilation,
            )
        )

        for _ in range(0, re_num_blocks[0] - 1):
            layers.append(
                SingleConv(
                    int(self.short_rate * re_stage_channels[0]),
                    int(self.short_rate * re_stage_channels[0]),
                    kernel_size=self.kernel_size,
                    pad=self.pad,
                    dilation=self.dilation,
                )
            )

        self.bottleneck = nn.Sequential(*layers)

        # tmp_width = re_stage_channels[0]
        for i in range(1, self.depth):
            # layers = nn.ModuleList()
            layers = []
            layers.append(
                SingleConv(
                    int(
                        self.short_rate
                        * (re_stage_channels[i - 1] + re_stage_channels[i])
                    ),
                    int(self.short_rate * re_stage_channels[i]),
                    kernel_size=self.kernel_size,
                    pad=self.pad,
                    dilation=self.dilation,
                )
            )

            for _ in range(0, re_num_blocks[i] - 1):
                layers.append(
                    SingleConv(
                        int(self.short_rate * re_stage_channels[i]),
                        int(self.short_rate * re_stage_channels[i]),
                        kernel_size=self.kernel_size,
                        pad=self.pad,
                        dilation=self.dilation,
                    )
                )
            setattr(self, f"de_layer{i - 1}", nn.Sequential(*layers))
            # layers.append(SingleConv(int(self.short_rate * re_stage_channels[i]),
            #                          int(self.short_rate * re_stage_channels[i+1]),
            #                          kernel_size=self.kernel_size, pad=self.pad,
            #                          dilation=self.dilation, match=False))
        if self.adw:
            for i in range(1, self.depth):
                setattr(
                    self,
                    f"alpha{i - 1}",
                    nn.Parameter(
                        torch.ones(int(self.short_rate * re_stage_channels[i]), 1, 1),
                        requires_grad=True,
                    ),
                )
                setattr(
                    self,
                    f"beta{i - 1}",
                    nn.Parameter(
                        torch.ones(
                            int(self.short_rate * re_stage_channels[i - 1]), 1, 1
                        ),
                        requires_grad=True,
                    ),
                )
        else:
            for i in range(1, self.depth):
                setattr(
                    self,
                    f"alpha{i - 1}",
                    nn.Parameter(
                        torch.ones(int(self.short_rate * re_stage_channels[i]), 1, 1),
                        requires_grad=False,
                    ),
                )
                setattr(
                    self,
                    f"beta{i - 1}",
                    nn.Parameter(
                        torch.ones(
                            int(self.short_rate * re_stage_channels[i - 1]), 1, 1
                        ),
                        requires_grad=False,
                    ),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = []
        x = self.en_layer0(x)
        shortcut.append(x)

        for i in range(1, self.depth):
            x = self.down(x)
            x = getattr(self, f"en_layer{i}")(x)
            if i != self.depth - 1:
                shortcut.append(x)

        refined_shortcut = []
        for i in range(0, self.depth - 1):
            refined_shortcut.append(getattr(self, f"short_layer{i}")(shortcut[i]))

        re_shortcut = refined_shortcut[::-1]
        x = self.bottleneck(x)

        for j in range(0, self.depth - 1):
            y = torch.concat(
                [
                    re_shortcut[j] * getattr(self, f"alpha{j}"),
                    self.up(x * getattr(self, f"beta{j}")),
                ],
                dim=1,
            )
            x = getattr(self, f"de_layer{j}")(y)

        output = self.seg_head(x)
        return output
