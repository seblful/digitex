import math

import torch
import torch.nn as nn
import torchvision.models as models


class HeatmapKeypointModel(nn.Module):
    def __init__(
        self,
        max_keypoints: int,
        backbone_out_channels: int,
        backbone_stride: int,
        deconv_channels: int,
        output_stride: int,
    ) -> None:
        """
        Heatmap-based keypoint prediction model.

        Args:
            backbone: Feature extraction backbone (e.g., ResNet50 without classification head)
            max_keypoints: Maximum number of keypoints to predict (defines output channels)
            backbone_out_channels: Number of output channels from the backbone
            deconv_channels: Number of channels in deconvolution layers
            output_stride: Target output stride relative to input (must be power of 2)
        """
        super().__init__()
        self.max_keypoints = max_keypoints

        # Verify output_stride is power of 2
        if not (output_stride & (output_stride - 1) == 0) and output_stride != 0:
            raise ValueError("output_stride must be a power of 2")

        # Create backbone
        self.backbone = self.create_backbone(freeze_initial_layers=True)

        # Create deconvolution layers
        self.deconv_layers = self.create_deconv_layers(
            backbone_out_channels, backbone_stride, deconv_channels, output_stride
        )

        # Create final prediction layer
        self.final_conv = nn.Conv2d(deconv_channels, max_keypoints, kernel_size=1)

        # # TODO maybe add sigmoid
        # self.final_conv = nn.Sequential(
        #     nn.Conv2d(deconv_channels, max_keypoints, kernel_size=1),
        #     nn.Sigmoid(),  # Ensures heatmap values in [0,1]
        # )

        # Initialize weights
        self._initialize_weights()

    def create_deconv_layers(
        self,
        backbone_out_channels: int,
        backbone_stride: int,
        deconv_channels: int,
        output_stride: int,
    ) -> nn.Sequential:
        deconv_layers = nn.Sequential()
        in_channels = backbone_out_channels

        # Add deconvolution blocks
        num_deconv_layers = int(math.log2(backbone_stride // output_stride))
        for _ in range(num_deconv_layers):
            deconv_layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    deconv_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False,
                )
            )
            deconv_layers.append(nn.BatchNorm2d(deconv_channels))
            deconv_layers.append(nn.ReLU(inplace=True))
            in_channels = deconv_channels

        return deconv_layers

    def create_backbone(
        self,
        freeze_initial_layers: bool = True,
    ) -> nn.Module:
        # Load pretrained model
        resnet = getattr(models, "resnet50")(weights="IMAGENET1K_V2")

        # Remove avgpool and fc layers
        backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Freeze initial layers if specified
        if freeze_initial_layers:
            for param in backbone.parameters():
                param.requires_grad = False

        return backbone

    def _initialize_weights(self) -> None:
        # Initialize deconv layers
        for m in self.deconv_layers.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Initialize final layer
        nn.init.normal_(self.final_conv.weight, std=0.001)
        nn.init.constant_(self.final_conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Heatmaps tensor of shape (B, max_keypoints, H//output_stride, W//output_stride)
        """
        features = self.backbone(x)
        upsampled = self.deconv_layers(features)
        heatmaps = self.final_conv(upsampled)

        return heatmaps
