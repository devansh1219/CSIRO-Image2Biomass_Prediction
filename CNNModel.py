import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module)

    def __init__(self, in_channels, out_channels, hidden_channels=None):
        super().__init__()
        hidden_channels = hidden_channels if hidden_channels is not None else out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):

    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()

        self.gate_transform = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )

        self.skip_transform = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )

        self.attention_map = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, gate_signal, skip_features):
        gate_proj = self.gate_transform(gate_signal)
        skip_proj = self.skip_transform(skip_features)

        fused = self.activation(gate_proj + skip_proj)
        weights = self.attention_map(fused)

        attended_features = skip_features * weights
        return attended_features, weights


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.layer(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, use_bilinear=True):
        super().__init__()

        if use_bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.upsample = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

        self.att_gate = AttentionGate(
            gate_channels=in_channels // 2,
            skip_channels=in_channels // 2,
            inter_channels=in_channels // 4
        )

    def forward(self, decoder_input, encoder_skip):
        decoder_input = self.upsample(decoder_input)

        height_diff = encoder_skip.size(2) - decoder_input.size(2)
        width_diff = encoder_skip.size(3) - decoder_input.size(3)

        decoder_input = F.pad(
            decoder_input,
            [width_diff // 2, width_diff - width_diff // 2,
             height_diff // 2, height_diff - height_diff // 2]
        )

        refined_skip, att_map = self.att_gate(decoder_input, encoder_skip)
        combined = torch.cat([refined_skip, decoder_input], dim=1)

        return self.conv(combined), att_map


class AttentionUNet(nn.Module):

    def __init__(self, in_channels=3, output_dim=5, base_filters=64, bilinear=True):
        super().__init__()

        self.encoder_start = DoubleConv(in_channels, base_filters)

        self.enc1 = Down(base_filters, base_filters * 2)
        self.enc2 = Down(base_filters * 2, base_filters * 4)
        self.enc3 = Down(base_filters * 4, base_filters * 8)

        factor = 2 if bilinear else 1
        self.enc4 = Down(base_filters * 8, base_filters * 16 // factor)

        self.dec1 = Up(base_filters * 16, base_filters * 8 // factor, bilinear)
        self.dec2 = Up(base_filters * 8, base_filters * 4 // factor, bilinear)
        self.dec3 = Up(base_filters * 4, base_filters * 2 // factor, bilinear)
        self.dec4 = Up(base_filters * 2, base_filters, bilinear)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Linear(base_filters, output_dim)

        self.saved_attention = []

    def forward(self, x):
        self.saved_attention.clear()

        s1 = self.encoder_start(x)
        s2 = self.enc1(s1)
        s3 = self.enc2(s2)
        s4 = self.enc3(s3)
        bottleneck = self.enc4(s4)

        x, a1 = self.dec1(bottleneck, s4)
        x, a2 = self.dec2(x, s3)
        x, a3 = self.dec3(x, s2)
        x, a4 = self.dec4(x, s1)

        self.saved_attention = [a1, a2, a3, a4]

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        return self.regressor(x)

    def get_attention_maps(self):
        return self.saved_attention
