import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)

class Generator(nn.Module):
    def __init__(self, conv_dim=64,num_blocks=6):
        super().__init__()

        # No makeup picture
        self.encoder_no_makeup = nn.Sequential(
            nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(conv_dim * 2, affine=True),
            nn.ReLU(inplace=True)
        )

        self.encoder_makeup = nn.Sequential(
            nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(conv_dim * 2, affine=True),
            nn.ReLU(inplace=True)
        )

        # In the middle, concat the 2 beanches and feed to several res block (5.1)
        self.fusion = nn.Sequential(
            nn.Conv2d(conv_dim * 4, conv_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(conv_dim * 4, affine=True),
            nn.ReLU(inplace=True),
            *[ResidualBlock(conv_dim * 4) for _ in range(num_blocks)]
        )

        self.decoder_makeup = nn.Sequential(
            nn.ConvTranspose2d(conv_dim * 4, conv_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(conv_dim * 2, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(conv_dim * 2, conv_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_dim, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        )
        self.decoder_no_makeup = nn.Sequential(
            nn.ConvTranspose2d(conv_dim * 4, conv_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(conv_dim * 2, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(conv_dim * 2, conv_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_dim, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        )
    def forward(self, no_makeup, makeup):
        feat_no_makeup = self.encoder_no_makeup(no_makeup)
        feat_makeup = self.encoder_makeup(makeup)
        combined_features = torch.cat([feat_no_makeup, feat_makeup], dim=1)
        fused = self.fusion(combined_features)
        out_makeup = self.decoder_makeup(fused)    
        out_no_makeup = self.decoder_no_makeup(fused)
        return {'makeup_output': out_makeup, 'No_makeup_output':out_no_makeup}
