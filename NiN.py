import torch
from torch import nn

def get_nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=strides, padding=padding),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
        nn.ReLU(inplace=True),
    )

class NiN(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.model = nn.Sequential(
            get_nin_block(in_channels=in_channels, out_channels=96, kernel_size=1, strides=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            get_nin_block(in_channels=96, out_channels=256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            get_nin_block(in_channels=256, out_channels=384, kernel_size=3, strides=1, padding=1),
            nn.Dropout(p=0.5),
            get_nin_block(in_channels=384, out_channels=out_channels, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )
    
    def forward(self, x):
        return self.model(x)
    
    def _validate(self):
        X = torch.rand(size=(1, self.in_channels, 224, 224))
        for layer in self.model:
            X = layer(X)
            print(layer.__class__.__name__, f"output shape:\t {X.shape}")
    
if __name__ == '__main__':
    model = NiN(3, 10)
    model._validate()