import torch
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.batch_norm(self.conv1(x)))
        x = self.batch_norm(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class Attention(nn.Module):
    def __init__(self, in_features, hidden_size):
        super(Attention, self).__init__()
        self.W = nn.Linear(in_features, hidden_size)
        self.U = nn.Linear(hidden_size, 1)

    def forward(self, x):
        u = torch.tanh(self.W(x))
        a = F.softmax(self.U(u), dim=1)
        return a * x

class CaptchaModel(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaModel, self).__init__()
        self.conv_1 = nn.Conv2d(3, 128, kernel_size=(3, 6), padding=(1, 1))
        self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.residual_1 = ResidualBlock(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=(3, 6), padding=(1, 1))
        self.pool_2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.residual_2 = ResidualBlock(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.linear_1 = nn.Linear(1152, 64)
        self.drop_1 = nn.Dropout(0.2)
        self.attention = Attention(64, 32)
        self.output = nn.Linear(64, num_chars + 1)

    def forward(self, images, targets=None):
        bs, _, _, _ = images.size()
        x = F.relu(self.conv_1(images))
        x = self.pool_1(x)
        x = self.residual_1(x)
        x = F.relu(self.conv_2(x))
        x = self.pool_2(x)
        x = self.residual_2(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(bs, x.size(1), -1)
        x = F.relu(self.linear_1(x))
        x = self.drop_1(x)
        x = self.attention(x)
        x = self.output(x)
        x = x.permute(1, 0, 2)

        if targets is not None:
            log_probs = F.log_softmax(x, 2)
            input_lengths = torch.full(
                size=(bs,), fill_value=log_probs.size(0), dtype=torch.int32
            )
            target_lengths = torch.full(
                size=(bs,), fill_value=targets.size(1), dtype=torch.int32
            )
            loss = nn.CTCLoss(blank=0)(
                log_probs, targets, input_lengths, target_lengths
            )
            return x, loss

        return x, None


if __name__ == "__main__":
    cm = CaptchaModel(19)
    img = torch.rand((1, 3, 50, 200))
    x, _ = cm(img, torch.rand((1, 5)))
