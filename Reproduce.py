import torch.nn as nn
from torch.nn import functional as F
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import imageio

def deconv(ch_in, ch_out, k_size, stride=2, pad=1, bn=True):
    layers = []
    layers.append(nn.ConvTranspose2d(ch_in, ch_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    return nn.Sequential(*layers)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

class Generator(nn.Module):
    def __init__(self, z_dim, image_size, conv_dim):
        super(Generator, self).__init__()
        self.fc = deconv(z_dim, conv_dim * 8, int(image_size / 16), 1, 0, bn=False)
        self.deconv1 = deconv(conv_dim * 8, conv_dim * 4, 4)
        self.deconv2 = deconv(conv_dim * 4, conv_dim * 2, 4)
        self.deconv3 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv4 = deconv(conv_dim, 3, 4, bn=False)

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.fc(z)
        out = F.leaky_relu(self.deconv1(out), 0.05)
        out = F.leaky_relu(self.deconv2(out), 0.05)
        out = F.leaky_relu(self.deconv3(out), 0.05)
        out = F.tanh(self.deconv4(out))
        return out


def Reproduce_hw3():
    image_size = 64
    g_conv_dim = 64
    z_dim = 512
    G_test = Generator(z_dim, image_size, g_conv_dim)
    G_test.load_state_dict(torch.load('generator_epoch_5127.pkl'))
    G_test = G_test.cuda()
    print("Loaded weights")
    fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(20, 10))
    for i in range(5):
        for j in range(10):
            random_z = torch.randn(32, z_dim)
            fixed_z = Variable(random_z).cuda()
            fake_images = G_test(fixed_z)
            axes[i][j].imshow(denorm(fake_images[0].cpu().permute(1, 2, 0).data).numpy())
            axes[i][j].axis('off')
    plt.tight_layout()
    plt.show()


Reproduce_hw3()