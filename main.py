import sys
from copy import deepcopy
import random
import numpy as np
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
from Discremnator import Discriminator
from Generator import Generator
from utils import denorm
import torchvision
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt


lr = 0.0002
max_epoch = 30
batch_size = 32
image_size = 64
g_conv_dim = 64
d_conv_dim = 64
log_step = 100
sample_step = 500
sample_num = 32
IMAGE_PATH = 'img_align_celeba'
SAMPLE_PATH = '../'


def main():
    seed = 43
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Training on GPU.")
    else:
        print("No GPU available, training on CPU.")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    dataset = ImageFolder(IMAGE_PATH, transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    z_dim = 512
    D = Discriminator(image_size)
    G = Generator(z_dim, image_size, g_conv_dim)
    D = D.to(device)
    G = G.to(device)
    criterion = nn.BCELoss().cuda()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    d_loss_list = []
    g_loss_list = []

    user_input = input("Type 'train' for training or 'load' to load weights")
    if user_input == "train":
        total_batch = len(data_loader.dataset) // batch_size
        fixed_z = Variable(torch.randn(sample_num, z_dim)).cuda()
        for epoch in range(max_epoch):
            print(epoch)
            g_loss_sum = 0
            d_loss_sum = 0
            size_of_data = 0
            for i, (images, labels) in enumerate(data_loader):
                image = Variable(images).to(device)
                size_of_data += image.shape[0]
                real_labels = Variable(torch.ones(batch_size)).to(device)
                fake_labels = Variable(torch.zeros(batch_size)).to(device)
                outputs = D(image)
                d_loss_real = criterion(outputs, real_labels)  # BCE
                d_loss_sum += d_loss_real.item()
                real_score = outputs
                z = Variable(torch.randn(batch_size, z_dim)).to(device)
                fake_images = G(z)
                outputs = D(fake_images)
                d_loss_fake = criterion(outputs, fake_labels)  # BCE
                fake_score = outputs
                d_loss = d_loss_real + d_loss_fake
                D.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                z = Variable(torch.randn(batch_size, z_dim)).to(device)
                fake_images = G(z)
                outputs = D(fake_images)
                g_loss = criterion(outputs, real_labels)
                g_loss_sum += g_loss.item()
                D.zero_grad()
                G.zero_grad()
                g_loss.backward()
                g_optimizer.step()

            gnerator_epoch = 'generator_epoch_zsize_512_ep' + str(epoch) + '.pkl'
            discriminator_epoch = 'discriminator_epoch_zsize_512_ep' + str(epoch) + '.pkl'
            torch.save(G.state_dict(), gnerator_epoch)
            torch.save(D.state_dict(), discriminator_epoch)
            d_loss_list.append(d_loss_sum / size_of_data)
            g_loss_list.append(g_loss_sum / size_of_data)


    with open('d_loss.txt', 'w') as f:
        for item in d_loss_list:
            f.write("%s\n" % item)
    with open('g_loss.txt', 'w') as f:
        for item in g_loss_list:
            f.write("%s\n" % item)


    if user_input == 'load':
        z_dim = 512
        G_test = Generator(z_dim, image_size, g_conv_dim)
        G_test.load_state_dict(torch.load('generator_epoch_zsize_512_ep8.pkl'))
        G_test = G_test.cuda()
        print("Loaded weights")
        seed = 43
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random_z = torch.randn(32, z_dim)
        fixed_z = Variable(random_z).cuda()
        fake_images = G_test(fixed_z)
        plt.imshow(denorm(fake_images[0].cpu().permute(1, 2, 0).data).numpy())
        plt.show()



        ## visualize latent space between 2 vectors
        interpolation = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        vec1 = torch.load('pic1.pt', map_location=torch.device('cpu'))
        vec2 = torch.load('pic2.pt', map_location=torch.device('cpu'))
        fig, axs = plt.subplots(1, 11, figsize=(22, 2))
        for i, a in enumerate(interpolation):
            inter_vec = a * vec1 + (1 - a) * vec2
            fixed_z = Variable(inter_vec).cuda()
            fake_images = G_test(fixed_z)
            axs[i].imshow(denorm(fake_images[0].cpu().permute(1, 2, 0).data).numpy())
            axs[i].axis('off')  # Hide axis#
        plt.tight_layout()
        plt.savefig('vector_space_between_two')
        plt.show()
        sys.exit()

        ##### visualize latent space with change in one dimantion
    #    perm_vec = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    #    perm_vec = [a * 4 for a in perm_vec]
    #    # Save images using imageio
    #    random_integers = [random.randint(0, 511) for _ in range(30)]
    #    for j in random_integers:
    #        fig, axs = plt.subplots(1, 10, figsize=(20, 2))
    #        # Generate and display each image
    #        for i, x in enumerate(perm_vec):
    #            copy_z = deepcopy(random_z)
    #            copy_z[:, j] = x
    #            print(random_z)
    #            fixed_z = Variable(copy_z).cuda()
    #            fake_images = G_test(fixed_z)
    #            # Display the image
    #            axs[i].imshow(denorm(fake_images[0].cpu().permute(1, 2, 0).data).numpy())
    #            axs[i].axis('off')  # Hide axis#
    #        plt.tight_layout()
    #        plt.savefig('vector_space_one_direction' + str(j))
    #        plt.show()
    #    sys.exit()

        #### visualize latent space with random walk
        fig, axs = plt.subplots(1, 10, figsize=(20, 2))
        for i in range(10):
            if i != 0:
                random_vector = np.random.choice([-1, 0, 1], size=512) * 2
                random_z += torch.from_numpy(random_vector)
            fixed_z = Variable(random_z).cuda()
            fake_images = G_test(fixed_z)
            axs[i].imshow(denorm(fake_images[0].cpu().permute(1, 2, 0).data).numpy())
            axs[i].axis('off')  # Hide axis#
        plt.tight_layout()
        plt.savefig('vector_space_random_walk_02')
        plt.show()


if __name__ == "__main__":
    main()




