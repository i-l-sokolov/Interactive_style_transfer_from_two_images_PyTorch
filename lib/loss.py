import torch.nn.functional as F
from torch import nn
import torch
import os
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from glob import glob


def gram_matrix(input):
    batch_size, h, w, f_map_num = input.size()  # batch size(=1)
    # b=number of feature maps
    # (h,w)=dimensions of a feature map (N=h*w)

    features = input.view(batch_size * h, w * f_map_num)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(batch_size * h * w * f_map_num)


class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLossOne(nn.Module):
    def __init__(self, target_feature1, target_feature2):
        super(StyleLossOne, self).__init__()
        self.mask1 = torch.ones_like(target_feature1)
        self.mask1[:, :, :, :self.mask1.shape[3] // 2] = 0
        self.target1 = gram_matrix(target_feature1 * self.mask1).detach()
        self.mask2 = torch.ones_like(target_feature2)
        self.mask2[:, :, :, self.mask1.shape[3] // 2:] = 0
        self.target2 = gram_matrix(target_feature2 * self.mask2).detach()

    def forward(self, input):
        input2_1 = torch.ones_like(input)
        input2_1[:, :, :, :input2_1.shape[3] // 2] = 0
        input2_2 = torch.ones_like(input)
        input2_2[:, :, :, input2_2.shape[3] // 2:] = 0
        G1 = gram_matrix(input * input2_1)
        G2 = gram_matrix(input * input2_2)
        self.loss = F.mse_loss(G1, self.target1) + F.mse_loss(G2, self.target2)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature1, target_feature2, alpha):
        super(StyleLoss, self).__init__()
        self.target1 = gram_matrix(target_feature1).detach()
        self.target2 = gram_matrix(target_feature2).detach()
        self.target = alpha * self.target1 + (1 - alpha) * self.target2

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)  # + F.mse_loss(G, self.target2))/2
        return input


def plotting_loss(all_losses):
    fig, ax = plt.subplots()
    for i in range(len(all_losses)):
        ax.plot(all_losses[i], label=str(i / 10))
    ax.legend()
    ax.set_ylim(bottom=0, top=50)
    ax.set_xlabel('step')
    ax.set_ylabel('weighted_loss')
    ax.set_title('Losses per step for different loss weights')
    fig.savefig('../results_images/loss_values.jpg')
    plt.close(fig)


def creating_gif(img2, img1):
    img2 = os.path.basename(img2).split('.')[0]
    img1 = os.path.basename(img1).split('.')[0]
    font_size = 20
    font = ImageFont.truetype('arial.ttf', font_size)
    images = []
    filenames = sorted(glob('../results_images/*.png'),
                       key=lambda x: int(os.path.basename(x).split('results_')[1].split('.png')[0]))
    for i, file in enumerate(filenames):
        image = Image.open(file).resize((256, 256))
        draw = ImageDraw.Draw(image)
        text = f'{img2}_{1 - i / 10:.1f}_{img1}_{i / 10:.1f}'
        draw.text((10, image.height - font_size * 1.3), text, font=font, fill=(255, 255, 255))
        images.append(image)
    imageio.mimsave('../results_images/results.gif', images, duration=1)
