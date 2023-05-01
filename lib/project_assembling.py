from transforms import get_loader, image_loader
from run_transfer import run_style_transfer
import torch
from torchvision import models
from tqdm import tqdm
from loss import plotting_loss, creating_gif


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 128
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

cnn = models.vgg19(weights='IMAGENET1K_V1').features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

loader = get_loader(imsize)


def init_run(cont_img, img1, img2, num_steps=500, style_weight=100000, content_weight=1, printing=False):
    style1_img = image_loader(img1, loader, device)
    style2_img = image_loader(img2, loader, device)
    content_img = image_loader(cont_img, loader, device)
    input_img = content_img.clone()
    losses_all = []
    for i in tqdm(range(11), desc='N of pic', ncols=75):
        i = i / 10
        _, loss_round = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                           content_img, style2_img, style1_img, i, input_img,
                           content_layers_default, style_layers_default, device,
                           num_steps=num_steps,
                           style_weight=style_weight,
                           content_weight=content_weight, printing=printing)
        losses_all.append(loss_round)
    plotting_loss(all_losses=losses_all)
    creating_gif(img1,img2)
