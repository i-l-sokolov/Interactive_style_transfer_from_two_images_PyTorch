import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import torch


def get_loader(imsize):
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize),
        transforms.ToTensor()])
    return loader

unloader = transforms.ToPILImage()


def image_loader(image_name, loader, device):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def imshow(tensor, title=None):
    """

    :param tensor:
    :param title:
    :return:
    """
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
