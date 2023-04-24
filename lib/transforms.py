import torchvision.transforms as transforms
import matplotlib.pyplot as plt

imsize = 128


loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.CenterCrop(imsize),
    transforms.ToTensor()])

unloader = transforms.ToPILImage()


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
