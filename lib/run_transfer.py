from torch import optim
from model import get_style_model_and_losses
from torchvision import transforms

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img1, style_img2, alpha, input_img,
                       content_layers, style_layers, device, num_steps=500, style_weight=100000, content_weight=1, printing=True):
    """Run the style transfer."""
    if printing:
        print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean, normalization_std,
                                                                     style_img1, style_img2, alpha, content_img,
                                                                     content_layers, style_layers, device)
    optimizer = get_input_optimizer(input_img)
    if printing:
        print('Optimizing..')
    run = [0]
    losses = []
    while run[0] <= num_steps:

        def closure():
            # correct the values
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()

            model(input_img)

            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            losses.append(loss.item())

            if (len(losses) == 1) or (losses[-1] == min(losses)):
                final_image = transforms.ToPILImage()(input_img.squeeze(0))
                final_image.save(f'../results_images/results_{alpha*10:.0f}.png')

            run[0] += 1
            if (run[0] % 50 == 0) and printing:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img, losses
