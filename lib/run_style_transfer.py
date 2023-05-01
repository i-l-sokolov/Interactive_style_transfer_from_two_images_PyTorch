from argparse import ArgumentParser
from project_assembling import init_run

parser = ArgumentParser(description='Neural Style Transfer Parser')
parser.add_argument('--content_image', type=str, default='../original_images/monalisa.jpg', help='number of optimization steps')
parser.add_argument('--style_image1', type=str, default='../original_images/scream.jpg', help='weight of style loss')
parser.add_argument('--style_image2', type=str, default='../original_images/picasso.jpg', help='weight of content loss')
parser.add_argument('--num_steps', type=int, default=500, help='number of optimization steps')
parser.add_argument('--style_weight', type=int, default=100000, help='weight of style loss')
parser.add_argument('--content_weight', type=int, default=1, help='weight of content loss')
parser.add_argument('--print', type=int, default=1, help='printing of losses during training')

args = parser.parse_args()

num_steps = args.num_steps
style_weight = args.style_weight
content_weight = args.content_weight
content_image = args.content_image
style_img1 = args.style_image1
style_img2 = args.style_image2
printing = bool(args.print)

if __name__ == '__main__':
    init_run(content_image, style_img2, style_img1, num_steps, style_weight, content_weight, printing=printing)
