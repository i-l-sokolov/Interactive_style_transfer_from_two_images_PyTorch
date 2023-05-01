import ipywidgets
import matplotlib.pyplot as plt
import os
from lib.project_assembling import init_run
from ipyfilechooser import FileChooser
from ipywidgets import VBox,HBox,IntSlider,Label,Layout, widgets
import numpy as np
from PIL import Image
from IPython.display import display, Javascript


class GeneralWidget():

    def __init__(self):
        self.file_chooser = self.get_filechooser()
        self.params = self.get_params()
        self.button = self.get_button()
        self.tab = self.run_widget()

    def get_filechooser(self):
        self.fc1 = FileChooser('../original_images/')
        self.fc2 = FileChooser('../original_images/')
        self.fc3 = FileChooser('../original_images/')
        file_chooser = ipywidgets.Accordion([self.fc1, self.fc2, self.fc3])
        file_chooser.set_title(0, 'content_image')
        file_chooser.set_title(1, 'style_image1')
        file_chooser.set_title(2, 'style_image2')
        return file_chooser

    def get_params(self):
        self.n_steps = IntSlider(value=500, min=50, max=1000, step=50)
        self.st_weight = IntSlider(value=100000, min=0, max=500000, step=50000)
        self.con_weight = IntSlider(value=1, min=1, max=50, step=1)
        num_step = HBox([Label('num_steps', layout=Layout(width='120px')), self.n_steps])
        style_weight = HBox([Label('style_weight', layout=Layout(width='120px')), self.st_weight])
        content_weight = HBox([Label('content_weight', layout=Layout(width='120px')), self.con_weight])
        params = VBox([num_step, style_weight, content_weight])
        return params

    def showing_loss(self, loss_pic='../results_images/loss_values.jpg'):
        fig, ax = plt.subplots(figsize=(8, 8))
        image = Image.open(loss_pic)
        ax.imshow(np.array(image))
        ax.axis('off')
        plt.show()

    def get_lossshowing(self):
        loss_showing = widgets.Output()
        with loss_showing:
            self.showing_loss()
        return loss_showing

    def get_gifwid(self):
        with open("../results_images/results.gif", "rb") as file:
            # read file as string into `image`
            image = file.read()

        gif = widgets.Image(
            value=image,
            format='gif'
        )
        return gif

    def get_button(self):
        button = widgets.Button(
            description='Run trasfering',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Start process',
            icon='check'  # (FontAwesome names without the `fa-` prefix)
        )
        button.on_click(self.onbutton_click)
        return button

    def onbutton_click(self, b):
        init_run(self.fc1.value, self.fc2.value, self.fc3.value,
                 self.n_steps.value, self.st_weight.value, self.con_weight.value)
        self.loss_showing = self.get_lossshowing()
        alpha = self.get_alpha()
        out_original = self.get_outorig()
        out_results = self.get_outres(alpha)
        self.results_box = self.get_finalbox(out_original, alpha, out_results)
        gif = self.get_gifwid()
        self.tab = self.get_tab(self.file_chooser, self.params, self.button,
                                self.loss_showing, self.results_box, gif)
        display(Javascript('IPython.notebook.execute_cell()'))

    def show_images(self, i):
        result_image = f'../results_images/results_{int(i*10)}.png'

        images = [self.fc2.value, result_image, self.fc3.value]

        titles = [os.path.basename(x) for x in images]

        titles = [x.split('.')[0] for x in titles]

        titles[1] = f'{titles[0]}_{1-i:.1f}_{titles[2]}_{i:.1f}'

        # Create a figure with three subplots
        fig, ax = plt.subplots(1, 3, figsize=(10, 4), sharey=True, sharex=True)

        # Load and resize each image and display it in a subplot
        for i, path in enumerate(images):
            image = Image.open(path)
            image = image.resize((256, 256))
            image = np.array(image)
            ax[i].imshow(image)
            ax[i].set_title(titles[i])
            ax[i].axis("off")

        # Show the figure
        plt.show()

    def show_original(self):
        fig, ax = plt.subplots(figsize=(3, 3))
        image = Image.open(self.fc1.value).resize((256, 256))
        ax.imshow(np.array(image))
        ax.axis('off')
        ax.set_title('original_image')
        plt.show()

    def get_outorig(self):
        out_original = widgets.Output()
        with out_original:
            self.show_original()
        return out_original

    def get_alpha(self):
        alpha = widgets.FloatSlider(min=0, max=1, step=0.1, description='pic2', layout=Layout(justify_content='center'))
        return alpha

    def get_outres(self, alpha):
        out_results = widgets.interactive_output(self.show_images, {'i': alpha})
        return out_results

    def get_finalbox(self, out_original, alpha, out_results):
        results_box = VBox([HBox([out_original, alpha], layout=Layout(align_items='center')), out_results])
        return results_box

    def get_tab(self, file_chooser, params, button, loss_showing, results_box, gif):
        tab = widgets.Tab()
        tab.children = [file_chooser, params, button, loss_showing, results_box, gif]
        tab.set_title(0, 'Choosing picuters')
        tab.set_title(1, 'Choosing params')
        tab.set_title(2, 'Start Run')
        tab.set_title(3, 'Loss_values')
        tab.set_title(4, 'RESULTS')
        tab.set_title(5, 'GIF')
        return tab

    def run_widget(self):
        first_wid = widgets.Output()

        # Getting first tab
        self.tab = self.get_tab(self.file_chooser, self.params, self.button, first_wid, first_wid, first_wid)

        return self.tab
