__author__ = 'bryan'

import matplotlib.pyplot as plt
import matplotlib.widgets as w
import skimage as ski
import skimage.io
import skimage.draw
import skimage.morphology
import skimage.color
from chirality_image_analysis.utility import *

class LassoTool:
    """Choose regions by hand that should be black
    in a binary image."""

    def __init__(self, image):
        self.image = image
        self.fig = showImage(self.image)
        self.ax = self.fig.gca()
        self.key_press_id = self.fig.canvas.mpl_connect('key_press_event', self.key_press)

        self.lasso_tool_on = False
        self.lasso = None

        self.label_fig_on = False
        self.labelFig = None

        plt.show()

    def key_press(self, event):
        if event.key == 'a':
            self.lasso_tool_on = not self.lasso_tool_on
            print 'Lasso Tool: ' , self.lasso_tool_on
            if self.lasso_tool_on:
                self.lasso = w.LassoSelector(self.ax, self.selector_callback, useblit=True)
                plt.draw()
            else:
                self.lasso.disconnect_events()
                self.lasso = None
                plt.draw()
        elif event.key == 'z':
            self.label_fig_on = not self.label_fig_on
            print 'Label Tool: ' , self.label_fig_on
            if self.label_fig_on:
                self.show_label_image()
            else:
                self.close_label_image()

    def show_label_image(self):
        self.labelFig = plt.figure()
        labels = ski.morphology.label(self.image) - 1
        ski.io.imshow(ski.color.label2rgb(labels))
        plt.show(block=False)

    def close_label_image(self):
        plt.close(self.labelFig)

    def selector_callback(self, verts):
        yindices = np.array([f[0] for f in verts])
        xindices = np.array([f[1] for f in verts])
        rr, cc = ski.draw.polygon(xindices, yindices)

        self.image[rr, cc] = False
        ski.io.imshow(self.image)
        plt.draw()