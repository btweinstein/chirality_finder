__author__ = 'bryan'

import matplotlib.pyplot as plt
import matplotlib.widgets as w
import skimage as ski
import skimage.io
import skimage.draw
import skimage.morphology
import skimage.color

from chirality_image_analysis.utility import *


class InteractiveSelector:
    """Choose regions by hand that should be black
    in a binary image."""

    def __init__(self, image):
        self.image = image
        self.prevImage = self.image.copy()
        self.fig = showImage(self.image)
        self.ax = self.fig.gca()

        self.isdone=False
        self.key_press_id = self.fig.canvas.mpl_connect('key_press_event', self.key_press)
        self.close_id = self.fig.canvas.mpl_connect('close_event', self.onclose)

        self.lasso = None
        self.cutter = None
        self.labelFig = None

        plt.show()

    def onclose(self, event):
        self.isdone=True
        print 'Done editing by hand!'

    def key_press(self, event):
        if event.key == 'a': # Lasso Black Tool
            print 'Black Lasso Tool: On'
            self.lasso = w.LassoSelector(self.ax, self.black_lasso_callback, useblit=True)
            plt.draw()
        if event.key == 'w': # Lasso White Tool
            print 'White Lasso Tool: On'
            self.lasso = w.LassoSelector(self.ax, self.white_lasso_callback, useblit=True)
            plt.draw()
        if event.key =='t': # Cut Tool
            print 'Cut Tool: On'
            self.cutter = w.LassoSelector(self.ax, self.cut_callback, useblit=True)
            plt.draw()
        if event.key == 'z': # Label Tool
            print 'Label Tool: Creating Plot'
            self.show_label_image()
        if event.key =='u': # Undo
            print 'Undoing last lasso...'
            self.undo_last_lasso()

    def undo_last_lasso(self):
        self.image = self.prevImage.copy()
        ski.io.imshow(self.image)
        plt.draw()

    def show_label_image(self):
        self.labelFig = plt.figure()
        labels = ski.morphology.label(self.image, neighbors=4, background=False) + 1
        ski.io.imshow(ski.color.label2rgb(labels, bg_label=0))
        plt.show(block=False)

    def cut_callback(self, verts):
        xindices = np.array([f[0] for f in verts], dtype=np.int)
        yindices = np.array([f[1] for f in verts], dtype=np.int)

        self.prevImage = self.image.copy()
        self.image[yindices, xindices] = False
        ski.io.imshow(self.image)

        self.cutter.disconnect_events()
        self.cutter = None
        plt.draw()

    def white_lasso_callback(self, verts):
        yindices = np.array([f[0] for f in verts])
        xindices = np.array([f[1] for f in verts])
        rr, cc = ski.draw.polygon(xindices, yindices)
        self.prevImage = self.image.copy()
        self.image[rr, cc] = True
        ski.io.imshow(self.image)

        self.lasso.disconnect_events()
        self.lasso = None
        plt.draw()

    def black_lasso_callback(self, verts):
        yindices = np.array([f[0] for f in verts])
        xindices = np.array([f[1] for f in verts])
        rr, cc = ski.draw.polygon(xindices, yindices)
        self.prevImage = self.image.copy()
        self.image[rr, cc] = False
        ski.io.imshow(self.image)

        self.lasso.disconnect_events()
        self.lasso = None
        plt.draw()