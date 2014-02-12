__author__ = 'bryan'

import mahotas as mh
import skimage as ski
import skimage.morphology
import skimage.measure

from utility import *


def findBranchPoints(binary):
    """Finds all branch points in a binary image. Useful
    for pruning a skeleton. """
    xbranch0  = np.array([[1,0,1],[0,1,0],[1,0,1]])
    xbranch1 = np.array([[0,1,0],[1,1,1],[0,1,0]])
    tbranch0 = np.array([[0,0,0],[1,1,1],[0,1,0]])
    tbranch1 = np.flipud(tbranch0)
    tbranch2 = tbranch0.T
    tbranch3 = np.fliplr(tbranch2)
    tbranch4 = np.array([[1,0,1],[0,1,0],[1,0,0]])
    tbranch5 = np.flipud(tbranch4)
    tbranch6 = np.fliplr(tbranch4)
    tbranch7 = np.fliplr(tbranch5)
    ybranch0 = np.array([[1,0,1],[0,1,0],[2,1,2]])
    ybranch1 = np.flipud(ybranch0)
    ybranch2 = ybranch0.T
    ybranch3 = np.fliplr(ybranch2)
    ybranch4 = np.array([[0,1,2],[1,1,2],[2,2,1]])
    ybranch5 = np.flipud(ybranch4)
    ybranch6 = np.fliplr(ybranch4)
    ybranch7 = np.fliplr(ybranch5)

    br = mh.morph.hitmiss(binary,xbranch0)
    br+= mh.morph.hitmiss(binary,xbranch1)

    br+= mh.morph.hitmiss(binary,tbranch0)
    br+= mh.morph.hitmiss(binary,tbranch1)
    br+= mh.morph.hitmiss(binary,tbranch2)
    br+= mh.morph.hitmiss(binary,tbranch3)
    br+= mh.morph.hitmiss(binary,tbranch4)
    br+= mh.morph.hitmiss(binary,tbranch5)
    br+= mh.morph.hitmiss(binary,tbranch6)
    br+= mh.morph.hitmiss(binary,tbranch7)

    br+= mh.morph.hitmiss(binary,ybranch0)
    br+= mh.morph.hitmiss(binary,ybranch1)
    br+= mh.morph.hitmiss(binary,ybranch2)
    br+= mh.morph.hitmiss(binary,ybranch3)
    br+= mh.morph.hitmiss(binary,ybranch4)
    br+= mh.morph.hitmiss(binary,ybranch5)
    br+= mh.morph.hitmiss(binary,ybranch6)
    br+= mh.morph.hitmiss(binary,ybranch7)

    return br

pruningLength = 20

def prune_skeleton(skeleton, showPictures=False):

    pruned = skeleton.copy()

    bp = findBranchPoints(pruned)
    if showPictures: showImage(bp)
    # Find the locations of the branch points
    (x_bp, y_bp) = np.where(bp)

    # Set the values equal to zero at the branch points
    pruned[x_bp, y_bp] = 0

    # Now remove the short connected components
    labels = ski.morphology.binary_closing()
    props = ski.measure.regionprops(labels)

    for p in props:
        coords = p.coords
        numPixels = coords.shape[0]
        if numPixels < pruningLength:
            pruned[coords[:, 0], coords[:, 1]] = 0
    if showPictures: showImage(pruned)
    return pruned