import exifread
import pylab as pl
import skimage as ski

def niceDisplay(image, display, scaleFactor=0.25):
    newImage = image.applyLayers()
    newImage.resize( int(newImage.width*scaleFactor), int(newImage.height*scaleFactor) ).save(display)
    
def getScale(imagePath):

    f = open(imagePath, 'rb')
    # Return Exif tags
    tags = exifread.process_file(f)
    f.close()

    scaleFraction = tags['Image XResolution'].values[0]
    pixelPerMM= float(scaleFraction.num)/float(scaleFraction.den)
    
    return 1/pixelPerMM

def showImage(image):
    pl.figure()
    ski.io.imshow(image, interpolation='None')