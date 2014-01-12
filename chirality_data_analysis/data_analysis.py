__author__ = 'bryan'

import matplotlib.pyplot as plt

def  makeChiralityPlot(chiralityData):
    """Makes a plot comparing the chirality of each labeled region."""

    f = plt.figure()
    numColors = len(chiralityData.label.unique())
    currentColor = 0
    for currentLabel in chiralityData.label.unique():
        labelData = chiralityData[chiralityData['label'] == currentLabel]
        labelData = labelData.sort(columns=['r'])
        # Somehow I still don't think we have a unique theta at every r. Let us fix that.
        # We can do that later. Let us apply this to the other replicate and see what happens.
        labelData.plot(x='r', y='rotated', style='+-', label=currentLabel, color=plt.cm.jet(1.*currentColor/numColors))
        currentColor += 1
    plt.legend()
    return f

def visualizeChiralityData(chiralityData):
    """Plots the different sectors"""
    fig = plt.figure()
    groups = chiralityData.groupby('label')
    numColors = len(groups)
    print numColors
    currentColor = 0
    for name, group in groups:
        group.plot(x='y',y='x', label=name, color=plt.cm.jet(1.*currentColor/numColors))
        currentColor += 1
    fig.gca().invert_yaxis()
    plt.legend()
    return fig