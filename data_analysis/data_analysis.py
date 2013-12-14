__author__ = 'bryan'

def  makeChiralityPlot(chiralityData):
    """Makes a plot comparing the chirality of each labeled region."""
    for currentLabel in chiralityData.label.unique():
        labelData = chiralityData[chiralityData['label'] == currentLabel]
        labelData = labelData.sort(columns=['r'])
        # Somehow I still don't think we have a unique theta at every r. Let us fix that.
        # We can do that later. Let us apply this to the other replicate and see what happens.
        labelData.plot(x='r', y='rotated', style='+')