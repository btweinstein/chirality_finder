__author__ = 'bryan'
"""Responsible for separating lineages in a binary image
by utilizing the geometry of the range expansion."""

class Circle:
    radius = None
    maxRadius = None
    minRadius = None
    xpoints = None
    ypoints = None

    def __init__(self, maxRadius, minRadius):
        self.radius = maxRadius

    def getPointsAtRadius
