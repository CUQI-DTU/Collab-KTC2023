import phantom
import matplotlib.pyplot as plt
import numpy as np

"""
Code for generating simple phantom images.

"""

for n in range(100):
    image = phantom.generate(inclusions = 2, # Number of inclusions
                             radius = 0.95, # Radius of the disc (full disc is 1)
                             inter_distance = 20, # Minimum l1 distance between inclusions (in pixels)
                             size = 256, # Image size
                             weights = np.array([4,4,1,1])) # Weights of the different inclusions [rectangles, ellipses, Voronoi cells, weird blobs]
    
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
        
    circ = plt.Circle((128,128), 128, fill = False, color='k', lw=2)
    plt.gca().add_artist(circ)
    plt.show()