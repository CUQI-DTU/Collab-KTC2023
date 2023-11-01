import numpy as np
import numpy.linalg as linalg
import numpy.random as random 
import math
import cv2

def generate(inclusions = 2,
             radius = 1,
             inter_distance = 0,
             size = 256,
             weights = None):
    
    if weights is None:
        weights = np.array([1,1,1,1])
    weights_sum = np.sum(weights)        

    accum_image = np.zeros(shape = (size,size)) 
    for i in range(inclusions):
        flag = True
        while flag:
            t = weights_sum*random.rand()
            accum = weights[0]
            flag = False
            if t <= accum and not flag:
                # Rectangle
                indicator = create_rectangle_indicator()
                flag = True
                
            accum += weights[1]
            if t <= accum and not flag:
                # ellipse
                indicator = create_ellipse_indicator()
                flag = True
                
            accum += weights[2]
            if t <= accum and not flag:
                # Voronoi
                indicator = create_Voronoi_indicator(10, corner = 0.8, ran = 0.6)
                flag = True

            accum += weights[3]
            if t <= accum and not flag:
                # Blob
                indicator = create_cosine_indicator(terms = 2)
                flag = True
                
            offset = random.uniform(-0.6, 0.6, size = (2))
            image = discretize(lambda x: indicator(x-offset))
            flag = (check_intersection(accum_image, image) or
                    check_radius(image, radius) or
                    check_intersection(_dilate(accum_image, r = inter_distance), image))
        
        val = random.randint(1,4) 
        accum_image += val*image
        
    return accum_image

# Discretize indicator function
def discretize(indicator, size = (256,256), lower_bounds = (-1,-1), upper_bounds = (1,1)):
    grid = np.zeros(shape = size, dtype=int)
    x_space = np.linspace(lower_bounds[0], upper_bounds[0], size[0])
    y_space = np.linspace(lower_bounds[1], upper_bounds[1], size[1])
    for (i, x) in enumerate(x_space):
        for (j, y) in enumerate(y_space):
            grid[i,j] = indicator(np.array([x,y]))
            
    return grid


# Create indicator functions
def _Voronoi_indicator(x, origin, points):
    dist = linalg.norm(x-origin)
    for point in points:
        if linalg.norm(x-point) < dist:
            return 0
    return 1

def create_Voronoi_indicator(num_additional_points = 0, corner = 0.8, ran = 0.6):
    origin = np.array([0,0])
    points = random.uniform(-ran,ran,(4+num_additional_points,2))
    points[0] = np.array([-corner,-corner])
    points[1] = np.array([corner,-corner])
    points[2] = np.array([-corner,corner])
    points[3] = np.array([corner,corner])
    
    return lambda x: _Voronoi_indicator(x, origin, points)

def create_rectangle_indicator():
    origin = np.array([0,0])
    points = np.zeros((4,2))
    v = random.uniform(low = 0.2, high = 0.5)
    h = random.uniform(low = 0.2, high = 0.5)
    
    theta = random.uniform(low = 0, high = 2*np.pi)
    R = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
    points[0] = R@np.array([0,-v])
    points[1] = R@np.array([0,v])
    points[2] = R@np.array([h,0])
    points[3] = R@np.array([-h,0])
    
    return lambda x: _Voronoi_indicator(x, origin, points)
    
def _cosine_indicator(x, magnitudes, frequences, offsets):
    r = linalg.norm(x)
    angle = math.atan2(x[0], x[1])
    for (m,f,o) in zip(magnitudes, frequences, offsets):
        if r > m*(math.cos(f*(angle - o)) + 1.5):
            return 0
    return 1

def create_cosine_indicator(terms = 2):
    magnitudes = random.uniform(low = 0.2, high = 0.4, size = (terms))
    frequences = random.randint(low = 1, high = 3, size = (terms))
    offsets = random.uniform(low = 0, high = 6.28, size = (terms))
    return lambda x: _cosine_indicator(x, magnitudes, frequences, offsets)


def _ellipse_indicator(x, origin, mat, scale = 1):
    r = (x-origin).T@mat@(x-origin)
    if r <= scale:
        return 1
    return 0

def create_ellipse_indicator():
    vec1 = random.randn(2)
    vec2 = random.randn(2)
    origin = np.array([0,0])
    
    mat = np.outer(vec1,vec1) + np.outer(vec2,vec2)
    
    return lambda x: _ellipse_indicator(x, origin, mat, scale = 0.03)

# Verify validity of phantom
def check_intersection(image_1, image_2):
    for i in range(image_1.shape[0]):
        for j in range(image_1.shape[0]):
            if image_1[i,j] != 0 and image_2[i,j] != 0:
                return True
    return False

def check_radius(image, r = 1, lower_bounds = (-1,-1), upper_bounds = (1,1)):
    x_space = np.linspace(lower_bounds[0], upper_bounds[0], image.shape[0])
    y_space = np.linspace(lower_bounds[1], upper_bounds[1], image.shape[1])
    
    for (i, x) in enumerate(x_space):
        for (j, y) in enumerate(y_space):
            if x*x+y*y >= r*r and image[i,j] != 0:
                return True
    return False
    
def _dilate(image, r = 1):
    kernel = np.ones((r, r))
    return cv2.dilate(image, kernel, iterations=1)
    