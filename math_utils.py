import numpy as np
import math

def lerp(a, b, x):
    return a * (1-x) + b * x


def clamp(x, max_x, min_x):
    if x > max_x:
        return max_x
    elif x < min_x:
        return min_x
    return x


def softmax(x, k) :
    c = np.max(x)
    exp_a = np.exp(k * (x-c))
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def softmax_pow(x, k):
    c = x / np.max(x)
    c = c ** k
    return c / np.sum(c)

def euclidean_distance(loc1, loc2):
    x1, y1, z1 = loc1
    x2, y2, z2 = loc2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return distance

def cartesian(longitude, latitude, elevation=0):
    # Convert to radians
    latitude = latitude * (math.pi / 180)
    longitude = longitude * (math.pi / 180)

    R = 6371  # 6378137.0 + elevation  # relative to centre of the earth
    X = R * math.cos(latitude) * math.cos(longitude)
    Y = R * math.cos(latitude) * math.sin(longitude)
    Z = R * math.sin(latitude)
    return (X, Y, Z)