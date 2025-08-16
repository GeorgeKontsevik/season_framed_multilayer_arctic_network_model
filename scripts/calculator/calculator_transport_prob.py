import numpy as np

def sigmoid(x, center, steepness=1):
    """
    Sigmoid function for smooth probability transition
    center: temperature at which probability is 0.5
    steepness: controls how sharp the transition is
    """
    return 1 / (1 + np.exp(-steepness * (x - center)))


def get_transport_probability(transport_type, temperature):
    """
    Calculate transport probability based on temperature using smooth transitions
    """
    if transport_type == "Aviation":
        if temperature < 0:
            return np.round(sigmoid(temperature, -50, 0.5), 5)
        else:
            return 1 - np.round(sigmoid(temperature, 40, 0.5), 5)

    elif transport_type == "Water transport":
        if temperature < 10:
            return np.round(sigmoid(temperature, -2, 0.8), 5)
        else:
            return 1 - np.round(sigmoid(temperature, +45, 0.8), 5)

    elif transport_type == "Regular road":
        if temperature < 0:
            return np.round(sigmoid(temperature, -60, 0.7), 5)
        else:
            return 1 - np.round(sigmoid(temperature, 40, 0.7), 5)

    elif transport_type == "Winter road":
        if temperature < 0:
            return np.round(sigmoid(temperature, -50, 0.8), 5)
        else:
            return 1 - np.round(sigmoid(temperature, 10, 0.8), 5)

    return -1e6
