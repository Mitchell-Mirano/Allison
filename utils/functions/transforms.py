import numpy as np

def transform_zero_to_one(features):
    min_value=np.min(features)
    max_value=np.max(features)
    return (features/(max_value-min_value)) -(min_value/(max_value-min_value))