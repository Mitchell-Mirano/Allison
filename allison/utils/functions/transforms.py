import numpy as np

def transform_zero_to_one(features):
    min_value=np.min(features)
    max_value=np.max(features)
    return (features/(max_value-min_value)) -(min_value/(max_value-min_value))

def get_numeric_labels(labels:np.array):
    rows,colums = labels.shape
    if rows < colums:
        labels = labels.T
    
    numeric_labels = []
    for label in labels:
        index = np.where(label == np.max(label))[0][0]
        numeric_labels.append(index+1)

    return np.array(numeric_labels)