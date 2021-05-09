import numpy as np



def r2_score(labels, predictions):
    errors=np.sum((labels-predictions)**2)
    varianza=labels.var()*len(labels)
    return (1-(errors/varianza))*100


def accuracy(labels,predictions):
    correct_predictions=0
    for label, labels_pred in zip(labels, predictions):
        if label==labels_pred:
            correct_predictions=correct_predictions +1

    return (correct_predictions/len(labels))*100


def recall(labels, predictions):
    targets=list(set(labels))
    tp=0
    fn=0

    for label, labels_pred in zip(labels, predictions):
        if label==labels_pred and label==targets[0]:
            tp=tp+1

        if label!=labels_pred and label==targets[0]:
            fn=fn +1
    return (tp/(tp+fn))*100

def precision(labels, predictions):
    targets=list(set(labels))
    tp=0
    ft=0

    for label, labels_pred in zip(labels, predictions):
        if label==labels_pred  and label==targets[0]:
            tp=tp +1
        if label!=labels_pred and  label==targets[1]:
            ft =ft +1
        
    return (tp/(tp+ft))*100


