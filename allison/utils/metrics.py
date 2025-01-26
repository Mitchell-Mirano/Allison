import numpy as np

def predict_labels(predictions):
    return np.where(predictions<=0.5,0,1)

def r2_score(labels, predictions):
    errors=np.sum((labels-predictions)**2)
    varianza=labels.var()*len(labels)
    return (1-(errors/varianza))*100


def accuracy(preds,labels):

    preds= preds
    labels= labels
    correct = np.sum(preds==labels)

    return (correct/len(preds))*100

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


