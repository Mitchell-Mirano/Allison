import numpy as np

def predict_labels(predictions):
    return np.where(predictions<=0.5,0,1)

def r2_score(y_pred,Y_train):
    sr = np.mean((y_pred-Y_train)**2)
    sy = np.mean((Y_train-np.mean(Y_train))**2)
    return 1-(sr/sy)


def accuracy(preds,labels):
    
    if labels.ndim==1:
        return np.mean(preds==labels)
    return np.mean(np.sum(preds==labels,axis=1)/labels.shape[1])

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


