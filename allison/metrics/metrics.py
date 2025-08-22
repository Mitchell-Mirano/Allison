import numpy as np
from allison.nn.tensor import Tensor



def r2_score(Y_true, Y_pred):

    if isinstance(Y_true, Tensor):
        Y_true = Y_true.data

    if isinstance(Y_pred, Tensor):
        Y_pred = Y_pred.data

    sr = np.mean((Y_true-Y_pred)**2)
    sy = np.mean((Y_true-np.mean(Y_true))**2)
    return 1-(sr/sy)


def accuracy(Y_true, Y_pred):

    if isinstance(Y_true, Tensor):
        Y_true = Y_true.data

    if isinstance(Y_pred, Tensor):
        Y_pred = Y_pred.data

    
    if Y_true.ndim==1:
        return np.mean(Y_true==Y_pred)
    return np.mean(np.sum(Y_true==Y_pred,axis=1)/Y_true.shape[1])

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



def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    classes = np.unique(y_true)
    cm = np.zeros((len(classes), len(classes)))
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            cm[i, j] = np.sum((y_true == c1) & (y_pred == c2))
    return cm



def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Reporte de clasificación similar a sklearn.metrics.classification_report.
    """
    classes = sorted(np.unique(y_true))
    report = {}
    total_true = len(y_true)

    # Métricas por clase
    for c in classes:
        true_pos = np.sum((y_true == c) & (y_pred == c))
        pred_pos = np.sum(y_pred == c)
        actual_pos = np.sum(y_true == c)

        precision = true_pos / pred_pos if pred_pos > 0 else 0.0
        recall = true_pos / actual_pos if actual_pos > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        report[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual_pos
        }

    # Promedio macro
    macro_precision = np.mean([report[c]["precision"] for c in classes])
    macro_recall = np.mean([report[c]["recall"] for c in classes])
    macro_f1 = np.mean([report[c]["f1"] for c in classes])

    # Promedio ponderado
    weights = np.array([report[c]["support"] for c in classes])
    weighted_precision = np.average([report[c]["precision"] for c in classes], weights=weights)
    weighted_recall = np.average([report[c]["recall"] for c in classes], weights=weights)
    weighted_f1 = np.average([report[c]["f1"] for c in classes], weights=weights)

    # Estilo sklearn: ancho fijo y espacio inicial
    header = f"{'':<12}{'precision':>9}{'recall':>9}{'f1-score':>9}{'support':>9}"
    lines = [header]
    
    # Líneas para cada clase
    for c in classes:
        lines.append(f"{str(c):<12}{report[c]['precision']:>9.2f}{report[c]['recall']:>9.2f}{report[c]['f1']:>9.2f}{report[c]['support']:>9}")
    
    lines.append("")
    
    # Fila de accuracy (sklearn también la incluye)
    accuracy = np.sum(y_true == y_pred) / total_true
    lines.append(f"{'accuracy':<12}{'':>9}{'':>9}{accuracy:>9.2f}{total_true:>9}")
    
    # Líneas de promedios con alineación correcta
    lines.append(f"{'macro avg':<12}{macro_precision:>9.2f}{macro_recall:>9.2f}{macro_f1:>9.2f}{total_true:>9}")
    lines.append(f"{'weighted avg':<12}{weighted_precision:>9.2f}{weighted_recall:>9.2f}{weighted_f1:>9.2f}{total_true:>9}")

    return "\n".join(lines)


