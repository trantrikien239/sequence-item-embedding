import numpy as np
from numpy import interp
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_fscore_support, roc_curve, auc
from sklearn.base import BaseEstimator
from matplotlib import pyplot as plt

from itertools import cycle

def evaluate(model, X_test, y_test):
    print('Evaluating model')
    print(classification_report(y_true=y_test, y_pred=model.predict(X_test)))
    metrics = auc_report(model, X_test, y_test)
    metrics['precision'], metrics['recall'], metrics['f1_score'], metrics['support'] = precision_recall_fscore_support(y_test, model.predict(X_test))
    return metrics

def auc_report(model, X, y_true):
    classes = model.classes_
    y_pred_classes = model.predict_proba(X)
    n_classes = len(classes)

    lw = 2
    for i in range(len(classes)):
        print(f"""{classes[i]}: {roc_auc_score(y_true=(y_true==classes[i]).astype(int), y_score=y_pred_classes[:,i])}""")

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true=(y_true==classes[i]).astype(int), y_score=y_pred_classes[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()

    plt.plot(fpr["macro"], tpr["macro"],
                label='macro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["macro"]),
                color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    metrics = {
        'macro_auc': roc_auc["macro"]
    }
    for i in range(n_classes):
        metrics[f'auc_{classes[i]}'] = roc_auc[i]
    return metrics