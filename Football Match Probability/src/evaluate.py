import tensorflow as tf
import matplotlib.pylab as plt
import numpy as np
from sklearn import metrics

target2int = {'away': 0, 'draw': 1, 'home': 2}

def evaluate(model, X, y):
    probs = model.predict(X)
    preds = np.argmax(probs, axis=1)
    report = metrics.classification_report(y, preds)
    print(report)
    logloss = metrics.log_loss(y, probs)
    print('Log loss:', logloss)
    cm = metrics.confusion_matrix(y, preds)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(target2int.keys()))
    disp.plot(cmap=plt.cm.Blues)
    plt.show()