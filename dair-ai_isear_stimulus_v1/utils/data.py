"""Module pour la gestion des données textuelles"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.5.4"
__author__ = "Votre Nom"

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from normalize_data import load_text_data

def load_data(dataset):
    """Charge les données textuelles pour EmotionNet"""
    if isinstance(dataset, str):
        # Si c'est un fichier numpy
        data = np.load(dataset)
        texts = data[:, 0]  # Adapté selon votre structure
        labels = data[:, 1]
    else:
        # Si c'est un dossier avec des fichiers texte
        texts, labels, _ = load_text_data(dataset)
    
    return texts, labels

# ... (adapter les autres fonctions pour le texte)
import os

def plot_confusion_matrix(phase, path, class_names):
    """Plots the confusion matrix using matplotlib.

    Parameter
    ---------
    phase : str
      String value indicating for what phase is the confusion matrix, i.e. training/validation/testing
    path : str
      Directory where the predicted and actual label NPY files reside
    class_names : str
      List consisting of the class names for the labels

    Returns
    -------
    conf : array, shape = [num_classes, num_classes]
      Confusion matrix
    accuracy : float
      Predictive accuracy
    """

    # List all the result files using os.listdir
    files = [f for f in os.listdir(path) if f.endswith('.npy')]

    labels = np.array([])

    for file in files:
        labels_batch = np.load(os.path.join(path, file))  # Charger les fichiers NPY
        labels = np.append(labels, labels_batch)

        if (files.index(file) / len(files)) % 0.2 == 0:
            print(
                "Done appending {}% of {}".format(
                    (files.index(file) / len(files)) * 100, len(files)
                )
            )

    labels = np.reshape(labels, newshape=(labels.shape[0] // 4, 4))

    print("Done appending NPY files.")

    # get the predicted labels
    predictions = labels[:, :2]

    # get the actual labels
    actual = labels[:, 2:]

    # create a TensorFlow session
    with tf.Session() as sess:

        # decode the one-hot encoded labels to single integer
        predictions = sess.run(tf.argmax(predictions, 1))
        actual = sess.run(tf.argmax(actual, 1))

    # get the confusion matrix based on the actual and predicted labels
    conf = confusion_matrix(y_true=actual, y_pred=predictions)

    # create a confusion matrix plot
    plt.imshow(conf, cmap=plt.cm.Purples, interpolation="nearest")

    # set the plot title
    plt.title("Confusion Matrix for {} Phase".format(phase))

    # legend of intensity for the plot
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    plt.tight_layout()
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")

    # show the plot
    plt.show()

    # get the accuracy of the phase
    accuracy = (conf[0][0] + conf[1][1]) / labels.shape[0]

    # return the confusion matrix and the accuracy
    return conf, accuracy
