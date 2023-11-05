from sklearn.model_selection import cross_val_score
from dataTransform import cleanData
from dataTransform import selection_testAndQuality
from sklearn.neural_network import MLPClassifier
import numpy as np

data = cleanData()
x, y, trainToX, testToX, trainToY, testToY = selection_testAndQuality(data)
MLP = MLPClassifier(hidden_layer_sizes=(10, 8, 6, 3),
                    activation="relu", solver="adam", max_iter=1500)

cv_score_mlp = cross_val_score(MLP, x, y, cv=5)

cv_acurracy_mlp = np.mean(cv_score_mlp)
print("El acierto promedio del Multy Layer Perceptron es: ",
      round(cv_acurracy_mlp, 5) * 100, "%")
