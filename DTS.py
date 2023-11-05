from dataTransform import cleanData
from dataTransform import selection_testAndQuality
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

data = cleanData()
x, y, trainToX, testToX, trainToY, testToY = selection_testAndQuality(data)

DTS = DecisionTreeClassifier(max_depth=6, random_state=0)
score_DTS = cross_val_score(DTS, x, y, cv=5)
acurracy_DTS = round(np.mean(score_DTS), 5) * 100

print("La presición promedio del DTS fue: ", acurracy_DTS, "%")
