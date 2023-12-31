from dataTransform import cleanData
from dataTransform import selection_testAndQuality
from sklearn.tree import DecisionTreeClassifier
from dataTransform import porcentajesAcierto
from MatrizConf import MatrizConf

data = cleanData()
x, y, trainToX, testToX, trainToY, testToY = selection_testAndQuality(data)

DTS = DecisionTreeClassifier(max_depth=5, random_state=20)
DTS.fit(trainToX, trainToY)

trainAcurracyDTS_Porcentual, testAcurracyDTS_Porcentual, acurracy_DTS, yTestPredictDTS = porcentajesAcierto(
    DTS, x, y, trainToX, testToX, trainToY, testToY)


print("La Score Cross-Validation del DTS fue: ", acurracy_DTS, "%")
print("La presición en train del DTS fue: ", trainAcurracyDTS_Porcentual, "%")
print("La presición en test del DTS fue: ", testAcurracyDTS_Porcentual, "%")

MatrizConf(testToY, yTestPredictDTS)
