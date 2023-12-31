from dataTransform import cleanData
from dataTransform import porcentajesAcierto
from sklearn.neural_network import MLPClassifier
from dataTransform import selection_testAndQuality
from MatrizConf import MatrizConf

data = cleanData()
x, y, trainToX, testToX, trainToY, testToY = selection_testAndQuality(data)
MLP = MLPClassifier(hidden_layer_sizes=(10, 8, 6, 3), alpha=0.5, random_state=20,
                    activation="relu", solver="adam", max_iter=1500)

MLP.fit(trainToX, trainToY)

trainAcurracyMLP_Porcentual, testAcurracyMLP_Porcentual, acurracy_MLP, yTestPredictMLP = porcentajesAcierto(
    MLP, x, y, trainToX, testToX, trainToY, testToY)

print("El Score Cross-Validation del MLP es: ", acurracy_MLP, "%")
print("La presición en train del MLP fue: ", trainAcurracyMLP_Porcentual, "%")
print("La presición en test del MLP fue: ", testAcurracyMLP_Porcentual, "%")

MatrizConf(testToY, yTestPredictMLP)
