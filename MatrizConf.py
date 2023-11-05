import seaborn as sb
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as mpp


def MatrizConf(yTest, yTestPrediction):
    Mc_IA = confusion_matrix(yTest, yTestPrediction)
    sb.heatmap(Mc_IA, annot=True, cmap="Blues", fmt="d")
    mpp.title("Matriz de confusion")
    mpp.xlabel("Predicciones")
    mpp.ylabel("Resultados")
    mpp.show()
