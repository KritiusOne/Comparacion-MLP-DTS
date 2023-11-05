import pandas as pd
from sklearn.model_selection import train_test_split
import Types
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import cross_val_score


def cleanData():
    data = pd.read_csv(Types.NAME_FILE_DATA, sep=Types.SEPARATION_FILE_DATA)

    data.drop(Types.USELESS_COLUMS, axis="columns", inplace=True)

    for col in data:
        if data[col].isnull().any():
            if data[col].dtype.name == "object":
                data[col] = data[col].fillna(data[col].mode())
            else:
                data[col] = data[col].fillna(data[col].mean())

    data = pd.get_dummies(
        data, columns=Types.COLUMNS_TO_TRANSFORM, drop_first=True)

    for i in Types.COLUMNS_TO_TRANSFORM:
        for col in data.filter(like=i):
            data[col] = data[col].astype(int)
    return data


def selection_testAndQuality(data):
    x = data.drop(Types.COLUMN_GOAL_ARR, axis=1)
    y = data[Types.COLUMN_GOAL]
    trainToX, testToX, trainToY, testToY = train_test_split(
        x, y, test_size=0.2, random_state=00000)
    return x, y, trainToX, testToX, trainToY, testToY


def porcentajesAcierto(IA, x, y, trainToX, testToX, trainToY, testToY):
    yTrainPredictMLP = IA.predict(trainToX)
    yTestPredictIA = IA.predict(testToX)

    trainAcurracyIA = accuracy_score(trainToY, yTrainPredictMLP)
    trainAcurracyIA_Porcentual = round(trainAcurracyIA, 8) * 100
    testAcurracyIA = accuracy_score(testToY, yTestPredictIA)
    testAcurracyIA_Porcentual = round(testAcurracyIA, 8) * 100

    score_IA = cross_val_score(IA, x, y, cv=5)
    acurracy_IA = np.mean(score_IA)

    return trainAcurracyIA_Porcentual, testAcurracyIA_Porcentual, round(acurracy_IA, 8) * 100, yTestPredictIA
