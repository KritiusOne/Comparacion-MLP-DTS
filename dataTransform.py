import pandas as pd
import category_encoders as ce
from sklearn.model_selection import train_test_split
import Types


def cleanData():
    data = pd.read_csv(Types.NAME_FILE_DATA, sep=Types.SEPARATION_FILE_DATA)
    # Eliminamos valores que no tengan relevancia
    data.drop(Types.USELESS_COLUMS, axis="columns", inplace=True)

    # Eliminamos valores en blanco
    data = data.dropna()

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
