import pandas as pd
import category_encoders as ce


def cleanData():
    data = pd.read_csv("Metabolic Syndrome.csv", sep=",")
    # Eliminamos valores que no tengan relevancia
    data.drop(["seqn"], axis="columns", inplace=True)
    print(data.shape)
    data = data.dropna()
    print(data.shape)

    # Para este caso, transformamos Sex, Marital y Race, me queda pendiente eliminar los magic strings
    # print(data.Sex[1]) -> esto es posible y valido
    data = pd.get_dummies(
        data, columns=["Sex", "Race", "Marital"], drop_first=True)
    coder = ce.OrdinalEncoder(cols=["Sex_Male"])
    data = coder.fit_transform(data)
    for col in data.filter(like="Race"):
        data[col] = data[col].astype(int)
    for col in data.filter(like="Marital"):
        data[col] = data[col].astype(int)

    return data