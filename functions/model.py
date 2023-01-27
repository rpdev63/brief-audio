import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import os
import numpy as np
from joblib import dump


def run_model(csv_file) :
    folder = os.getcwd() + "/data/datasets/"
    df = pd.read_csv(folder + csv_file)
    #nettoyer
    df.dropna(axis='columns', inplace=True)
    tmp = df.shape[1]
    print("{} colonnes ont été supprimés car les valeurs étaient aberrantes".format(tmp - df.shape[1]))

    #définir les features et les targets
    y = df["target"]
    X = df.select_dtypes(include=['int', 'float'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y)
    print("Il y a {} données pour notre set de training.\nIl y a {} données pour notre set de test".format(len(X_train), len(X_test)))

    #Standardiser
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #Evaluer avec 3 modèles de ML
    launch_model("SVC", SVC(), X_train, y_train, X_test, y_test)
    launch_model("RegressionLogistique", LogisticRegression(), X_train, y_train, X_test, y_test)
    launch_model("GaussianNB", GaussianNB(), X_train, y_train, X_test, y_test)


def launch_model(name, model, X_train, y_train, X_test, y_test):
    print("============ ",name, " ============")
    model.fit(X_train, y_train)

    #Prédictions avec le set de test
    predictions = model.predict(X_test)    
    sc = model.score(X_test, y_test)
    print("Score de précision: {:.3f}.".format(sc))
    
    # Plot the confusion matrix
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Matrice de confusion:\n{disp.confusion_matrix}")
    path = os.getcwd() + r"/models/"
    if not os.path.exists(path):
        os.makedirs(path)
    dump(model, path + name)

    