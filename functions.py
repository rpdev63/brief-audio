# Constituer un dataset
import os
import pickle
import numpy as np
from features_functions import compute_features
import pandas as pd
import random
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def generate_sounds(duree, size, folder):
    
    path = os.getcwd()+'/data/'
    os.mkdir(path+folder)
    fe = 44100
    amp = 0.1
    list_freq = [random.randint(0, 20000) for i in range(size)]
    t = np.arange(0, duree, 1/fe)
    
    for i in range(len(list_freq)):
        sinus = amp*np.sin(2*np.pi*(list_freq[i])*t)
        file_sinus = 'data/'+folder+'/sinus'+str(i)+'-f-'+str(list_freq[i])
        fichier_sinus = open(file_sinus, 'wb')
        pickle.dump(sinus, fichier_sinus)
        fichier_sinus.close()
        
    for i in range(len(list_freq)):
        bb = amp*np.random.randn(duree*fe)
        file_bb = 'data/'+folder+'/bb'+str(i)
        fichier_bb = open(file_bb, 'wb')
        pickle.dump(bb, fichier_bb)
        fichier_bb.close()  
    
    print("Sons artifiels générés !")




def generate_dataset(directory) :
    print("Génération du dataframe / Lecture des fichiers audio")
    data_dir = os.getcwd() + r"/data/" + directory + ""
    signals = os.listdir(data_dir)
    df = pd.DataFrame()
    i = 0

    for signal in signals : 
        i += 1
        file = open(data_dir + signal, 'rb')
        input_sig = pickle.load(file)

        # Compute the signal in three domains
        sig_sq = input_sig**2
        sig_t = input_sig / np.sqrt(sig_sq.sum())
        sig_f = np.absolute(np.fft.fft(sig_t))
        sig_c = np.absolute(np.fft.fft(sig_f))
        features_list = []
        N_feat, features_list = compute_features(sig_t, sig_f[:sig_t.shape[0]//2], sig_c[:sig_t.shape[0]//2])
        
        if "sinus" in signal :
            features_list.append("sinus")
        elif "bb" in signal :
            features_list.append("blanc")
            
        df[signal] = features_list
        
    df_final = df.transpose()
    labels = [ "feature" + str(i) for i in range(N_feat + 1) ]
    df_final.columns = labels
    df_final.rename(columns={df_final.columns[-1]:'target'}, inplace=True)
    df_final.to_csv('data/'+directory+"son.csv", index=False)
    print("fichier csv généré")
    

def predict_with_SVM(csv_file) :
    df = pd.read_csv(csv_file)
    print(df.head())

    #Predictions :
    print('\n')
    print("Entrainement du modèle \n ============== \n")

    tmp = df.shape[1]
    df.dropna(axis='columns', inplace=True)
    print("{} colonnes ont été supprimés car les valeurs étaient aberrantes".format(tmp - df.shape[1]))
    y = df["target"]
    X = df.select_dtypes(include=['int', 'float'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    print("Il y a {} données pour notre set de training.\nIl y a {} données pour notre set de test".format(len(X_train), len(X_test)))

    #Entrainement avec le set de training
    model = SVC()
    model.fit(X_train, y_train)

    #Prédictions avec le set de test
    predictions = model.predict(X_test)
    print("Comme prédictions, nous obtenons les résultats suivants : \n{}".format(predictions))

    #Calculer la précision
    sc = model.score(X_test, y_test)
    print("En comparant nos valeurs prédites avec celles attendues, nous obtenons le score de précision suivant : \n{}.".format(sc))

    # Plot the confusion matrix
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Matrice de confusion:\n{disp.confusion_matrix}")