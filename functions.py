# Constituer un dataset
import os
import pickle
import numpy as np
from features_functions import compute_features
import pandas as pd
import pickle
import random


def generation_data(duree, size, folder):
    
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




def get_data(directory, number) :
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
        if i == number :
            break
    df_final = df.transpose()
    df_final.rename(columns={df.columns[-1]:'target'})

    return df_final

