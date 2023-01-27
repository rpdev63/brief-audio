import os
import numpy as np
import pandas as pd
import random
from functions.features import compute_features
import warnings
import re
from scipy.io.wavfile import write, read
import shutil


def generate_sounds(size, folder):        
    path = os.getcwd() + r'/data/'
    if not os.path.exists(path):
        os.makedirs(path)
    os.mkdir(path+folder)
    print(f"Création du répertoire {path + folder}")
    fe = 44100
    list_freq = [random.randint(0, 20000) for i in range(size)]
    list_amp = [random.randint(0, 100) for i in range(size)]
    list_duree = [random.randint(1, 9) for i in range(size)]    
    for i in range(len(list_freq)):
        t = np.arange(0, list_duree[i], 1/fe)
        sinus = (list_amp[i]/100)*np.sin(2*np.pi*(list_freq[i])*t)
        file_sinus = 'data/'+folder+'/sinus'+str(i)+'-f-'+str(list_freq[i])
        write(file_sinus+'.wav', 44100, sinus)
    for i in range(len(list_freq)):
        bb = (list_amp[i]/100)*np.random.randn(list_duree[i]*fe)
        file_bb = 'data/'+folder+'/blanc'+str(i)
        write(file_bb+'.wav', 44100, bb)
    print("Sons artifiels générés !")


def generate_dataset(directory, all_files=False) :
    #deactivate runtime warning ( division by zero )
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print("Génération d'un fichier .csv en cours")
    data_dir = os.getcwd() + r"/data/" + directory  + "/"
    signals = os.listdir(data_dir)
    df = pd.DataFrame()
    i = 0
    for signal in signals : 
        #get the target
        if all_files : 
            df_ref = pd.read_csv(r'data/testing.csv')
            name = signal[1:]
            try : 
                row = df_ref.loc[df_ref['filename'] == name]
                target = row.iat[0, 3]                
            except : 
                print("Donnée manquantes pour labelliser")
        else :
            target = re.sub(r'\d+', '', signal)
            target = re.sub(r'\..*', '', target)
        
        #get 
        sample_rate, input_sig = read(data_dir + signal)
        if (target != "sinus") & (target != "blanc"):
            input_sig = input_sig / 32767

        # Compute the signal in three domains
        sig_sq = input_sig**2
        sig_t = input_sig / np.sqrt(sig_sq.sum())
        sig_f = np.absolute(np.fft.fft(sig_t))
        sig_c = np.absolute(np.fft.fft(sig_f))
        features_list = []
        N_feat, features_list = compute_features(sig_t, sig_f[:sig_t.shape[0]//2], sig_c[:sig_t.shape[0]//2], sample_rate)
        
        #add target value        
        features_list.append(target)        
        df_tmp = pd.DataFrame({signal:features_list})
        df = pd.concat((df, df_tmp),axis=1)

    #finalize dataset and generate csv file
    df_final = df.transpose()
    labels = [ "feature" + str(i) for i in range(N_feat + 1) ]
    df_final.columns = labels
    df_final.rename(columns={df_final.columns[-1]:'target'}, inplace=True)
    path = os.getcwd() + r'/data/datasets/'
    if not os.path.exists(path):
        os.makedirs(path)
    df_final.to_csv(path + directory + ".csv", index=False)
    print("\nFichier csv généré ! \n")


def classify_sounds(choice1, choice2, size=10):
    # creat dataframe with csv file, choose your csv file
    df = pd.read_csv(r'data/testing.csv')

    # file name in csv and file name in wav folder d'ont have excatly the same name, we need to add a Y at the beginning
    df['filename'] = df['filename'].map('Y{}'.format)

    # write the path for the sound library
    sound_library_path = '/data/sound_library'

    # choose two sound styles from the available labels
    label_1 = choice1
    label_2 = choice2

    # create variables for later
    name_1 = label_1.replace(' ', '_')
    name_1 = name_1.lower()
    name_2 = label_2.replace(' ','_')
    name_2 = name_2.lower()

    # select label 1 and label 2 sounds, for this we filter df label columns
    label_1_df = df.query('label == "{}"'.format(label_1))
    label_2_df = df.query('label == "{}"'.format(label_2))

    # creat the folrder for stores sounds files
    path = 'data/{}_{}_sounds'.format(name_1, name_2)
    os.makedirs(path, exist_ok = True)

    # Loop for select and copy in a folder all label 1 sounds
    for index, elem in enumerate(label_1_df.filename):
        src = os.getcwd()+'{}/{}'.format(sound_library_path, elem)
        dst = os.getcwd()+'/{}/'.format(path)
        shutil.copy(src, dst+'{}{}.wav'.format(name_1,index))
        if index >= size :
            break

    # Loop for select and copy in a folder all label 2 sounds
    for index, elem in enumerate(label_2_df.filename):
        src = os.getcwd()+'/{}/{}'.format(sound_library_path, elem)
        dst = os.getcwd()+'/{}/'.format(path)
        shutil.copy(src, dst+'{}{}.wav'.format(name_2,index))
        if index >= size :
            break

