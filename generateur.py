import numpy as np
import os
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
        file_sinus = 'data/'+folder+'/sinus'+str(i)
        fichier_sinus = open(file_sinus, 'wb')
        pickle.dump(sinus, fichier_sinus)
        fichier_sinus.close()
        
    for i in range(len(list_freq)):
        bb = amp*np.random.randn(duree*fe)
        file_bb = 'data/'+folder+'/bb'+str(i)
        fichier_bb = open(file_bb, 'wb')
        pickle.dump(bb, fichier_bb)
        fichier_bb.close()  

generation_data(2, 500, '2sec')