import pandas as pd
import os
import shutil

# creat dataframe with csv file, choose your csv file
df = pd.read_csv('testing.csv')

# file name in csv and file name in wav folder d'ont have excatly the same name, we need to add a Y at the beginning
df['filename'] = df['filename'].map('Y{}'.format)

# write the path for the sound library
sound_library_path = '/data/sound_library'

# choose two sound styles from the available labels
label_1 = 'Motorcycle'
label_2 = 'Skateboard'

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

# Loop for select and copy in a folder all label 2 sounds
for index, elem in enumerate(label_2_df.filename):
    src = os.getcwd()+'/{}/{}'.format(sound_library_path, elem)
    dst = os.getcwd()+'/{}/'.format(path)
    shutil.copy(src, dst+'{}{}.wav'.format(name_2,index))