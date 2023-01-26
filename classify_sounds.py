import pandas as pd
import os
import shutil

# creat dataframe with csv file
df = pd.read_csv('testing.csv')

# file name in csv and file name in wav folder d'ont have excatly the same name, we need to add a Y at the beginning
df['filename'] = df['filename'].map('Y{}'.format)

# we want to select just car and truck sound, for this we filter df label columns
car_df = df.query('label == "Car"')
truck_df = df.query('label == "Truck"')

# creat folder for store sounds files
path = 'data/car_truck_sounds'
os.makedirs(path, exist_ok = True)

# We wan to move all sounds with car and trucks in forlder and rename it
for index, elem in enumerate(car_df.filename):
    src = os.getcwd()+'/data/sound_library/{}'.format(elem)
    dst = os.getcwd()+'/data/car_truck_sounds/'
    shutil.copy(src, dst+'car{}.wav'.format(index))

for index, elem in enumerate(truck_df.filename):
    src = os.getcwd()+'/data/sound_library/{}'.format(elem)
    dst = os.getcwd()+'/data/car_truck_sounds/'
    shutil.copy(src, dst+'truck{}.wav'.format(index))