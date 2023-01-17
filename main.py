import pandas as pd
from functions import generate_sounds, generate_dataset, predict_with_SVM

duration = 2
samples = 100
directory = str(duration) +"sec/"

# generate_sounds(duration, samples, directory)
generate_dataset(directory=directory)
csv_file = 'data/' + directory + 'son.csv'
predict_with_SVM(csv_file)
