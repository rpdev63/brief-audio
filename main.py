import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from functions import get_data, generation_data

generation_data(2,100, "2sec")

df = get_data("2sec/",200)

df.to_csv("son.csv")
