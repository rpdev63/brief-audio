from functions import get_data, generation_data


# generation_data(2,200, "2sec")

df = get_data("2sec/",200)

df.to_csv("son.csv")
print(df)



