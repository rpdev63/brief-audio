import pandas as pd
from functions import generate_sounds, generate_dataset, predict_with_SVM, generate_dataset_from_sounds

trigger1, trigger2, trigger3 = False, False, False
error = "Mauvaise saisie, veuillez recommencer."
generate_dataset_from_sounds()
# while ( not trigger1 ):
#     print("Voulez vous créer une nouvelle base de donnée ? (o)ui / (n)on")
#     answers = ['o','n']
#     choice= input('Votre réponse : ')
#     if (choice in answers) :
#         trigger1 = True
#         if choice == "o":
#             #Prompt the duration
#             # while (not trigger2) :
#             #     print("Veuillez entrer une durée en secondes : ( min : 1, max : 9 )")
#             #     try : 
#             #         duration = int(input('Votre réponse : '))
#             #     except :
#             #         print(error)
#             #         continue 
#             #     else :
#             #         if (duration >= 1) & (duration <= 9) :
#             #             trigger2 = True
#             #         else :
#             #             print(error)                        
#             #Prompt the sample quantity
#             while (not trigger3) :
#                 print("Veuillez saisir un nombre d'échantillon : ( min : 1, max : 200 )")
#                 try :
#                     n_samples = int(input('Votre réponse : '))
#                 except :
#                     print(error)
#                     continue 
#                 else :
#                     if (n_samples >= 1) & (n_samples <= 200) :
#                         trigger3 = True
#                     else :
#                         print(error)
            
#             directory = f"{n_samples}samples/"
#             generate_sounds(n_samples, directory)
#             generate_dataset(directory=directory)
#     else :
#         print(error)  



# csv_file = 'data/' + directory + 'son.csv'
# predict_with_SVM(csv_file)
