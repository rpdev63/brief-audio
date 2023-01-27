from functions.database import generate_sounds, generate_dataset, classify_sounds
from functions.model import run_model
import os

trigger1, trigger2, trigger3, trigger4, trigger5, trigger6 = False, False, False, False, False, False
error = "Mauvaise saisie, veuillez recommencer."
lexique = {
         '1':'Bicycle',
         '2':'Skateboard',
         '3':'Screaming',
         '4':'Car',
         '5':'Bus',
         '6':'Truck',
         '7':'Motorcycle',
}         

while ( not trigger1 ):
    print('''Veuillez faire un choix ? (1, 2, 3 ou (q)uitter) ?
    1 : créer une base de donnée de bruits blancs / sinus
    2 : créer une base de donnée en choisissant 2 types de sons
    3 : créer une base de donnée avec tout le contenu de data/sound_library
    4 : entrainer un modèle sur les données''')
    choice= input('Votre réponse : ')
    answers = ['1','2','3','4','q']
    if (choice in answers) :
        trigger1 = True
        match choice:
            # 1 : créer une base de donnée de bruits blancs / sinus
            case "1":
                while (not trigger2) :
                    print("Veuillez saisir un nombre d'échantillon : ( min : 1, max : 200 )")
                    try :
                        n_samples = int(input('Votre réponse : '))
                    except :
                        print('\n' + error + '\n')
                        continue 
                    else :
                        if (n_samples >= 1) & (n_samples <= 200) :
                            trigger2 = True
                            folder = str(n_samples) + "samples" 
                            generate_sounds(n_samples, folder)
                            generate_dataset(folder)
                        else :
                            print('\n' + error + '\n')            
            case "2":
                # 2 : créer une base de donnée en choisissant 2 types de sons
                while (not trigger3) :
                    print("Choix 1 : Entrez un nombre correspondant au type de son. ")
                    for key, val in lexique.items() :
                        print(f"{key} : {val}" )
                    choice1= input('Votre réponse : ')
                    if (choice1 in lexique.keys()) :
                        trigger3 = True
                    else :
                        print('\n' + error + '\n')
                while (not trigger4) :
                    print("Choix 2 : Entrez un nombre correspondant au type de son. ")
                    choice2= input('Votre réponse : ')
                    if (choice2 in lexique.keys()) & (choice2 != choice1):
                        trigger4 = True
                    else :
                        print('\n' + error + '\n')
                while (not trigger5) :
                    print("Veuillez saisir un nombre d'échantillon : ( min : 1, max : 2000 )")
                    try :
                        n_samples = int(input('Votre réponse : '))
                    except :
                        print('\n' + error + '\n')
                        continue 
                    else :
                        if (n_samples >= 1) & (n_samples <= 200) :
                            trigger5 = True
                            classify_sounds(lexique[choice1], lexique[choice2], n_samples)
                            name1 = lexique[choice1].lower()
                            name2 = lexique[choice2].lower()
                            generate_dataset(f"{name1}_{name2}"+"_sounds") 
                        else :
                            print('\n' + error + '\n')                            
            case "3":
                generate_dataset("sound_library", all_files=True)       
            case "4":
                # 3 : entrainer un modèle sur les données
                datasets_dir = os.getcwd() + r"/data/datasets" 
                datasets = os.listdir(datasets_dir)
                dict_csv = {}
                while (not trigger6) :
                    print("Choisissez un dataset dans la liste suivante ( entrer un chiffre ) : ")
                    for index, elem in enumerate(datasets):
                        print("\t" + str(index +1) + ': ' + elem)
                        dict_csv[index + 1] = elem
                    choice1= input('Votre réponse : ')
                    if int(choice1) in dict_csv.keys() :
                        trigger6 = True
                    else :
                        print('\n' + error + '\n')  
                print(dict_csv)
                run_model(dict_csv[int(choice1)]) 
    else :
        print('\n' + error + '\n')