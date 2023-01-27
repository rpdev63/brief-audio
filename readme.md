# Brief Audio

Par Justine & Rémi

Explications de nos travaux :
https://docs.google.com/document/d/11LW6PRWsq84z4KHhM7VpuMjgivTahSS1VUZixkGnYHI/edit#


## Procédure d'installation 

1 ) Cloner le repo git :    

  git clone https://github.com/rpdev63/brief-audio.git
  cd brief-audio

2 ) Créer un environnement virtuel sous anaconda ( si necessaire installer anaconda https://www.anaconda.com/products/distribution ) :

  conda create --name myEnvName python 
  
3 ) Activer l'environnement virtuel

  conda activate myEnvName
  
4 ) Lire le fichier requirements.txt pour installer les librairies python

  conda install --yes --file requirements.txt
  
5 ) Suivez ce lien et copiez / coller le dossier data à la racine du projet https://drive.google.com/drive/folders/1Oz58S0TNKTBnC4ZvN3kVqvSA-wAEV1aR

6 ) Mettez vos fichiers de sons dans le dossier data/sound_library
  
7 ) Lancer l'interface

  python main.py
