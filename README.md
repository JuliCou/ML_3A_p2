## ML_3A_p2

Learning analytics: https://analyse.kmi.open.ac.uk/open_dataset#about
Projet Machine Learning 3A

# Contributors:

Justine Coutelier

Julie Courgibet

Nicolas Glomot

# Analysis dataframes - Features engineering

code python: getting_main_dataframe.py

Obtention fichier csv de plus de 6 millions de lignes

# Machine learning

initial_predict.py permet une première prédiction à partir des données de studentInfo principalement. Score assez faible (45%)

model_time_dataframe.py crée des dataframes pour 10 temps différents (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100% de la durée du cours).

On obtient les scores de prédictions suivants (dataframe type 1 : données de connexion + notes ; dataframe type 2 : ajout des données contenues dans studentInfo).

Les scores de précisions suivant ont été calculés en cross-validation (3 étapes). Modèle xgboost avec optimisation de paramètres :

0% : type 1 précision = 0.5875166666666667 - type 2 précision : 0.6026751626336121

10% : type 1 précision = 0.5913227568078973 - type 2 précision : 0.6007418735880748

20% : type 1 précision = 0.6448313781656134 - type 2 précision : 0.6511824747784355

30% : type 1 précision = 0.6890744753810236 - type 2 précision : 0.689902763999480

40% : type 1 précision = 0.7259839367845765 - type 2 précision : 0.7281931271847806

50% : type 1 précision = 0.7628942256217462 - type 2 précision : 0.7629861767392349

60% : type 1 précision = 0.7958457338618671 - type 2 précision : 0.7970116105065767

70% : type 1 précision = 0.7944650495139648 - type 2 précision : 0.7958457338618671

80% : type 1 précision = 0.8431255165686132 - type 2 précision : 0.8435244044165594

90% : type 1 précision = 0.8620560881509368 - type 2 précision : 0.8612890174456801

100% : type 1 précision = 0.8697880356860684 - type 2 précision : 0.8693583609935804


Pour 100%, la note d'examen n'est pas prise en compte.

# API

api.py : permet d'afficher les résultats

# RAF

1/ Afficher scores de résultats dans l'API => Fait (partie prof)

2/ predict_proba (à ajouter dans l'API) => Fait

3/ Utiliser modèle comme random forest pour faire du learning analytics (expliquer pourquoi tel élève est en difficulté) Features importance https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
=> Fait

4/ Data viz => Idées ? (peut être amélioré)