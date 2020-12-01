from matplotlib import pyplot
import pandas as pd
import pickle
from xgboost import plot_importance, plot_tree
import seaborn as sns
import eli5


# Importation dataframes
time = range(0, 110, 10)
dic_df = {t:[] for t in time}
dic_df[time[0]] = pd.read_csv('dataframes/df_t0.csv', header=0, sep=",", encoding="ISO-8859-1")
dic_df[time[1]] = pd.read_csv('dataframes/df_t1.csv', header=0, sep=",", encoding="ISO-8859-1")
dic_df[time[2]] = pd.read_csv('dataframes/df_t2.csv', header=0, sep=",", encoding="ISO-8859-1")
dic_df[time[3]] = pd.read_csv('dataframes/df_t3.csv', header=0, sep=",", encoding="ISO-8859-1")
dic_df[time[4]] = pd.read_csv('dataframes/df_t4.csv', header=0, sep=",", encoding="ISO-8859-1")
dic_df[time[5]] = pd.read_csv('dataframes/df_t5.csv', header=0, sep=",", encoding="ISO-8859-1")
dic_df[time[6]] = pd.read_csv('dataframes/df_t6.csv', header=0, sep=",", encoding="ISO-8859-1")
dic_df[time[7]] = pd.read_csv('dataframes/df_t7.csv', header=0, sep=",", encoding="ISO-8859-1")
dic_df[time[8]] = pd.read_csv('dataframes/df_t8.csv', header=0, sep=",", encoding="ISO-8859-1")
dic_df[time[9]] = pd.read_csv('dataframes/df_t9.csv', header=0, sep=",", encoding="ISO-8859-1")
dic_df[time[10]] = pd.read_csv('dataframes/df_t10.csv', header=0, sep=",", encoding="ISO-8859-1")

# Importation des 10 modèles
dic_models = {t:[] for t in time}
dic_models[time[0]] = pickle.load(open('models/model_t0.sav', 'rb'))
dic_models[time[1]] = pickle.load(open('models/model_t1.sav', 'rb'))
dic_models[time[2]] = pickle.load(open('models/model_t2.sav', 'rb'))
dic_models[time[3]] = pickle.load(open('models/model_t3.sav', 'rb'))
dic_models[time[4]] = pickle.load(open('models/model_t4.sav', 'rb'))
dic_models[time[5]] = pickle.load(open('models/model_t5.sav', 'rb'))
dic_models[time[6]] = pickle.load(open('models/model_t6.sav', 'rb'))
dic_models[time[7]] = pickle.load(open('models/model_t7.sav', 'rb'))
dic_models[time[8]] = pickle.load(open('models/model_t8.sav', 'rb'))
dic_models[time[9]] = pickle.load(open('models/model_t9.sav', 'rb'))
dic_models[time[10]] = pickle.load(open('models/model_t10.sav', 'rb'))

# Show features importance
# Choix date - i = 10 => 100%
i = 10

# Affichage des valeurs
columns = dic_df[time[i]].drop(['final_result', '_id'], axis=1).columns
for idx, col in enumerate(dic_df[time[i]].drop(['final_result', '_id'], axis=1).columns):
    print(col, " : ", str(dic_models[time[i]].feature_importances_[idx]))

# Graphique
# https://xgboost.readthedocs.io/en/latest/python/python_api.html
features_importance_df = pd.DataFrame({"Features" : columns, "Importance" : dic_models[time[i]].feature_importances_})
features_importance_df = features_importance_df.sort_values(by="Importance", ascending=False)
fi_df = features_importance_df.iloc[range(10)]
sns.barplot(x='Features', y="Importance", data=fi_df)
pyplot.show()

# Graphique avec autre méthode - résultat différent
# avec plot_importance(dic_models[time[i]])
importance = dic_models[time[i]].get_booster().get_score(importance_type='weight')
features_importance_df2 = pd.DataFrame({"Features":list(importance.keys()), "Importance":list(importance.values())})
features_importance_df2 = features_importance_df2.sort_values(by="Importance", ascending=False)
fi_df2 = features_importance_df2.iloc[range(10)]
sns.barplot(x='Features', y="Importance", data=fi_df2)
pyplot.show()

# Understanding prediction with features
predictions = dic_models[time[i]].predict(dic_df[time[i]].drop(['final_result', '_id'], axis=1))

for j in range(len(predictions)):
    expl = eli5.explain_prediction_xgboost(dic_models[time[i]], dic_df[time[i]].drop(['final_result', '_id'], axis=1).iloc[j])
    features_importance_eleve = {"Features" : [], "Weights" : []}
    for i in range(len(expl.targets)):
        if expl.targets[i].target == predictions[j]:
            posFeatures = expl.targets[i].feature_weights.pos[:4]
            negFeatures = expl.targets[i].feature_weights.neg[:3]
            features_importance_eleve["Features"] += [posFeatures[k].feature for k in range(1, len(posFeatures))]
            features_importance_eleve["Features"] += [negFeatures[k].feature for k in range(0, len(negFeatures))]
            features_importance_eleve["Weights"] += [posFeatures[k].weight for k in range(1, len(posFeatures))]
            features_importance_eleve["Weights"] += [negFeatures[k].weight for k in range(0, len(negFeatures))]
