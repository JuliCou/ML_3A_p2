import os
from copy import deepcopy
from flask import Flask, request, render_template, json
import pandas as pd
import pickle
import eli5
import seaborn as sns
from matplotlib import pyplot
import random
from sklearn.metrics import accuracy_score


app = Flask(__name__)

# Importation des données
studentInfo = pd.read_csv('studentInfo.csv', header=0, sep=",", encoding="ISO-8859-1")
assessments = pd.read_csv('assessments.csv', header=0, sep=",", encoding="ISO-8859-1")
courses = pd.read_csv('courses.csv', header=0, sep=",", encoding="ISO-8859-1")
studentAssessment = pd.read_csv('studentAssessment.csv', header=0, sep=",", encoding="ISO-8859-1")
studentRegistration = pd.read_csv('studentRegistration.csv', header=0, sep=",", encoding="ISO-8859-1")
studentVle = pd.read_csv('studentVle.csv', header=0, sep=",", encoding="ISO-8859-1")
vle = pd.read_csv('vle.csv', header=0, sep=",", encoding="ISO-8859-1")

# Importation des dataset selon le temps
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

# Importation des datasets complets selon le temps
time = range(0, 110, 10)
dic_c_df = {t:[] for t in time}
dic_c_df[time[0]] = pd.read_csv('complete_dataframes/df_t0.csv', header=0, sep=",", encoding="ISO-8859-1")
dic_c_df[time[1]] = pd.read_csv('complete_dataframes/df_t1.csv', header=0, sep=",", encoding="ISO-8859-1")
dic_c_df[time[2]] = pd.read_csv('complete_dataframes/df_t2.csv', header=0, sep=",", encoding="ISO-8859-1")
dic_c_df[time[3]] = pd.read_csv('complete_dataframes/df_t3.csv', header=0, sep=",", encoding="ISO-8859-1")
dic_c_df[time[4]] = pd.read_csv('complete_dataframes/df_t4.csv', header=0, sep=",", encoding="ISO-8859-1")
dic_c_df[time[5]] = pd.read_csv('complete_dataframes/df_t5.csv', header=0, sep=",", encoding="ISO-8859-1")
dic_c_df[time[6]] = pd.read_csv('complete_dataframes/df_t6.csv', header=0, sep=",", encoding="ISO-8859-1")
dic_c_df[time[7]] = pd.read_csv('complete_dataframes/df_t7.csv', header=0, sep=",", encoding="ISO-8859-1")
dic_c_df[time[8]] = pd.read_csv('complete_dataframes/df_t8.csv', header=0, sep=",", encoding="ISO-8859-1")
dic_c_df[time[9]] = pd.read_csv('complete_dataframes/df_t9.csv', header=0, sep=",", encoding="ISO-8859-1")
dic_c_df[time[10]] = pd.read_csv('complete_dataframes/df_t10.csv', header=0, sep=",", encoding="ISO-8859-1")

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

# Liste des élèves
identifiant_student = list(dic_df[time[0]].id_student.unique())
code_module = list(studentInfo.code_module.unique())
code_presentation = list(studentInfo.code_presentation.unique())


@app.route('/formulaire_prof', methods=['POST', 'GET'])
def formulaire_prof():
    """Create profile function"""
    if request.method == 'POST':
        donnees = request.form
        module = donnees["code_module"]
        presentation = donnees["code_presentation"]
        date = int(donnees["date_perc"])
        # Récupération données pour affichage tableau de bord du prof
        students = studentInfo[(studentInfo.code_module==module)&(studentInfo.code_presentation==presentation)]
        dfStudents = dic_c_df[date][(studentInfo.code_module==module)&(studentInfo.code_presentation==presentation)]
        dfStudents = dic_df[date][dic_df[date]["_id"].isin(list(dfStudents["_id"]))]
        # Prédictions
        target_classes = dic_models[date].classes_
        predictions = dic_models[date].predict(dfStudents.drop(['final_result', '_id'], axis=1))
        accuracy = accuracy_score(predictions, list(dfStudents['final_result']))
        proba = dic_models[date].predict_proba(dfStudents.drop(['final_result', '_id'], axis=1))
        donnees = {"code_module":module, "code_presentation":presentation, "date":date, "precision": accuracy}
        data_OK = []
        data_alerte = []
        for i in range(dfStudents.shape[0]):
            dt = {}
            dt["id_student"] = students.iloc[i].id_student
            dt["_id"] = dfStudents.iloc[i]._id
            dt["final_result"] = students.iloc[i].final_result
            dt["predicted_result"] = predictions[i]
            dt["proba"] = [{"classe" : target_classes[j], "proba": proba[i][j]} for j in range(len(proba[i]))]
            if predictions[i]=="Withdrawn" or predictions[i]=="Fail":
                data_alerte.append(dt)
            else:
                data_OK.append(dt)
        # Feature Importance - modèle général
        importance = dic_models[date].get_booster().get_score(importance_type='weight')
        features_importance_df2 = pd.DataFrame({"Features":list(importance.keys()), "Importance":list(importance.values())})
        features_importance_df2 = features_importance_df2.sort_values(by="Importance", ascending=False)
        fi_df2 = features_importance_df2.iloc[range(10)]
        fig = pyplot.figure(figsize=(15, 7))
        fig = sns.barplot(x='Features', y="Importance", data=fi_df2)
        nom_image = "fig_prof_" + str(random.randint(0, 10)) + ".jpeg"
        fig.figure.savefig("static/" + nom_image)
        donnees["img"] = "../static/" + nom_image
        return render_template('tableau_prof.html', donnees=donnees, data_OK=data_OK, data_alerte=data_alerte)

    return render_template('index.html')


@app.route('/formulaire_eleve', methods=['POST', 'GET'])
def formulaire_eleve():
    """Create profile function"""
    if request.method == 'POST':
        donnees = request.form
        student = int(donnees["id_student"])
        date = int(donnees["date_perc"])
        if student in identifiant_student:
            infoEleve = studentInfo[studentInfo.id_student == student]
            donneesEleve = dic_df[date][dic_df[date].id_student == student]
            # Prédictions
            target_classes = dic_models[date].classes_
            predictions = dic_models[date].predict(donneesEleve.drop(['final_result', '_id'], axis=1))
            proba = dic_models[date].predict_proba(donneesEleve.drop(['final_result', '_id'], axis=1))
            # Learning analytics
            f_importance = []
            for j in range(len(predictions)):
                expl = eli5.explain_prediction_xgboost(dic_models[date], donneesEleve.drop(['final_result', '_id'], axis=1).iloc[j])
                features_importance_eleve = {"Features" : [], "Weights" : [], "img" : ""}
                for i in range(len(expl.targets)):
                    if expl.targets[i].target == predictions[j]:
                        posFeatures = expl.targets[i].feature_weights.pos[:4]
                        negFeatures = expl.targets[i].feature_weights.neg[:3]
                        features_importance_eleve["Features"] += [posFeatures[k].feature for k in range(1, len(posFeatures))]
                        features_importance_eleve["Features"] += [negFeatures[k].feature for k in range(0, len(negFeatures))]
                        features_importance_eleve["Weights"] += [posFeatures[k].weight for k in range(1, len(posFeatures))]
                        features_importance_eleve["Weights"] += [negFeatures[k].weight for k in range(0, len(negFeatures))]
                feature_importance_df = pd.DataFrame(features_importance_eleve)
                fig = pyplot.figure(figsize=(15, 7))
                fig = sns.barplot(x='Features', y="Weights", data=feature_importance_df)
                nom_image = "fig" + str(j) + str(random.randint(0, 10)) + ".jpeg"
                fig.figure.savefig("static/" + nom_image)
                features_importance_eleve["img"] = "../static/" + nom_image
                f_importance.append(features_importance_eleve)
            # Construction données à afficher
            donnees_etudiant = {"id_student": student, "date" : date}
            data_OK = []
            data_alerte = []
            for i in range(donneesEleve.shape[0]):
                if predictions[i] == "Withdrawn" or predictions[i] == "Fail":
                    data_alerte.append({"code_presentation": infoEleve.iloc[i]["code_presentation"],
                                        "code_module":infoEleve.iloc[i].code_module,
                                        "prediction":predictions[i],
                                        "final_result":infoEleve.iloc[i].final_result,
                                        "proba": [{"classe" : target_classes[j], "proba": proba[i][j]} for j in range(len(proba[i]))],
                                        "f_importance" : f_importance[i]})
                else:
                    data_OK.append({"code_presentation": infoEleve.iloc[i]["code_presentation"],
                                    "code_module":infoEleve.iloc[i].code_module,
                                    "prediction":predictions[i],
                                    "final_result":infoEleve.iloc[i].final_result,
                                    "proba": [{"classe" : target_classes[j], "proba": proba[i][j]} for j in range(len(proba[i]))],
                                    "f_importance" : f_importance[i]})
            return render_template("tableau_eleve.html", data_eleve=donnees_etudiant, data_OK=data_OK, data_alerte=data_alerte, feature_importance=f_importance)
        else:
            return render_template("page_erreur.html")

    return render_template('index.html')


@app.route('/prof_voir_eleve', methods=['POST', 'GET'])
def prof_voir_eleve():
    """Create profile function"""
    if request.method == 'POST':
        # Récupération données
        donnees = request.form
        data_eleve = donnees["data_eleve"]
        data_general = donnees["data_general"]
        data_eleve = eval(data_eleve)
        data_general = eval(data_general)
        # Recherche données
        student = int(data_eleve["id_student"])
        date = int(data_general["date"])
        # Recherche dataframe
        infoEleve = studentInfo[(studentInfo.id_student == student)&(studentInfo.code_module==data_general["code_module"])&(studentInfo.code_presentation==data_general["code_presentation"])]
        dfStudents = dic_df[date][dic_df[date]._id==data_eleve["_id"]]
        # Prédictions
        target_classes = dic_models[date].classes_
        predictions = dic_models[date].predict(dfStudents.drop(['final_result', '_id'], axis=1))
        proba = dic_models[date].predict_proba(dfStudents.drop(['final_result', '_id'], axis=1))
        # Learning analytics
        f_importance = []
        expl = eli5.explain_prediction_xgboost(dic_models[date], dfStudents.drop(['final_result', '_id'], axis=1).iloc[0])
        features_importance_eleve = {"Features" : [], "Weights" : [], "img" : ""}
        for i in range(len(expl.targets)):
            if expl.targets[i].target == predictions[0]:
                posFeatures = expl.targets[i].feature_weights.pos[:4]
                negFeatures = expl.targets[i].feature_weights.neg[:3]
                biaisPos = 0
                for k in range(len(posFeatures)):
                    if "BIAS" in posFeatures[k].feature:
                        biaisPos = k
                features_importance_eleve["Features"] += [posFeatures[k].feature for k in range(0, len(posFeatures)) if k != biaisPos]
                features_importance_eleve["Features"] += [negFeatures[k].feature for k in range(0, len(negFeatures))]
                features_importance_eleve["Weights"] += [posFeatures[k].weight for k in range(0, len(posFeatures)) if k != biaisPos]
                features_importance_eleve["Weights"] += [negFeatures[k].weight for k in range(0, len(negFeatures))]
        feature_importance_df = pd.DataFrame(features_importance_eleve)
        fig = pyplot.figure(figsize=(15, 7))
        fig = sns.barplot(x='Features', y="Weights", data=feature_importance_df)
        nom_image = "fig0" + str(random.randint(0, 10)) + ".jpeg"
        fig.figure.savefig("static/" + nom_image)
        features_importance_eleve["img"] = "../static/" + nom_image
        f_importance.append(features_importance_eleve)
        # Construction données à afficher
        donnees_etudiant = {"id_student": student, "date" : date}
        data_OK = []
        data_alerte = []
        if predictions[0] == "Withdrawn" or predictions[0] == "Fail":
            data_alerte.append({"code_presentation": infoEleve.iloc[0]["code_presentation"],
                                "code_module":infoEleve.iloc[0].code_module,
                                "prediction":predictions[0],
                                "final_result":infoEleve.iloc[0].final_result,
                                "proba": [{"classe" : target_classes[j], "proba": proba[0][j]} for j in range(len(proba[0]))],
                                "f_importance" : f_importance[0]})
        else:
            data_OK.append({"code_presentation": infoEleve.iloc[0]["code_presentation"],
                            "code_module":infoEleve.iloc[0].code_module,
                            "prediction":predictions[0],
                            "final_result":infoEleve.iloc[0].final_result,
                            "proba": [{"classe" : target_classes[j], "proba": proba[0][j]} for j in range(len(proba[0]))],
                            "f_importance" : f_importance[0]})
        return render_template("tableau_eleve.html", data_eleve=donnees_etudiant, data_OK=data_OK, data_alerte=data_alerte, feature_importance=f_importance)
    return render_template('index.html')


@app.route('/index', methods=['POST', 'GET'])
def routage():
    """Routing function"""
    if request.method == 'POST':
        route = request.form.getlist("choix")

        if route[0] == 'profil_prof':
            dic = {"code_module": code_module, "code_presentation":code_presentation, "date_perc": list(range(0, 110, 10))}
            return render_template('dashboard_prof.html', dic=dic)
        if route[0] == 'profil_eleve':
            dic = {"id_student" : identifiant_student, "date_perc": list(range(0, 110, 10))}
            return render_template('dashboard_eleve.html', dic=dic)

    return render_template("index.html")


@app.route('/')
def index():
    """Index function"""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
