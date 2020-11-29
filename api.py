import os
from copy import deepcopy
from flask import Flask, request, render_template, json
import pandas as pd
import pickle


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

# Prédictions
# model_t0.predict(df_t0.drop(['final_result', '_id'], axis=1))

# Liste des élèves
identifiant_student = list(studentInfo.id_student.unique())
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
        dfStudents = dic_df[date][(studentInfo.code_module==module)&(studentInfo.code_presentation==presentation)]
        # Prédictions
        predictions = dic_models[date].predict(dfStudents.drop(['final_result', '_id'], axis=1))
        donnees = {"code_module":module, "code_presentation":presentation, "date":date}
        data_OK = []
        data_alerte = []
        for i in range(dfStudents.shape[0]):
            dt = {}
            dt["id_student"] = students.iloc[i].id_student
            dt["final_result"] = students.iloc[i].final_result
            dt["predicted_result"] = predictions[i]
            if predictions[i]=="Withdrawn" or predictions[i]=="Fail":
                data_alerte.append(dt)
            else:
                data_OK.append(dt)
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
            donneesEleve = dic_df[date][studentInfo.id_student == student]
            # data = {}
            # for i in range(donneesEleve.shape[0]):
            #     # data[i] = {}
            #     # for col in donneesEleve.columns:
            #     #     data[i][col] = donneesEleve.col.iloc[i]
            predictions = dic_models[date].predict(donneesEleve.drop(['final_result', '_id'], axis=1))
            donnees_etudiant = {"id_student": student, "date" : date}
            data_OK = []
            data_alerte = []
            for i in range(donneesEleve.shape[0]):
                if predictions[i] == "Withdrawn" or predictions[i] == "Fail":
                    data_alerte.append({"code_presentation": infoEleve.iloc[i]["code_presentation"], "code_module":infoEleve.iloc[i].code_module, "prediction":predictions[i], "final_result":infoEleve.iloc[i].final_result})
                else:
                    data_OK.append({"code_presentation": infoEleve.iloc[i]["code_presentation"], "code_module":infoEleve.iloc[i].code_module, "prediction":predictions[i], "final_result":infoEleve.iloc[i].final_result})
            return render_template("tableau_eleve.html", data_eleve=donnees_etudiant, data_OK=data_OK, data_alerte=data_alerte)
        else:
            return render_template("page_erreur.html")

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
