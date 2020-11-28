import pandas as pd
import copy
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
import pickle


###########################
#### PREMIERE PARTIE ######
###########################
### PREPARATION DONNEES ###
###########################

# Importation des données
studentInfo = pd.read_csv('studentInfo.csv', header=0, sep=",", encoding="ISO-8859-1")
assessments = pd.read_csv('assessments.csv', header=0, sep=",", encoding="ISO-8859-1")
courses = pd.read_csv('courses.csv', header=0, sep=",", encoding="ISO-8859-1")
studentAssessment = pd.read_csv('studentAssessment.csv', header=0, sep=",", encoding="ISO-8859-1")
studentRegistration = pd.read_csv('studentRegistration.csv', header=0, sep=",", encoding="ISO-8859-1")
studentVle = pd.read_csv('studentVle.csv', header=0, sep=",", encoding="ISO-8859-1")
vle = pd.read_csv('vle.csv', header=0, sep=",", encoding="ISO-8859-1")

# Assessment
assessments.date = assessments.date.fillna(0)
assessments = assessments.merge(courses, on=["code_module", "code_presentation"])
assessments.date = [assessments.date.iloc[i] if assessments.date.iloc[i] != 0 else assessments.module_presentation_length.iloc[i] for i in range(assessments.shape[0])]
assessments = assessments.drop("module_presentation_length", 1)

# vle
vle = vle.merge(courses, on=["code_module", "code_presentation"])
vle.week_from = vle.week_from.fillna(0)
vle.week_to = vle.week_to.fillna(0)
vle.week_to = [vle.week_to[i]*7 if vle.week_to[i] != 0 else vle.module_presentation_length[i] for i in range(vle.shape[0])]
vle.week_from = [int(x) for x in vle.week_from]
vle.week_to = [int(x) for x in vle.week_to]
vle = vle.drop("module_presentation_length", 1)

# Student Info
studentInfo.disability = [1 if x == 'Y' else 0 for x in studentInfo.disability]

studentInfo.highest_education = studentInfo.highest_education.replace('No Formal quals', 0).replace('Lower Than A Level', 1)
studentInfo.highest_education = studentInfo.highest_education.replace('A Level or Equivalent', 2).replace('HE Qualification', 3)
studentInfo.highest_education = studentInfo.highest_education.replace('Post Graduate Qualification', 5)

studentInfo.imd_band = studentInfo.imd_band.replace('0-10%', 5).replace('10-20', 15).replace('20-30%', 25)
studentInfo.imd_band = studentInfo.imd_band.replace('30-40%', 35).replace('40-50%', 45).replace('50-60%', 55)
studentInfo.imd_band = studentInfo.imd_band.replace('60-70%', 65).replace('70-80%', 75).replace('80-90%', 85)
studentInfo.imd_band = studentInfo.imd_band.replace('90-100%', 95)
studentInfo.imd_band = studentInfo.imd_band.fillna(studentInfo.imd_band.mean())
studentInfo.imd_band = [int(x) for x in studentInfo.imd_band]

studentInfo.age_band = studentInfo.age_band.replace('55<=', 60).replace('35-55', 45).replace('0-35', 20)

# Student Registration
studentRegistration = studentRegistration.merge(courses, on=["code_module", "code_presentation"])
studentRegistration.date_unregistration = [int(studentRegistration.date_unregistration[i]) if not(np.isnan(studentRegistration.date_unregistration[i])) else studentRegistration.module_presentation_length[i] + 1 for i in range(studentRegistration.shape[0])]
studentRegistration = studentRegistration.drop("module_presentation_length", 1)

# Student Assessment
studentAssessment = studentAssessment.merge(assessments, on="id_assessment").merge(courses, on=["code_module", "code_presentation"])
studentAssessment["is_late"] = [1 if studentAssessment.date_submitted[i] > studentAssessment.date[i] else 0 for i in range(studentAssessment.shape[0])]
studentAssessment = studentAssessment.drop("module_presentation_length", 1).drop("date", 1)
# Traitement des na
studentAssessment = studentAssessment.merge(studentInfo, on=["code_module", "code_presentation", "id_student"])
new_score = []
for i in range(studentAssessment.shape[0]):
    if not(np.isnan(studentAssessment.score[i])):
        new_score.append(studentAssessment.score[i])
    else:
        if studentAssessment.final_result[i] == "Withdrawn" or studentAssessment.final_result[i] == "Fail":
            new_score.append(0)
        else:
            new_score.append(studentAssessment.score.mean())

studentAssessment["score"] = new_score

# Suppression des colonnes inutiles issues des merges
studentAssessment = studentAssessment.drop("code_module", 1).drop("code_presentation", 1).drop("assessment_type", 1).drop("weight", 1)
studentAssessment = studentAssessment.drop('gender', 1).drop('region', 1).drop('highest_education', 1).drop('imd_band', 1).drop('age_band', 1)
studentAssessment = studentAssessment.drop('num_of_prev_attempts', 1).drop('studied_credits', 1).drop('disability', 1).drop('final_result', 1)

# Student Vle
nStudentVle = studentVle.groupby(["code_module", "code_presentation", "id_student", "id_site", "date"]).agg(
    {
         'sum_click' : ["sum", "count"]
    }
)
nStudentVle.columns = ['nb_click_total', 'count']
nStudentVle = nStudentVle.reset_index()

# Importation dataframe - features
df = pd.read_csv('data_time2.csv', header=0, sep=",", encoding="ISO-8859-1")

# Création clé primaire - identifiant
X = df[["code_module", "code_presentation", "id_student"]]
X = df.groupby(["code_module", "code_presentation", "id_student"]).groups
dic2 = {"code_module" : [], "code_presentation" : [], "id_student" : []}
for key, item in X.items():
    dic2["code_module"].append(key[0])
    dic2["code_presentation"].append(key[1])
    dic2["id_student"].append(key[2])
X = pd.DataFrame(dic2)

Y = []
for i in range(X.shape[0]):
    mod = X.code_module.iloc[i]
    stu = X.id_student.iloc[i]
    pres = X.code_presentation.iloc[i]
    # results
    res = studentInfo[(studentInfo.code_module==mod)&(studentInfo.code_presentation==pres)&(studentInfo.id_student==stu)].final_result.iloc[0]
    Y.append(res)

X["final_result"] = Y

X["_id"] = range(X.shape[0])
df = df.merge(X, on=["id_student", "code_presentation", "code_module"], how="inner")

# Encodage des colonnes code_presentation et code_module
d = pd.get_dummies(df[["code_presentation", "code_module"]])
df[d.columns] = d
df = df.drop(["code_presentation", "code_module"], axis=1)

# Remplacement NaN infinity
df = df.fillna(0)
df = df.replace(np.nan, 0)

# Add Features with data about student
student = studentInfo.merge(studentRegistration, on=["code_module", "code_presentation", "id_student"])
student["_id"] = X["_id"]
student = student.drop(["date_unregistration", "code_module", "code_presentation", "id_student", "final_result"], axis=1)
student = student.fillna(0)
student = student.replace([np.inf, -np.inf], np.nan)
student.fillna(student.mean(), inplace=True)
student = pd.get_dummies(student)
student = student.drop(['gender_F'], axis=1)


# Predict for different time
time = df.date.unique()
time = range(0, 110, 10)

def get_dataframe(df_main, time_t, df_student):
    # Construction dataframe avec dernier temps
    df_t = pd.DataFrame(columns=df.columns)
    for i in range(X.shape[0]):
        df_ = df[df._id ==X["_id"].iloc[i]]
        df_ = df_[df_.date < time_t]
        if df_.shape[0] == 0:
            df_ = df[df._id ==X["_id"].iloc[i]]
        df_ = df_.iloc[[-1]]
        df_t = pd.concat((df_t, df_), axis=0, ignore_index=True)
    # Nettoyage dataframe (dtypes)
    df_dtypes = df.dtypes
    for col in df.columns:
        if df_dtypes[col] != df_t.dtypes[col]:
            df_t[col] = df_t[col].astype(df_dtypes[col])
    # Ajout features Student
    df_t = df_t.merge(df_student, on="_id", how="inner")
    return df_t

dic_df = {t: pd.DataFrame(columns=df.columns) for t in time}
for t in time:
    dic_df[t] = get_dataframe(df, t, student)

# Train et test split
X_train, X_test = train_test_split(X, test_size=0.33, random_state=42)
num_class = len(dic_df[time[0]].final_result.unique())
for idx, t in enumerate(time):
    df_train = dic_df[t][dic_df[t]._id.isin(X_train._id)]
    df_test = dic_df[t][dic_df[t]._id.isin(X_test._id)]
    model = XGBClassifier(learning_rate=0.02,
                            n_estimators=500,
                            objective='multi:softmax',
                            num_class=num_class,
                            silent=True,
                            nthread=1,
                            min_child_weight=3,
                            gamma=0.2,
                            subsample=0.4,
                            colsample_bytree=0.6,
                            max_depth=5)
    model.fit(dic_df[t].drop(["final_result", "_id"], axis=1), dic_df[t][["final_result"]])
    # Save dataframe and model
    df_test.to_csv("dataframes/df_t" + str(idx)+ ".csv", index=False, header=True)
    pickle.dump(model, open("models/model_t" + str(idx)+ ".sav", 'wb'))