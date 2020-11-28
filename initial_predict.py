import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


## Function for reporting best parameters grid search
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

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
# vle = vle.drop("week_from", 1)
# vle = vle.drop("week_to", 1)

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
# .drop("semester", 1).drop("year", 1)
studentRegistration.date_unregistration = [int(studentRegistration.date_unregistration[i]) if not(np.isnan(studentRegistration.date_unregistration[i])) else studentRegistration.module_presentation_length[i] + 1 for i in range(studentRegistration.shape[0])]
studentRegistration = studentRegistration.drop("module_presentation_length", 1)

# Student Assessment
studentAssessment = studentAssessment.merge(assessments, on="id_assessment").merge(courses, on=["code_module", "code_presentation"])
studentAssessment["is_late"] = [1 if studentAssessment.date_submitted[i] > studentAssessment.date[i] else 0 for i in range(studentAssessment.shape[0])]
# inutile!! studentAssessment["is_banked"] = [1 if x == 1 else 0 for x in studentAssessment["is_banked"]]
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




###########################
########### M L ###########
###########################


# First model: Predictions on day 1
student = studentInfo.merge(studentRegistration, on=["code_module", "code_presentation", "id_student"])
student = student.drop("date_unregistration", axis=1)
nStudentVleBefore = nStudentVle[nStudentVle.date < 1]
# First aggregation
nStudentVleBefore1 = nStudentVleBefore.groupby(["code_module", "code_presentation", "id_student", "date"]).agg(
    {
         'nb_click_total' : ["sum", "mean", "std"],
         'id_site' : ["count"]
    }
)
nStudentVleBefore1.columns = ['nb_click_total', 'mean_click', "std_click", "nb_ressources"]
nStudentVleBefore1 = nStudentVleBefore1.reset_index()
# Other data with 1st aggregation
nStudentVleBefore2 = nStudentVleBefore1.groupby(["code_module", "code_presentation", "id_student"]).agg(
    {
         'date' : ["count", "min", "max"],
         'nb_ressources' : ["mean", "max", "min", "std"]
    }
)
nStudentVleBefore2.columns = ['nb_active_days', "1st_day", "last_day", 'mean_nb_ressources', "max_nb_ressources", "min_nb_ressources", "std_nb_ressources"]
nStudentVleBefore2 = nStudentVleBefore2.reset_index()
# 2nd aggregation
nStudentVleBefore3 = nStudentVleBefore.groupby(["code_module", "code_presentation", "id_student"]).agg(
    {
         'nb_click_total' : ["sum", "mean", "std"],
         'id_site' : ["count"]
    }
)
nStudentVleBefore3.columns = ['nb_click_total', 'mean_click', "std_click", "nb_ressources"]
nStudentVleBefore3 = nStudentVleBefore3.reset_index()

# Results of these aggregations
nStudentVleBefore = nStudentVleBefore2.merge(nStudentVleBefore3, on=["code_module", "code_presentation", "id_student"])

# Third aggregation
activeVLEBefore = vle[vle.week_from < 1]
nVLE = activeVLEBefore.groupby(["code_module", "code_presentation"]).agg(
    {
        "id_site" : ["count"]
    }
)
nVLE.columns = ["nb_total_ressources"]
nVLE = nVLE.reset_index()

nStudentVleBefore = nStudentVleBefore.merge(nVLE, on=["code_module", "code_presentation"])

# New features
nStudentVleBefore["per_ressources"] = nStudentVleBefore["nb_ressources"] / nStudentVleBefore["nb_total_ressources"]
nStudentVleBefore = nStudentVleBefore.drop("nb_total_ressources", axis=1)

# Final datatable for 1st model
studentTotal = student.merge(nStudentVleBefore, on=["code_module", "code_presentation", "id_student"], how='outer')
studentTotal = studentTotal.fillna(0)

# Recherche valeur na
studentTotal = studentTotal.replace([np.inf, -np.inf], np.nan)
studentTotal.fillna(studentTotal.mean(), inplace=True)

# Label encoder for target
studentTotal["Success"] = [1 if x=="Pass" or x=="Distinction" else 0 for x in studentTotal.final_result]
studentTotal["final_result"] = studentTotal["final_result"].replace("Fail", 0).replace("Withdrawn", 1).replace("Pass", 2).replace("Distinction", 3)

# One hot encoder for all the variables
# >>> studentTotal.shape
# (32593, 25)
studentTotal = pd.get_dummies(studentTotal)
# >>> studentTotal.shape
# (32593, 47)

# Définition variables X et y
uselessCol = ['gender_F', "id_student", "Success"]
uselessCol.append("final_result")
X = studentTotal.drop(uselessCol, axis=1).values
y = studentTotal[["Success"]].values

# Recherches des meilleurs paramètres RF
params = {
        'n_estimators':[50, 100, 200, 500],
        "bootstrap": [True, False],
        'max_depth': [None, 1, 5, 10, 20]
       }
n_iter_search = 20
rf = RandomForestClassifier()
random_search_rf = RandomizedSearchCV(rf, param_distributions=params,
                                   n_iter=n_iter_search, cv=5, iid=False)
random_search_rf.fit(X, y)
report(random_search_rf.cv_results_)
# Model with rank: 1
# Mean validation score: 0.459 (std: 0.020)
# Parameters: {'n_estimators': 500, 'max_depth': 1, 'bootstrap': False}

# Avec Success
# Model with rank: 1
# Mean validation score: 0.642 (std: 0.022)
# Parameters: {'n_estimators': 50, 'max_depth': 5, 'bootstrap': False}

# Recherches des meilleurs paramètres XGBoost
params = {
        'n_estimators':[50, 100, 200, 500],
        'min_child_weight': [1, 3, 5],
        'gamma': [0.2, 0.5, 1, 2],
        'subsample': [0.4, 0.6, 0.8],
        'colsample_bytree': [0.4, 0.6, 0.8],
        'max_depth': [1, 5, 10, 20]
       }
num_class = len(studentTotal["final_result"].unique())
xgb = XGBClassifier(learning_rate=0.02, objective='multi:softmax',
                    num_class=num_class, silent=True, nthread=1)
folds = 3
param_comb = 30
skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 2019)
random_search_xgb = RandomizedSearchCV(xgb,
                                       param_distributions=params,
                                       n_iter=param_comb,
                                       n_jobs=4,
                                       cv=skf.split(X, y),
                                       verbose=3,
                                       random_state=2019)

random_search_xgb.fit(X, y)
report(random_search_xgb.cv_results_)
# Model with rank: 1
# Mean validation score: 0.506 (std: 0.007)
# Parameters: {'subsample': 0.8, 'n_estimators': 500, 'min_child_weight': 5, 'max_depth': 5, 'gamma': 0.5, 'colsample_bytree': 0.6}

# Cross-validation
nSplits = 10
kf = StratifiedKFold(n_splits=nSplits, random_state=2019)
cv_accuracy_rf = 0.0
cv_accuracy_lr = 0.0
cv_accuracy_xgb = 0.0

col_new_data = studentTotal.drop(uselessCol, axis=1).columns
# col_results_x = ["lr_cls1", "lr_cls2", "lr_cls3", "lr_cls4"]
# col_results_y = ["rf_cls1", "rf_cls2", "rf_cls3", "rf_cls4"]
col_results_x = ["lr_cls1", "lr_cls2"]
col_results_y = ["rf_cls1", "rf_cls2"]

# col_results_z = ["xgb_cls1", "xgb_cls2", "xgb_cls3", "xgb_cls4"]

new_data = pd.DataFrame(columns=col_new_data)
results = pd.DataFrame(columns=col_results_x+col_results_y) # +col_results_z

for train_index, cv_index in kf.split(X, y):
    # Division data
    X_train, X_cv = X[train_index], X[cv_index]
    y_train, y_cv = y[train_index], y[cv_index]
    # Ajout dans dataframe
    new_data_x = pd.DataFrame(X_cv, columns=col_new_data)
    new_data = pd.concat((new_data, new_data_x), axis=0, ignore_index=True)
    # Models
    lr = LogisticRegressionCV()
    rf = RandomForestClassifier(n_estimators=random_search_rf.best_estimator_.n_estimators, bootstrap=random_search_rf.best_estimator_.bootstrap, max_depth=random_search_rf.best_estimator_.max_depth)
    rf = RandomForestClassifier(n_estimators=30, bootstrap=False, max_depth=1)
    # xgb = XGBClassifier(learning_rate=0.02,
    #                 n_estimators=random_search_xgb.best_estimator_.n_estimators,
    #                 objective='multi:softmax',
    #                 num_class=num_class,
    #                 silent=True,
    #                 nthread=1,
    #                 min_child_weight=random_search_xgb.best_estimator_.min_child_weight,
    #                 gamma=random_search_xgb.best_estimator_.gamma,
    #                 subsample=random_search_xgb.best_estimator_.subsample,
    #                 colsample_bytree=random_search_xgb.best_estimator_.colsample_bytree,
    #                 max_depth=random_search_xgb.best_estimator_.max_depth)
    # Training
    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    # xgb.fit(X_train, y_train)
    # Predictions
    y_pred_lr = lr.predict(X_cv)
    y_pred_rf = rf.predict(X_cv)
    # y_pred_xgb = xgb.predict(X_cv)
    cv_accuracy_lr += accuracy_score(y_cv, y_pred_lr)
    cv_accuracy_rf += accuracy_score(y_cv, y_pred_rf)
    # cv_accuracy_xgb += accuracy_score(y_cv, y_pred_xgb)
    # Prediction probabilités
    y_pred_lr = lr.predict_proba(X_cv)
    y_pred_rf = rf.predict_proba(X_cv)
    # y_pred_xgb = xgb.predict_proba(X_cv)
    # Ajout dans dataframe
    results_x = pd.DataFrame(y_pred_lr, columns=col_results_x)
    results_y = pd.DataFrame(y_pred_rf, columns=col_results_y)
    # results_z = pd.DataFrame(y_pred_xgb, columns=col_results_z)
    results = pd.concat((results, pd.concat([results_x, results_y], axis=1)), axis=0, ignore_index=True) # , results_z

print('CV accuracy LR: ' + str(cv_accuracy_lr /nSplits))
print('CV accuracy RF: ' + str(cv_accuracy_rf /nSplits))
# print('CV accuracy XGB: ' + str(cv_accuracy_xgb /nSplits))

# Avec les 4 catégories
# >>> print('CV accuracy LR: ' + str(cv_accuracy_lr /nSplits))
# CV accuracy LR: 0.4534381470128688
# >>> print('CV accuracy RF: ' + str(cv_accuracy_rf /nSplits))
# CV accuracy RF: 0.4580684781744283

# Définition variables X et y
new_complete_data = pd.concat([new_data, results], axis=1)
colMerge = []
for col in studentTotal.columns:
    if col in new_complete_data.columns:
        colMerge.append(col)
data_lastM = new_complete_data.merge(studentTotal, on=colMerge, how="inner")
data_lastM = data_lastM[col_results_x+col_results_y]

X = data_lastM.values

nSplits = 10
kf = StratifiedKFold(n_splits=nSplits, random_state=2019)
cv_accuracy = 0.0

for train_index, cv_index in kf.split(X, y):
    # Division data
    X_train, X_cv = X[train_index], X[cv_index]
    y_train, y_cv = y[train_index], y[cv_index]
    # Apprentissage
    lr = LogisticRegressionCV()
    lr.fit(X_train, y_train)
    # Predictions
    y_pred = lr.predict(X_cv)
    cv_accuracy += accuracy_score(y_cv, y_pred)

print('CV accuracy LR: ' + str(cv_accuracy /nSplits))


