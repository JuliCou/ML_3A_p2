import pandas as pd
import copy
import numpy as np


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

############################
##### DEUXIEME PARTIE ######
############################
### FEATURES ENGINEERING ###
############################

# Organising data
# Empty dataframe
dic = {"code_module": [], "id_student": [], "code_presentation": [], "date": []}
dic = {**dic, "nb_total_click": [], "nb_total_click_last10D": [], "perc_ressources_total": []}
dic = {**dic, "perc_ressources_last10D": [], "click_activity_evolution": [], "ressources_activity_evolution": []}
dic = {**dic, "perc_assignements_late": [], "perc_not_returned_ass": []}
dic = {**dic, 'nb_active_days': [], '1st_day': [], 'last_day': [], 'mean_nb_ressources':[]}
dic = {**dic, 'perc_active_days': []}
dic = {**dic, 'max_nb_ressources': [], 'min_nb_ressources': [], 'std_nb_ressources' : []}
dic = {**dic, 'mean_click_day': [], 'mean_click_active_day' : []}
dic = {**dic, 'mean_click_day_10': [], 'mean_click_active_day_10' : []}
dic = {**dic, 'mean_assignments' : [], 'std_assignments' : [], 'validation' : []}

data2 = pd.DataFrame(columns=list(dic.keys()))

# Other dataframes
studentAssessment = studentAssessment.merge(assessments, on="id_assessment")
studentAssessment["weighted_score"] = studentAssessment["score"] * studentAssessment["weight"] / 100
studentExams = studentAssessment[studentAssessment.assessment_type == "Exam"]
studentAssessments = studentAssessment[studentAssessment.assessment_type != "Exam"]

assessments_pre_exam = assessments[assessments.assessment_type != "Exam"]


# Adding data
for i in range(studentInfo.shape[0]):
    # Variables
    mod = studentInfo.code_module.iloc[i]
    stu = studentInfo.id_student.iloc[i]
    pres = studentInfo.code_presentation.iloc[i]
    # Module presentation length
    length = courses[(courses.code_module==mod)&(courses.code_presentation==pres)].module_presentation_length.iloc[0]
    # student VLE corresponding to the 3 variables
    stuVLE = nStudentVle[(nStudentVle.code_module == mod) & (nStudentVle.id_student == stu) & (nStudentVle.code_presentation == pres)]
    stuVLE = stuVLE.sort_values(by=["date"])
    # module VLE
    modVLE = vle[(vle.code_module==mod) & (vle.code_presentation==pres)]
    # Registration
    reg = studentRegistration[(studentRegistration.code_module == mod) & (studentRegistration.id_student == stu) & (studentRegistration.code_presentation == pres)]
    # Assessments
    assessmentsMod = assessments_pre_exam[(assessments_pre_exam.code_module == mod) & (assessments_pre_exam.code_presentation == pres)]
    assessmentsMod = assessmentsMod.sort_values(by=["date"])
    assessmentsStu = studentAssessments[(studentAssessments.id_student == stu) & (studentAssessments.code_module == mod) & (studentAssessments.code_presentation == pres)]
    assessmentsStu = assessmentsStu.sort_values(by=["date"])
    # Liste date
    date = list(stuVLE["date"]) + list(assessmentsStu["date"])
    date = list(dict.fromkeys(date))
    date.sort()
    date = list(range(min(0, stuVLE.date.min()), min(length, reg.date_unregistration.iloc[0])))
    # Pour chaque date
    for dt in date:
        # Recherche activité et notes
        activity = stuVLE[stuVLE.date <= dt]
        activityL10d = stuVLE[(stuVLE.date <= dt) & (stuVLE.date >= dt-10)]
        activityL20d = stuVLE[(stuVLE.date <= dt) & (stuVLE.date >= dt-20)]
        notes = assessmentsStu[assessmentsStu.date <= dt]
        notesMod = assessmentsMod[assessmentsMod.date <= dt]
        modVLE_active = modVLE[((modVLE.week_from <= dt) | (modVLE.week_from == 0)) & (modVLE.week_to >= dt)]
        # Dictionnaire
        dic["code_module"].append(mod)
        dic["id_student"].append(stu)
        dic["code_presentation"].append(pres)
        dic["date"].append(dt/length*100)
        dic["nb_total_click"].append(activity.nb_click_total.sum())
        dic["nb_total_click_last10D"].append(activityL10d.nb_click_total.sum())
        dic["perc_ressources_total"].append(len(activity.id_site.unique())/len(modVLE_active.id_site.unique())*100)
        dic["perc_ressources_last10D"].append(len(activityL10d.id_site.unique())/len(modVLE_active.id_site.unique())*100)
        dic["click_activity_evolution"].append(activityL10d.nb_click_total.sum()/(activityL20d.nb_click_total.sum()-activityL10d.nb_click_total.sum()+1))
        dic["ressources_activity_evolution"].append(len(activityL10d.id_site.unique())/(len(activityL20d.id_site.unique())-len(activityL10d.id_site.unique())+1))
        if notesMod.shape[0] != 0:
            dic["perc_assignements_late"].append(notes.is_late.sum()/notesMod.shape[0])
            dic["perc_not_returned_ass"].append((notesMod.shape[0]-notes.shape[0])/notesMod.shape[0])
        else:
            dic["perc_assignements_late"].append(0)
            dic["perc_not_returned_ass"].append(0)
        dic['nb_active_days'].append(date.index(dt))
        if dt <= 0:
            dic['perc_active_days'].append(0)
        else:
            dic['perc_active_days'].append(date.index(dt)/dt)
        # Statistics with ressources - id_site
        if activity.shape[0] != 0:
            dic['1st_day'].append(min(activity.date))
            dic['last_day'].append(max(activity.date))
            act = activity.groupby(["code_module", "code_presentation", "id_student", "date"]).agg(
                {
                    'id_site' : ["count"]
                }
            )
            act.columns = ['nb_ressources']
            act = act.reset_index()
            dic['mean_nb_ressources'].append(act.nb_ressources.mean())
            dic['max_nb_ressources'].append(act.nb_ressources.max())
            dic['min_nb_ressources'].append(act.nb_ressources.min())
            if act.shape[0] > 1:
                dic['std_nb_ressources'].append(act.nb_ressources.std())
            else:
                dic['std_nb_ressources'].append(0)
        else:
            dic['1st_day'].append(np.nan)
            dic['last_day'].append(np.nan)
            dic['mean_nb_ressources'].append(0)
            dic['max_nb_ressources'].append(0)
            dic['min_nb_ressources'].append(0)
            dic['std_nb_ressources'].append(0)
        # Statistics with number of clicks - general
        act = activity.groupby(["code_module", "code_presentation", "id_student", "date"]).agg(
                {
                    'nb_click_total' : ["sum"]
                }
            )
        act.columns = ['nb_total_click']
        act = act.reset_index()
        if act.shape[0] > 0:
            dic['mean_click_active_day'].append(act.nb_total_click.mean())
            if dt <= 0:
                dic['mean_click_day'].append(act.nb_total_click.mean())
            else:
                dic['mean_click_day'].append(act.nb_total_click.sum()/dt)
        else:
            dic['mean_click_active_day'].append(0)
            dic['mean_click_day'].append(0)
        # Statistics with number of clicks - last 10 days
        act = act.sort_values(by=["date"])
        act_last10 = act.iloc[range(max(0, act.shape[0]-10), act.shape[0])]
        act = act[act.date >= dt-10]
        if act.shape[0] > 0:
            dic['mean_click_active_day_10'].append(act_last10.nb_total_click.mean())
            if dt <= 0:
                dic['mean_click_day_10'].append(act.nb_total_click.mean())
            else:
                dic['mean_click_day_10'].append(act.nb_total_click.sum()/10)
        else:
            dic['mean_click_day_10'].append(0)
            dic['mean_click_active_day_10'].append(0)
        # Assignments - notes
        if notesMod.shape[0] == 0:
            dic['mean_assignments'].append(50)
            dic['std_assignments'].append(0)
            dic['validation'].append(1)
        else:
            nt = notesMod.drop(["assessment_type", "code_module", "code_presentation", "date"], axis=1).merge(notes, on=["id_assessment", "weight"], how="outer")
            nt = nt.fillna(0)
            dic['mean_assignments'].append(nt.weighted_score.sum()/(nt.weight.sum()/100))
            if nt.shape[0] > 1:
                dic['std_assignments'].append(nt.score.std())
            else:
                dic['std_assignments'].append(0)
            if nt.weighted_score.sum()/(nt.weight.sum()/100) >= 50:
                dic['validation'].append(1)
            else:
                dic['validation'].append(0)

data2 = pd.DataFrame(dic)
data2.to_csv("data_time.csv", index=False, header=True)


data_time_agg = data2.groupby(["id_student", "code_module", "code_presentation"]).agg({"id_student":["count"]})
data_time_agg.columns = ["nb"]
data_time_agg = data_time_agg.reset_index()
data_time_agg.shape

dic_not_present = {"id_student":[], "code_presentation":[], "code_module":[]}

for i in range(studentInfo.shape[0]):
    if data_time_agg[(data_time_agg.id_student==studentInfo.iloc[i].id_student)&(data_time_agg.code_presentation==studentInfo.iloc[i].code_presentation)&(data_time_agg.code_module==studentInfo.iloc[i].code_module)].shape[0] == 0:
        dic_not_present["id_student"].append(studentInfo.iloc[i].id_student)
        dic_not_present["code_module"].append(studentInfo.iloc[i].code_module)
        dic_not_present["code_presentation"].append(studentInfo.iloc[i].code_presentation)

not_present = pd.DataFrame(dic_not_present)
not_present = not_present.merge(studentInfo, on=["id_student", "code_module", "code_presentation"], how="inner")

for i in range(not_present.shape[0]):
    # Variables
    mod = not_present.code_module.iloc[i]
    stu = not_present.id_student.iloc[i]
    pres = not_present.code_presentation.iloc[i]
    # Module presentation length
    length = courses[(courses.code_module==mod)&(courses.code_presentation==pres)].module_presentation_length.iloc[0]
    # student VLE corresponding to the 3 variables
    stuVLE = nStudentVle[(nStudentVle.code_module == mod) & (nStudentVle.id_student == stu) & (nStudentVle.code_presentation == pres)]
    stuVLE = stuVLE.sort_values(by=["date"])
    # module VLE
    modVLE = vle[(vle.code_module==mod) & (vle.code_presentation==pres)]
    # Registration
    reg = studentRegistration[(studentRegistration.code_module == mod) & (studentRegistration.id_student == stu) & (studentRegistration.code_presentation == pres)]
    # Assessments
    assessmentsMod = assessments_pre_exam[(assessments_pre_exam.code_module == mod) & (assessments_pre_exam.code_presentation == pres)]
    assessmentsMod = assessmentsMod.sort_values(by=["date"])
    assessmentsStu = studentAssessments[(studentAssessments.id_student == stu) & (studentAssessments.code_module == mod) & (studentAssessments.code_presentation == pres)]
    assessmentsStu = assessmentsStu.sort_values(by=["date"])
    # Dic
    dic["code_module"].append(mod)
    dic["id_student"].append(stu)
    dic["code_presentation"].append(pres)
    dic["date"].append(0)
    dic["nb_total_click"].append(stuVLE.nb_click_total.sum())
    dic["nb_total_click_last10D"].append(stuVLE.nb_click_total.sum())
    dic["perc_ressources_total"].append(len(stuVLE.id_site.unique())/len(modVLE.id_site.unique())*100)
    dic["perc_ressources_last10D"].append(len(stuVLE.id_site.unique())/len(modVLE.id_site.unique())*100)
    dic["click_stuVLE_evolution"].append(stuVLE.nb_click_total.sum()/(stuVLE.nb_click_total.sum()-stuVLE.nb_click_total.sum()+1))
    dic["ressources_stuVLE_evolution"].append(len(stuVLE.id_site.unique())/(len(stuVLE.id_site.unique())-len(stuVLE.id_site.unique())+1))
    if assessmentsMod.shape[0] != 0:
        dic["perc_assignements_late"].append(assessmentsStu.is_late.sum()/assessmentsMod.shape[0])
        dic["perc_not_returned_ass"].append((assessmentsMod.shape[0]-assessmentsStu.shape[0])/assessmentsMod.shape[0])
    else:
        dic["perc_assignements_late"].append(0)
        dic["perc_not_returned_ass"].append(0)
    dic['nb_active_days'].append(0)
    dic['perc_active_days'].append(0)
    # Statistics with ressources - id_site
    if stuVLE.shape[0] != 0:
        dic['1st_day'].append(min(stuVLE.date))
        dic['last_day'].append(max(stuVLE.date))
        act = stuVLE.groupby(["code_module", "code_presentation", "id_student", "date"]).agg(
            {
                'id_site' : ["count"]
            }
        )
        act.columns = ['nb_ressources']
        act = act.reset_index()
        dic['mean_nb_ressources'].append(act.nb_ressources.mean())
        dic['max_nb_ressources'].append(act.nb_ressources.max())
        dic['min_nb_ressources'].append(act.nb_ressources.min())
        if act.shape[0] > 1:
            dic['std_nb_ressources'].append(act.nb_ressources.std())
        else:
            dic['std_nb_ressources'].append(0)
    else:
        dic['1st_day'].append(np.nan)
        dic['last_day'].append(np.nan)
        dic['mean_nb_ressources'].append(0)
        dic['max_nb_ressources'].append(0)
        dic['min_nb_ressources'].append(0)
        dic['std_nb_ressources'].append(0)
    # Statistics with number of clicks - general
    act = stuVLE.groupby(["code_module", "code_presentation", "id_student", "date"]).agg(
            {
                'nb_click_total' : ["sum"]
            }
        )
    act.columns = ['nb_total_click']
    act = act.reset_index()
    if act.shape[0] > 0:
        dic['mean_click_active_day'].append(act.nb_total_click.mean())
        dic['mean_click_day'].append(act.nb_total_click.mean())
    else:
        dic['mean_click_active_day'].append(0)
        dic['mean_click_day'].append(0)
    # Statistics with number of clicks - last 10 days
    act = act.sort_values(by=["date"])
    act_last10 = act.iloc[range(max(0, act.shape[0]-10), act.shape[0])]
    if act.shape[0] > 0:
        dic['mean_click_active_day_10'].append(act_last10.nb_total_click.mean())
        dic['mean_click_day_10'].append(act.nb_total_click.mean())
    else:
        dic['mean_click_day_10'].append(0)
        dic['mean_click_active_day_10'].append(0)
    # Assignments - assessmentsStu
    if assessmentsMod.shape[0] == 0:
        dic['mean_assignments'].append(50)
        dic['std_assignments'].append(0)
        dic['validation'].append(1)
    else:
        nt = assessmentsMod.drop(["assessment_type", "code_module", "code_presentation", "date"], axis=1).merge(assessmentsStu, on=["id_assessment", "weight"], how="outer")
        nt = nt.fillna(0)
        dic['mean_assignments'].append(nt.weighted_score.sum()/(nt.weight.sum()/100))
        if nt.shape[0] > 1:
            dic['std_assignments'].append(nt.score.std())
        else:
            dic['std_assignments'].append(0)
        if nt.weighted_score.sum()/(nt.weight.sum()/100) >= 50:
            dic['validation'].append(1)
        else:
            dic['validation'].append(0)


## ENREGISTREMENT DU DATAFRAME

data2 = pd.DataFrame(dic)
data2.to_csv("data_time2.csv", index=False, header=True)