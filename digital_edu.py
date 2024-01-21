import pandas as pd
df=pd.read_csv('train.csv')
df.drop(['id','bdate','has_photo','has_mobile','followers_count','graduation','relation','life_main','people_main','city','last_seen','occupation_name','career_start','langs','career_end'],axis=1,inplace=True)

df['education_form'].fillna('Full-time',inplace=True)
df[list(pd.get_dummies(df['education_form']).columns)]=pd.get_dummies(df['education_form'])
df.drop(['education_form'],axis=1,inplace=True)
def sex_apply(sex):
    if sex == 2:
        return 0 
    return 1
df['sex'] = df['sex'].apply(sex_apply)

print(df['education_status'].value_counts())

def edu_status_apply(edu_status):
    if edu_status == 'Undergraduate applicant':
        return 0
    elif edu_status == "Student (Bochelor's)" or edu_status == "Student (Specialist)" or edu_status == "Student(Master's)":
        return 1
    elif edu_status == "Alumnus (Bochelor's)" or edu_status == "Alumnus (Specialist)" or edu_status == "Alumnus(Master's)":
        return 2
    elif edu_status == 'Condidate of Sciences':
        return 3
    else:
        return 4

df['education_status'] = df['education_status'].apply(edu_status_apply)

def Langs_apply(Langs):
    if langs.find('Русский') != -1:
        return 1
    return 0
df['langs'] = df['langs'].apply(Langs_apply)
df['occupation_type'].fillna('university',inplace = True)
def ocu_type_apply(ocu_type):
    if ocu_type == 'university':
        return 0
    return 1
df['occupation_type']=df['occupation_type'].apply(ocu_type_apply)
print(df['occupation_type'].value_counts())    
df.info()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

X= df.drop('result',axis=1)
y=df['result']
X_train,X_test,y_train, y_test=train_test_split(X,y,test_size=0.1)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

classifier=KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
print(y_test)
print(y_pred)
