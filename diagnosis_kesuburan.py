import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Dataset_1/fertility.csv')
X = df.drop(['Season', 'Diagnosis'], axis = 1)
X = pd.get_dummies(
    X, columns = ['Childish diseases', 'Accident or serious trauma',
    'Surgical intervention', 'High fevers in the last year',
    'Frequency of alcohol consumption', 'Smoking habit'],
    drop_first = True
)
y = df['Diagnosis']

# print(y.value_counts())

# no need split train and test data due to the number of data observation is small
# and imbalance Diagnosis Normal = 88, and Diagonisis Altered = 12

# method 1. Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver='liblinear')
log_reg.fit(X, y)

# print('Logistic Regression Accuracy = {}%'.format(round(log_reg.score(X, y) * 100, 2)))

# method 2. SVM
from sklearn.svm import SVC
svm = SVC(kernel = 'rbf', gamma = 0.5)
svm.fit(X, y)

# print('SVM Accuracy = {}%'.format(round(svm.score(X, y) * 100, 2)))

# method 3. Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators = 25, criterion = 'entropy')
rf_clf.fit(X, y)

# print('Random Forest Classifier Accuracy = {}%'.format(round(rf_clf.score(X, y) * 100, 3)))

# ['Age', 'Number of hours spent sitting per day', 'Childish diseases_yes',
#        'Accident or serious trauma_yes', 'Surgical intervention_yes',
#        'High fevers in the last year_more than 3 months ago',
#        'High fevers in the last year_no',
#        'Frequency of alcohol consumption_hardly ever or never',
#        'Frequency of alcohol consumption_once a week',
#        'Frequency of alcohol consumption_several times a day',
#        'Frequency of alcohol consumption_several times a week',
#        'Smoking habit_never', 'Smoking habit_occasional']

# Arin prediction
arin_atribute = [29, 5, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
arin_log_reg = log_reg.predict([arin_atribute])
arin_svm = svm.predict([arin_atribute])
arin_rf_clf = rf_clf.predict([arin_atribute])
print('Arin, prediksi kesuburan: {} (Logistic Regession)'.format(arin_log_reg[0]))
print('Arin, prediksi kesuburan: {} (SVM)'.format(arin_svm[0]))
print('Arin, prediksi kesuburan: {} (Random Forest Classifier)'.format(arin_rf_clf[0]))
print(' ')

# Bebi prediction
bebi_atribute = [31, 24, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0]
bebi_log_reg = log_reg.predict([bebi_atribute])
bebi_svm = svm.predict([bebi_atribute])
bebi_rf_clf = rf_clf.predict([bebi_atribute])
print('Bebi, prediksi kesuburan: {} (Logistic Regession)'.format(bebi_log_reg[0]))
print('Bebi, prediksi kesuburan: {} (SVM)'.format(bebi_svm[0]))
print('Bebi, prediksi kesuburan: {} (Random Forest Classifier)'.format(bebi_rf_clf[0]))
print(' ')

# Caca prediction
caca_atribute = [25, 7, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
caca_log_reg = log_reg.predict([caca_atribute])
caca_svm = svm.predict([caca_atribute])
caca_rf_clf = rf_clf.predict([caca_atribute])
print('Caca, prediksi kesuburan: {} (Logistic Regession)'.format(caca_log_reg[0]))
print('Caca, prediksi kesuburan: {} (SVM)'.format(caca_svm[0]))
print('Caca, prediksi kesuburan: {} (Random Forest Classifier)'.format(caca_rf_clf[0]))
print(' ')

# Dini prediction
dini_atribute = [28, 24, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
dini_log_reg = log_reg.predict([dini_atribute])
dini_svm = svm.predict([dini_atribute])
dini_rf_clf = rf_clf.predict([dini_atribute])
print('Dini, prediksi kesuburan: {} (Logistic Regession)'.format(dini_log_reg[0]))
print('Dini, prediksi kesuburan: {} (SVM)'.format(caca_svm[0]))
print('Dini, prediksi kesuburan: {} (Random Forest Classifier)'.format(dini_rf_clf[0]))
print(' ')

# Enno prediction
enno_atribute = [42, 8, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0]
enno_log_reg = log_reg.predict([enno_atribute])
enno_svm = svm.predict([enno_atribute])
enno_rf_clf = rf_clf.predict([enno_atribute])
print('Enno, prediksi kesuburan: {} (Logistic Regession)'.format(enno_log_reg[0]))
print('Enno, prediksi kesuburan: {} (SVM)'.format(enno_svm[0]))
print('Enno, prediksi kesuburan: {} (Random Forest Classifier)'.format(enno_rf_clf[0]))
print(' ')

# Dummy prediction (Case Altered target take one from dataset (data index-1))
dummy_atribute = [35, 6, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
dummy_log_reg = log_reg.predict([dummy_atribute])
dummy_svm = svm.predict([dummy_atribute])
dummy_rf_clf = rf_clf.predict([dummy_atribute])
print('Dummy (sebut saja Bunga), prediksi kesuburan: {} (Logistic Regession)'.format(dummy_log_reg[0]))
print('Dummy (sebut saja Bunga), prediksi kesuburan: {} (SVM)'.format(dummy_svm[0]))
print('Dummy (sebut saja Bunga), prediksi kesuburan: {} (Random Forest Classifier)'.format(dummy_rf_clf[0]))