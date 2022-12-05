import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

train = pd.read_csv('data/heart_train.csv')
test = pd.read_csv('data/heart_test.csv')

# Train the Model 
clf = LogisticRegression(penalty='l2', C=0.1)
clf.fit(train.drop('target', axis = 1), train['target'])
y_pred = clf.predict(test.drop('target', axis = 1))
y_pred_proba = clf.predict_proba(test.drop('target', axis = 1))[::,1]

# Test the model using AOC-ROC Graph
def auc_roc_plot(y_test, y_pred_proba):
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.title('AUC-ROC plot')
    plt.savefig('model_results.png', dpi=120)

def accuracy(y_test, y_pred):
  """
  Calculuates accuracy y_test and y_preds.
  """
  return metrics.accuracy_score(y_test, y_pred)

auc_roc_plot(test['target'],y_pred_proba)  
accuracy = accuracy(test['target'], y_pred)
print('Accuracy Score: ',accuracy)

# Write metrics to file
with open('metrics.txt', 'w') as outfile:
    outfile.write(f'\nAccuracy Score = {accuracy}.')