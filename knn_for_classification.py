import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import mlflow

#This data set is available here: https://www.kaggle.com/amolbhivarkar/knn-for-classification-using-scikit-learn/data?select=diabetes.csv

df=pd.read_csv("diabetes.csv")
df.head(10)

df.info()
df.corr()
df.describe()

corr_df = df.corr(method='pearson')

plt.matshow(corr_df)
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_df, annot=True)
plt.show()

data=df.drop("Outcome",axis=1)
target=df["Outcome"]

data.head()
target.head()

X_train,X_test,y_train,y_test=train_test_split(data,target,test_size=0.2,random_state=123)

#define the model and parameters
knn = KNeighborsClassifier()

parameters = {"n_neighbors":range(1,10),
              "leaf_size":[1,3,5],
              "algorithm":["auto", "ball_tree", "kd_tree", "brute"],
              "n_jobs":[-1],
              "metric":["euclidean","manhattan","minkowski"]}
#Fit the model
model = GridSearchCV(knn, param_grid=parameters)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("knn_diabet")

with mlflow.start_run() as run:    
    model.fit(X_train,y_train)   
    mlflow.sklearn.autolog()
    metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="val_")
    y_pred = model.predict(X_test)
    fscore=f1_score(y_test,y_pred)   
    mlflow.log_metric("f1-score",fscore)
    model_name = "knn_model"
    artifact_path="artifacts"                     
    mlflow.sklearn.log_model(sk_model=model,artifact_path=artifact_path,registered_model_name=model_name)               
    runID=run.info.run_uuid
    mlflow.register_model("runs:/"+runID+"/"+artifact_path,"knn")
    
prediction=model.predict(X_test)

print(model.best_params_)
print(model.score(X_test,y_test))
print(model.best_score_)
print(model.best_index_)
print(model.cv_results_)
print(sorted(model.cv_results_.keys()))

y_pred = model.predict(X_test)
print(confusion_matrix(y_test,y_pred))

print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted']))

print(classification_report(y_test,y_pred))
