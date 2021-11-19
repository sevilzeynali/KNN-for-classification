# KNN Classification

It is a  Python program for knn classification of diabetes dataset which is available [here](https://www.kaggle.com/amolbhivarkar/knn-for-classification-using-scikit-learn/data?select=diabetes.csv)

## Installing and requirements
You need to install :

 - Python
 - Pandas
 - Matplotlib
 - Seaborn
 - Sklearn
 - MLflow
  
## How does it work
To run this program you shoud do in a terminal or conda environment
```
knn_for_classification.py
 ```
 for tracking the model with MLflow you can type this localhost in your browser:
 ```
 http://localhost:5000
 ```
 ## Data set
 Data set is available [here](https://www.kaggle.com/amolbhivarkar/knn-for-classification-using-scikit-learn/data?select=diabetes.csv) in Kaggle.
## More information
This program uses GridSearchCV to select the best Hyperparameter 's of KNN classifier.
Classification report of this model is :
 ```
					precision    recall  f1-score   support

           0       0.77      0.88      0.82        96
           1       0.73      0.57      0.64        58

    accuracy                           0.76       154
   macro avg       0.75      0.72      0.73       154
weighted avg       0.76      0.76      0.75       154

 ```
 
 
 

