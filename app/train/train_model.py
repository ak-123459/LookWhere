
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,cross_val_score
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import pickle
import joblib
from sklearn.model_selection import GridSearchCV


dataset = pd.read_csv("C:\\Users\\techma\\OneDrive\Documents\\face_angle_dataset.csv")

# Assuming your dataset is already loaded into `dataset`
# dataset = dataset.sample(n=130, random_state=1)

# 2. Initialize Logistic Regression model
model = LogisticRegression(max_iter=200,C= 0.01, penalty= "l2", solver='lbfgs')

# 3. Cross-validation using StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

print("Cross-validation scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# 4. Fit and evaluate on tests set (optional)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\nEvaluation on Test Set:")
print("Test size:", len(y_test))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Recall Score:", recall_score(y_test, y_pred, average="weighted"))
print("Precision Score:", precision_score(y_test, y_pred, average="weighted"))
print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot()


joblib.dump(model, '../pretrained_model/logistic_model.joblib')

joblib.dump(le, '../pretrained_model/label_encoder.joblib')




param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']
}

grid = GridSearchCV(LogisticRegression(max_iter=200), param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best cross-validation score:", grid.best_score_)




