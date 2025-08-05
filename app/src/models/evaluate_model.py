from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)

    print("Evaluation on Test Set:")
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("Recall Score:", recall_score(y_test, y_pred, average="weighted"))
    print("Precision Score:", precision_score(y_test, y_pred, average="weighted"))
    print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot()
    plt.show()
