from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib
from app.configs.config import MODEL_PATH, ENCODER_PATH, CV_SPLITS, RANDOM_STATE, LOGISTIC_PARAMS


def train_model(X_train, y_train, label_encoder):

    model = LogisticRegression(**LOGISTIC_PARAMS)

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

    print("Cross-validation scores:", cv_scores)
    print("Mean CV Accuracy:", cv_scores.mean())

    model.fit(X_train, y_train)

    # Save model and encoder
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    return model
