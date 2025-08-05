from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def preprocess_data(dataset):

    le = LabelEncoder()

    X = dataset.drop(["face_side"], axis=1)
    y = dataset['face_side']

    y = le.fit_transform(y)

    # 1. Optional: Apply train-tests split (if you still want a tests set)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1, shuffle=True)

    return X_train,X_test,y_train,y_test,le