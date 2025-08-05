# Paths
DATASET_PATH="app/data/preprocessed/face_angle_dataset.csv"
MODEL_PATH="app/models/face_side_det/logistic_model.joblib"
ENCODER_PATH="app/models/face_side_det/label_encoder.joblib"
CAPTURED_IMAGES_PATH = "app/captured_images/"



# Splitting
TEST_SIZE = 0.15
RANDOM_STATE = 32
CV_SPLITS = 5



# Model hyperparameters
LOGISTIC_PARAMS = {
    'max_iter': 200,
    'C': 0.01,
    'penalty': 'l2',
    'solver': 'lbfgs'
}



# Grid Search tuning space
PARAM_GRID = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']
}



FACEINSIGHT_PARAMS = {'name':"buffalo_s"}