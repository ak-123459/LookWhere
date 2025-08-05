from configs.config import MODEL_PATH,ENCODER_PATH
from camera_inference import FacePosePredictor
import os

print("current cwd:-",os.getcwd())

if __name__ == "__main__":

    predictor = FacePosePredictor(
        model_path=MODEL_PATH,
        label_encoder_path=ENCODER_PATH,
        ctx_id=0  # Set to 0 for CPU or >0 for GPU
    )
    predictor.run_webcam()
