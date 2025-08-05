import joblib
import numpy as np
import time
import logging
from insightface.app import FaceAnalysis
import cv2
import albumentations as A
import os
from configs.config import FACEINSIGHT_PARAMS
from utils import beep ,error_beep
from configs.config import CAPTURED_IMAGES_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to console
        # Uncomment below to log to file:
        # logging.FileHandler("pose_predictor.log", mode='w')
    ]
)




class FacePosePredictor:

    def __init__(self, model_path, label_encoder_path, ctx_id=0):
        logging.info("Initializing FacePosePredictor...")

        # Load trained model
        start_time = time.time()
        self.model = joblib.load(model_path)
        logging.info(f"Model loaded from '{model_path}' in {time.time() - start_time:.3f} seconds")

        # Load label encoder
        start_time = time.time()
        self.label_encoder = joblib.load(label_encoder_path)
        logging.info(f"LabelEncoder loaded from '{label_encoder_path}' in {time.time() - start_time:.3f} seconds")

        # Initialize face analysis model
        logging.info("Initializing InsightFace model...")
        self.app = FaceAnalysis(name=FACEINSIGHT_PARAMS['name'])
        self.app.prepare(ctx_id=ctx_id)
        self.step = 0
        self.face_sides = {}
        self.save_img_path = CAPTURED_IMAGES_PATH
        logging.info("InsightFace model initialized.")



    @staticmethod
    def apply_augmentations(image,is_augmented: bool=False,save_path:str="./",image_name:str="") -> bool:

        # Define augmentations
        augmentations = {
            "flip_image1": A.HorizontalFlip(p=1.0),
            "brightness_contrast_image2": A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.4, p=1.0),
            "gaussian_blur_image3": A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            "affine_transform_image4": A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=1.0),
            "noise_image5": A.GaussNoise(var_limit=(10.0, 70.0), p=1.0),
            "clahe_image6": A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            "coarse_dropout_image7": A.CoarseDropout(max_holes=2, max_height=8, max_width=8, min_holes=1, fill_value=0,
                                                     p=1.0),
            "grid_distortion_image8": A.GridDistortion(num_steps=3, distort_limit=0.1, p=1.0)
        }

        os.makedirs(save_path, exist_ok=True)

        cv2.imwrite(save_path+image_name,image)

        # Apply all augmentations and collect results
        augmented_images = []
        for name, transform in augmentations.items():

            image_name = name+image_name

            if save_path and image_name:
                save_full_path = os.path.join(save_path, image_name)


            try:

                augmented = transform(image=image)["image"]
                cv2.imwrite(save_full_path,augmented)
                is_augmented = True

            except:

                print(f"Failed to apply {name}. Skipping.")

                continue

        return is_augmented

    @staticmethod
    def save_captured_frame(image,save_path:str="./",image_name:str=""):

        os.makedirs(save_path, exist_ok=True)

        cv2.imwrite(save_path + image_name, image)

        return image


    def predict_pose(self, frame):

            logging.debug("Running face detection...")
            start_time = time.time()

            faces = self.app.get(frame)

            if not faces:
                logging.info("No face detected in the frame.")
                return frame, None

            detection_time = time.time() - start_time
            logging.debug(f"Face detection completed in {detection_time:.3f} seconds")

            if(len(faces) ==1):

                face = faces[0]
                yaw, pitch, roll = face.pose
                features = np.array([[yaw, pitch, roll]])

                # Prediction
                logging.debug("Making prediction...")
                start_pred = time.time()
                pred_class_index = self.model.predict(features)[0]
                pred_label = self.label_encoder.inverse_transform([pred_class_index])[0]
                pred_prob = self.model.predict_proba(features)[0]
                confidence = round(np.max(pred_prob),2)
                pred_time = time.time() - start_pred
                logging.info(f"Prediction: {pred_label} (Confidence: {confidence:.2f}) in {pred_time:.3f} seconds")
                logging.debug(f"Pose: Yaw={yaw:.2f}, Pitch={pitch:.2f}, Roll={roll:.2f}")

                if(confidence>0.80 and pred_label not in self.face_sides):


                    frame = self.save_captured_frame(frame,save_path=self.save_img_path,image_name=pred_label+".jpg")

                    if(frame is None):

                        return

                    self.face_sides.update({pred_label: frame})  # add each aug image in dict
                    self.step += 1  # increment step
                    beep()  # play a beep sound on step done



                stats = {
                    "Status": "Running",
                    "Steps Completed":self.step,
                    "Detected": pred_label,
                    "Confidence":confidence,
                }

                frame = self._draw_texts(frame,stats)

                return frame, pred_label

            else:
                 error_beep()

                 return frame,None



    @staticmethod
    def _draw_texts(frame, stats_dict, position=(10, 10), width=250, height=120):
        overlay = frame.copy()
        cv2.rectangle(overlay, position, (position[0] + width, position[1] + height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        y_offset = 20
        color = (0, 255, 0) if stats_dict["Confidence"] > 0.8 else (0, 0, 255)

        for i, (label, val) in enumerate(stats_dict.items()):
            text = f"{label}: {val}"
            cv2.putText(frame, text, (position[0] + 10, position[1] + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            y_offset += 20

        return frame



    def run_webcam(self):
        logging.info("Starting webcam...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            logging.error("Error: Cannot open webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to read frame from webcam.")
                break

            annotated_frame, _ = self.predict_pose(frame)

            if(self.step==5):
                break

            cv2.imshow("Live Prediction", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Quitting webcam loop.")
                break

        cap.release()
        cv2.destroyAllWindows()
        logging.info("Webcam released and windows closed.")
