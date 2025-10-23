
import os
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from insightface.app import FaceAnalysis


class FacePoseDatasetBuilder:
    def __init__(self, image_dir, dataset_path="face_pose_dataset.csv", show_images=False, device_id=0):
        """
        Initializes the dataset builder.

        Args:
            image_dir (str): Path to folder containing images.
            dataset_path (str): Path to CSV dataset file.
            show_images (bool): Whether to display annotated images.
            device_id (int): Device ID for InsightFace (0 = CPU, >0 = GPU).
        """
        self.image_dir = Path(image_dir)
        self.dataset_path = Path(dataset_path)
        self.show_images = show_images
        self.device_id = device_id
        self.app = self._load_model()
        self.dataset = self._load_or_create_dataset()

    # -------------------- Model Loading --------------------
    def _load_model(self):
        print("🔹 Loading InsightFace model (buffalo_s)...")
        app = FaceAnalysis(name="buffalo_s")
        app.prepare(ctx_id=self.device_id)
        print("✅ Model loaded successfully.")
        return app

    # -------------------- Dataset Handling --------------------
    def _load_or_create_dataset(self):
        if self.dataset_path.exists():
            df = pd.read_csv(self.dataset_path)
            print(f"✅ Loaded existing dataset with {len(df)} records.")
        else:
            df = pd.DataFrame(columns=["yaw", "pitch", "roll", "face_side"])
            print("📄 No existing dataset found — starting new one.")
        return df

    def _save_dataset(self):
        self.dataset.to_csv(self.dataset_path, index=False)
        print(f"💾 Dataset saved to {self.dataset_path} ({len(self.dataset)} total records).")

    # -------------------- Helper Methods --------------------
    @staticmethod
    def _get_face_side(filename):
        # Example: RF_001.jpg → RF
        return filename.split('_')[0] if '_' in filename else "Unknown"

    def _process_image(self, img_path):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Could not read {img_path}")
            return None

        faces = self.app.get(img)
        if not faces:
            print(f"🚫 No face detected in {img_path.name}")
            return None

        face = faces[0]
        yaw, pitch, roll = face.pose

        if self.show_images:
            self._display_face(img, face, yaw, pitch, roll)

        return yaw, pitch, roll

    def _display_face(self, img, face, yaw, pitch, roll):
        """Display bounding box and pose information on image."""
        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Yaw: {yaw:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, f"Pitch: {pitch:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, f"Roll: {roll:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Detected Face", img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

    # -------------------- Main Dataset Builder --------------------
    def build(self):
        images = [f for f in self.image_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        print(f"📸 Found {len(images)} images in {self.image_dir}")

        for img_path in tqdm(images, desc="Processing images"):
            face_side = self._get_face_side(img_path.stem)
            result = self._process_image(img_path)
            if result:
                yaw, pitch, roll = result
                new_row = {"yaw": yaw, "pitch": pitch, "roll": roll, "face_side": face_side}
                self.dataset = pd.concat([self.dataset, pd.DataFrame([new_row])], ignore_index=True)

        self._save_dataset()



# -------------------- Run the Pipeline --------------------
if __name__ == "__main__":
    image_folder = r"D:\\video_frames"
    output_csv = "D:\\video_frames\\yaw_pitch_roll_dataset.csv"

    builder = FacePoseDatasetBuilder(image_folder, output_csv, show_images=False, device_id=0)
    builder.build()
