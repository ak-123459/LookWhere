# Auto Face Pose Capture using Logistic Regression

## 🚀 Problem Statement

In our computer vision application, we wanted to **add a person’s face data into a database** by capturing it from a camera. Typically, this requires capturing images from multiple facial angles manually — such as:
- Left profile
- Right profile
- Upward (top) view
- Downward (bottom) view
- Center (frontal) view

Manually clicking images while instructing a person to rotate their head is time-consuming and error-prone.

### 🎯 Objective

**Automate the face pose capturing process** using machine learning. Once the camera starts, the model should automatically identify the current head position and capture the face when it matches any of the five desired directions.

---

## 🧠 ML Solution

To solve this, we used **InsightFace's face pose estimation** features — extracting 3 key pose parameters:
- `yaw`
- `roll`
- `pitch`

Using these, we created a dataset where each face image was labeled based on its pose direction. Then, we trained a **Logistic Regression model** using a total of **150 features** derived from the face landmarks and pose data.

---

## 📊 Model Training & Evaluation

- ✅ Model: **Logistic Regression**
- 📈 Accuracy: **~90%**
- 🔢 Features Used: **150 pose-related features**
- 🧪 Evaluation: Based on test set performance and live camera testing.

---

## 📦 Project Structure
``` 
 ├──app
    ├── captured_face_sides/            # all captured face images
    ├── configs/        # all configuration
    ├── data/           # all data related scripts
    ├── models/      
    ├── src/               
    ├── ..
    ├── .. 
    ├── live.py
    ├── main.py
           
└── README.md            # Project documentation
```


---

## 🎥 Demo Video

Watch the project demo here: [Demo Video on YouTube](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)


## 🎥 Real-time Demo

Run the following to test the model on a live webcam feed:

```bash
git clone https://github.com/ak-123459/LookWhere.git
```

```bash
cd LookWhere
```

```bash
pip install -r requirements.txt
````


```bash
python app/live.py
```


### Train and validate the model

````
python app/main.py
````


## 🧠 Model
**Algorithm**: Logistic Regression

**Input**: Pose features (yaw, pitch, roll, etc.)

**Output**: Face direction label

## 📂 Configuration
Hyperparameters and model configs can be found and edited in config.yaml.

## 📃 License
This project is licensed under the **MIT** License.




