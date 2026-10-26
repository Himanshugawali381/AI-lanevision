
import os

# Set before importing TensorFlow.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO


BACKEND_DIR = Path(__file__).resolve().parent
MODEL_DIR = BACKEND_DIR / "models"
DATASET_DIR = Path(
    r"C:\Users\LENOVO\Desktop\lane-images\ai-project\dataset"
).resolve()
ORIGINAL_DIR = DATASET_DIR / "original_images"

# New inference outputs are kept separate from the existing dataset results.
OUTPUT_DIR = DATASET_DIR / "live_inference"


class SteeringAnglePredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(
            str(model_path),
            {
                "Custom>scaled_atan": lambda x: 2 * tf.atan(x),
            },
            compile=False,
        )

    def predict_angle(self, img):
        img = np.expand_dims(img, axis=0)
        angle_predict = self.model.predict(img, verbose=0)
        angle_predict = np.asarray(angle_predict).reshape(-1)
        return float(angle_predict[0] * (180 / np.pi))


class ImageSegmentationPredictor:
    def __init__(self, lane_model_path, object_model_path, conf=0.5):
        self.lane_model = YOLO(str(lane_model_path))
        self.object_model = YOLO(str(object_model_path))
        self.conf = conf

    def get_prediction(self, result):
        predict_freq = {}

        if result.boxes is None:
            return predict_freq

        class_labels = result.boxes.cls.cpu().numpy().astype(int)

        for class_id in class_labels:
            label = result.names.get(int(class_id), str(class_id))
            predict_freq[label] = predict_freq.get(label, 0) + 1

        return predict_freq

    def annotate_image(self, image, lane_result, object_result):
        annotated = lane_result.plot(img=image.copy())
        annotated = object_result.plot(img=annotated)
        return annotated


class SelfDrivingCar:
    def __init__(
        self,
        steering_model,
        segmentation_models,
        dataset_path,
        output_path=None,
    ):
        self.steering_model = steering_model
        self.segmentation_models = segmentation_models
        self.dataset_path = Path(dataset_path).resolve()

        self.original_path = self.dataset_path / "original_images"

        self.output_path = (
            Path(output_path).resolve()
            if output_path is not None
            else self.dataset_path
        )

        self.annotated_path = self.output_path / "annotated_images"
        self.annotated_path.mkdir(parents=True, exist_ok=True)

    def start_car(self):
        if not self.original_path.is_dir():
            raise FileNotFoundError(
                f"Original images folder not found: {self.original_path}"
            )

        supported_extensions = {".jpg", ".jpeg", ".png"}

        imgs_path = [
            path
            for path in self.original_path.iterdir()
            if path.is_file()
            and path.suffix.lower() in supported_extensions
        ]

        # Preserve numeric frame ordering when filenames are numeric.
        def sort_key(path):
            try:
                return (0, int(path.stem))
            except ValueError:
                return (1, path.name.lower())

        imgs_path.sort(key=sort_key)

        if not imgs_path:
            raise FileNotFoundError(
                f"No supported images found in {self.original_path}"
            )

        self.output_path.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_path / "predictions.csv"

        columns = [
            "img_id",
            "img_name",
            "annotated_img",
            "steering_angle",
            "objects_predict",
            "lane_predict",
        ]

        with ThreadPoolExecutor(max_workers=3) as executor:
            with csv_path.open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()

                for index, img_path in enumerate(imgs_path, start=1):
                    image = cv2.imread(str(img_path))

                    if image is None:
                        print(f"Cannot open image: {img_path}")
                        continue

                    # Keep the existing steering preprocessing.
                    steering_img = image[-150:]
                    steering_img = cv2.resize(steering_img, (200, 66))

                    future_angle = executor.submit(
                        self.steering_model.predict_angle,
                        steering_img,
                    )

                    future_lane = executor.submit(
                        self.segmentation_models.lane_model.predict,
                        image,
                        conf=self.segmentation_models.conf,
                        verbose=False,
                    )

                    future_objects = executor.submit(
                        self.segmentation_models.object_model.predict,
                        image,
                        conf=self.segmentation_models.conf,
                        verbose=False,
                    )

                    steering_angle = future_angle.result()
                    lane_result = future_lane.result()[0]
                    object_result = future_objects.result()[0]

                    lane_prediction = (
                        self.segmentation_models.get_prediction(lane_result)
                    )
                    object_prediction = (
                        self.segmentation_models.get_prediction(object_result)
                    )

                    annotated_image = self.segmentation_models.annotate_image(
                        image,
                        lane_result,
                        object_result,
                    )

                    annotated_name = f"{img_path.stem}_annotated.jpg"
                    annotated_file = self.annotated_path / annotated_name

                    if not cv2.imwrite(str(annotated_file), annotated_image):
                        raise RuntimeError(
                            f"Could not save annotated image: {annotated_file}"
                        )

                    # Store relative paths so the output is portable.
                    writer.writerow({
                        "img_id": index,
                        "img_name": str(
                            Path("original_images") / img_path.name
                        ),
                        "annotated_img": str(
                            Path("annotated_images") / annotated_name
                        ),
                        "steering_angle": steering_angle,
                        "objects_predict": object_prediction,
                        "lane_predict": lane_prediction,
                    })

                    file.flush()

                    print(
                        f"{index}/{len(imgs_path)} | "
                        f"{img_path.name} | "
                        f"Steering angle: {steering_angle:.2f} degrees"
                    )

        print("Inference completed.")
        print(f"Predictions saved to: {csv_path}")
        print(f"Annotated images saved to: {self.annotated_path}")


def main():
    required_files = [
        MODEL_DIR / "model.39-0.01.keras",
        MODEL_DIR / "Yolo11n-trained.pt",
        MODEL_DIR / "yolo26n-seg.pt",
    ]

    missing_files = [
        path for path in required_files if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing model file(s):\n"
            + "\n".join(str(path) for path in missing_files)
        )

    if not ORIGINAL_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset image folder not found: {ORIGINAL_DIR}"
        )

    steering_model = SteeringAnglePredictor(
        MODEL_DIR / "model.39-0.01.keras"
    )

    segmentation_models = ImageSegmentationPredictor(
        MODEL_DIR / "Yolo11n-trained.pt",
        MODEL_DIR / "yolo26n-seg.pt",
        conf=0.5,
    )

    car = SelfDrivingCar(
        steering_model,
        segmentation_models,
        DATASET_DIR,
        output_path=OUTPUT_DIR,
    )

    car.start_car()


if __name__ == "__main__":
    main()
