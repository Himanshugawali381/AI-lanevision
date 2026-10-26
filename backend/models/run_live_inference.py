
from pathlib import Path
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from main import (
    SteeringAnglePredictor,
    ImageSegmentationPredictor,
    SelfDrivingCar,
)

BACKEND_DIR = Path(__file__).resolve().parent
MODEL_DIR = BACKEND_DIR / "models"
DATASET_DIR = BACKEND_DIR / "dataset"
OUTPUT_DIR = DATASET_DIR / "live_inference"


def main():
    required_files = [
        MODEL_DIR / "model.39-0.01.keras",
        MODEL_DIR / "Yolo11n-trained.pt",
        MODEL_DIR / "yolo26n-seg.pt",
    ]

    missing = [str(path) for path in required_files if not path.is_file()]

    if missing:
        raise FileNotFoundError(
            "Missing model files:\n" + "\n".join(missing)
        )

    if not (DATASET_DIR / "original_images").is_dir():
        raise FileNotFoundError(
            f"Original images folder not found: "
            f"{DATASET_DIR / 'original_images'}"
        )

    steering_model = SteeringAnglePredictor(
        str(MODEL_DIR / "model.39-0.01.keras")
    )

    segmentation_models = ImageSegmentationPredictor(
        str(MODEL_DIR / "Yolo11n-trained.pt"),
        str(MODEL_DIR / "yolo26n-seg.pt"),
        conf=0.5,
    )

    car = SelfDrivingCar(
        steering_model,
        segmentation_models,
        DATASET_DIR,
        output_path=OUTPUT_DIR,
    )

    car.start_car()

    print(f"Live inference completed. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
