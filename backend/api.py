
import csv
import subprocess
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(
    r"C:\Users\LENOVO\Desktop\lane-images\ai-project\dataset"
).resolve()

ORIGINAL_DIR = DATASET_DIR / "original_images"
ORIGINAL_ANNOTATED_DIR = DATASET_DIR / "annotated_images"
ORIGINAL_CSV = DATASET_DIR / "predictions.csv"

LIVE_OUTPUT_DIR = DATASET_DIR / "live_inference"
LIVE_ANNOTATED_DIR = LIVE_OUTPUT_DIR / "annotated_images"
LIVE_CSV = LIVE_OUTPUT_DIR / "predictions.csv"

LIVE_RUNNER = BACKEND_DIR / "main.py"
INFERENCE_LOG = BACKEND_DIR / "live_inference.log"


# --------------------------------------------------
# APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="LaneVision API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------
# INFERENCE STATE
# --------------------------------------------------

inference_process = None
inference_log_file = None
inference_lock = threading.Lock()
inference_error = None
inference_completed = False

# The dashboard initially displays your existing predictions.
active_csv = ORIGINAL_CSV
active_annotated_dir = ORIGINAL_ANNOTATED_DIR


# --------------------------------------------------
# CSV AND IMAGE HELPERS
# --------------------------------------------------

def load_predictions(csv_path=None):
    path = Path(csv_path or active_csv)

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Predictions CSV not found: {path}",
        )

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            return list(csv.DictReader(file))

    except (OSError, csv.Error) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read predictions CSV: {exc}",
        ) from exc


def resolve_image(csv_value, fallback_dir):
    """Resolve a CSV image path within its permitted image folder."""

    if not csv_value:
        return None

    raw = str(csv_value).strip().strip('"').strip("'")
    normalized = raw.replace("\\", "/")
    recorded_path = Path(normalized)
    fallback_dir = fallback_dir.resolve()

    candidates = []

    # Try the original absolute path if it still exists.
    if recorded_path.is_absolute():
        candidates.append(recorded_path)

    # Support relative paths in the CSV.
    candidates.extend([
        DATASET_DIR / normalized,
        fallback_dir / normalized,
        fallback_dir / Path(normalized).name,
    ])

    for candidate in candidates:
        try:
            resolved = candidate.resolve()

            # Only serve files from the permitted image directory.
            if (
                resolved.is_relative_to(fallback_dir)
                and resolved.is_file()
            ):
                return resolved

        except (OSError, RuntimeError, ValueError):
            continue

    return None


def make_frame(row, index):
    original = resolve_image(
        row.get("img_name"),
        ORIGINAL_DIR,
    )

    annotated = resolve_image(
        row.get("annotated_img"),
        active_annotated_dir,
    )

    original_value = str(
        row.get("img_name") or ""
    ).replace("\\", "/")

    annotated_value = str(
        row.get("annotated_img") or ""
    ).replace("\\", "/")

    return {
        "index": index,
        "img_id": row.get("img_id") or str(index + 1),
        "original_name": Path(original_value).name,
        "annotated_name": Path(annotated_value).name,
        "original_available": original is not None,
        "annotated_available": annotated is not None,
        "original_url": (
            f"/api/images/original/{index}"
            if original else None
        ),
        "annotated_url": (
            f"/api/images/annotated/{index}"
            if annotated else None
        ),
        "steering_angle": row.get("steering_angle", "0"),
        "objects_predict": row.get("objects_predict", "{}"),
        "lane_predict": row.get("lane_predict", "{}"),
    }


# --------------------------------------------------
# HEALTH AND DATA ENDPOINTS
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "LaneVision API is running",
        "health": "/api/health",
        "frames": "/api/frames",
        "start_inference": "/api/inference/start",
        "inference_status": "/api/inference/status",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "dataset_dir": str(DATASET_DIR),
        "csv_exists": ORIGINAL_CSV.is_file(),
        "original_images_dir_exists": ORIGINAL_DIR.is_dir(),
        "annotated_images_dir_exists": ORIGINAL_ANNOTATED_DIR.is_dir(),
        "live_inference_available": LIVE_CSV.is_file(),
    }


@app.get("/api/frames")
def get_frames():
    rows = load_predictions()

    return {
        "total": len(rows),
        "frames": [
            make_frame(row, index)
            for index, row in enumerate(rows)
        ],
    }


@app.get("/api/images/{kind}/{index}")
def get_image(kind: str, index: int):
    if kind not in {"original", "annotated"}:
        raise HTTPException(
            status_code=400,
            detail="Image type must be original or annotated.",
        )

    rows = load_predictions()

    if index < 0 or index >= len(rows):
        raise HTTPException(
            status_code=404,
            detail="Frame not found.",
        )

    row = rows[index]

    if kind == "original":
        image_path = resolve_image(
            row.get("img_name"),
            ORIGINAL_DIR,
        )
    else:
        image_path = resolve_image(
            row.get("annotated_img"),
            active_annotated_dir,
        )

    if image_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"{kind.capitalize()} image not found for frame {index}.",
        )

    return FileResponse(image_path)


# --------------------------------------------------
# LIVE INFERENCE ENDPOINTS
# --------------------------------------------------

@app.post("/api/inference/start")
def start_inference():
    global inference_process
    global inference_log_file
    global inference_error
    global inference_completed

    with inference_lock:
        if (
            inference_process is not None
            and inference_process.poll() is None
        ):
            return {
                "status": "running",
                "message": "Inference is already running.",
            }

        if not LIVE_RUNNER.is_file():
            raise HTTPException(
                status_code=500,
                detail=f"Inference script not found: {LIVE_RUNNER}",
            )

        if not ORIGINAL_DIR.is_dir():
            raise HTTPException(
                status_code=404,
                detail=f"Original images folder not found: {ORIGINAL_DIR}",
            )

        if not ORIGINAL_CSV.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Existing predictions CSV not found: {ORIGINAL_CSV}",
            )

        inference_error = None
        inference_completed = False

        # Remove previous live results so an unsuccessful new run
        # cannot accidentally activate stale output.
        if LIVE_CSV.exists():
            LIVE_CSV.unlink()

        if LIVE_ANNOTATED_DIR.exists():
            import shutil
            shutil.rmtree(LIVE_ANNOTATED_DIR)

        LIVE_ANNOTATED_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            inference_log_file = INFERENCE_LOG.open(
                "w",
                encoding="utf-8",
            )

            inference_process = subprocess.Popen(
                [
                    sys.executable,
                    str(LIVE_RUNNER),
                ],
                cwd=str(BACKEND_DIR),
                stdout=inference_log_file,
                stderr=subprocess.STDOUT,
            )

        except OSError as exc:
            if inference_log_file is not None:
                inference_log_file.close()
                inference_log_file = None

            raise HTTPException(
                status_code=500,
                detail=f"Could not start inference: {exc}",
            ) from exc

    return {
        "status": "running",
        "message": "Live inference started.",
    }


@app.get("/api/inference/status")
def inference_status():
    global inference_process
    global inference_log_file
    global inference_error
    global inference_completed
    global active_csv
    global active_annotated_dir

    with inference_lock:
        if inference_process is None:
            return {
                "status": "idle",
                "message": "No inference run has been started.",
            }

        return_code = inference_process.poll()

        if return_code is None:
            return {
                "status": "running",
                "message": "TensorFlow and YOLO are processing images.",
            }

        if inference_log_file is not None:
            inference_log_file.close()
            inference_log_file = None

        if return_code != 0:
            try:
                inference_error = INFERENCE_LOG.read_text(
                    encoding="utf-8",
                    errors="replace",
                )[-6000:]
            except OSError:
                inference_error = "Could not read the inference log."

            return {
                "status": "failed",
                "message": "Inference failed.",
                "details": inference_error,
            }

        if not LIVE_CSV.is_file() or not LIVE_ANNOTATED_DIR.is_dir():
            return {
                "status": "failed",
                "message": "Inference completed without the required output files.",
            }

        # Activate the new dataset only once, after a successful run.
        if not inference_completed:
            try:
                with LIVE_CSV.open(
                    "r",
                    encoding="utf-8-sig",
                    newline="",
                ) as file:
                    new_rows = list(csv.DictReader(file))

                if not new_rows:
                    return {
                        "status": "failed",
                        "message": "The new predictions CSV is empty.",
                    }

            except (OSError, csv.Error) as exc:
                return {
                    "status": "failed",
                    "message": f"Could not validate inference output: {exc}",
                }

            active_csv = LIVE_CSV
            active_annotated_dir = LIVE_ANNOTATED_DIR
            inference_completed = True

        return {
            "status": "completed",
            "message": "Live inference completed successfully.",
            "total": len(load_predictions()),
        }
