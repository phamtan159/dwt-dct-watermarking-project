import json
import math
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


FRAME_ROOT = Path(os.environ.get("FRAME_ROOT", "data/processed/frames"))
MOUTH_ROOT = Path(os.environ.get("MOUTH_ROOT", "data/processed/mouth"))
MEDIAPIPE_DIR = Path(os.environ.get("MEDIAPIPE_DIR", "data/annotations/mediapipe"))
FACE_LANDMARKER_MODEL = Path(
    os.environ.get("FACE_LANDMARKER_MODEL", "pretrained/mediapipe/face_landmarker.task")
)

OUTER_MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
INNER_MOUTH = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
FEATURE_POINTS = {
    "left_mouth": 61,
    "right_mouth": 291,
    "upper_lip_inner": 13,
    "lower_lip_inner": 14,
    "upper_lip_outer": 0,
    "lower_lip_outer": 17,
    "nose_tip": 1,
    "chin": 152,
}


def read_image(path: Path):
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, image) -> bool:
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        return False
    path.write_bytes(encoded.tobytes())
    return True


class FaceLandmarkerAdapter:
    def __init__(self, model_path: Path):
        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def process(self, rgb_image):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_image))
        result = self.landmarker.detect(image)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]

    def close(self):
        self.landmarker.close()


class SolutionsFaceMeshAdapter:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
        )

    def process(self, rgb_image):
        result = self.face_mesh.process(rgb_image)
        if not result.multi_face_landmarks:
            return None
        return result.multi_face_landmarks[0].landmark

    def close(self):
        self.face_mesh.close()


def create_landmark_detector():
    if FACE_LANDMARKER_MODEL.exists():
        return FaceLandmarkerAdapter(FACE_LANDMARKER_MODEL), "mediapipe_tasks_face_landmarker"
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        return SolutionsFaceMeshAdapter(), "mediapipe_face_mesh"
    return None, "opencv_fallback_mouth_crop"


def distance(a, b):
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)


def normalized_point(landmark):
    return {"x": float(landmark.x), "y": float(landmark.y), "z": float(landmark.z)}


def visual_features(points):
    left = points["left_mouth"]
    right = points["right_mouth"]
    upper_inner = points["upper_lip_inner"]
    lower_inner = points["lower_lip_inner"]
    upper_outer = points["upper_lip_outer"]
    lower_outer = points["lower_lip_outer"]
    nose_tip = points["nose_tip"]
    chin = points["chin"]

    lip_width = max(distance(left, right), 1e-6)
    inner_opening = distance(upper_inner, lower_inner)
    outer_opening = distance(upper_outer, lower_outer)
    face_height = max(distance(nose_tip, chin), 1e-6)

    return {
        "lip_width": round(lip_width, 6),
        "mouth_opening": round(inner_opening, 6),
        "mouth_opening_ratio": round(inner_opening / lip_width, 6),
        "jaw_opening_proxy": round(outer_opening / face_height, 6),
        "lip_rounding_proxy": round(inner_opening / lip_width, 6),
        "labiodental_contact_proxy": round(1.0 - min(1.0, outer_opening / lip_width), 6),
        "tongue_landmarks_available": False,
    }


def process_video_frames(name, face_mesh):
    if face_mesh is None:
        return process_video_frames_fallback(name)

    in_dir = FRAME_ROOT / name
    out_dir = MOUTH_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_items = []
    for frame_index, filename in enumerate(sorted(os.listdir(in_dir))):
        image_path = in_dir / filename
        image = read_image(image_path)
        if image is None:
            frame_items.append({"frame": filename, "frame_index": frame_index, "face_detected": False})
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        landmarks = face_mesh.process(rgb)
        if not landmarks:
            frame_items.append({"frame": filename, "frame_index": frame_index, "face_detected": False})
            continue

        h, w, _ = image.shape
        xs = [int(landmarks[i].x * w) for i in OUTER_MOUTH + INNER_MOUTH]
        ys = [int(landmarks[i].y * h) for i in OUTER_MOUTH + INNER_MOUTH]

        y1, y2 = max(0, min(ys)), min(h, max(ys))
        x1, x2 = max(0, min(xs)), min(w, max(xs))

        crop_written = False
        if y2 > y1 and x2 > x1:
            crop = image[y1:y2, x1:x2]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop = cv2.resize(crop, (88, 88))
            write_image(out_dir / filename, crop)
            crop_written = True

        points = {
            name: normalized_point(landmarks[index])
            for name, index in FEATURE_POINTS.items()
        }

        frame_items.append(
            {
                "frame": filename,
                "frame_index": frame_index,
                "face_detected": True,
                "crop_written": crop_written,
                "mouth_bbox_px": [x1, y1, x2, y2],
                "points": points,
                "visual_features": visual_features(points),
            }
        )

    return frame_items


def fallback_mouth_bbox(image):
    h, w, _ = image.shape
    x1 = int(w * 0.30)
    x2 = int(w * 0.70)
    y1 = int(h * 0.58)
    y2 = int(h * 0.82)
    return x1, y1, x2, y2, "center_lower_face_region"


def fallback_visual_features(x1, y1, x2, y2, image_shape):
    h, w, _ = image_shape
    crop_w = max(x2 - x1, 1)
    crop_h = max(y2 - y1, 1)
    return {
        "lip_width": round(crop_w / max(w, 1), 6),
        "mouth_opening": round(crop_h / max(h, 1), 6),
        "mouth_opening_ratio": round(crop_h / max(crop_w, 1), 6),
        "jaw_opening_proxy": round(crop_h / max(h, 1), 6),
        "lip_rounding_proxy": round(crop_h / max(crop_w, 1), 6),
        "labiodental_contact_proxy": None,
        "tongue_landmarks_available": False,
    }


def process_video_frames_fallback(name):
    in_dir = FRAME_ROOT / name
    out_dir = MOUTH_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_items = []
    for frame_index, filename in enumerate(sorted(os.listdir(in_dir))):
        image_path = in_dir / filename
        image = read_image(image_path)
        if image is None:
            frame_items.append({"frame": filename, "frame_index": frame_index, "face_detected": False})
            continue

        x1, y1, x2, y2, method = fallback_mouth_bbox(image)
        crop = image[y1:y2, x1:x2]
        crop_written = False
        if crop.size > 0:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop = cv2.resize(crop, (88, 88))
            write_image(out_dir / filename, crop)
            crop_written = True

        frame_items.append(
            {
                "frame": filename,
                "frame_index": frame_index,
                "face_detected": True,
                "crop_written": crop_written,
                "mouth_bbox_px": [x1, y1, x2, y2],
                "fallback_method": method,
                "points": None,
                "visual_features": fallback_visual_features(x1, y1, x2, y2, image.shape),
            }
        )

    return frame_items


def iter_frame_dirs(frame_root: Path):
    if not frame_root.exists():
        return []
    dirs = []
    for directory in sorted(path for path in frame_root.rglob("*") if path.is_dir()):
        if any(path.suffix.lower() in {".jpg", ".jpeg", ".png"} for path in directory.iterdir() if path.is_file()):
            dirs.append(directory)
    return dirs


def main():
    MEDIAPIPE_DIR.mkdir(parents=True, exist_ok=True)
    if not FRAME_ROOT.exists():
        print(f"No frame directory found: {FRAME_ROOT}")
        return

    face_mesh, model_name = create_landmark_detector()
    if face_mesh is None:
        print(
            "WARNING: no MediaPipe FaceLandmarker model found at "
            f"{FACE_LANDMARKER_MODEL}; using OpenCV fallback mouth crop."
        )

    frame_dirs = iter_frame_dirs(FRAME_ROOT)
    if not frame_dirs:
        print(f"No speaker frame directories found in {FRAME_ROOT}")
        if face_mesh is not None:
            face_mesh.close()
        return

    for in_dir in frame_dirs:
        if in_dir.parent == FRAME_ROOT:
            print(f"Skip {in_dir}: frames must be inside a speaker folder")
            continue
        name = in_dir.relative_to(FRAME_ROOT).as_posix()

        frames = process_video_frames(name, face_mesh)
        payload = {
            "video_id": name,
            "model": model_name,
            "features_version": "mouth_visual_attributes_v1",
            "frames": frames,
        }
        output_path = MEDIAPIPE_DIR / Path(name).with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        detected = sum(1 for item in frames if item.get("face_detected"))
        print(f"Wrote {output_path} ({detected}/{len(frames)} frames detected)")

    if face_mesh is not None:
        face_mesh.close()


if __name__ == "__main__":
    main()
