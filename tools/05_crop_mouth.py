import cv2, os
import mediapipe as mp

MOUTH = [61,146,91,181,84,17,314,405,321,375,291]

for name in os.listdir("data/processed/frames"):
    in_dir = f"data/processed/frames/{name}"
    out_dir = f"data/processed/mouth/{name}"

    os.makedirs(out_dir, exist_ok=True)

    face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True
    )

    for f in sorted(os.listdir(in_dir)):
        img = cv2.imread(f"{in_dir}/{f}")
        res = face.process(img)

        if res.multi_face_landmarks:
            h, w, _ = img.shape
            pts = res.multi_face_landmarks[0].landmark

            xs = [int(pts[i].x * w) for i in MOUTH]
            ys = [int(pts[i].y * h) for i in MOUTH]

            y1, y2 = max(0, min(ys)), min(h, max(ys))
            x1, x2 = max(0, min(xs)), min(w, max(xs))

            if y2 > y1 and x2 > x1:
                crop = img[y1:y2, x1:x2]
                
                # Changed for AV-Hubert compatibility
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) # grayscale
                crop = cv2.resize(crop, (88, 88)) # 88x88 instead of 96x96

                cv2.imwrite(f"{out_dir}/{f}", crop)
