import sys

import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import mediapipe as mp

delegate = BaseOptions.Delegate.GPU if sys.argv[1] == "gpu" else BaseOptions.Delegate.CPU
mode = vision.RunningMode.VIDEO if sys.argv[2] == "video" else vision.RunningMode.IMAGE
print("mediapipe", mp.__version__, delegate, mode, flush=True)
opts = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task", delegate=delegate),
    running_mode=mode,
    num_hands=1,
)
with vision.HandLandmarker.create_from_options(opts) as lm:
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((480, 640, 3), np.uint8))
    r = lm.detect_for_video(img, 0) if mode == vision.RunningMode.VIDEO else lm.detect(img)
    print("OK hands:", len(r.hand_landmarks), flush=True)
