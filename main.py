from utils import split_image, join_images, print_fps, get_valid_objects_in_zone
from camera import Camera
import cv2
from ultralytics import YOLO
import torch


PARTS = 2
INFERENCE_PARAMS = {
    "conf": 0.4, 
    "iou": 0.4,
    "agnostic_nms": True,
    "verbose": False
}
MODEL_PATH = r"models\yolo26n_320_ncnn_model"



valid_objects_in_zone = get_valid_objects_in_zone(PARTS)


# --- Model ---
model = YOLO(MODEL_PATH, task="detect")  

# --- Camera ---
cam = Camera(use_opencv=False, show=False)

while True:

    # STEP 1: CAPTURE FRAME
    """ frame = cam.read()
    if frame is None:
        break """
    frame = cv2.imread(r"files\20260926_011716.jpg")

    #STEP 2: SPLIT IMAGE
    images = split_image(frame, parts=PARTS)

    #STEP 3: NCNN INFERENCE (IT ONLY ALLOWS INDIVIDUAL BATCHES)
    results = [
        model(source=image, **INFERENCE_PARAMS)[0]
        for image in images
    ]

    is_valid = True
    #STEP 4: DRAW INVALID OBJECT BOXES
    for i, result in enumerate(results):

        boxes = result.boxes
        valid_objects = valid_objects_in_zone[i]

        invalid_mask = ~torch.isin(
            boxes.cls,
            torch.tensor(
                valid_objects,
                device=boxes.cls.device
            )
        )

        if invalid_mask.any():
            is_valid = False

            for x1, y1, x2, y2 in (
                boxes.xyxy[invalid_mask]
                .int()
                .cpu()
                .tolist()
            ):
                cv2.rectangle(
                    images[i],
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )

    #STEP 5: JOIN IMAGES
    frame = join_images(images, parts=PARTS)

    """
        RENDER IMAGE
    """
    cv2.imshow("Imagen", frame)
    
    
    #FULL IMAGE  STATE
    #print(is_valid)
   

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()