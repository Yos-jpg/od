import torch
import cv2
import numpy as np
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
# ----------------------------
# 1️⃣ Device selection (GPU if available)
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ----------------------------
# 2️⃣ Load pretrained model
# ----------------------------
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT  # (COCO_V1 dataset)
model = fasterrcnn_resnet50_fpn(weights=weights)
model.to(device).eval()

# Class labels
labels = weights.meta["categories"]

# ----------------------------
# 3️⃣ OpenCV video capture
# ----------------------------

img = "..."  # Change to your image path

cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

# Optional: reduce resolution for FPS boost
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ----------------------------
# 4️⃣ Detection loop
# ----------------------------
# frame_count = 0
# frame_skip = 2  # Process every 2nd frame for FPS boost

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optional frame skipping
    # frame_count += 1
    # if frame_count % frame_skip != 0:
        cv2.imshow("Live Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Convert frame to RGB and tensor
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Run model
    with torch.no_grad():
        outputs = model(img_tensor)

    # Extract detections
    pred_classes = outputs[0]["labels"].cpu().numpy()
    pred_scores = outputs[0]["scores"].cpu().numpy()
    pred_boxes = outputs[0]["boxes"].cpu().numpy()

    # Draw boxes for confident predictions
    for cls, score, box in zip(pred_classes, pred_scores, pred_boxes):
        if score >= 0.6:  # confidence threshold
            x1, y1, x2, y2 = box.astype(int)
            label = f"{labels[cls]}: {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Live Object Detection", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# 5️⃣ Cleanup
# ----------------------------
cap.release()
cv2.destroyAllWindows()


# import torch
# print(torch.__version__)         # PyTorch version
# print(torch.cuda.is_available())  # Should be True if CUDA works
# print(torch.version.cuda)
