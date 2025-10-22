import sdl3
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


sdl3.SDL_Init(sdl3.SDL_INIT_VIDEO)
window = sdl3.SDL_CreateWindow(
    b"ROV Pilot UI", 1920, 1080, sdl3.SDL_WINDOW_RESIZABLE)
renderer = sdl3.SDL_CreateRenderer(window, None)

# ---------------- ROV State ----------------
# Insert class for each truster, depth, heading, speed, strafe, yaw, heave, estop
# it will prolly take a lot of space so just use demo values for now


class State:
    thrusters = [1, -0.75, 0.5, 1.0]  # demo values
    depth = 12.34
    heading = 182
    speed = 0.42
    estop = False
    forward = 0.1
    strafe = -0.2
    yaw = 0.0
    heave = 0.3


state = State()
THRUSTER_NAMES = ["FL", "FR", "BL", "BR"]

# ---------------- Video Source ----------------
# Replace with your RTSP stream (or 0 for webcam)
VIDEO_SOURCE = 0
# VIDEO_SOURCE = "rtsp://username:password@ip:port/stream"
cap = cv2.VideoCapture(VIDEO_SOURCE)


class Frame():
    def get_frame(self):
        ok, frame = cap.read()
        if not ok:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR → RGB
        frame = cv2.resize(frame, (640, 480))           # scale to fit UI

        img_tensor = torch.from_numpy(
            frame).permute(2, 0, 1).float() / 255.0
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

        h, w, _ = frame.shape
        surf = np.zeros((h, w, 4), dtype=np.uint8)
        surf[:, :, :3] = frame
        surf[:, :, 3] = 255

        tex = sdl3.SDL_CreateTexture(renderer, sdl3.SDL_PIXELFORMAT_ABGR8888,
                                     sdl3.SDL_TEXTUREACCESS_STREAMING, w, h)
        sdl3.SDL_UpdateTexture(tex, None, surf.ctypes.data, surf.strides[0])
        return tex, w, h


frame = Frame().get_frame



class DT():
    def draw_text(self, text, x, y, font_scale=0.5, color=(255, 255, 255)):
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        img = np.zeros((th + 6, tw + 6, 3), dtype=np.uint8)
        cv2.putText(img, text, (3, th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, 1, cv2.LINE_AA)

        surf = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        surf[:, :, :3] = img[:, :, ::-1]
        surf[:, :, 3] = 255

        h, w, _ = surf.shape
        tex = sdl3.SDL_CreateTexture(renderer, sdl3.SDL_PIXELFORMAT_ABGR8888,
                                     sdl3.SDL_TEXTUREACCESS_STREAMING, w, h)
        sdl3.SDL_UpdateTexture(tex, None, surf.ctypes.data, surf.strides[0])
        dst = sdl3.SDL_FRect(x, y, w, h)
        sdl3.SDL_RenderTexture(renderer, tex, None, dst)
        sdl3.SDL_DestroyTexture(tex)


draw = DT().draw_text



class UI():
    def draw_ui(self):
        sdl3.SDL_SetRenderDrawColor(renderer, 14, 18, 30, 255)
        sdl3.SDL_RenderClear(renderer)

        vid_x, vid_y, vid_w, vid_h = 240, 40, 640, 480
        texinfo = frame()  # Fixed: Added parentheses to call the function
        if texinfo:
            tex, w, h = texinfo
            dst = sdl3.SDL_FRect(vid_x, vid_y, vid_w, vid_h)
            sdl3.SDL_RenderTexture(renderer, tex, None, dst)
            sdl3.SDL_DestroyTexture(tex)
        else:
            sdl3.SDL_SetRenderDrawColor(renderer, 40, 40, 40, 255)
            sdl3.SDL_RenderFillRect(
                renderer, sdl3.SDL_FRect(vid_x, vid_y, vid_w, vid_h))
            draw("[No Video]", vid_x + 20, vid_y + 20, 0.6)

        # Telemetry
        draw(f"Depth: {state.depth:.2f}m", vid_x, vid_y + vid_h + 10, 0.6)
        draw(f"Heading: {state.heading:.0f}°",
             vid_x + 220, vid_y + vid_h + 10, 0.6)
        draw(f"Speed: {state.speed:.2f}m/s",
             vid_x + 420, vid_y + vid_h + 10, 0.6)

        # Thruster bars (vertical %)
        thr_x = vid_x + vid_w + 36
        thr_y = vid_y
        bar_w, bar_h, spacing = 34, 140, 54
        for i, name in enumerate(THRUSTER_NAMES):
            val = float(state.thrusters[i])     # -1..+1
            percent = int(val * 100)
            x = thr_x + i * spacing

            rect_bg = sdl3.SDL_FRect(x, thr_y, bar_w, bar_h)
            sdl3.SDL_SetRenderDrawColor(renderer, 60, 60, 80, 255)
            sdl3.SDL_RenderFillRect(renderer, rect_bg)

            mid_y = thr_y + bar_h // 2
            sdl3.SDL_SetRenderDrawColor(renderer, 200, 200, 200, 200)
            sdl3.SDL_RenderLine(renderer, x, mid_y, x + bar_w - 1, mid_y)

            if val >= 0:
                fill_h = int((bar_h // 2) * val)
                if fill_h > 0:
                    rect_fill = sdl3.SDL_FRect(
                        x, mid_y - fill_h, bar_w, fill_h)
                    sdl3.SDL_SetRenderDrawColor(renderer, 60, 180, 80, 255)
                    sdl3.SDL_RenderFillRect(renderer, rect_fill)
            else:
                fill_h = int((bar_h // 2) * (-val))
                if fill_h > 0:
                    rect_fill = sdl3.SDL_FRect(x, mid_y, bar_w, fill_h)
                    sdl3.SDL_SetRenderDrawColor(renderer, 200, 80, 80, 255)
                    sdl3.SDL_RenderFillRect(renderer, rect_fill)

            draw(f"{name} {percent:+d}%", x - 6, thr_y + bar_h + 8, 0.5)

        sb_y = vid_y + vid_h + 44
        draw(f"Fwd:{state.forward:+.2f} Strf:{state.strafe:+.2f} "
             f"Yaw:{state.yaw:+.2f} Heave:{state.heave:+.2f}",
             vid_x, sb_y, 0.5)
        draw(f"ESTOP: {'ON' if state.estop else 'OFF'}",
             vid_x + 520, sb_y, 0.5)

        # Fixed: Moved SDL_RenderPresent inside the draw_ui method
        sdl3.SDL_RenderPresent(renderer)


UserI = UI().draw_ui



running = True
event = sdl3.SDL_Event()
while running:
    while sdl3.SDL_PollEvent(event):
        if event.type == sdl3.SDL_EVENT_QUIT:
            running = False

    # Fixed: Actually call the draw function
    UserI()

cap.release()
sdl3.SDL_Quit()
