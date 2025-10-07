// ===============================
// C++ SDL3 + OpenCV + ZeroMQ UI Skeleton
// ===============================
// This code demonstrates how to:
// 1. Capture video with OpenCV
// 2. Send frames to a Python PyTorch server via ZeroMQ
// 3. Receive detection results and draw them
// 4. Render with SDL3
//
// Requires: OpenCV, SDL3, ZeroMQ, nlohmann/json
//
// Python server code is included below for reference.

#include <opencv2/opencv.hpp>
#include <SDL3/SDL.h>
#include <zmq.hpp>           // ZeroMQ for IPC
#include <nlohmann/json.hpp> // For JSON serialization

using json = nlohmann::json;

int main()
{
    // SDL3 and OpenCV setup
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *window = SDL_CreateWindow("ROV Pilot UI", 1920, 1080, SDL_WINDOW_RESIZABLE);
    SDL_Renderer *renderer = SDL_CreateRenderer(window, nullptr, 0);

    cv::VideoCapture cap(0); // or RTSP stream
    if (!cap.isOpened())
        return -1;

    // ZeroMQ setup
    zmq::context_t context(1);
    zmq::socket_t socket(context, zmq::socket_type::req);
    socket.connect("tcp://127.0.0.1:5555"); // Python server

    while (true)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        // Resize and convert to RGB
        cv::resize(frame, frame, cv::Size(640, 480));
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

        // Send frame to Python (as bytes)
        std::vector<uchar> buf;
        cv::imencode(".jpg", frame, buf);
        zmq::message_t request(buf.data(), buf.size());
        socket.send(request, zmq::send_flags::none);

        // Receive detection results (JSON)
        zmq::message_t reply;
        socket.recv(reply, zmq::recv_flags::none);
        std::string json_str(static_cast<char *>(reply.data()), reply.size());
        json detections = json::parse(json_str);

        // Draw detections
        for (const auto &det : detections)
        {
            int x1 = det["x1"], y1 = det["y1"], x2 = det["x2"], y2 = det["y2"];
            std::string label = det["label"];
            float score = det["score"];
            if (score >= 0.6)
            {
                cv::rectangle(frame, {x1, y1}, {x2, y2}, {0, 255, 0}, 2);
                cv::putText(frame, label, {x1, y1 - 5}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 0}, 2);
            }
        }

        // Convert to SDL texture and render (implement as needed)
        // ...

        // SDL event loop, UI drawing, etc.
        // ...
    }

    cap.release();
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}

// ===============================
// Python PyTorch Model Server (ZeroMQ)
// ===============================
// Save this as detect_server.py and run it before the C++ app.
/*
import zmq
import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()
labels = weights.meta["categories"]

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

while True:
    msg = socket.recv()
    img_array = np.frombuffer(msg, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(frame).permute(2,0,1).float()/255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
    pred_classes = outputs[0]["labels"].cpu().numpy()
    pred_scores = outputs[0]["scores"].cpu().numpy()
    pred_boxes = outputs[0]["boxes"].cpu().numpy()
    results = []
    for cls, score, box in zip(pred_classes, pred_scores, pred_boxes):
        if score >= 0.6:
            x1, y1, x2, y2 = box.astype(int)
            results.append({
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                "label": labels[cls], "score": float(score)
            })
    import json
    socket.send_string(json.dumps(results))
*/
