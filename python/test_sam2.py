import cv2
import numpy as np
import torch
import time
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2WebcamTracker:
    def __init__(self, checkpoint, model_cfg, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Device selected: {self.device}")

        # Load the SAM2 model
        print("Loading SAM2 model...")
        self.sam2_model = build_sam2(model_cfg, checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)

        # Initialize webcam
        print("Initializing webcam...")
        self.cap = cv2.VideoCapture(0)
        self.click_points = []
        self.click_labels = []

        # Add mouse callback
        cv2.namedWindow("SAM2 Webcam")
        cv2.setMouseCallback("SAM2 Webcam", self.on_mouse_click)

    def on_mouse_click(self, event, x, y, flags, param):
        """Handle mouse click events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click: Add clicked point with label 1
            print(f"Mouse clicked at ({x}, {y})")
            self.click_points.append([x, y])
            self.click_labels.append(1)

    def get_mask_center(self, mask):
        """Calculate the center of the mask."""
        indices = np.argwhere(mask)
        if indices.size == 0:
            return None  # Return None if no mask
        center = indices.mean(axis=0).astype(int)
        return center[1], center[0]  # (x, y) format

    def apply_transparent_mask(self, frame, mask, color=(0, 255, 0), alpha=0.5):
        """Overlay a transparent mask on the image."""
        overlay = frame.copy()
        for i in range(3):
            overlay[:, :, i] = np.where(
                mask == 1,
                frame[:, :, i] * (1 - alpha) + color[i] * alpha,
                frame[:, :, i]
            )
        return overlay

    def overlay_info(self, frame, fps, detection_time, latency):
        """Overlay FPS, detection time, and latency information on the frame."""
        info_text = f"FPS: {fps:.2f} | Detection Time: {detection_time:.2f}s | Latency: {latency:.2f}s"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    def run(self):
        """Main loop to process webcam frames and run SAM2 tracking."""
        print("Starting webcam processing...")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            while True:
                frame_start_time = time.time()  # Start time for latency measurement

                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame from webcam. Exiting...")
                    break

                # Initialize FPS timer
                start_time = time.time()

                self.predictor.set_image(frame)
                if self.click_points:
                    # Convert click points to numpy arrays
                    input_points = np.array(self.click_points)
                    input_labels = np.array(self.click_labels)

                    # Run SAM2 prediction
                    detection_start = time.time()
                    masks, scores, _ = self.predictor.predict(
                        point_coords=input_points,
                        point_labels=input_labels,
                        multimask_output=False
                    )
                    detection_time = time.time() - detection_start

                    # Process predicted masks
                    for i, mask in enumerate(masks):
                        mask_center = self.get_mask_center(mask)
                        self.click_points[i] = mask_center

                    # Display the mask on the frame
                    if masks is not None and len(masks) > 0:
                        frame = self.apply_transparent_mask(frame, masks[0])

                    # Draw the clicked points
                    for point in input_points:
                        cv2.circle(frame, (point[0], point[1]), 5, (0, 0, 255), -1)
                else:
                    detection_time = 0

                # Calculate FPS and latency
                latency = time.time() - frame_start_time
                fps = 1 / (time.time() - start_time)

                # Overlay info on the frame

                frame = self.overlay_info(frame, fps, detection_time, latency)

                # Display the frame
                cv2.imshow("SAM2 Webcam", frame)

                # Handle keyboard input
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    print("Exiting...")
                    break  # Exit on 'q'
                elif key & 0xFF == ord('x'):
                    # Clear click points on 'x'
                    print("Clearing clicked points...")
                    self.click_points.clear()
                    self.click_labels.clear()

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Paths for SAM2 model
    checkpoint_path = "../checkpoints/sam2.1_hiera_large.pt"
    model_config_path = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"

    # Initialize and run the tracker
    tracker = SAM2WebcamTracker(checkpoint_path, model_config_path)
    tracker.run()
