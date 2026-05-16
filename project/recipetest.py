import cv2
import time
import subprocess
from collections import Counter
from ultralytics import YOLO

# Load TensorRT model
model = YOLO("/home/ys9072/jetson_orin_nano_super_yolov11-master/weights/best.engine")

# Open camera
cap = cv2.VideoCapture(0)

stable_detected = []
start_detect_time = None

# Detection must stay stable for 5 seconds
TARGET_SECONDS = 5

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Failed to read camera frame.")
        break

    # YOLO inference
    results = model(frame, conf=0.25)

    # Draw detection boxes
    annotated_frame = results[0].plot()

    detected = []

    # Extract detected class names
    for box in results[0].boxes:
        cls = int(box.cls[0])
        name = results[0].names[cls]
        detected.append(name)

    # Sort for stable comparison
    detected = sorted(detected)

    # If something is detected
    if detected:

        # Same detection continues
        if detected == stable_detected:

            if start_detect_time is None:
                start_detect_time = time.time()

            elapsed = time.time() - start_detect_time

            # Show timer
            cv2.putText(
                annotated_frame,
                f"Stable Detection: {elapsed:.1f}s",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Stable for enough time
            if elapsed >= TARGET_SECONDS:

                print("\n=== Stable Objects Detected ===")
                print(detected)

                detected_count = dict(Counter(detected))

                print("\n=== Detected Ingredient Count ===")
                print(detected_count)

                prompt = f"""
Detected ingredients:
{detected_count}

Suggest one simple dish using these detected ingredients as the ONLY possible main ingredients.

CRITICAL RULES (MUST FOLLOW):
- Main ingredients MUST contain ONLY items from Detected ingredients.
- DO NOT add ANY extra items to Main ingredients under any circumstance.
- Main ingredients must be a STRICT subset of detected ingredients.
- If an ingredient is not in detected ingredients, it cannot appear in Main ingredients.

You may use additional supporting ingredients (such as onion, garlic, egg, cheese, butter, vegetables, or noodles) ONLY if they are essential for the dish.

IMPORTANT RULES:
- Salt, pepper, oil, soy sauce, sugar, spices, herbs MUST NOT be listed as "Additional ingredients".
- These MUST be included ONLY under "Seasonings/Sauce".
- Do NOT treat seasonings or cooking oils as ingredients.
- Do not duplicate items across sections.

Provide:
- A concise dish name
- Main ingredients (ONLY from detected ingredients, no exceptions)
- Additional supporting ingredients (real food ingredients only)
- Seasonings/Sauce (must include salt, pepper, oils, sauces, spices if used)
- Estimated calories for one serving
- A short step-by-step recipe with 3-5 numbered steps

Answer in English and keep it concise.

Format:
Dish:
Main ingredients:
Additional ingredients:
Seasonings/Sauce:
Estimated calories:
Recipe:
1.
2.
3.
Comment:
"""

                # Close camera window
                cap.release()
                cv2.destroyAllWindows()

                print("\n=== Sending To Gemma ===")

                # Run Gemma
                result = subprocess.run(
                    ["ollama", "run", "gemma3", prompt],
                    capture_output=True,
                    text=True
                )

                print("\n=== Gemma Response ===")
                print(result.stdout)

                break

        else:
            # New detection
            stable_detected = detected
            start_detect_time = time.time()

    else:
        # Reset if nothing detected
        stable_detected = []
        start_detect_time = None

        cv2.putText(
            annotated_frame,
            "No detection... waiting",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # Show camera window
    cv2.imshow("YOLO Inference", annotated_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
