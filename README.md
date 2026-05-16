# On-Device AI Ingredient Recipe Recommendation

카메라로 식재료를 인식하고, 감지된 재료를 기반으로 간단한 레시피와 예상 칼로리를 추천하는 On-Device AI 프로젝트입니다.

YOLO11s 모델을 사용해 식재료를 탐지하고, TensorRT로 변환한 모델을 Jetson 환경에서 실행합니다.  
감지된 재료는 Gemma3 모델에 전달되어 레시피 추천 결과로 출력됩니다.

---

## Features

- 실시간 카메라 기반 식재료 인식
- YOLO11s 기반 객체 탐지
- TensorRT 엔진 모델을 이용한 추론
- 5초 안정 탐지 필터링
- 감지된 재료 개수 계산
- Gemma3 기반 레시피 및 예상 칼로리 추천
- 터미널 결과 출력

---

## Project Structure

```text
project/
├── 2.convert_model_to_TensorRT.py
├── recipetest.py
├── weights/
│   ├── best.pt
│   └── best.engine
├── test.jpg
└── README.md
```

---

## Files

| File | Description |
|---|---|
| `2.convert_model_to_TensorRT.py` | YOLO 모델 `best.pt`를 TensorRT 엔진 모델로 변환하는 코드 |
| `recipetest.py` | 카메라 실행, 식재료 탐지, 레시피 추천을 수행하는 메인 코드 |
| `best.pt` | 학습된 YOLO11s 모델 |
| `best.engine` | TensorRT로 변환된 추론용 모델 |
| `test.jpg` | TensorRT 변환 후 테스트 추론에 사용할 이미지 |

---

## Requirements

본 프로젝트는 Jetson Orin Nano 환경에서 실행하는 것을 기준으로 작성되었습니다.

필요한 주요 환경은 다음과 같습니다.

- Python
- OpenCV
- Ultralytics YOLO
- TensorRT
- Ollama
- Gemma3
- Camera

Python 패키지는 아래 명령어로 설치할 수 있습니다.

```bash
pip install ultralytics opencv-python
```

Ollama에서 Gemma3 모델을 사용할 수 있도록 준비합니다.

```bash
ollama pull gemma3
```

---

## Model

본 프로젝트에서는 YOLO11s 기반 식재료 탐지 모델을 사용합니다.

학습된 모델 파일은 다음 경로에 위치해야 합니다.

```text
weights/best.pt
```

TensorRT 변환 후 생성되는 엔진 모델은 다음 경로에 위치해야 합니다.

```text
weights/best.engine
```

코드 내부의 모델 경로는 실행 환경에 맞게 수정해야 합니다.

예시:

```python
model = YOLO("weights/best.engine")
```

---

## How to Run

### 1. Clone Repository

```bash
git clone <repository-url>
cd <repository-name>
```

---

### 2. Install Dependencies

```bash
pip install ultralytics opencv-python
```

---

### 3. Prepare Gemma3

```bash
ollama pull gemma3
```

Ollama가 정상적으로 설치되어 있고, `gemma3` 모델이 실행 가능한 상태여야 합니다.

---

### 4. Convert YOLO Model to TensorRT

학습된 `best.pt` 모델을 TensorRT 엔진 모델로 변환합니다.

```bash
python 2.convert_model_to_TensorRT.py
```

변환이 완료되면 `best.engine` 파일이 생성됩니다.

---

### 5. Run Main Program

카메라를 연결한 뒤 메인 코드를 실행합니다.

```bash
python recipetest.py
```

실행 후 카메라 앞에 음식 재료를 보여주면 YOLO 모델이 식재료를 탐지합니다.

같은 탐지 결과가 5초 동안 유지되면 해당 재료가 최종 확정되고, Gemma3가 레시피와 예상 칼로리를 생성합니다.

---

## Output Example

```text
=== Stable Objects Detected ===
['rice', 'kimchi', 'pork']

=== Detected Ingredient Count ===
{'rice': 1, 'kimchi': 1, 'pork': 1}

=== Sending To Gemma ===

=== Gemma Response ===
Dish:
Kimchi Pork Rice Bowl

Main ingredients:
rice, kimchi, pork

Additional ingredients:
None

Seasonings/Sauce:
soy sauce, sesame oil, pepper

Estimated calories:
Approximately 550 kcal

Recipe:
1. Cook the pork in a pan until browned.
2. Add kimchi and stir-fry together.
3. Serve over warm rice.

Comment:
A simple dish using the detected ingredients as the main ingredients.
```

---


## Notes

- `best.pt`, `best.engine` 파일 경로는 실행 환경에 맞게 수정해야 합니다.
- 카메라 장치 번호가 다를 경우 `cv2.VideoCapture(0)`의 숫자를 변경해야 합니다.
- TensorRT 변환은 Jetson 환경에서 실행하는 것을 권장합니다.
- Ollama와 Gemma3가 설치되어 있어야 레시피 추천 기능이 동작합니다.
- 모델 파일 용량이 큰 경우 GitHub 업로드 시 Git LFS 사용을 권장합니다.
- 현재 결과는 터미널에 출력됩니다.

—
