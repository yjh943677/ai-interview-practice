# 🎤 AI Interview Practice

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange?logo=google&logoColor=white)
![DeepFace](https://img.shields.io/badge/DeepFace-0.0.x-red)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-blueviolet)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey?logo=windows)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 한국어

### 프로젝트 소개

웹캠 하나로 **실시간 화상 면접 연습**을 도와주는 Python 애플리케이션입니다.  
시선·자세·표정·고개 기울기·눈 깜빡임을 동시에 분석하고, 세션 종료 후 종합 점수와 구간별 피드백 리포트를 제공합니다.

```
웹캠 실행 → 3초 카운트다운 → 실시간 분석 → q 키 종료 → 리포트 자동 출력
```

---

### 주요 기능

| 기능 | 설명 |
|------|------|
| 👁️ **시선 분석** | MediaPipe FaceMesh 홍채 중심 추적. 연속 3프레임 이상 이탈 시 불안정 판정 |
| 🧍 **자세 분석** | 어깨 기울기 각도(±10°) + 손 위치 감지 |
| 😊 **표정 분석** | DeepFace 백그라운드 스레드 실행. angry/fear/sad → 불안정, neutral/happy → 안정 |
| 🙆 **고개 기울기** | solvePnP로 yaw(±20°) / pitch(±15°) / roll(±10°) 실시간 추정 |
| 👀 **눈 깜빡임** | EAR 기반 감지. 정상 범위 10~25회/분. 범위 이탈 시 경고 |
| 🎬 **카운트다운** | 면접 시작 전 3초 카운트다운 (분석 미포함) |
| 📊 **실시간 UI** | 우측 상단 상태 패널 (초록 ● 안정 / 빨강 ● 불안정) |
| 🏆 **종합 점수** | 시선 30% + 자세 40% + 표정 30% 가중합산 |
| 📄 **리포트 저장** | 세션 종료 시 PNG + PDF 그래프 자동 저장 |

---

### 시스템 구조

```
interview_app.py
│
├── GazeAnalyzer          # 홍채 비율 계산 → 시선 안정성 판정
├── PostureAnalyzer       # 어깨 각도 + 손 위치 → 자세 안정성 판정
├── ExpressionAnalyzer    # DeepFace 백그라운드 스레드 → 표정 안정성 판정
├── HeadPoseAnalyzer      # solvePnP yaw/pitch/roll → 고개 안정성 판정
├── BlinkAnalyzer         # EAR 기반 눈 깜빡임 → BPM 계산 및 판정
├── ScoreEngine           # 프레임 누적 → 항목별 / 종합 점수 산출
├── SessionRecorder       # 시간축 데이터 기록 → 리포트 그래프 출력
│
├── run_countdown()       # 3초 카운트다운 UI
├── draw_status_panel()   # 우측 상태 패널 렌더링
├── draw_status_bar()     # 하단 수치 정보 바 렌더링
└── main()                # 웹캠 루프 진입점
```

---

### 설치 방법

**1. 저장소 클론**

```bash
git clone https://github.com/yjh943677/ai-interview-practice.git
cd ai-interview-practice
```

**2. 가상환경 생성 (권장)**

```bash
# conda 사용 시 (Python 3.10 권장)
conda create -p ./venv python=3.10 -y
conda activate ./venv

# 또는 venv 사용 시
python -m venv venv
venv\Scripts\activate
```

**3. 의존 패키지 설치**

```bash
pip install opencv-python mediapipe==0.10.9 deepface matplotlib numpy
pip install tensorflow-cpu==2.13.0 protobuf>=3.20.3,<4 tf-keras
```

> **Windows 주의사항:**  
> - 사용자 홈 경로에 한글이 포함된 경우 DeepFace 모델 다운로드 시 인코딩 오류가 발생할 수 있습니다.  
> - 가상환경을 영문 경로(예: `C:\envs\interview_env`)에 생성하는 것을 권장합니다.  
> - 최초 실행 시 DeepFace 감정 인식 모델(~6MB)이 자동 다운로드됩니다.

---

### 실행 방법

```bash
python interview_app.py
```

| 키 | 동작 |
|----|------|
| `q` | 면접 세션 종료 → 리포트 자동 출력 |

---

### 결과 예시

세션 종료(`q` 키) 후 다음이 자동 생성됩니다.

**① 터미널 텍스트 피드백**
```
========================================================
  INTERVIEW SESSION FEEDBACK
========================================================
  Duration         : 120s  (2987 frames)
  Gaze Score       : 88.3 / 100
  Posture Score    : 91.5 / 100
  Expression Score : 79.2 / 100
  TOTAL SCORE      : 86.8 / 100
  Avg Blink Rate   : 17.4 /min  (normal: 10~25)
--------------------------------------------------------
  [Gaze]      Stable throughout!
  [Posture]   Unstable periods:
              34s ~ 41s  (7s)
  [Expression] Unstable periods:
              88s ~ 95s  (7s)
  [Head Pose] Stable throughout!

  >> Excellent! You are well-prepared for the interview.
========================================================
  Graph saved -> C:/interview/session_report.png
             -> C:/interview/session_report.pdf
```

**② matplotlib 리포트 그래프 (`session_report.png` / `.pdf`)**

- **그래프 1~4**: Gaze / Posture / Expression / Head Pose 안정성 시계열
  - 파란/초록/주황/보라 선: 5초 이동 평균 안정성 (%)
  - 회색 점선: 70% 기준선
  - 색상 음영: 70% 이상 구간 (안정)
  - **빨간 음영**: 70% 미만 구간 (불안정 하이라이트)
- **그래프 5**: Gaze / Posture / Expression / Total 점수 막대 차트

---

### 개발 환경

| 항목 | 버전 |
|------|------|
| OS | Windows 11 |
| Python | 3.10 |
| opencv-python | 4.13.0 |
| mediapipe | 0.10.9 |
| deepface | 0.0.99 |
| tensorflow-cpu | 2.13.0 |
| numpy | 1.24.3 |
| matplotlib | 3.10.x |

---

---

## English

### Project Overview

A Python application that provides **real-time AI-powered mock interview practice** using just a webcam.  
It simultaneously analyzes gaze, posture, facial expression, head pose, and blink rate, then generates a comprehensive score report with per-interval feedback at the end of each session.

```
Launch → 3-second countdown → Real-time analysis → Press q → Auto report
```

---

### Key Features

| Feature | Description |
|---------|-------------|
| 👁️ **Gaze Analysis** | MediaPipe FaceMesh iris tracking. Unstable if deviation lasts 3+ consecutive frames |
| 🧍 **Posture Analysis** | Shoulder tilt angle (±10°) + hand position detection |
| 😊 **Expression Analysis** | DeepFace in background thread. angry/fear/sad → unstable, neutral/happy → stable |
| 🙆 **Head Pose** | Real-time yaw(±20°) / pitch(±15°) / roll(±10°) via solvePnP |
| 👀 **Blink Rate** | EAR-based detection. Normal range: 10~25 BPM |
| 🎬 **Countdown** | 3-second countdown before session starts (no analysis during countdown) |
| 📊 **Live UI** | Top-right status panel (green ● stable / red ● unstable) |
| 🏆 **Total Score** | Weighted sum: Gaze 30% + Posture 40% + Expression 30% |
| 📄 **Report Export** | Auto-saves PNG + PDF graph on session end |

---

### System Architecture

```
interview_app.py
│
├── GazeAnalyzer          # Iris ratio calculation → gaze stability
├── PostureAnalyzer       # Shoulder angle + hand position → posture stability
├── ExpressionAnalyzer    # DeepFace background thread → expression stability
├── HeadPoseAnalyzer      # solvePnP yaw/pitch/roll → head stability
├── BlinkAnalyzer         # EAR-based blink detection → BPM and stability
├── ScoreEngine           # Frame accumulation → per-category and total scores
├── SessionRecorder       # Time-series recording → report graph generation
│
├── run_countdown()       # 3-second countdown UI
├── draw_status_panel()   # Right-side status panel renderer
├── draw_status_bar()     # Bottom info bar renderer
└── main()                # Webcam loop entry point
```

---

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/yjh943677/ai-interview-practice.git
cd ai-interview-practice
```

**2. Create a virtual environment (recommended)**

```bash
# Using conda (Python 3.10 recommended)
conda create -p ./venv python=3.10 -y
conda activate ./venv

# Or using venv
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install opencv-python mediapipe==0.10.9 deepface matplotlib numpy
pip install tensorflow-cpu==2.13.0 "protobuf>=3.20.3,<4" tf-keras
```

> **Windows Notes:**  
> - If your home directory path contains non-ASCII characters, DeepFace model download may fail due to encoding errors.  
> - It is recommended to create the virtual environment under an ASCII-only path (e.g. `C:\envs\interview_env`).  
> - On first run, the DeepFace emotion model (~6MB) will be downloaded automatically.

---

### How to Run

```bash
python interview_app.py
```

| Key | Action |
|-----|--------|
| `q` | End session → auto-generate report |

---

### Report Output

After pressing `q`, the following are generated automatically.

**① Terminal text feedback**

```
========================================================
  INTERVIEW SESSION FEEDBACK
========================================================
  Duration         : 120s  (2987 frames)
  Gaze Score       : 88.3 / 100
  Posture Score    : 91.5 / 100
  Expression Score : 79.2 / 100
  TOTAL SCORE      : 86.8 / 100
  Avg Blink Rate   : 17.4 /min  (normal: 10~25)
  ...
  >> Excellent! You are well-prepared for the interview.
```

**② matplotlib report graph (`session_report.png` / `.pdf`)**

- **Graphs 1~4**: Time-series stability for Gaze / Posture / Expression / Head Pose
  - Colored line: 5-second moving average stability (%)
  - Gray dashed line: 70% threshold
  - Colored fill: stable zones (≥70%)
  - **Red fill**: unstable zones (<70% highlight)
- **Graph 5**: Bar chart for Gaze / Posture / Expression / Total scores

---

### Development Environment

| Item | Version |
|------|---------|
| OS | Windows 11 |
| Python | 3.10 |
| opencv-python | 4.13.0 |
| mediapipe | 0.10.9 |
| deepface | 0.0.99 |
| tensorflow-cpu | 2.13.0 |
| numpy | 1.24.3 |
| matplotlib | 3.10.x |
