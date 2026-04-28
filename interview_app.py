"""
AI 화상 면접 연습 프로그램 - 4단계: 고개 자세 + 눈 깜빡임 + UI 개선
------------------------------------------------------------------------------
실행 방법: python interview_app.py
종료      : 화면에서 q 키 입력 → 자동으로 그래프 + 피드백 출력
추가 설치 : pip install deepface tf-keras
"""

import io
import queue
import sys
import threading
import time
from collections import deque

# Windows cp949 터미널에서 이모지 출력 오류 방지
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

print("[1/3] Loading MediaPipe...", flush=True)
import cv2
import mediapipe as mp
import numpy as np

print("[2/3] Loading DeepFace + TensorFlow (first run may take a while)...", flush=True)
from deepface import DeepFace as _DeepFace

print("[3/3] Loading matplotlib...", flush=True)
import matplotlib.pyplot as plt

print("All libraries loaded. Starting application...\n", flush=True)

# ── MediaPipe 솔루션 초기화 ───────────────────────────────────────────────
mp_face_mesh      = mp.solutions.face_mesh
mp_pose           = mp.solutions.pose
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ── 튜닝 상수 ─────────────────────────────────────────────────────────────
SHOULDER_ANGLE_MAX = 10.0
INSTABILITY_FRAMES = 12
HISTORY_WINDOW     = 30
EXPR_ANALYZE_EVERY = 15

# 점수 가중치
WEIGHT_GAZE    = 0.30
WEIGHT_POSTURE = 0.40
WEIGHT_EXPR    = 0.30

# 홍채 랜드마크 인덱스
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER  = 473
RIGHT_EYE_BOUND   = {"left": 33,  "right": 133, "top": 159, "bottom": 145}
LEFT_EYE_BOUND    = {"left": 362, "right": 263, "top": 386, "bottom": 374}

# 표정 분류
STABLE_EMOTIONS   = {"neutral", "happy"}
UNSTABLE_EMOTIONS = {"angry", "fear", "sad", "disgust", "surprise"}

# 고개 자세 임계값 (도)
YAW_MAX   = 20.0
PITCH_MAX = 15.0
ROLL_MAX  = 10.0

# 고개 자세 계산용 3D 기준점 (정면 얼굴 모델)
_MODEL_POINTS = np.array([
    [0.0,    0.0,    0.0   ],   # 코끝 (1)
    [0.0,   -330.0, -65.0 ],   # 턱 끝 (152)
    [-225.0, 170.0, -135.0],   # 왼쪽 눈 왼쪽 끝 (263)
    [225.0,  170.0, -135.0],   # 오른쪽 눈 오른쪽 끝 (33)
    [-150.0,-150.0, -125.0],   # 입 왼쪽 끝 (287)
    [150.0, -150.0, -125.0],   # 입 오른쪽 끝 (57)
], dtype=np.float64)
_MODEL_IDX = [1, 152, 263, 33, 287, 57]

# EAR 계산용 눈 랜드마크 인덱스
_RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
_LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]


# ── 시선 분석기 ───────────────────────────────────────────────────────────
class GazeAnalyzer:
    """픽셀 기반 Safe Zone으로 시선 이탈 감지 (Rogers et al., 2018).
    화면 중앙 상단 가로 40%, 세로 35% 영역을 기준으로 연속 3초 이탈 시 1회 카운트.
    세션 5회 이상 이탈 → warn=True."""
    SAFE_W_RATIO = 0.40
    SAFE_H_RATIO = 0.35
    DEVIATE_SEC  = 3.0
    MAX_EVENTS   = 5

    def __init__(self):
        self._deviate_start: float | None = None
        self._last_count_end: float       = -999.0
        self.deviate_events: int          = 0

    def safe_zone(self, img_w, img_h):
        """Safe Zone (x1, x2, y1, y2) 픽셀 좌표 반환."""
        cx = img_w / 2
        hw = img_w * self.SAFE_W_RATIO / 2
        return int(cx - hw), int(cx + hw), 0, int(img_h * self.SAFE_H_RATIO)

    def analyze(self, face_landmarks, img_w, img_h, now: float):
        """(warn, iris_px, is_unstable, safe_zone) 반환."""
        lm = face_landmarks.landmark
        try:
            iris_x = (lm[RIGHT_IRIS_CENTER].x + lm[LEFT_IRIS_CENTER].x) / 2 * img_w
            iris_y = (lm[RIGHT_IRIS_CENTER].y + lm[LEFT_IRIS_CENTER].y) / 2 * img_h
        except Exception:
            self._deviate_start = None
            return False, (img_w / 2, img_h / 2), False, self.safe_zone(img_w, img_h)

        x1, x2, y1, y2 = self.safe_zone(img_w, img_h)
        raw_deviated = not (x1 <= iris_x <= x2 and y1 <= iris_y <= y2)

        if raw_deviated:
            if self._deviate_start is None:
                self._deviate_start = now
            # 연속 3초 이탈이고, 마지막 카운트 이후 새 이탈이면 1회 추가
            elif (now - self._deviate_start >= self.DEVIATE_SEC and
                  self._deviate_start > self._last_count_end):
                self.deviate_events += 1
                self._last_count_end  = now
                self._deviate_start   = now   # 다음 3초 카운트를 위해 리셋
        else:
            self._deviate_start = None

        warn = self.deviate_events >= self.MAX_EVENTS
        return warn, (iris_x, iris_y), raw_deviated, (x1, x2, y1, y2)


# ── 자세 분석기 ───────────────────────────────────────────────────────────
class PostureAnalyzer:
    """어깨 기울기 5초 지속 판정 + FaceMesh 바운딩박스 기반 손 제스처 감지."""
    SHOULDER_PERSIST_SEC = 5.0
    GESTURE_WARN_COUNT   = 3

    # 손목·손가락 끝 Pose 랜드마크 인덱스
    _HAND_LM = [
        mp_pose.PoseLandmark.LEFT_WRIST,   mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_INDEX,   mp_pose.PoseLandmark.RIGHT_INDEX,
        mp_pose.PoseLandmark.LEFT_PINKY,   mp_pose.PoseLandmark.RIGHT_PINKY,
        mp_pose.PoseLandmark.LEFT_THUMB,   mp_pose.PoseLandmark.RIGHT_THUMB,
    ]

    def __init__(self):
        self._tilt_start: float | None = None
        self._tilt_warn                = False
        self.gesture_count: int        = 0
        self._hand_was_in_face         = False

    def _shoulder_angle(self, lm):
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        return abs(np.degrees(np.arctan2(rs.y - ls.y, abs(rs.x - ls.x))))

    def _face_bbox(self, face_lm, img_w, img_h):
        xs = [p.x * img_w for p in face_lm.landmark]
        ys = [p.y * img_h for p in face_lm.landmark]
        return int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))

    def _hand_in_face(self, pose_lm, x1, x2, y1, y2, img_w, img_h):
        lm = pose_lm.landmark
        for idx in self._HAND_LM:
            pt = lm[idx]
            if pt.visibility < 0.5:
                continue
            px, py = pt.x * img_w, pt.y * img_h
            if x1 <= px <= x2 and y1 <= py <= y2:
                return True
        return False

    def analyze(self, pose_landmarks, now: float,
                face_lm=None, img_w: int = 0, img_h: int = 0):
        lm     = pose_landmarks.landmark
        issues = []
        angle  = self._shoulder_angle(lm)

        # 어깨 기울기: 5초 이상 지속될 때만 불안정 판정
        if angle > SHOULDER_ANGLE_MAX:
            if self._tilt_start is None:
                self._tilt_start = now
            elif now - self._tilt_start >= self.SHOULDER_PERSIST_SEC:
                self._tilt_warn = True
                issues.append(f"Shoulder tilt {angle:.1f}deg")
        else:
            self._tilt_start = None
            self._tilt_warn  = False

        # 손 제스처: FaceMesh 바운딩박스 침범 → 엣지 트리거로 카운트
        hand_now = False
        if face_lm is not None and img_w > 0 and img_h > 0:
            x1, x2, y1, y2 = self._face_bbox(face_lm, img_w, img_h)
            hand_now = self._hand_in_face(pose_landmarks, x1, x2, y1, y2, img_w, img_h)
            if hand_now and not self._hand_was_in_face:
                self.gesture_count += 1
            if hand_now:
                issues.append("Hand near face")
        self._hand_was_in_face = hand_now

        gesture_warn = self.gesture_count >= self.GESTURE_WARN_COUNT
        if gesture_warn:
            issues.append(f"Hand gesture x{self.gesture_count}")

        unstable = self._tilt_warn or hand_now
        warn     = self._tilt_warn or gesture_warn
        return warn, angle, issues, unstable


# ── 고개 자세 분석기 ──────────────────────────────────────────────────────
class HeadPoseAnalyzer:
    """FaceMesh 랜드마크로 yaw / pitch / roll 을 추정합니다.
    solvePnP 로 회전 벡터를 구한 뒤 오일러 각으로 변환합니다."""

    def __init__(self):
        self.history: deque[bool] = deque(maxlen=HISTORY_WINDOW)

    def _euler(self, face_landmarks, img_w, img_h):
        lm = face_landmarks.landmark
        img_pts = np.array(
            [(lm[i].x * img_w, lm[i].y * img_h) for i in _MODEL_IDX],
            dtype=np.float64,
        )
        focal   = img_w
        cam_mat = np.array(
            [[focal, 0,     img_w / 2],
             [0,     focal, img_h / 2],
             [0,     0,     1        ]], dtype=np.float64,
        )
        dist = np.zeros((4, 1))
        ok, rvec, _ = cv2.solvePnP(
            _MODEL_POINTS, img_pts, cam_mat, dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        # 오일러 각 (pitch, yaw, roll)
        sy    = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        yaw   = np.degrees(np.arctan2(-rmat[1, 0], -rmat[0, 0]))  # 거울 모드 보정
        roll  = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
        return yaw, pitch, roll

    def analyze(self, face_landmarks, img_w, img_h):
        """(warn, yaw, pitch, roll, is_unstable) 반환."""
        try:
            yaw, pitch, roll = self._euler(face_landmarks, img_w, img_h)
        except Exception:
            self.history.append(False)
            return False, 0.0, 0.0, 0.0, False

        unstable = (
            abs(yaw)   > YAW_MAX   or
            abs(pitch) > PITCH_MAX or
            abs(roll)  > ROLL_MAX
        )
        self.history.append(unstable)
        warn = sum(self.history) >= INSTABILITY_FRAMES
        return warn, yaw, pitch, roll, unstable


# ── 눈 깜빡임 분석기 ──────────────────────────────────────────────────────
class BlinkAnalyzer:
    """10초 단위 구간 기반 깜빡임 분석.
    과긴장: 10초에 7회 초과 (≈42/min, 평균+2.5SD, Bentivoglio 1997 / Haak 2008)
    긴장 경직: 10초에 1회 이하 (집중 시 감소, Doughty 2001)"""
    EAR_THRESHOLD    = 0.20
    INTERVAL_SEC     = 10.0
    BLINK_HIGH_PER10 = 7   # 10초에 7회 초과 → 과긴장
    BLINK_LOW_PER10  = 1   # 10초에 1회 이하 → 긴장 경직

    def __init__(self):
        self._eye_closed      = False
        self._interval_start: float | None = None
        self._interval_blinks = 0
        self._last_status     = "normal"   # "normal" / "hyper" / "rigid"
        self.total_blinks     = 0
        self._blink_times: deque[float] = deque()  # BPM 표시용 60초 rolling

    @staticmethod
    def _ear(lm, indices, img_w, img_h):
        pts = np.array([(lm[i].x * img_w, lm[i].y * img_h) for i in indices])
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3]) + 1e-6
        return (A + B) / (2.0 * C)

    def analyze(self, face_landmarks, img_w, img_h, now: float):
        """(bpm, is_unstable, blinked_this_frame) 반환."""
        lm  = face_landmarks.landmark
        ear = (self._ear(lm, _RIGHT_EYE_EAR, img_w, img_h) +
               self._ear(lm, _LEFT_EYE_EAR,  img_w, img_h)) / 2.0

        if self._interval_start is None:
            self._interval_start = now

        blinked = False
        if ear < self.EAR_THRESHOLD:
            self._eye_closed = True
        elif self._eye_closed:
            self._eye_closed = False
            self._interval_blinks += 1
            self.total_blinks += 1
            self._blink_times.append(now)
            blinked = True

        # 10초 구간 평가 후 초기화
        if now - self._interval_start >= self.INTERVAL_SEC:
            if self._interval_blinks > self.BLINK_HIGH_PER10:
                self._last_status = "hyper"
            elif self._interval_blinks <= self.BLINK_LOW_PER10:
                self._last_status = "rigid"
            else:
                self._last_status = "normal"
            self._interval_start  = now
            self._interval_blinks = 0

        # BPM 표시용 (60초 rolling)
        while self._blink_times and self._blink_times[0] < now - 60.0:
            self._blink_times.popleft()
        elapsed_window = min(now, 60.0) if now > 0 else 1.0
        bpm = len(self._blink_times) / elapsed_window * 60.0

        unstable = self._last_status != "normal"
        return bpm, unstable, blinked


# ── 표정 분석기 ───────────────────────────────────────────────────────────
class ExpressionAnalyzer:
    def __init__(self):
        self._lock            = threading.Lock()
        self._latest_emotion  = "neutral"
        self._is_unstable     = False
        self._frame_queue     = queue.Queue(maxsize=1)
        self._running         = True
        self.history: deque[bool] = deque(maxlen=HISTORY_WINDOW)

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while self._running:
            try:
                frame = self._frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                result  = _DeepFace.analyze(
                    frame, actions=["emotion"],
                    enforce_detection=False, silent=True)
                emotion = result[0]["dominant_emotion"].lower()
            except Exception:
                emotion = "neutral"
            with self._lock:
                self._latest_emotion = emotion
                self._is_unstable    = emotion in UNSTABLE_EMOTIONS

    def submit_frame(self, frame: np.ndarray):
        try:
            self._frame_queue.put_nowait(frame.copy())
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass

    def get_state(self):
        with self._lock:
            return self._latest_emotion, self._is_unstable

    def update_history(self, is_unstable: bool):
        self.history.append(is_unstable)
        return sum(self.history) >= INSTABILITY_FRAMES

    def stop(self):
        self._running = False


# ── 점수 산출 엔진 ────────────────────────────────────────────────────────
class ScoreEngine:
    """시선 30% + 자세 40% + 표정 30%. HeadPose / Blink 는 표시만."""

    def __init__(self):
        self.gaze_stable    = 0;  self.gaze_total    = 0
        self.posture_stable = 0;  self.posture_total = 0
        self.expr_stable    = 0;  self.expr_total    = 0

    def update(self, gaze_unstable, posture_unstable, expr_unstable,
               face_detected, pose_detected):
        if face_detected:
            self.gaze_total += 1
            if not gaze_unstable:
                self.gaze_stable += 1
            self.expr_total += 1
            if not expr_unstable:
                self.expr_stable += 1
        if pose_detected:
            self.posture_total += 1
            if not posture_unstable:
                self.posture_stable += 1

    @property
    def gaze_score(self):
        return 100 * self.gaze_stable / self.gaze_total if self.gaze_total else 0.0

    @property
    def posture_score(self):
        return 100 * self.posture_stable / self.posture_total if self.posture_total else 0.0

    @property
    def expr_score(self):
        return 100 * self.expr_stable / self.expr_total if self.expr_total else 100.0

    @property
    def total_score(self):
        return (self.gaze_score    * WEIGHT_GAZE +
                self.posture_score * WEIGHT_POSTURE +
                self.expr_score    * WEIGHT_EXPR)


# ── 세션 기록기 ───────────────────────────────────────────────────────────
class SessionRecorder:
    def __init__(self):
        self.timestamps:          list[float] = []
        self.gaze_stable_log:     list[bool]  = []
        self.posture_stable_log:  list[bool]  = []
        self.expr_stable_log:     list[bool]  = []
        self.head_stable_log:     list[bool]  = []

    def record(self, elapsed, gaze_unstable, posture_unstable,
               expr_unstable, head_unstable):
        self.timestamps.append(elapsed)
        self.gaze_stable_log.append(not gaze_unstable)
        self.posture_stable_log.append(not posture_unstable)
        self.expr_stable_log.append(not expr_unstable)
        self.head_stable_log.append(not head_unstable)

    def plot_and_report(self, score_engine: ScoreEngine,
                        blink_analyzer: BlinkAnalyzer):
        if not self.timestamps:
            print("No recorded data.")
            return

        t = np.array(self.timestamps)
        g = np.array(self.gaze_stable_log,    dtype=float)
        p = np.array(self.posture_stable_log, dtype=float)
        e = np.array(self.expr_stable_log,    dtype=float)
        h = np.array(self.head_stable_log,    dtype=float)

        fps_est  = len(t) / (t[-1] + 1e-6)
        win      = max(1, int(fps_est * 5))
        kernel   = np.ones(win) / win
        g_smooth = np.convolve(g, kernel, mode="same")
        p_smooth = np.convolve(p, kernel, mode="same")
        e_smooth = np.convolve(e, kernel, mode="same")
        h_smooth = np.convolve(h, kernel, mode="same")

        THRESHOLD = 70

        def draw_stability_graph(ax, t, smooth, color, ylabel):
            pct = smooth * 100
            ax.plot(t, pct, color=color, linewidth=1.5)
            ax.axhline(THRESHOLD, color="gray", linestyle="--", linewidth=0.8)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 105)
            ax.fill_between(t, pct, alpha=0.15, color=color,
                            where=(pct >= THRESHOLD))
            ax.fill_between(t, pct, alpha=0.15, color="red",
                            where=(pct < THRESHOLD))

        fig, axes = plt.subplots(5, 1, figsize=(12, 13))
        fig.suptitle("Interview Session Report", fontsize=14, fontweight="bold")

        draw_stability_graph(axes[0], t, g_smooth, "#2196F3", "Gaze (%)")
        draw_stability_graph(axes[1], t, p_smooth, "#4CAF50", "Posture (%)")
        draw_stability_graph(axes[2], t, e_smooth, "#FF9800", "Expression (%)")
        draw_stability_graph(axes[3], t, h_smooth, "#9C27B0", "Head Pose (%)")

        # ⑤ 점수 요약 막대
        labels = ["Gaze\n(30%)", "Posture\n(40%)", "Expression\n(30%)", "Total"]
        scores = [score_engine.gaze_score, score_engine.posture_score,
                  score_engine.expr_score, score_engine.total_score]
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
        bars   = axes[4].bar(labels, scores, color=colors, width=0.5)
        axes[4].set_ylim(0, 110)
        axes[4].set_ylabel("Score")
        for bar, score in zip(bars, scores):
            axes[4].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 1.5,
                         f"{score:.1f}", ha="center", va="bottom", fontsize=10)

        for ax in axes[:4]:
            ax.set_xlabel("Time (s)")
            ax.grid(True, alpha=0.3)
        axes[4].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig("C:/interview/session_report.png", dpi=120)
        plt.savefig("C:/interview/session_report.pdf")
        plt.show(block=True)

        # ── 텍스트 피드백 ────────────────────────────────────────────────
        duration = t[-1]
        total_bpm = blink_analyzer.total_blinks / (duration / 60.0) if duration > 0 else 0
        print("\n" + "=" * 56)
        print("  INTERVIEW SESSION FEEDBACK")
        print("=" * 56)
        print(f"  Duration         : {duration:.0f}s  ({len(t)} frames)")
        print(f"  Gaze Score       : {score_engine.gaze_score:.1f} / 100")
        print(f"  Posture Score    : {score_engine.posture_score:.1f} / 100")
        print(f"  Expression Score : {score_engine.expr_score:.1f} / 100")
        print(f"  TOTAL SCORE      : {score_engine.total_score:.1f} / 100")
        lo = BlinkAnalyzer.BLINK_LOW_PER10  * 6   # 10초 기준 → 분당 환산
        hi = BlinkAnalyzer.BLINK_HIGH_PER10 * 6
        print(f"  Avg Blink Rate   : {total_bpm:.1f} /min  "
              f"(normal: {lo}~{hi})")
        print("-" * 56)

        def find_unstable_intervals(stable_log, timestamps, min_dur=1.5):
            intervals, in_bad, start = [], False, 0.0
            for s, ts in zip(stable_log, timestamps):
                if not s and not in_bad:
                    in_bad, start = True, ts
                elif s and in_bad:
                    if ts - start >= min_dur:
                        intervals.append((start, ts))
                    in_bad = False
            if in_bad and timestamps[-1] - start >= min_dur:
                intervals.append((start, timestamps[-1]))
            return intervals

        for label, log in [
            ("Gaze",       self.gaze_stable_log),
            ("Posture",    self.posture_stable_log),
            ("Expression", self.expr_stable_log),
            ("Head Pose",  self.head_stable_log),
        ]:
            bad = find_unstable_intervals(log, self.timestamps)
            if bad:
                print(f"  [{label}] Unstable periods:")
                for s, e in bad:
                    print(f"            {s:.0f}s ~ {e:.0f}s  ({e-s:.0f}s)")
            else:
                print(f"  [{label}] Stable throughout!")

        total = score_engine.total_score
        if total >= 85:
            comment = "Excellent! You are well-prepared for the interview."
        elif total >= 70:
            comment = "Good. Minor improvements recommended."
        elif total >= 50:
            comment = "Fair. Practice maintaining gaze, posture, and calm expression."
        else:
            comment = "Needs improvement. Focus on camera contact, posture, and expression."
        print(f"\n  >> {comment}")
        print("=" * 56)
        print(f"  Graph saved -> C:/interview/session_report.png")
        print(f"             -> C:/interview/session_report.pdf")
        print("=" * 56 + "\n")


# ── UI 드로잉 헬퍼 ────────────────────────────────────────────────────────
def draw_status_panel(frame, states: dict):
    """우측 상단에 반투명 상태 패널을 그립니다.
    states = {"Gaze": True, "Posture": False, ...}  (True = 안정, False = 불안정)
    """
    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = 0.55
    thick     = 1
    pad       = 10
    row_h     = 28
    panel_w   = 160
    panel_h   = pad * 2 + row_h * len(states)
    img_h, img_w = frame.shape[:2]

    x0 = img_w - panel_w - 10
    y0 = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, (label, stable) in enumerate(states.items()):
        color  = (60, 200, 60) if stable else (60, 60, 220)
        symbol = "OK" if stable else " !"
        y      = y0 + pad + row_h * i + 18
        # 색상 원
        cv2.circle(frame, (x0 + 14, y - 5), 6, color, -1, cv2.LINE_AA)
        # 레이블 + 상태
        cv2.putText(frame, f"{label:<12}{symbol}", (x0 + 26, y),
                    font, scale, (230, 230, 230), thick, cv2.LINE_AA)


def draw_status_bar(frame, gaze_ratio, shoulder_angle, emotion,
                    yaw, pitch, roll, bpm, fps, score_engine: ScoreEngine):
    """하단 상태 바."""
    h, w    = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 36), (w, h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    text = (f"FPS {fps:4.1f}  |  "
            f"Gaze X:{gaze_ratio[0]:.0f} Y:{gaze_ratio[1]:.0f}  |  "
            f"Shoulder {shoulder_angle:.1f}deg  |  "
            f"Expr: {emotion:<9}|  "
            f"Head Y:{yaw:+.0f} P:{pitch:+.0f} R:{roll:+.0f}  |  "
            f"Blink:{bpm:.0f}/min  |  "
            f"Score G:{score_engine.gaze_score:.0f} "
            f"P:{score_engine.posture_score:.0f} "
            f"E:{score_engine.expr_score:.0f} "
            f"T:{score_engine.total_score:.0f}")
    cv2.putText(frame, text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 190, 190), 1, cv2.LINE_AA)


def draw_landmarks(frame, face_lm, pose_lm):
    if face_lm:
        mp_drawing.draw_landmarks(
            frame, face_lm, mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
        mp_drawing.draw_landmarks(
            frame, face_lm, mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
    if pose_lm:
        mp_drawing.draw_landmarks(
            frame, pose_lm, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())


# ── 카운트다운 ────────────────────────────────────────────────────────────
def run_countdown(cap, window_title: str):
    font, steps = cv2.FONT_HERSHEY_SIMPLEX, ["3", "2", "1", "Start!"]
    for label in steps:
        deadline = time.time() + (0.5 if label == "Start!" else 1.0)
        while time.time() < deadline:
            ret, frame = cap.read()
            if not ret:
                return
            frame        = cv2.flip(frame, 1)
            img_h, img_w = frame.shape[:2]
            cx, cy       = img_w // 2, img_h // 2
            is_start     = (label == "Start!")
            fscale       = 3.0 if is_start else 4.0
            thick        = 8   if is_start else 10
            (tw, th), bl = cv2.getTextSize(label, font, fscale, thick)
            radius  = max(tw, th) // 2 + 40
            overlay = frame.copy()
            cv2.circle(overlay, (cx, cy), radius, (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            cv2.putText(frame, label,
                        (cx - tw // 2, cy + th // 2 - bl // 2),
                        font, fscale, (255, 255, 255), thick, cv2.LINE_AA)
            cv2.imshow(window_title, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return


# ── 메인 루프 ─────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    gaze_analyzer    = GazeAnalyzer()
    posture_analyzer = PostureAnalyzer()
    head_analyzer    = HeadPoseAnalyzer()
    blink_analyzer   = BlinkAnalyzer()
    expr_analyzer    = ExpressionAnalyzer()
    score_engine     = ScoreEngine()
    recorder         = SessionRecorder()

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose = mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    WIN_TITLE = "AI Interview Practice - Phase 4  (q: quit)"

    print("=" * 56)
    print("  AI Interview Practice - Phase 4")
    print("  Gaze + Posture + Expression + Head + Blink")
    print("=" * 56)
    print("  [q] Quit and show report")
    print("=" * 56)

    run_countdown(cap, WIN_TITLE)

    session_start = time.time()
    frame_count   = 0

    try:
        while True:
            frame_start = time.time()
            ret, frame  = cap.read()
            if not ret:
                print("[ERROR] Failed to capture frame.")
                break

            frame        = cv2.flip(frame, 1)
            img_h, img_w = frame.shape[:2]
            rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elapsed      = time.time() - session_start
            now          = elapsed
            frame_count += 1

            # ── 분석 ────────────────────────────────────────────────────
            face_result = face_mesh.process(rgb)
            pose_result = pose.process(rgb)

            if frame_count % EXPR_ANALYZE_EVERY == 0 and face_result.multi_face_landmarks:
                expr_analyzer.submit_frame(frame)

            # 기본값
            gaze_warn        = False;  gaze_ratio       = (0.5, 0.5)
            gaze_unstable    = False
            posture_warn     = False;  shoulder_ang     = 0.0
            pose_issues      = [];     posture_unstable = False
            head_warn        = False;  yaw = pitch = roll = 0.0
            head_unstable    = False
            bpm              = 0.0;    blink_unstable   = False
            face_lm          = None

            gaze_safe_zone = None
            if face_result.multi_face_landmarks:
                face_lm = face_result.multi_face_landmarks[0]
                gaze_warn, gaze_ratio, gaze_unstable, gaze_safe_zone = \
                    gaze_analyzer.analyze(face_lm, img_w, img_h, now)
                head_warn, yaw, pitch, roll, head_unstable = head_analyzer.analyze(
                    face_lm, img_w, img_h)
                bpm, blink_unstable, _ = blink_analyzer.analyze(
                    face_lm, img_w, img_h, now)

            if pose_result.pose_landmarks:
                posture_warn, shoulder_ang, pose_issues, posture_unstable = \
                    posture_analyzer.analyze(pose_result.pose_landmarks, now,
                                             face_lm, img_w, img_h)

            emotion, expr_unstable_raw = expr_analyzer.get_state()
            expr_warn     = expr_analyzer.update_history(expr_unstable_raw)
            expr_unstable = expr_unstable_raw

            # ── 점수 & 기록 ──────────────────────────────────────────────
            face_detected = face_result.multi_face_landmarks is not None
            pose_detected = pose_result.pose_landmarks is not None
            score_engine.update(gaze_unstable, posture_unstable, expr_unstable,
                                face_detected, pose_detected)
            recorder.record(elapsed, gaze_unstable, posture_unstable,
                            expr_unstable, head_unstable)

            # ── 랜드마크 시각화 ──────────────────────────────────────────
            draw_landmarks(frame, face_lm, pose_result.pose_landmarks)

            # ── Safe Zone 시각화 ─────────────────────────────────────────
            if gaze_safe_zone is not None:
                sx1, sx2, sy1, sy2 = gaze_safe_zone
                sz_color = (60, 60, 220) if gaze_unstable else (60, 200, 60)
                cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), sz_color, 2, cv2.LINE_AA)
                cv2.putText(frame, "Safe Zone", (sx1 + 4, sy2 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, sz_color, 1, cv2.LINE_AA)
                # 홍채 중심 점
                ix, iy = int(gaze_ratio[0]), int(gaze_ratio[1])
                cv2.circle(frame, (ix, iy), 5, sz_color, -1, cv2.LINE_AA)
                # 이탈 이벤트 카운트
                cv2.putText(frame, f"Gaze out: {gaze_analyzer.deviate_events}/5",
                            (sx1 + 4, max(sy2 + 18, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, sz_color, 1, cv2.LINE_AA)

            # ── 우측 상태 패널 (경고 박스 대체) ─────────────────────────
            panel_states = {
                "Gaze":    not gaze_warn,
                "Posture": not posture_warn,
                "Expr":    not expr_warn,
                "Head":    not head_warn,
                "Blink":   not blink_unstable,
            }
            draw_status_panel(frame, panel_states)

            # ── 하단 상태 바 & 경과 시간 ─────────────────────────────────
            fps = 1.0 / max(time.time() - frame_start, 1e-6)
            draw_status_bar(frame, gaze_ratio, shoulder_ang, emotion,
                            yaw, pitch, roll, bpm, fps, score_engine)

            cv2.putText(frame, f"{elapsed:.0f}s",
                        (img_w - 70, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (200, 200, 200), 2, cv2.LINE_AA)

            cv2.imshow(WIN_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        expr_analyzer.stop()
        cap.release()
        face_mesh.close()
        pose.close()
        recorder.plot_and_report(score_engine, blink_analyzer)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
