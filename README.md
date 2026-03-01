# TKD Analysis — Head Tracking with Auto Re-tracking

## 개요

최대 2명의 머리 위치를 추적하는 프로그램입니다.
한 번 클릭하면 OpenCV CSRT 트래커가 자동으로 머리를 따라가며,
추적이 끊겼을 때 템플릿 매칭으로 자동 재탐색합니다.
호모그래피(Homography)를 통해 비스듬한 카메라 영상을 Bird's-Eye 직각 좌표로 변환합니다.

---

## 파일 구조

```
TKD_analysis/
├── webcam_head_tracking_retrack.py   # 실시간 웹캠 추적
├── webcam_head_tracking_trail.py     # 실시간 웹캠 추적 + 궤적(Trail) 시각화
├── video_head_tracking_retrack.py    # 녹화 영상 파일 추적
├── requirements.txt                  # Python 패키지 목록
└── README.md
```

---

## 설치

```bash
pip install -r requirements.txt
```

---

## 실행

### 웹캠 (실시간)
```bash
python webcam_head_tracking_retrack.py
python webcam_head_tracking_retrack.py --cam-id 1 --conf-thr 0.5
```

### 웹캠 + 궤적(Trail) 시각화
```bash
python webcam_head_tracking_trail.py
python webcam_head_tracking_trail.py --max-trail 300
```

| 추가 인자 | 기본값 | 설명 |
|-----------|--------|------|
| `--max-trail` | 600 | 저장할 최대 궤적 점 수 (30fps 기준 약 20초) |

- HEAD 1 궤적: **파란색** 선
- HEAD 2 궤적: **빨간색** 선
- 오래된 경로일수록 흐릿하게 페이드 처리
- 카메라 뷰 + Bird's-Eye 뷰 양쪽에 궤적 표시
- **T 키**: 궤적만 초기화 (추적은 계속 유지)

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--cam-id` | 0 | 카메라 번호 |
| `--rect-w` | 480 | Bird's-Eye 뷰 가로 크기 (px) |
| `--rect-h` | 480 | Bird's-Eye 뷰 세로 크기 (px) |
| `--search-scale` | 3.5 | 재탐색 영역 확장 배수 |
| `--conf-thr` | 0.40 | 템플릿 매칭 유사도 임계값 (0~1) |
| `--fail-max` | 90 | FAILED 전환까지 허용 실패 프레임 수 |
| `--out-dir` | output | 결과 저장 폴더 |

### 녹화 영상 파일
```bash
python video_head_tracking_retrack.py --video my_video.mp4
python video_head_tracking_retrack.py --video my_video.mp4 --speed 0.5
python video_head_tracking_retrack.py --video my_video.mp4 --start-frame 300
```

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--video` | **(필수)** | 입력 영상 파일 경로 (mp4, avi, mov 등) |
| `--speed` | 1.0 | 재생 속도 배율 (0.5=절반, 2.0=두 배) |
| `--start-frame` | 0 | 시작 프레임 번호 |
| `--loop` | False | 영상 끝에서 처음으로 반복 |
| `--rect-w` | 480 | Bird's-Eye 뷰 가로 크기 (px) |
| `--rect-h` | 480 | Bird's-Eye 뷰 세로 크기 (px) |
| `--search-scale` | 3.5 | 재탐색 영역 확장 배수 |
| `--conf-thr` | 0.40 | 템플릿 매칭 유사도 임계값 (0~1) |
| `--fail-max` | 90 | FAILED 전환까지 허용 실패 프레임 수 |
| `--out-dir` | output | 결과 저장 폴더 |

---

## 사용 방법

### Step 1 — 캘리브레이션 (호모그래피 기준점 설정)
실제 공간에서 직사각형을 이루는 꼭짓점 4개를 화면에서 순서대로 클릭합니다.
```
(1) 왼쪽 위 → (2) 오른쪽 위 → (3) 오른쪽 아래 → (4) 왼쪽 아래
```
클릭 완료 후 **ENTER** 키를 누르면 호모그래피가 확정됩니다.

> **영상 파일의 경우** 프로그램이 일시정지 상태로 시작됩니다.
> 첫 프레임에서 캘리브레이션을 완료한 뒤 SPACE로 재생하세요.

### Step 2 — 머리 추적 시작
- **1** 또는 **2** 키로 슬롯(HEAD 1 / HEAD 2) 선택
- 화면에서 추적할 머리 위치를 **왼쪽 클릭** → 자동 추적 시작

---

## 추적 상태

| 상태 | 색상 | 설명 |
|------|------|------|
| TRACKING | 마젠타 / 시안 | 정상 추적 중 |
| SEARCHING | 주황 | 추적 끊김 → 템플릿 매칭으로 자동 재탐색 중 |
| FAILED | 빨강 | 재탐색 실패 → 수동 재클릭 필요 |

---

## 키 조작

### 공통
| 키 | 동작 |
|----|------|
| `1` / `2` | 슬롯 선택 (HEAD 1 / HEAD 2) |
| Left-click | 선택 슬롯 트래커 초기화 / 재초기화 |
| Right-click | 선택 슬롯 추적 중지 |
| `C` | 선택 슬롯 추적 중지 |
| `R` | 전체 초기화 (재캘리브레이션) |
| `S` | 스냅샷 저장 |
| `Q` / `ESC` | 종료 & CSV 저장 |

### 영상 파일 전용
| 키 | 동작 |
|----|------|
| `SPACE` | 일시정지 / 재개 |
| `←` LEFT | 10프레임 뒤로 (일시정지 중) |
| `→` RIGHT | 10프레임 앞으로 (일시정지 중) |
| `F` | 1프레임 앞으로 (일시정지 중) |
| `[` | 재생 속도 절반 |
| `]` | 재생 속도 두 배 |

---

## CSV 출력 컬럼

| 컬럼 | 설명 |
|------|------|
| frame | 원본 영상 프레임 번호 |
| timestamp / video_time_sec | 경과 시간 (초) |
| head_id | 머리 번호 (1 또는 2) |
| raw_x, raw_y | 원본 화면 픽셀 좌표 |
| rect_x, rect_y | 호모그래피 보정 후 Bird's-Eye 좌표 |
| rect_x_norm, rect_y_norm | 0~1 정규화 좌표 |
| homography_applied | 호모그래피 적용 여부 (0/1) |
| state | 추적 상태 (tracking / searching / failed / none) |
| retrack_score | 재탐색 시 템플릿 매칭 유사도 점수 |

---

## 의존성

- Python 3.8 이상
- opencv-python >= 4.8.0
- numpy >= 1.24.0
