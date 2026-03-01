# TKD Analysis — 영상 기반 두부 추적 및 운동학 분석

> 태권도(또는 1:1 대인 스포츠) 영상에서 두 선수의 머리 위치를 자동 추적하고,
> 이동 속도·방향전환·운동 효율성 등 운동학적 지표를 정량 분석하는 파이썬 도구 모음입니다.

---

## 파일 구조

```
TKD_analysis/
├── webcam_head_tracking_retrack.py    # 실시간 웹캠 추적
├── webcam_head_tracking_trail.py      # 실시간 웹캠 추적 + 궤적(Trail) 시각화
├── video_head_tracking_retrack.py     # 녹화 영상 파일 추적  ← CSV 생성
├── video_head_tracking_trail.py       # 녹화 영상 파일 추적 + 궤적(Trail) 시각화
├── motion_kinematics_analysis.py      # 추적 CSV → 운동학 분석 및 시각화
├── requirements.txt                   # Python 패키지 목록
└── README.md
```

---

## 분석 흐름

```
[영상 파일]
    │
    ▼
video_head_tracking_retrack.py   ── 머리 위치 추적 → output/head_track_*.csv
    │
    ▼
motion_kinematics_analysis.py    ── CSV 읽기 → 운동학 계산 → 차트·통계 저장
    │
    ▼
output/motion_analysis_<날짜>/
    ├── summary_report.csv         선수별 요약 통계
    ├── speed_accel.png            속도·가속도 시계열
    ├── trajectory.png             이동 궤적
    ├── cod_analysis.png           방향전환(COD) 분석
    ├── work_rate.png              라운드별 운동 부하
    ├── centroid.png               두 선수 무게중심 궤적
    ├── correlation.png            운동학 변수 상관행렬
    └── radar_comparison.png       선수 비교 레이더 차트
```

---

## 설치

```bash
pip install -r requirements.txt
```

---

## 1단계 — 머리 추적 (CSV 생성)

### 웹캠 실시간 추적
```bash
python webcam_head_tracking_retrack.py
python webcam_head_tracking_retrack.py --cam-id 1 --conf-thr 0.5
```

### 웹캠 + 궤적(Trail) 시각화
```bash
python webcam_head_tracking_trail.py
python webcam_head_tracking_trail.py --max-trail 300
```

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--cam-id` | 0 | 카메라 번호 |
| `--max-trail` | 600 | 저장할 최대 궤적 점 수 (30fps 기준 약 20초) |
| `--rect-w` | 480 | Bird's-Eye 뷰 가로 크기 (px) |
| `--rect-h` | 480 | Bird's-Eye 뷰 세로 크기 (px) |
| `--search-scale` | 3.5 | 재탐색 영역 확장 배수 |
| `--conf-thr` | 0.40 | 템플릿 매칭 유사도 임계값 (0~1) |
| `--fail-max` | 90 | FAILED 전환까지 허용 실패 프레임 수 |
| `--out-dir` | output | 결과 저장 폴더 |

### 녹화 영상 파일 추적
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

### 녹화 영상 파일 + 궤적(Trail) 시각화
```bash
python video_head_tracking_trail.py --video my_video.mp4
python video_head_tracking_trail.py --video my_video.mp4 --max-trail 300
```

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--video` | **(필수)** | 입력 영상 파일 경로 |
| `--max-trail` | 600 | 저장할 최대 궤적 점 수 |
| `--speed` | 1.0 | 재생 속도 배율 |
| `--start-frame` | 0 | 시작 프레임 번호 |

- HEAD 1 궤적: **파란색** 선 / HEAD 2 궤적: **빨간색** 선
- 오래된 경로일수록 흐릿하게 페이드 처리
- 궤적은 오른쪽 **Bird's-Eye 뷰에서만** 표시
- **T 키**: 궤적만 초기화 (추적은 계속 유지)

---

### 사용 방법

#### Step 1 — 캘리브레이션 (호모그래피 기준점 설정)
비스듬한 카메라 영상을 Bird's-Eye(정사각형 평면) 좌표로 변환하기 위해,
실제 공간에서 직사각형을 이루는 꼭짓점 4개를 화면에서 순서대로 클릭합니다.

```
(1) 왼쪽 위 → (2) 오른쪽 위 → (3) 오른쪽 아래 → (4) 왼쪽 아래
```

클릭 완료 후 **ENTER** 키를 누르면 호모그래피가 확정됩니다.

> **영상 파일의 경우** 프로그램이 일시정지 상태로 시작됩니다.
> 첫 프레임에서 캘리브레이션을 완료한 뒤 SPACE로 재생하세요.

#### Step 2 — 머리 추적 시작
- **1** 또는 **2** 키로 슬롯 선택 (HEAD 1 / HEAD 2)
- 화면에서 추적할 머리 위치를 **왼쪽 클릭** → 자동 추적 시작

---

### 추적 상태

| 상태 | 색상 | 설명 |
|------|------|------|
| TRACKING | 마젠타 / 시안 | 정상 추적 중 |
| SEARCHING | 주황 | 추적 끊김 → 템플릿 매칭으로 자동 재탐색 중 |
| FAILED | 빨강 | 재탐색 실패 → 수동 재클릭 필요 |

---

### 키 조작

#### 공통
| 키 | 동작 |
|----|------|
| `1` / `2` | 슬롯 선택 (HEAD 1 / HEAD 2) |
| Left-click | 선택 슬롯 트래커 초기화 / 재초기화 |
| Right-click | 선택 슬롯 추적 중지 |
| `C` | 선택 슬롯 추적 중지 |
| `R` | 전체 초기화 (재캘리브레이션) |
| `S` | 스냅샷 저장 |
| `Q` / `ESC` | 종료 & CSV 저장 |

#### 영상 파일 전용
| 키 | 동작 |
|----|------|
| `SPACE` | 일시정지 / 재개 |
| `←` LEFT | 10프레임 뒤로 (일시정지 중) |
| `→` RIGHT | 10프레임 앞으로 (일시정지 중) |
| `F` | 1프레임 앞으로 (일시정지 중) |
| `[` | 재생 속도 절반 |
| `]` | 재생 속도 두 배 |

---

### CSV 출력 컬럼

| 컬럼 | 설명 |
|------|------|
| frame | 원본 영상 프레임 번호 |
| timestamp / video_time_sec | 경과 시간 (초) |
| head_id | 머리 번호 (1 또는 2) |
| raw_x, raw_y | 원본 화면 픽셀 좌표 |
| rect_x, rect_y | 호모그래피 보정 후 Bird's-Eye 좌표 (픽셀) |
| rect_x_norm, rect_y_norm | 0~1 정규화 좌표 |
| homography_applied | 호모그래피 적용 여부 (0/1) |
| state | 추적 상태 (tracking / searching / failed / none) |
| retrack_score | 재탐색 시 템플릿 매칭 유사도 점수 |

---

## 2단계 — 운동학 분석 (motion_kinematics_analysis.py)

`video_head_tracking_retrack.py`가 생성한 CSV를 읽어
선수의 이동 패턴을 운동학적으로 분석하고 차트와 통계를 저장합니다.

### 실행

```bash
# 기본 실행
python motion_kinematics_analysis.py \
    --csv output/head_track_20240101_120000.csv

# 코트 크기 및 라운드 단위 지정
python motion_kinematics_analysis.py \
    --csv output/head_track_20240101_120000.csv \
    --court-w 8 --court-h 8 --round-sec 120

# COD 임계각·고강도 기준 조정
python motion_kinematics_analysis.py \
    --csv output/head_track_20240101_120000.csv \
    --cod-thr 30 --hi-pct 80
```

### 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--csv` | **(필수)** | 추적 스크립트가 저장한 CSV 경로 |
| `--out-dir` | output | 결과 저장 상위 폴더 |
| `--court-w` | 14.0 | 실제 코트 가로 (m) |
| `--court-h` | 8.0 | 실제 코트 세로 (m) |
| `--rect-w` | 480 | 추적 스크립트의 RECT_W 값 (px) |
| `--rect-h` | 480 | 추적 스크립트의 RECT_H 값 (px) |
| `--round-sec` | 120 | 라운드 구분 단위 (초) |
| `--cod-thr` | 45.0 | 방향전환 판정 임계각 (도) |
| `--hi-pct` | 75 | 고강도 구간 기준 속도 백분위 |

### 분석 항목 7가지

#### 1. 이동 속도 및 가속도 분석
프레임 간 좌표 변화로 순간 속도와 가속도를 계산합니다.

```
속도   = 이동 거리 / 시간 간격      (m/s)
가속도 = 속도 변화량 / 시간 간격    (m/s²)
```

"이 선수는 공격 진입 시 평균 2.3m/s, 최대 가속도 4.1m/s²" 같은 수치를 뽑아
민첩성을 객관적으로 평가할 수 있고, 선수 간 비교나 훈련 전후 변화 측정에 바로 쓸 수 있습니다.

#### 2. 방향전환(COD) 능력 평가
이동 벡터의 각도 변화율을 통해 방향전환 빈도, 속도, 각도를 정량화합니다.

```
방향각    = arctan2(Δy, Δx)
방향 변화량 ≥ cod-thr(도)  →  COD 이벤트로 기록
```

방향전환이 빠른 선수가 카운터 공격에 유리한데,
이를 수치로 훈련 목표를 설정할 수 있습니다.

| 산출 지표 | 설명 |
|-----------|------|
| cod_count | 전체 방향전환 횟수 |
| cod_per_min | 분당 방향전환 빈도 (민첩성 지수) |
| 전환 시 speed_ms | 방향전환 순간의 속도 |

#### 3. 이동 효율성 분석
실제 이동 거리 대비 직선 거리의 비율로 이동 효율성을 측정합니다.

```
이동 효율성(%) = 직선 거리 / 실제 이동 거리 × 100
```

효율이 낮으면 불필요한 움직임이 많다는 뜻이므로, 스텝 교정의 근거가 됩니다.

#### 4. 라운드별 운동 부하(Work Rate) 측정
단위 시간당 이동 거리와 속도 변화를 통해 운동 강도를 구간별로 산출합니다.
체력 배분 전략 수립, 고강도 구간(high-intensity zone) 진입 빈도 분석이 가능합니다.

| 지표 | 설명 |
|------|------|
| total_dist_m | 라운드 내 총 이동 거리 |
| avg_speed_ms | 라운드 평균 속도 |
| hi_zone_frac | 고강도 구간 프레임 비율 |
| cod_count | 라운드 내 방향전환 횟수 |

축구에서 쓰는 총 이동 거리·스프린트 횟수 분석과 동일한 개념입니다.

#### 5. 무게중심(Centroid) 이동 패턴
양 선수의 좌표를 결합하면 경기 전체의 무게중심(centroid) 이동 궤적을 그릴 수 있습니다.

```
무게중심 = (Player1 좌표 + Player2 좌표) / 2
```

경기 흐름의 주도권이 어디에 있었는지를 시각적으로 보여줍니다.
무게중심이 한 선수 쪽으로 치우칠수록 해당 선수가 공간을 주도하고 있음을 의미합니다.

#### 6. 운동학적 변수와 경기력 상관분석
위에서 도출한 변수들(속도, 가속도, 방향전환율, 이동거리 등)과
득점·승패 간 상관관계 및 회귀분석이 가능합니다.

"가속도 상위 25% 선수의 승률이 70%"와 같은
**근거 기반 훈련 방향 설정**에 활용됩니다.

속도·가속도·방향전환·이동 거리 간의 Pearson 상관행렬과
선수 비교 레이더 차트를 함께 제공합니다.

#### 7. 부상 위험 예측
급격한 감속, 과도한 방향전환, 비대칭 이동 패턴 등은 부상 위험 지표가 됩니다.

| 위험 신호 | 설명 |
|-----------|------|
| 급격한 감속 | 짧은 시간 내 가속도 급감 |
| 과도한 COD | 방향전환 각도·빈도 임계값 초과 |
| 비대칭 이동 | 좌우·전후 이동 거리 불균형 |

시계열로 모니터링하면 특정 선수의 피로 누적이나
부상 전조를 조기에 감지할 수 있습니다.

### 출력 파일

| 파일 | 내용 |
|------|------|
| `summary_report.csv` | 선수별 종합 통계 (속도·가속도·COD·효율성·운동부하) |
| `efficiency_by_round.csv` | 라운드별 이동 효율성 |
| `work_rate_by_round.csv` | 라운드별 운동 부하 상세 |
| `cod_events.csv` | 방향전환 이벤트 발생 위치·시각·속도 목록 |
| `centroid.csv` | 프레임별 무게중심 좌표 및 선수 간 거리 |
| `speed_accel.png` | 속도·가속도·방향변화 시계열 그래프 |
| `trajectory.png` | 이동 궤적 + COD 이벤트 위치 |
| `cod_analysis.png` | 방향변화 분포·속도 산점도·누적 COD |
| `work_rate.png` | 라운드별 이동거리·속도·효율성·고강도 비율 |
| `centroid.png` | 무게중심 궤적·X/Y 시계열·선수 간 거리 |
| `correlation.png` | 운동학 변수 상관행렬 히트맵 |
| `radar_comparison.png` | 선수 비교 레이더 차트 |

---

## 의존성

```
# 추적 스크립트
opencv-python >= 4.8.0
numpy >= 1.24.0

# 운동학 분석 스크립트 (추가 필요)
pandas >= 2.0.0
matplotlib >= 3.7.0
scipy >= 1.10.0
```

```bash
pip install -r requirements.txt
```

---

## 추천 논문 제목

본 도구를 활용하여 작성할 수 있는 연구 제목 예시입니다.

### 방법론 개발 중심
> **영상 기반 자동 추적 시스템을 활용한 태권도 선수의 이동 패턴 및 운동학적 변수 분석**
> *Analysis of Movement Patterns and Kinematic Variables in Taekwondo Athletes Using a Video-Based Automatic Tracking System*

### 선수 평가·비교 중심
> **이동 속도·방향전환 빈도·운동 효율성 지표를 활용한 태권도 선수 민첩성 정량 평가**
> *Quantitative Agility Assessment of Taekwondo Athletes Using Movement Speed, Change-of-Direction Frequency, and Locomotor Efficiency*

### 경기력 예측 중심
> **태권도 경기 중 운동학적 변수(속도·COD·이동효율)와 경기 결과의 관계 분석**
> *Relationship Between Kinematic Variables (Speed, COD, Movement Efficiency) and Match Outcomes in Taekwondo Competition*

### 훈련 효과 중심
> **영상 분석 기반 운동학 지표를 활용한 태권도 훈련 전후 이동 패턴 변화 연구**
> *Changes in Movement Patterns Before and After Taekwondo Training Using Video-Based Kinematic Indicators*

### 전술 분석 중심
> **두부 추적 기반 무게중심 이동 궤적을 활용한 태권도 경기 전술 패턴 분석**
> *Tactical Pattern Analysis in Taekwondo Matches Using Center-of-Mass Trajectory Derived from Head Tracking*
