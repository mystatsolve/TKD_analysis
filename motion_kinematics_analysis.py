"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          운동학 분석 도구  (Kinematic Analysis Tool)                        ║
║          ── video_head_tracking_retrack.py 출력 CSV 전용 ──                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

[입력 데이터]
  반드시 video_head_tracking_retrack.py 로 생성된 CSV 파일을 사용하십시오.
  해당 스크립트는 영상에서 최대 2명의 머리 위치를 CSRT 트래커 + 템플릿
  매칭 재탐색으로 추적하며, 호모그래피(Bird's-Eye 변환)를 적용한 뒤
  아래 컬럼을 포함하는 CSV를 저장합니다.

  CSV 컬럼 구조 (video_head_tracking_retrack.py 출력):
  ┌─────────────────────┬──────────────────────────────────────────────────┐
  │ 컬럼명              │ 설명                                              │
  ├─────────────────────┼──────────────────────────────────────────────────┤
  │ frame               │ 원본 영상의 프레임 번호 (1-based)                 │
  │ video_time_sec      │ 영상 기준 시각 (초). 일부 버전은 timestamp        │
  │ head_id             │ 선수 식별자 (1 또는 2)                            │
  │ raw_x, raw_y        │ 원본 카메라 좌표 (픽셀)                           │
  │ rect_x, rect_y      │ Bird's-Eye 뷰 좌표 (픽셀, 0~RECT_W/RECT_H)      │
  │ rect_x_norm,        │ Bird's-Eye 좌표 정규화값 (0.0~1.0)               │
  │   rect_y_norm       │                                                  │
  │ homography_applied  │ 호모그래피 적용 여부 (0/1)                        │
  │ state               │ 추적 상태: tracking / searching / failed / none  │
  │ retrack_score       │ 템플릿 매칭 재탐색 신뢰도 점수                    │
  └─────────────────────┴──────────────────────────────────────────────────┘

  ※ state == 'tracking' 인 행만 분석에 사용됩니다.
     (searching/failed/none 구간은 위치 오차가 크므로 자동 제외)

[분석 파이프라인]
  CSV 로드  →  좌표 변환(픽셀→m)  →  운동학 계산(속도·가속도·방향각)
    →  COD 이벤트 감지  →  라운드별 효율성·운동부하 산출
    →  무게중심 계산(2인)  →  요약 통계  →  시각화 7종 저장

[분석 항목]
  1. 이동 속도 및 가속도     프레임 간 좌표 차이 / 시간 간격으로 계산
  2. 방향 전환(COD) 평가     이동 벡터 각도 변화율 기반 이벤트 감지
  3. 이동 효율성             직선거리 / 실이동거리 × 100 (%)
  4. 운동 부하(Work Rate)    라운드 단위 거리·속도·고강도 구간 비율
  5. 무게중심 이동 패턴      양 선수 좌표 결합 → 경기 주도권 시각화
  6. 운동학적 변수 상관분석  속도·가속도·COD·효율성 간 상관행렬

[실행 예시]
  # 기본 실행 (코트 14m×8m, 라운드 120초 단위)
  python motion_kinematics_analysis.py \\
      --csv output/video_headtrack_20240101_120000.csv

  # 코트 크기·라운드 단위 직접 지정
  python motion_kinematics_analysis.py \\
      --csv output/video_headtrack_20240101_120000.csv \\
      --court-w 14 --court-h 8 --round-sec 120

  # COD 임계각 30도, 고강도 기준 80백분위
  python motion_kinematics_analysis.py \\
      --csv output/video_headtrack_20240101_120000.csv \\
      --cod-thr 30 --hi-pct 80

[필요 패키지]
  pip install pandas numpy matplotlib scipy

[결과물 위치]  out_dir/motion_analysis_<YYYYMMDD_HHMMSS>/
  summary_report.csv       선수별 종합 통계
  efficiency_by_round.csv  라운드별 이동 효율성
  work_rate_by_round.csv   라운드별 운동 부하
  cod_events.csv           방향전환 이벤트 목록
  centroid.csv             무게중심 시계열 (2인)
  speed_accel.png          속도·가속도·방향변화 시계열
  trajectory.png           이동 궤적 + COD 이벤트
  cod_analysis.png         방향전환 분석 3종
  work_rate.png            라운드별 운동 부하 4종
  centroid.png             무게중심 이동 패턴 3종
  correlation.png          운동학적 변수 상관행렬
  radar_comparison.png     선수 비교 레이더 차트
"""

import os
import sys
import warnings
import argparse
import time

warnings.filterwarnings('ignore')  # 불필요한 경고 메시지 억제

import numpy as np           # 수치 계산 (배열 연산, arctan2, sqrt 등)
import pandas as pd          # CSV 로드 및 데이터프레임 조작
import matplotlib.pyplot as plt          # 시각화 기반
import matplotlib.patches as mpatches   # 범례 커스텀 패치 (예약)
from matplotlib.gridspec import GridSpec # 복합 레이아웃 (예약)

# scipy.stats: COD 산점도 추세선(linregress) 에 사용
# 미설치 시 추세선만 생략되고 나머지 분석은 정상 동작
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print('[WARN] scipy 미설치 — COD 추세선 생략 (pip install scipy)')

# ── 한글 폰트 설정 ──────────────────────────────────────────────────────────
# matplotlib 기본 폰트는 한글을 지원하지 않으므로 시스템에서 우선순위대로 탐색
import matplotlib
try:
    from matplotlib import font_manager
    # Windows: Malgun Gothic / Linux: NanumGothic / macOS: AppleGothic 순으로 시도
    _candidates = ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'DejaVu Sans']
    _installed  = {f.name for f in font_manager.fontManager.ttflist}
    for _fc in _candidates:
        if _fc in _installed:
            matplotlib.rc('font', family=_fc)
            break
    # 음수 기호(-) 가 깨지는 문제 방지
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception:
    pass  # 폰트 설정 실패 시 기본 폰트로 계속 진행

plt.style.use('dark_background')  # 어두운 배경 스타일 (스포츠 분석 보고서 스타일)


# ══════════════════════════════════════════════════════════════════════════════
# ■ 전역 설정 상수
#   --court-w / --court-h 인자로 런타임 오버라이드 가능
# ══════════════════════════════════════════════════════════════════════════════
COURT_W_M     = 14.0  # 실제 코트 가로 (m) — Bird's-Eye X축 전체 길이에 해당
COURT_H_M     = 8.0   # 실제 코트 세로 (m) — Bird's-Eye Y축 전체 길이에 해당

# video_head_tracking_retrack.py의 RECT_W, RECT_H 기본값과 반드시 일치해야 함
# 다른 값으로 실행했다면 --rect-w / --rect-h 인자로 덮어씌워야 함
RECT_W_PX     = 480   # Bird's-Eye 뷰 가로 픽셀
RECT_H_PX     = 480   # Bird's-Eye 뷰 세로 픽셀

ROUND_SEC     = 120   # 라운드 구분 단위 (초). Work Rate 구간 크기
COD_ANGLE_THR = 45.0  # 방향전환(COD) 판정 임계각 (도). 이 값 이상이면 COD 이벤트
HI_SPEED_PCT  = 75    # 고강도(High-Intensity) 속도 임계 백분위. 전체 속도의 상위 25%

# 선수별 시각화 색상 (BGR → matplotlib RGB hex)
HEAD_COLORS = {1: '#FF66FF', 2: '#00FFFF'}  # Player1: 마젠타, Player2: 시안
HEAD_LABELS = {1: 'Player 1', 2: 'Player 2'}


# ══════════════════════════════════════════════════════════════════════════════
# ■ Step 1 — 데이터 로드 & 전처리
# ══════════════════════════════════════════════════════════════════════════════
def load_and_clean(csv_path: str, rect_w: int, rect_h: int,
                   court_w: float, court_h: float) -> pd.DataFrame:
    """
    video_head_tracking_retrack.py 가 저장한 CSV를 읽어 분석용 DataFrame으로
    변환합니다.

    처리 순서:
      1) CSV 파일 읽기
      2) state == 'tracking' 행만 남김
         → 'searching' / 'failed' / 'none' 구간은 추적 오류가 포함될 수 있어 제외
      3) 컬럼명 통일: 구버전의 'timestamp' → 'video_time_sec'
      4) 수치형 변환 (문자열 'nan' 포함 대응)
      5) Bird's-Eye 픽셀 좌표 → 실제 거리 (m) 변환
         공식: x_m = rect_x ÷ (RECT_W_PX / court_w)
               즉 픽셀당 미터 = court_w / RECT_W_PX

    Parameters
    ----------
    csv_path : video_head_tracking_retrack.py 출력 CSV 절대/상대 경로
    rect_w   : Bird's-Eye 뷰 가로 픽셀 수 (추적 스크립트의 RECT_W 값)
    rect_h   : Bird's-Eye 뷰 세로 픽셀 수 (추적 스크립트의 RECT_H 값)
    court_w  : 실제 코트 가로 (m)
    court_h  : 실제 코트 세로 (m)

    Returns
    -------
    pd.DataFrame
        x_m, y_m 컬럼이 추가된 정제 DataFrame
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # 컬럼명 앞뒤 공백 제거

    # ── state 필터 ──────────────────────────────────────────────────────────
    # video_head_tracking_retrack.py 는 각 프레임에 state 컬럼을 기록함:
    #   'tracking'  : CSRT 트래커가 정상 추적 중 → 분석에 사용
    #   'searching' : 트래커 실패 후 템플릿 매칭 재탐색 중 → 위치 불신뢰로 제외
    #   'failed'    : 재탐색 한계 초과, 클릭 재초기화 필요 → 제외
    #   'none'      : 아직 초기화 안 된 슬롯 → 제외
    df = df[df['state'] == 'tracking'].copy()

    # ── 컬럼명 통일 ─────────────────────────────────────────────────────────
    # video_head_tracking_retrack.py 의 CSV_HEADER 에는 'video_time_sec' 이지만
    # 파생 스크립트(trail 버전 등)는 'timestamp' 를 사용하므로 양쪽 모두 처리
    if 'timestamp' in df.columns and 'video_time_sec' not in df.columns:
        df = df.rename(columns={'timestamp': 'video_time_sec'})

    # ── 수치 변환 ───────────────────────────────────────────────────────────
    # CSV 에 'nan' 문자열이 저장되는 경우(호모그래피 미적용 프레임 등)를
    # errors='coerce' 로 NaN 으로 자동 변환
    for col in ['rect_x', 'rect_y', 'video_time_sec', 'head_id']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 핵심 컬럼에 NaN이 있는 행 제거
    df = df.dropna(subset=['rect_x', 'rect_y', 'video_time_sec', 'head_id'])
    df['head_id'] = df['head_id'].astype(int)  # 선수 번호를 정수형으로

    # ── 좌표 단위 변환: 픽셀 → 미터 ────────────────────────────────────────
    # Bird's-Eye 뷰는 rect_w × rect_h 픽셀 영역에 실제 court_w × court_h (m)를 매핑
    # 따라서 1픽셀 = court_w/rect_w 미터
    # 예) rect_w=480, court_w=14m → 1px = 0.0292m = 2.92cm
    df['x_m'] = df['rect_x'] / (rect_w / court_w)
    df['y_m'] = df['rect_y'] / (rect_h / court_h)

    # head_id 기준으로 정렬 후 각 선수 내에서 프레임 순으로 정렬
    df = df.sort_values(['head_id', 'frame']).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ■ Step 2 — 운동학 변수 계산 (속도·가속도·방향각)
# ══════════════════════════════════════════════════════════════════════════════
def compute_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """
    head_id 별로 프레임 간 운동학 변수를 계산하여 컬럼을 추가합니다.

    추가 컬럼:
      step_dist_m  : 프레임 간 이동 거리 (m)
                     공식: sqrt( (x[i]-x[i-1])² + (y[i]-y[i-1])² )
      speed_ms     : 순간 속도 (m/s)
                     공식: step_dist_m / Δt
      accel_ms2    : 순간 가속도 (m/s²)
                     공식: (speed[i] - speed[i-1]) / Δt
      direction    : 이동 방향각 (도, -180~180)
                     공식: degrees( arctan2(Δy, Δx) )
                     0°=오른쪽, 90°=위쪽, -90°=아래쪽, ±180°=왼쪽
      ang_change   : 연속 프레임 간 방향 변화량 (도, 0~180)
                     180° 초과분은 360°-값으로 보정하여 항상 최소 회전각 반환

    Notes
    -----
    - 각 선수(head_id)는 독립적으로 계산 (groupby)
    - 첫 번째 행은 이전 프레임이 없으므로 NaN
    - 프레임 간격이 0인 경우(중복 프레임 등) NaN 처리
    """
    parts = []
    for hid, grp in df.groupby('head_id'):
        g = grp.sort_values('video_time_sec').copy()

        # 연속 프레임 간 시간 간격 (초)
        dt = g['video_time_sec'].diff()   # diff(): i번째 - (i-1)번째

        # 연속 프레임 간 좌표 변위 (m)
        dx = g['x_m'].diff()
        dy = g['y_m'].diff()

        # ── 이동 거리 (유클리드 거리) ──────────────────────────────────────
        dist = np.sqrt(dx**2 + dy**2)

        # ── 순간 속도 (m/s) ────────────────────────────────────────────────
        # Δt <= 0 인 경우(타임스탬프 오류) NaN 처리
        speed = (dist / dt).where(dt > 0, np.nan)

        # ── 순간 가속도 (m/s²) ─────────────────────────────────────────────
        # 속도의 변화율: Δv / Δt
        # 양수 = 가속, 음수 = 감속
        accel = (speed.diff() / dt).where(dt > 0, np.nan)

        # ── 이동 방향각 (도) ────────────────────────────────────────────────
        # arctan2(dy, dx): y축 방향이 아래가 양수인 좌표계를 감안
        # 반환값 범위: -180° ~ +180°
        direction = np.degrees(np.arctan2(dy, dx))

        # ── 방향 변화량 (최소 회전각, 0~180°) ──────────────────────────────
        # 단순 차이를 구하면 -359° ~ +359° 범위가 나올 수 있으므로
        # abs 후 180° 초과분은 360° 에서 빼서 '최단 회전 방향' 기준으로 환산
        raw_chg    = direction.diff().abs()
        ang_change = raw_chg.where(raw_chg <= 180, 360 - raw_chg)

        # 계산 결과를 그룹 DataFrame 에 추가
        g['step_dist_m'] = dist.values
        g['speed_ms']    = speed.values
        g['accel_ms2']   = accel.values
        g['direction']   = direction.values
        g['ang_change']  = ang_change.values
        parts.append(g)

    return pd.concat(parts).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# ■ Step 3 — 방향전환(COD: Change of Direction) 이벤트 감지
# ══════════════════════════════════════════════════════════════════════════════
def detect_cod_events(df: pd.DataFrame, angle_thr: float) -> pd.DataFrame:
    """
    연속 프레임 간 방향 변화량이 angle_thr(도) 이상인 행을 COD 이벤트로
    추출합니다.

    COD(Change of Direction)는 스포츠 과학에서 민첩성 측정의 핵심 지표로,
    이 함수에서는 단순 각도 임계값 기반으로 정량화합니다.

    Parameters
    ----------
    df        : compute_kinematics() 실행 후 ang_change 컬럼이 있는 DataFrame
    angle_thr : COD 판정 최소 방향 변화각 (도). 권장값: 30~60°

    Returns
    -------
    pd.DataFrame
        COD 이벤트 행만 모은 DataFrame (원본 컬럼 전체 포함)

    활용 방법:
      - cod_count / 경기 시간(분) = 분당 방향전환 횟수 → 민첩성 지수
      - COD 발생 시점의 speed_ms → 고속 방향전환 능력 평가
      - 경기 시간 축 상의 분포 → 피로도에 따른 방향전환 빈도 변화 추적
    """
    parts = []
    for hid, grp in df.groupby('head_id'):
        # ang_change 가 임계값 이상인 행만 추출
        ev = grp[grp['ang_change'] >= angle_thr].copy()
        parts.append(ev)

    if parts:
        return pd.concat(parts).reset_index(drop=True)
    # COD 이벤트가 하나도 없는 경우 빈 DataFrame 반환
    return pd.DataFrame(columns=df.columns)


# ══════════════════════════════════════════════════════════════════════════════
# ■ Step 4-A — 이동 효율성 분석 (라운드별)
# ══════════════════════════════════════════════════════════════════════════════
def compute_efficiency(df: pd.DataFrame, round_sec: int) -> pd.DataFrame:
    """
    라운드(시간 구간) 단위로 이동 효율성을 계산합니다.

    이동 효율성 = 직선 거리 / 실제 이동 거리 × 100 (%)

    해석:
      100% : 완전 직선 이동 (최단 경로)
      낮을수록: 지그재그·불필요한 왕복이 많음 → 체력 낭비
      훈련 목표: 공격/수비 전환 시 효율성 수치로 스텝 교정

    공식 상세:
      total_dist_m  = Σ step_dist_m  (구간 내 모든 프레임 간 거리의 합)
      straight_dist = ||end_pos - start_pos||  (유클리드 직선 거리)
      efficiency    = straight_dist / total_dist × 100

    Parameters
    ----------
    df        : compute_kinematics() 결과 DataFrame
    round_sec : 라운드 단위 초. 예) 120 → 2분씩 구간 분할

    Returns
    -------
    pd.DataFrame
        컬럼: head_id, round, t_start, t_end,
              total_dist_m, straight_dist_m, efficiency_pct
    """
    rows = []
    for hid, grp in df.groupby('head_id'):
        grp   = grp.sort_values('video_time_sec')
        max_t = grp['video_time_sec'].max()

        # 0초부터 max_t 까지 round_sec 간격으로 경계 배열 생성
        # 예) max_t=310, round_sec=120 → [0, 120, 240, 360]
        edges = np.arange(0, max_t + round_sec, round_sec)

        for i in range(len(edges) - 1):
            # 현재 라운드 구간 [edges[i], edges[i+1]) 추출
            seg = grp[(grp['video_time_sec'] >= edges[i]) &
                      (grp['video_time_sec'] <  edges[i+1])]

            # 2개 미만 프레임이면 거리 계산 불가 → 건너뜀
            if len(seg) < 2:
                continue

            # 실제 이동 거리: 각 프레임 간 거리의 합
            total_dist = seg['step_dist_m'].sum()

            # 직선 거리: 구간 첫 좌표 ~ 끝 좌표 사이의 유클리드 거리
            start_pt = seg.iloc[0][['x_m', 'y_m']].values.astype(float)
            end_pt   = seg.iloc[-1][['x_m', 'y_m']].values.astype(float)
            straight = float(np.linalg.norm(end_pt - start_pt))

            # 효율성 비율 (0 나눗셈 방지)
            efficiency = (straight / total_dist * 100) if total_dist > 0 else np.nan

            rows.append({
                'head_id':        hid,
                'round':          i + 1,          # 1-based 라운드 번호
                't_start':        edges[i],
                't_end':          edges[i + 1],
                'total_dist_m':   round(total_dist, 3),
                'straight_dist_m': round(straight, 3),
                'efficiency_pct': round(efficiency, 2)
                                  if not np.isnan(efficiency) else np.nan,
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# ■ Step 4-B — 라운드별 운동 부하(Work Rate) 분석
# ══════════════════════════════════════════════════════════════════════════════
def compute_work_rate(df: pd.DataFrame, round_sec: int, hi_pct: float) -> pd.DataFrame:
    """
    라운드(시간 구간) 단위로 운동 부하 지표를 산출합니다.

    축구의 GPS 추적 분석과 동일한 개념:
      - 총 이동 거리 = 체력 소비량의 대리 지표
      - 고강도 구간 비율(hi_zone_frac) = 스프린트/고속 이동 비중
      - COD 횟수 = 순간적 방향전환 부하

    고강도 임계 속도(hi_speed_thr):
      전체 데이터(양 선수, 전 구간)의 속도 분포에서 hi_pct 백분위값 사용
      → 상대적 기준이므로 영상 특성에 관계없이 일관성 있는 비교 가능

    Parameters
    ----------
    df        : compute_kinematics() 결과
    round_sec : 라운드 구간 크기 (초)
    hi_pct    : 고강도 판정 백분위 (0~100). 기본 75 → 상위 25% 속도

    Returns
    -------
    pd.DataFrame
        컬럼: head_id, round, t_start, t_end,
              total_dist_m, avg_speed_ms, max_speed_ms, max_accel_ms2,
              hi_zone_frac, hi_speed_thr, cod_count
    """
    # 전체 데이터 기준 고강도 속도 임계값 1회만 계산
    global_hi_thr = float(np.nanpercentile(df['speed_ms'].dropna(), hi_pct))

    rows = []
    for hid, grp in df.groupby('head_id'):
        grp   = grp.sort_values('video_time_sec')
        max_t = grp['video_time_sec'].max()
        edges = np.arange(0, max_t + round_sec, round_sec)

        for i in range(len(edges) - 1):
            seg = grp[(grp['video_time_sec'] >= edges[i]) &
                      (grp['video_time_sec'] <  edges[i+1])]
            if len(seg) < 2:
                continue

            # 고강도 구간: 속도가 임계값 이상인 프레임
            hi_mask = seg['speed_ms'] >= global_hi_thr
            # 고강도 비율: 고강도 프레임 수 / 전체 프레임 수
            hi_frac = hi_mask.sum() / len(seg)

            # COD 횟수: 방향전환 임계각 이상인 프레임 수
            cod_cnt = int((seg['ang_change'] >= COD_ANGLE_THR).sum())

            rows.append({
                'head_id':       hid,
                'round':         i + 1,
                't_start':       edges[i],
                't_end':         edges[i + 1],
                'total_dist_m':  round(seg['step_dist_m'].sum(), 3),
                'avg_speed_ms':  round(seg['speed_ms'].mean(), 3),
                'max_speed_ms':  round(seg['speed_ms'].max(), 3),
                # abs(): 감속 구간도 최대 가속도 크기로 환산
                'max_accel_ms2': round(seg['accel_ms2'].abs().max(), 3),
                'hi_zone_frac':  round(hi_frac, 4),  # 0.0~1.0
                'hi_speed_thr':  round(global_hi_thr, 3),
                'cod_count':     cod_cnt,
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# ■ Step 5 — 무게중심(Centroid) 계산
# ══════════════════════════════════════════════════════════════════════════════
def compute_centroid(df: pd.DataFrame) -> pd.DataFrame:
    """
    두 선수가 동시에 추적된 프레임에서 무게중심(중간점)을 계산합니다.

    무게중심 = (Player1 좌표 + Player2 좌표) / 2

    스포츠 분석 활용:
      - 무게중심이 한쪽으로 치우친 구간 = 해당 선수가 공간 주도
      - 두 선수 간 거리(dist_between_m) 변화 = 압박/후퇴 전술 패턴 분석
      - 무게중심 궤적이 코트 중앙 → 균형 잡힌 경기 흐름

    구현 방식:
      pivot_table 로 같은 frame 번호의 head_id=1, head_id=2 좌표를 같은
      행으로 묶은 뒤, 양쪽 모두 존재하는 프레임만 사용

    Parameters
    ----------
    df : compute_kinematics() 결과 (head_id 1·2 모두 포함 필요)

    Returns
    -------
    pd.DataFrame
        cx_m, cy_m (무게중심), dist_between_m (선수 간 거리) 포함
        선수가 1명뿐이면 빈 DataFrame 반환
    """
    if df['head_id'].nunique() < 2:
        return pd.DataFrame()  # 단독 선수 데이터

    # frame 을 인덱스, head_id 를 컬럼으로 하는 피벗 테이블 생성
    # 결과: x_m_1, y_m_1, x_m_2, y_m_2, video_time_sec_1 등의 컬럼
    pivot = df.pivot_table(
        index='frame', columns='head_id',
        values=['x_m', 'y_m', 'video_time_sec'],
        aggfunc='first'
    )
    pivot.columns = [f'{v}_{int(k)}' for v, k in pivot.columns]

    # 두 선수 모두 좌표가 있는 컬럼 확인
    needed = ['x_m_1', 'y_m_1', 'x_m_2', 'y_m_2']
    if not all(c in pivot.columns for c in needed):
        return pd.DataFrame()

    # 양쪽 모두 NaN 없는 프레임만 사용
    both = pivot.dropna(subset=needed).reset_index()

    # 시각(video_time_sec) 컬럼 정리
    time_col = 'video_time_sec_1' if 'video_time_sec_1' in both.columns else None
    if time_col:
        both = both.rename(columns={time_col: 'video_time_sec'})
    else:
        # 없으면 원본 df 의 frame→time 매핑으로 채움
        t_map = (df[['frame', 'video_time_sec']]
                 .drop_duplicates('frame')
                 .set_index('frame'))
        both = both.join(t_map, on='frame')

    # 무게중심: 두 좌표의 산술 평균
    both['cx_m'] = (both['x_m_1'] + both['x_m_2']) / 2
    both['cy_m'] = (both['y_m_1'] + both['y_m_2']) / 2

    # 두 선수 간 유클리드 거리 (m)
    both['dist_between_m'] = np.sqrt(
        (both['x_m_1'] - both['x_m_2'])**2 +
        (both['y_m_1'] - both['y_m_2'])**2
    )
    return both.sort_values('video_time_sec').reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# ■ Step 6 — 선수별 요약 통계 산출
# ══════════════════════════════════════════════════════════════════════════════
def compute_summary(df: pd.DataFrame, eff_df: pd.DataFrame,
                    work_df: pd.DataFrame, cod_df: pd.DataFrame) -> pd.DataFrame:
    """
    모든 분석 결과를 선수별 하나의 요약 행으로 집약합니다.

    포함 지표:
      속도계열  : avg/max/std_speed_ms
      가속도계열: avg/max_accel_ms2
      거리      : total_dist_m, duration_sec
      COD       : cod_count, cod_per_min (분당 횟수)
      효율성    : avg_efficiency_pct (라운드 평균)
      운동부하  : avg_dist_per_round, avg_hi_zone_frac
    """
    rows = []
    for hid, grp in df.groupby('head_id'):
        # 각 선수의 속도·가속도 유효값만 추출
        spd = grp['speed_ms'].dropna()
        acc = grp['accel_ms2'].dropna()

        # 하위 분석 결과에서 해당 선수 행만 필터링
        e = eff_df[eff_df['head_id'] == hid]
        w = work_df[work_df['head_id'] == hid]
        c = cod_df[cod_df['head_id'] == hid] if len(cod_df) > 0 else pd.DataFrame()

        # 경기 지속 시간: 첫 프레임 ~ 마지막 프레임 시각 차이
        dur = grp['video_time_sec'].max() - grp['video_time_sec'].min()

        rows.append({
            'head_id':            hid,
            'label':              HEAD_LABELS.get(hid, f'Player {hid}'),
            # ── 속도 ──
            'avg_speed_ms':       round(float(spd.mean()), 3) if len(spd) > 0 else np.nan,
            'max_speed_ms':       round(float(spd.max()),  3) if len(spd) > 0 else np.nan,
            'std_speed_ms':       round(float(spd.std()),  3) if len(spd) > 0 else np.nan,
            # ── 가속도 ──
            'avg_accel_ms2':      round(float(acc.mean()),     3) if len(acc) > 0 else np.nan,
            'max_accel_ms2':      round(float(acc.abs().max()), 3) if len(acc) > 0 else np.nan,
            # ── 이동 거리·시간 ──
            'total_dist_m':       round(float(grp['step_dist_m'].sum()), 2),
            'duration_sec':       round(float(dur), 1),
            # ── 방향전환(COD) ──
            'cod_count':          len(c),
            # 분당 COD 횟수: 경기 시간(분) 으로 나누어 정규화
            'cod_per_min':        round(len(c) / (dur / 60), 2) if dur > 0 else 0,
            # ── 이동 효율성 (라운드 평균) ──
            'avg_efficiency_pct': round(float(e['efficiency_pct'].mean()), 2)
                                   if len(e) > 0 else np.nan,
            # ── Work Rate (라운드 평균) ──
            'avg_dist_per_round': round(float(w['total_dist_m'].mean()), 2)
                                   if len(w) > 0 else np.nan,
            'avg_hi_zone_frac':   round(float(w['hi_zone_frac'].mean()), 4)
                                   if len(w) > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# ■ 시각화 함수들
# ══════════════════════════════════════════════════════════════════════════════

def plot_speed_accel(df: pd.DataFrame, out_path: str):
    """
    (1) 속도·가속도·방향변화 시계열 그래프 저장

    3개 서브플롯:
      ① 순간 속도 (m/s) vs 시간
      ② 순간 가속도 (m/s²) vs 시간
         양수=가속, 음수=감속
      ③ 이동 방향 변화량 (도) vs 시간
         빨간 점선 = COD 임계각

    두 선수의 곡선이 같은 축에 중첩되어 비교 가능
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)
    fig.suptitle('이동 속도 및 가속도 분석', fontsize=14, fontweight='bold')

    for hid, grp in df.groupby('head_id'):
        c   = HEAD_COLORS.get(hid, 'white')
        lbl = HEAD_LABELS.get(hid, f'Player {hid}')
        g   = grp.sort_values('video_time_sec')
        t   = g['video_time_sec']

        axes[0].plot(t, g['speed_ms'],   color=c, alpha=0.8, lw=0.7, label=lbl)
        axes[1].plot(t, g['accel_ms2'],  color=c, alpha=0.8, lw=0.7, label=lbl)
        axes[2].plot(t, g['ang_change'], color=c, alpha=0.8, lw=0.7, label=lbl)

    axes[0].set_ylabel('속도 (m/s)');    axes[0].set_title('순간 속도')
    axes[1].set_ylabel('가속도 (m/s²)'); axes[1].set_title('순간 가속도')
    axes[2].set_ylabel('방향 변화 (°)'); axes[2].set_title('이동 방향 변화율')

    # COD 임계각 기준선: 이 선 위에 있는 구간이 방향전환 이벤트
    axes[2].axhline(COD_ANGLE_THR, color='red', ls='--', lw=0.8,
                    label=f'COD 임계 ({COD_ANGLE_THR}°)')
    axes[2].set_xlabel('시간 (s)')

    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [저장] {out_path}')


def plot_trajectory(df: pd.DataFrame, court_w: float, court_h: float,
                    cod_df: pd.DataFrame, out_path: str):
    """
    (2) 이동 궤적 시각화

    각 선수별 서브플롯:
      - 점 색상: 시간 경과 (plasma 컬러맵, 밝을수록 나중 시각)
      - 빨간 X: COD 이벤트 발생 위치
      - 초록 원: 경기 시작 위치
      - 노란 사각형: 경기 종료 위치

    제목에 해당 선수의 총 이동 거리(m) 표시
    """
    heads = sorted(df['head_id'].unique())
    ncols = min(len(heads), 2)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 7))
    if ncols == 1:
        axes = [axes]
    fig.suptitle('이동 궤적 및 방향전환(COD) 이벤트', fontsize=14, fontweight='bold')

    for ax_i, hid in enumerate(heads[:2]):
        ax  = axes[ax_i]
        c   = HEAD_COLORS.get(hid, 'white')
        lbl = HEAD_LABELS.get(hid, f'Player {hid}')
        grp = df[df['head_id'] == hid].sort_values('video_time_sec')

        # 시간을 0~1로 정규화하여 컬러맵에 매핑
        t_n = ((grp['video_time_sec'] - grp['video_time_sec'].min()) /
               (grp['video_time_sec'].max() - grp['video_time_sec'].min() + 1e-9))

        # 경로 점 (시간 색상 인코딩)
        ax.scatter(grp['x_m'], grp['y_m'], c=t_n, cmap='plasma',
                   s=3, alpha=0.7, zorder=2, label='경로')
        # 연결선 (희미하게)
        ax.plot(grp['x_m'].values, grp['y_m'].values,
                color=c, alpha=0.25, lw=0.5, zorder=1)

        # COD 이벤트 위치 표시
        if len(cod_df) > 0:
            csub = cod_df[cod_df['head_id'] == hid]
            if len(csub) > 0:
                ax.scatter(csub['x_m'], csub['y_m'], color='red', s=20,
                           marker='x', zorder=5, label=f'COD ({len(csub)})')

        # 시작점(초록 원) / 끝점(노란 사각형)
        ax.scatter([grp['x_m'].iloc[0]],  [grp['y_m'].iloc[0]],
                   color='lime',   s=80, zorder=6, marker='o', label='시작')
        ax.scatter([grp['x_m'].iloc[-1]], [grp['y_m'].iloc[-1]],
                   color='yellow', s=80, zorder=6, marker='s', label='끝')

        ax.set_xlim(-0.5, court_w + 0.5)
        ax.set_ylim(-0.5, court_h + 0.5)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_title(f'{lbl}  |  총 {grp["step_dist_m"].sum():.1f} m')
        ax.set_aspect('equal', adjustable='box')
        ax.legend(fontsize=7); ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [저장] {out_path}')


def plot_cod_analysis(df: pd.DataFrame, cod_df: pd.DataFrame, out_path: str):
    """
    (3) 방향전환(COD) 능력 평가 3종 패널

    ① 방향 변화 분포 히스토그램
       - 빨간 점선: COD 임계각
       - 임계각 오른쪽 면적 = COD 이벤트 비율

    ② COD 발생 시 속도 vs 방향변화 산점도
       - 오른쪽 상단: 고속 상황에서 급격한 방향전환 → 높은 민첩성
       - 추세선(r값): 속도↑ 일수록 방향전환↓ 경향 확인 가능

    ③ 시간대별 누적 COD 계단 그래프
       - 기울기가 가파를수록 해당 구간에 방향전환 집중
       - 후반부로 갈수록 기울기가 완만해지면 피로 누적 가능성
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('방향전환(COD) 능력 평가', fontsize=14, fontweight='bold')

    # ── ① 방향 변화 분포 ──────────────────────────────────────────────────
    ax = axes[0]
    for hid, grp in df.groupby('head_id'):
        c   = HEAD_COLORS.get(hid, 'white')
        lbl = HEAD_LABELS.get(hid, f'Player {hid}')
        ax.hist(grp['ang_change'].dropna(), bins=40,
                color=c, alpha=0.55, label=lbl, density=True)
    ax.axvline(COD_ANGLE_THR, color='red', ls='--', lw=1,
               label=f'임계값 ({COD_ANGLE_THR}°)')
    ax.set_xlabel('방향 변화각 (도)'); ax.set_ylabel('밀도')
    ax.set_title('방향 변화 분포'); ax.legend(fontsize=7); ax.grid(alpha=0.25)

    # ── ② COD 시 속도 vs 방향변화 산점도 ────────────────────────────────
    ax = axes[1]
    if len(cod_df) > 0:
        for hid in sorted(cod_df['head_id'].unique()):
            c   = HEAD_COLORS.get(hid, 'white')
            lbl = HEAD_LABELS.get(hid, f'Player {hid}')
            sub = cod_df[cod_df['head_id'] == hid].dropna(
                subset=['speed_ms', 'ang_change'])
            ax.scatter(sub['speed_ms'], sub['ang_change'],
                       color=c, alpha=0.5, s=15, label=lbl)

            # scipy 설치 시 선형 회귀 추세선 추가
            if HAS_SCIPY and len(sub) >= 3:
                slope, intercept, r, p, _ = stats.linregress(
                    sub['speed_ms'].dropna(), sub['ang_change'].dropna())
                xr = np.linspace(sub['speed_ms'].min(), sub['speed_ms'].max(), 50)
                ax.plot(xr, slope * xr + intercept,
                        color=c, ls='--', lw=1, alpha=0.7,
                        label=f'r={r:.2f}')
    ax.set_xlabel('속도 (m/s)'); ax.set_ylabel('방향 변화각 (도)')
    ax.set_title('COD 발생 시 속도 vs 방향변화')
    ax.legend(fontsize=7); ax.grid(alpha=0.25)

    # ── ③ 시간대별 누적 COD 계단 그래프 ─────────────────────────────────
    ax = axes[2]
    if len(cod_df) > 0:
        for hid in sorted(cod_df['head_id'].unique()):
            c   = HEAD_COLORS.get(hid, 'white')
            lbl = HEAD_LABELS.get(hid, f'Player {hid}')
            sub = cod_df[cod_df['head_id'] == hid].sort_values('video_time_sec')
            # step 그래프: COD 발생할 때마다 1씩 계단형으로 증가
            ax.step(sub['video_time_sec'], np.arange(1, len(sub) + 1),
                    color=c, where='post', label=f'{lbl} (총 {len(sub)})')
    ax.set_xlabel('시간 (s)'); ax.set_ylabel('누적 COD 횟수')
    ax.set_title('시간대별 누적 COD'); ax.legend(fontsize=7); ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [저장] {out_path}')


def plot_work_rate(work_df: pd.DataFrame, eff_df: pd.DataFrame, out_path: str):
    """
    (4) 라운드별 운동 부하 분석 4종 패널

    ① 총 이동 거리 막대 (라운드별 체력 소비 비교)
    ② 평균 속도 추이 (라운드별 페이스 변화)
    ③ 이동 효율성 (라운드별 불필요 움직임 변화)
    ④ 고강도 구간 비율 + COD 횟수 (이중 축)
       막대: 고강도 구간 비율(%), 꺾은선: COD 횟수
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('라운드별 운동 부하 (Work Rate) 분석', fontsize=14, fontweight='bold')

    ids   = sorted(work_df['head_id'].unique())
    width = 0.35  # 막대 너비

    # ── ① 총 이동 거리 ───────────────────────────────────────────────────
    ax = axes[0, 0]
    for i, hid in enumerate(ids):
        sub = work_df[work_df['head_id'] == hid].sort_values('round')
        # 선수가 2명일 때 막대가 겹치지 않도록 오프셋 계산
        off = (i - len(ids) / 2 + 0.5) * width
        ax.bar(sub['round'] + off, sub['total_dist_m'], width=width,
               color=HEAD_COLORS.get(hid, 'white'), alpha=0.8,
               label=HEAD_LABELS.get(hid, f'P{hid}'))
    ax.set_xlabel('라운드'); ax.set_ylabel('이동 거리 (m)')
    ax.set_title('라운드별 총 이동 거리')
    ax.legend(fontsize=7); ax.grid(alpha=0.25, axis='y')

    # ── ② 평균 속도 추이 ─────────────────────────────────────────────────
    ax = axes[0, 1]
    for hid in ids:
        sub = work_df[work_df['head_id'] == hid].sort_values('round')
        ax.plot(sub['round'], sub['avg_speed_ms'],
                color=HEAD_COLORS.get(hid, 'white'), marker='o', ms=4,
                label=HEAD_LABELS.get(hid, f'P{hid}'))
    ax.set_xlabel('라운드'); ax.set_ylabel('평균 속도 (m/s)')
    ax.set_title('라운드별 평균 속도 추이')
    ax.legend(fontsize=7); ax.grid(alpha=0.25)

    # ── ③ 이동 효율성 ────────────────────────────────────────────────────
    ax = axes[1, 0]
    for hid in eff_df['head_id'].unique():
        sub = eff_df[eff_df['head_id'] == hid].sort_values('round')
        ax.plot(sub['round'], sub['efficiency_pct'],
                color=HEAD_COLORS.get(hid, 'white'), marker='s', ms=4,
                label=HEAD_LABELS.get(hid, f'P{hid}'))
    ax.set_xlabel('라운드'); ax.set_ylabel('이동 효율성 (%)')
    ax.set_title('라운드별 이동 효율성\n(직선거리 / 실제이동거리 × 100)')
    ax.set_ylim(0, 102); ax.legend(fontsize=7); ax.grid(alpha=0.25)

    # ── ④ 고강도 구간 + COD (이중 축) ────────────────────────────────────
    ax  = axes[1, 1]
    ax2 = ax.twinx()  # 오른쪽 y축: COD 횟수
    for i, hid in enumerate(ids):
        sub = work_df[work_df['head_id'] == hid].sort_values('round')
        off = (i - len(ids) / 2 + 0.5) * width
        # 왼쪽 축: 고강도 구간 비율 (%)
        ax.bar(sub['round'] + off, sub['hi_zone_frac'] * 100, width=width,
               color=HEAD_COLORS.get(hid, 'white'), alpha=0.6,
               label=f'{HEAD_LABELS.get(hid, "P"+str(hid))} HI%')
        # 오른쪽 축: COD 횟수 꺾은선
        ax2.plot(sub['round'], sub['cod_count'],
                 color=HEAD_COLORS.get(hid, 'white'), marker='^', ms=5,
                 ls='--', alpha=0.8,
                 label=f'{HEAD_LABELS.get(hid, "P"+str(hid))} COD')
    ax.set_xlabel('라운드'); ax.set_ylabel('고강도 구간 비율 (%)')
    ax2.set_ylabel('COD 횟수')
    ax.set_title('고강도 구간 비율 및 방향전환 횟수')
    # 두 축의 범례 통합
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6)
    ax.grid(alpha=0.25, axis='y')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [저장] {out_path}')


def plot_centroid(centroid_df: pd.DataFrame, court_w: float, court_h: float,
                  out_path: str):
    """
    (5) 무게중심 이동 패턴 3종 패널

    ① 무게중심 궤적 (시간 컬러맵)
       코트 위에서 경기 주도권이 이동한 방향/위치 시각화

    ② 무게중심 X/Y 시계열
       가로·세로 방향의 주도권 변화 분리 추적

    ③ 두 선수 간 거리 시계열
       빨간 점선: 전체 평균 거리
       거리가 줄어드는 구간 → 압박 국면
       거리가 늘어나는 구간 → 공간 분리·후퇴 국면
    """
    if centroid_df is None or len(centroid_df) == 0:
        print('  [건너뜀] 2인 동시 추적 데이터 없음 → 무게중심 분석 불가')
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('무게중심 이동 패턴 (2인 결합)', fontsize=14, fontweight='bold')

    # ── ① 궤적 ──────────────────────────────────────────────────────────
    ax = axes[0]
    t_n = ((centroid_df['video_time_sec'] - centroid_df['video_time_sec'].min()) /
           (centroid_df['video_time_sec'].max() - centroid_df['video_time_sec'].min() + 1e-9))
    sc = ax.scatter(centroid_df['cx_m'], centroid_df['cy_m'],
                    c=t_n, cmap='plasma', s=5, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='시간 정규화')
    ax.set_xlim(0, court_w); ax.set_ylim(0, court_h)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title('무게중심 이동 궤적')
    ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=0.2)

    # ── ② X/Y 시계열 ─────────────────────────────────────────────────────
    ax = axes[1]
    t  = centroid_df['video_time_sec']
    ax.plot(t, centroid_df['cx_m'], color='#FF8844', lw=0.8, alpha=0.9, label='CX (가로)')
    ax.plot(t, centroid_df['cy_m'], color='#44AAFF', lw=0.8, alpha=0.9, label='CY (세로)')
    ax.set_xlabel('시간 (s)'); ax.set_ylabel('위치 (m)')
    ax.set_title('무게중심 X/Y 시계열'); ax.legend(); ax.grid(alpha=0.25)

    # ── ③ 선수 간 거리 ────────────────────────────────────────────────────
    ax = axes[2]
    mean_dist = centroid_df['dist_between_m'].mean()
    ax.plot(t, centroid_df['dist_between_m'], color='#88FF44', lw=0.8, alpha=0.9)
    ax.axhline(mean_dist, color='red', ls='--', lw=1,
               label=f'평균 {mean_dist:.2f} m')
    ax.set_xlabel('시간 (s)'); ax.set_ylabel('거리 (m)')
    ax.set_title('선수 간 거리'); ax.legend(); ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [저장] {out_path}')


def plot_correlation(df: pd.DataFrame, out_path: str):
    """
    (6) 운동학적 변수 상관행렬 히트맵

    분석 변수:
      speed_ms     순간 속도
      accel_ms2    순간 가속도
      ang_change   방향 변화량
      step_dist_m  프레임 간 이동 거리

    색상 해석:
      초록(+1 근처): 강한 양의 상관 — 변수가 함께 증가
      빨강(-1 근처): 강한 음의 상관 — 한쪽이 증가하면 다른 쪽 감소
      노랑(0 근처) : 상관 없음

    각 셀에 상관계수(r) 값 직접 표시. |r|>0.5 이면 검은 글씨
    """
    var_keys   = ['speed_ms', 'accel_ms2', 'ang_change', 'step_dist_m']
    var_labels = ['속도\n(m/s)', '가속도\n(m/s²)', '방향변화\n(°)', '단계거리\n(m)']

    heads = sorted(df['head_id'].unique())
    ncols = min(len(heads), 2)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    if ncols == 1:
        axes = [axes]
    fig.suptitle('운동학적 변수 상관 분석', fontsize=14, fontweight='bold')

    for ax_i, hid in enumerate(heads[:2]):
        ax  = axes[ax_i]
        grp = df[df['head_id'] == hid][var_keys].dropna()
        lbl = HEAD_LABELS.get(hid, f'Player {hid}')

        if len(grp) < 10:
            ax.text(0.5, 0.5, '데이터 부족', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        corr = grp.corr().values  # Pearson 상관계수 행렬 (n×n)

        # 히트맵: RdYlGn 컬러맵, -1~+1 범위
        im = ax.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        n = len(var_keys)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(var_labels, fontsize=8)
        ax.set_yticklabels(var_labels, fontsize=8)

        # 각 셀에 r값 텍스트 표시
        for i in range(n):
            for j in range(n):
                # |r| > 0.5: 배경이 진하므로 검은 글씨, 그 외: 흰 글씨
                txt_color = 'black' if abs(corr[i, j]) > 0.5 else 'white'
                ax.text(j, i, f'{corr[i, j]:.2f}',
                        ha='center', va='center',
                        fontsize=9, color=txt_color, fontweight='bold')

        ax.set_title(f'{lbl}  상관행렬  (n={len(grp):,})')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [저장] {out_path}')


def plot_radar(summary_df: pd.DataFrame, out_path: str):
    """
    (7) 선수 비교 레이더(Spider) 차트

    6개 축: 평균속도 / 최대속도 / 최대가속도 / COD/분 / 이동효율 / 고강도비율

    정규화 방법:
      각 축을 두 선수 중 최대값으로 나누어 0~1 범위로 변환
      → 단위가 다른 지표를 같은 축에 표시 가능
      → 실제 값 비교는 summary_report.csv 참조

    단독 선수 데이터인 경우 레이더 차트 생략 (비교 의미 없음)
    """
    if len(summary_df) < 2:
        return  # 선수 1명이면 비교 불가

    metrics = ['avg_speed_ms', 'max_speed_ms', 'max_accel_ms2',
               'cod_per_min', 'avg_efficiency_pct', 'avg_hi_zone_frac']
    labels  = ['평균속도', '최대속도', '최대가속도',
               'COD/분', '이동효율', '고강도\n구간비율']

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.suptitle('선수 비교 레이더 차트', fontsize=14, fontweight='bold')

    # 6개 축의 각도 위치 (0~2π 균등 분할)
    angles  = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 마지막을 첫 번째와 연결하여 폐곡선 완성

    # 선수별 지표값 배열 (NaN → 0 처리)
    vals_arr = np.array(
        [[row.get(m, 0) or 0 for m in metrics] for _, row in summary_df.iterrows()],
        dtype=float
    )
    # 축별 최대값으로 정규화
    vmax = np.nanmax(vals_arr, axis=0)
    vmax[vmax == 0] = 1  # 0 나눗셈 방지
    vals_norm = vals_arr / vmax

    for i, (_, row) in enumerate(summary_df.iterrows()):
        hid = int(row['head_id'])
        c   = HEAD_COLORS.get(hid, 'white')
        lbl = row['label']
        v   = vals_norm[i].tolist() + vals_norm[i][:1].tolist()  # 폐곡선
        ax.plot(angles, v, color=c, lw=2, label=lbl)
        ax.fill(angles, v, color=c, alpha=0.15)  # 내부 채우기

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [저장] {out_path}')


# ══════════════════════════════════════════════════════════════════════════════
# ■ 메인 진입점
# ══════════════════════════════════════════════════════════════════════════════
def main():
    """
    CLI 인자 파싱 → 분석 파이프라인 순차 실행 → 결과 저장

    파이프라인 순서:
      [1/7] 데이터 로드 & 좌표 변환
      [2/7] 운동학 변수 계산 (속도·가속도·방향각)
      [3/7] COD 이벤트 감지
      [4/7] 라운드별 효율성·운동 부하 산출
      [5/7] 무게중심 계산
      [6/7] 요약 통계 산출 & CSV 저장
      [7/7] 시각화 7종 저장

    모든 결과는 out_dir/motion_analysis_<timestamp>/ 에 저장됩니다.
    """
    global COD_ANGLE_THR, HI_SPEED_PCT  # CLI 인자로 런타임 오버라이드

    parser = argparse.ArgumentParser(
        description='운동학 분석 — video_head_tracking_retrack.py 출력 CSV 전용',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    # ── 필수 인자 ──
    parser.add_argument('--csv',
                        type=str, required=True,
                        help='video_head_tracking_retrack.py 가 저장한 CSV 경로')
    # ── 선택 인자 (기본값 = 전역 상수) ──
    parser.add_argument('--out-dir',
                        type=str, default='output',
                        help='결과 저장 상위 폴더 (기본: output)')
    parser.add_argument('--court-w',
                        type=float, default=COURT_W_M,
                        help=f'실제 코트 가로 m (기본: {COURT_W_M})')
    parser.add_argument('--court-h',
                        type=float, default=COURT_H_M,
                        help=f'실제 코트 세로 m (기본: {COURT_H_M})')
    parser.add_argument('--rect-w',
                        type=int, default=RECT_W_PX,
                        help=f"추적 스크립트의 RECT_W 픽셀 (기본: {RECT_W_PX})")
    parser.add_argument('--rect-h',
                        type=int, default=RECT_H_PX,
                        help=f"추적 스크립트의 RECT_H 픽셀 (기본: {RECT_H_PX})")
    parser.add_argument('--round-sec',
                        type=int, default=ROUND_SEC,
                        help=f'라운드 구분 단위 초 (기본: {ROUND_SEC})')
    parser.add_argument('--cod-thr',
                        type=float, default=COD_ANGLE_THR,
                        help=f'방향전환 임계각 도 (기본: {COD_ANGLE_THR})')
    parser.add_argument('--hi-pct',
                        type=float, default=HI_SPEED_PCT,
                        help=f'고강도 속도 백분위 (기본: {HI_SPEED_PCT})')
    args = parser.parse_args()

    # CLI 인자로 전역 상수 덮어쓰기
    COD_ANGLE_THR = args.cod_thr
    HI_SPEED_PCT  = args.hi_pct

    # 결과 저장 폴더: 실행 시각 포함하여 중복 방지
    ts      = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.out_dir, f'motion_analysis_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    print('=' * 65)
    print('  운동학 분석 도구  (video_head_tracking_retrack.py CSV 기반)')
    print('=' * 65)
    print(f'  입력 CSV : {args.csv}')
    print(f'  코트 크기: {args.court_w} m × {args.court_h} m')
    print(f'  라운드   : {args.round_sec} 초 단위')
    print(f'  COD 임계 : {COD_ANGLE_THR}°')
    print(f'  HI 기준  : 상위 {int(HI_SPEED_PCT)}% 속도')
    print(f'  결과 폴더: {out_dir}')
    print()

    # ── [1/7] 로드 & 전처리 ─────────────────────────────────────────────
    print('[1/7] 데이터 로드 및 좌표 변환 중...')
    df = load_and_clean(args.csv, args.rect_w, args.rect_h,
                        args.court_w, args.court_h)
    n_heads = df['head_id'].nunique()
    print(f'      유효 프레임: {len(df):,}  |  선수 수: {n_heads}')

    # state='tracking' 데이터가 너무 적으면 분석 불가
    if len(df) < 10:
        print('[ERROR] 유효 데이터(state=tracking)가 너무 적습니다.')
        print('        video_head_tracking_retrack.py 로 영상을 다시 추적하거나')
        print('        호모그래피 캘리브레이션이 완료된 영상인지 확인하세요.')
        return

    # ── [2/7] 운동학 계산 ───────────────────────────────────────────────
    print('[2/7] 속도·가속도·방향각 계산 중...')
    df = compute_kinematics(df)

    # ── [3/7] COD 이벤트 감지 ───────────────────────────────────────────
    print('[3/7] 방향전환(COD) 이벤트 감지 중...')
    cod_df = detect_cod_events(df, COD_ANGLE_THR)
    print(f'      COD 이벤트: {len(cod_df):,}  (임계각 ≥ {COD_ANGLE_THR}°)')

    # ── [4/7] 라운드별 효율성·운동 부하 ────────────────────────────────
    print('[4/7] 라운드별 효율성·운동 부하 계산 중...')
    eff_df  = compute_efficiency(df, args.round_sec)
    work_df = compute_work_rate(df, args.round_sec, HI_SPEED_PCT)

    # ── [5/7] 무게중심 ──────────────────────────────────────────────────
    print('[5/7] 무게중심 계산 중...')
    centroid_df = compute_centroid(df)
    if len(centroid_df) > 0:
        mean_d = centroid_df['dist_between_m'].mean()
        print(f'      동시 추적 프레임: {len(centroid_df):,}  '
              f'(평균 선수 간 거리: {mean_d:.2f} m)')
    else:
        print('      단독 선수 데이터 — 무게중심 분석 생략')

    # ── [6/7] 요약 통계 & CSV 저장 ──────────────────────────────────────
    print('[6/7] 요약 통계 산출 중...')
    summary_df = compute_summary(df, eff_df, work_df, cod_df)

    # CSV 저장 (BOM 포함 UTF-8 → 한글 엑셀 호환)
    summary_df.to_csv(os.path.join(out_dir, 'summary_report.csv'),
                      index=False, encoding='utf-8-sig')
    eff_df.to_csv(os.path.join(out_dir, 'efficiency_by_round.csv'),
                  index=False, encoding='utf-8-sig')
    work_df.to_csv(os.path.join(out_dir, 'work_rate_by_round.csv'),
                   index=False, encoding='utf-8-sig')
    if len(cod_df) > 0:
        # COD 이벤트 CSV: 분석에 필요한 컬럼만 선택
        save_cols = [c for c in ['frame', 'video_time_sec', 'head_id',
                                  'x_m', 'y_m', 'speed_ms', 'ang_change']
                     if c in cod_df.columns]
        cod_df[save_cols].to_csv(os.path.join(out_dir, 'cod_events.csv'),
                                  index=False, encoding='utf-8-sig')
    if len(centroid_df) > 0:
        centroid_df.to_csv(os.path.join(out_dir, 'centroid.csv'),
                           index=False, encoding='utf-8-sig')

    # 콘솔 출력
    print()
    print('  ── 선수별 요약 통계 ─────────────────────────────────────────')
    pd.set_option('display.float_format', '{:.3f}'.format)
    print(summary_df.to_string(index=False))
    print()

    # ── [7/7] 시각화 ────────────────────────────────────────────────────
    print('[7/7] 시각화 생성 중...')
    plot_speed_accel(df,
        os.path.join(out_dir, 'speed_accel.png'))
    plot_trajectory(df, args.court_w, args.court_h, cod_df,
        os.path.join(out_dir, 'trajectory.png'))
    plot_cod_analysis(df, cod_df,
        os.path.join(out_dir, 'cod_analysis.png'))
    plot_work_rate(work_df, eff_df,
        os.path.join(out_dir, 'work_rate.png'))
    plot_centroid(centroid_df, args.court_w, args.court_h,
        os.path.join(out_dir, 'centroid.png'))
    plot_correlation(df,
        os.path.join(out_dir, 'correlation.png'))
    plot_radar(summary_df,
        os.path.join(out_dir, 'radar_comparison.png'))

    print()
    print('=' * 65)
    print(f'  분석 완료! → {out_dir}')
    print('=' * 65)


if __name__ == '__main__':
    main()
