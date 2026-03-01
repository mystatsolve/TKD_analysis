"""
Head Tracking with Auto Re-tracking - 2 People
──────────────────────────────────────────────────────────────────
[프로그램 개요]
  CSRT 트래커가 머리를 잃어버렸을 때(LOST),
  마지막으로 저장된 머리 이미지(템플릿)를 사용해
  자동으로 재탐색(Template Matching)합니다.

[상태 머신 (State Machine) - 슬롯별]
  none      : 초기화 전 (아직 클릭 안 함)
  tracking  : CSRT 트래커 정상 추적 중
  searching : 추적 끊김 → 템플릿 매칭으로 재탐색 중
  failed    : 재탐색 실패 누적 → 수동 재클릭 필요

[재탐색 원리]
  1. 추적 성공 중 매 프레임: 현재 박스 영역 이미지를 템플릿으로 저장
  2. 추적 실패(LOST) 시: 마지막 위치 주변 확장 영역에서 템플릿 탐색
  3. 유사도(TM_CCOEFF_NORMED) > 임계값이면 해당 위치로 트래커 재초기화
  4. SEARCH_FAIL_MAX 프레임 연속 실패 시 FAILED 상태로 전환

[Controls]
  Click x4    : 기준점 설정 (캘리브레이션)
  ENTER       : 호모그래피 확정
  1 / 2       : 슬롯 선택 (HEAD 1 / HEAD 2)
  Left-click  : 선택 슬롯 트래커 초기화 / 재초기화
  Right-click : 선택 슬롯 추적 중지
  C           : 선택 슬롯 추적 중지
  R           : 전체 초기화
  S           : 스냅샷 저장
  Q / ESC     : 종료 & CSV 저장
"""

import os, csv, time, warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# ■ 설정 상수
# ══════════════════════════════════════════════════════════════════════════════
CAM_ID          = 0                 # 카메라 번호
OUT_DIR         = 'ViTPose/output'  # 결과 저장 폴더
RECT_W          = 480               # Bird's-Eye 뷰 가로 크기 (픽셀)
RECT_H          = 480               # Bird's-Eye 뷰 세로 크기 (픽셀)
TRACK_BOX_W     = 80                # 트래커 초기 박스 너비 (픽셀)
TRACK_BOX_H     = 80                # 트래커 초기 박스 높이 (픽셀)
NUM_HEADS       = 2                 # 동시 추적 최대 인원 수

# ── 재탐색 관련 설정 ──────────────────────────────────────────────────────────
SEARCH_SCALE    = 3.5   # 재탐색 영역 = 마지막 박스 크기 × 이 배수
                        # 클수록 더 넓은 범위를 탐색하지만 느림
SEARCH_CONF_THR = 0.40  # 템플릿 매칭 유사도 임계값 (0.0~1.0)
                        # 이 값 이상이면 재탐색 성공으로 판단
SEARCH_FAIL_MAX = 90    # 이 프레임 수 연속 재탐색 실패 시 FAILED 상태로 전환
                        # (예: 30fps 카메라 기준 약 3초)
TEMPLATE_UPDATE_INTERVAL = 5  # 트래킹 중 템플릿을 갱신하는 프레임 간격
                               # 작을수록 최신 외형 반영, 너무 작으면 오류 누적
# ─────────────────────────────────────────────────────────────────────────────

# ── 슬롯 상태 정의 ─────────────────────────────────────────────────────────────
STATE_NONE      = 'none'       # 미초기화
STATE_TRACKING  = 'tracking'   # 정상 추적 중
STATE_SEARCHING = 'searching'  # 재탐색 중
STATE_FAILED    = 'failed'     # 재탐색 실패 (수동 개입 필요)

# ── CSV 헤더 ──────────────────────────────────────────────────────────────────
CSV_HEADER = [
    'frame', 'timestamp', 'head_id',
    'raw_x', 'raw_y',
    'rect_x', 'rect_y',
    'rect_x_norm', 'rect_y_norm',
    'homography_applied',
    'state',            # 이 프레임의 슬롯 상태 (tracking/searching/failed/none)
    'retrack_score',    # 재탐색 시 템플릿 매칭 유사도 (없으면 nan)
]

# ── 색상 정의 (BGR) ────────────────────────────────────────────────────────────
HEAD_COLORS = [
    (255,   0, 255),  # HEAD 1: 마젠타 (정상 추적)
    (  0, 255, 255),  # HEAD 2: 시안   (정상 추적)
]
SEARCH_COLOR  = (0, 200, 255)   # 주황: 재탐색 중
FAILED_COLOR  = (0,  50, 255)   # 빨강: 재탐색 실패
BOX_COLORS    = [(200, 0, 200), (0, 200, 200)]


# ══════════════════════════════════════════════════════════════════════════════
# ■ 호모그래피 유틸리티
# ══════════════════════════════════════════════════════════════════════════════
def compute_homography(src_pts, rect_w, rect_h):
    """카메라 기준점 4개 → Bird's-Eye 직사각형으로 변환하는 H 행렬 계산."""
    dst_pts = np.float32([
        [0,      0      ],
        [rect_w, 0      ],
        [rect_w, rect_h ],
        [0,      rect_h ],
    ])
    H, _ = cv2.findHomography(np.float32(src_pts), dst_pts, method=0)
    return H


def apply_homography(H, px, py):
    """단일 점 (px, py)을 호모그래피 H로 변환."""
    out = cv2.perspectiveTransform(np.float32([[[px, py]]]), H)
    return float(out[0][0][0]), float(out[0][0][1])


# ══════════════════════════════════════════════════════════════════════════════
# ■ 재탐색 (Template Matching) 핵심 함수
# ══════════════════════════════════════════════════════════════════════════════
def search_with_template(frame, template, last_x, last_y, last_w, last_h,
                          scale, conf_thr, cam_w, cam_h):
    """
    템플릿 매칭으로 화면에서 머리를 재탐색합니다.

    [동작 원리]
    1. 마지막으로 알려진 위치(last_x, last_y) 주변에
       scale배 크기의 탐색 영역(ROI)을 설정
    2. 해당 ROI 안에서 저장된 template 이미지와 가장 비슷한 위치를 탐색
    3. 유사도(score)가 conf_thr 이상이면 해당 위치 반환

    매개변수:
        frame            : 현재 카메라 프레임 (numpy array)
        template         : 마지막 성공 프레임에서 저장한 머리 이미지
        last_x, last_y   : 마지막으로 알려진 머리 중심 좌표
        last_w, last_h   : 마지막 트래킹 박스 크기
        scale            : 탐색 영역 확장 배수
        conf_thr         : 유사도 임계값
        cam_w, cam_h     : 카메라 프레임 너비/높이

    반환값:
        성공: (found_cx, found_cy, score) - 발견된 중심 좌표와 유사도
        실패: None
    """
    th, tw = template.shape[:2]  # 템플릿 높이, 너비

    # ── 탐색 ROI(Region of Interest) 계산 ──
    # 마지막 위치 주변 scale배 확장한 영역
    half_sw = int(last_w * scale / 2)  # 탐색 영역 절반 너비
    half_sh = int(last_h * scale / 2)  # 탐색 영역 절반 높이

    # 화면 경계를 벗어나지 않도록 클램핑
    roi_x1 = max(0, int(last_x) - half_sw)
    roi_y1 = max(0, int(last_y) - half_sh)
    roi_x2 = min(cam_w, int(last_x) + half_sw)
    roi_y2 = min(cam_h, int(last_y) + half_sh)

    # ROI가 템플릿보다 작으면 탐색 불가
    if (roi_y2 - roi_y1) < th or (roi_x2 - roi_x1) < tw:
        return None

    # ── 템플릿 매칭 실행 ──
    search_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    # TM_CCOEFF_NORMED: 정규화된 상관계수 방식
    #   결과값 범위: -1(역상관) ~ 1(완전일치)
    #   1에 가까울수록 템플릿과 유사
    result = cv2.matchTemplate(search_roi, template, cv2.TM_CCOEFF_NORMED)

    # 결과 맵에서 최대값 위치 찾기
    # min_val, max_val: 최소/최대 유사도 점수
    # min_loc, max_loc: 최소/최대 위치 (TM_CCOEFF_NORMED는 max가 최적)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < conf_thr:
        # 유사도가 임계값 미만 → 찾지 못함
        return None

    # 템플릿 좌상단 위치(max_loc)를 ROI 좌표 → 전체 프레임 좌표로 변환
    found_x1 = roi_x1 + max_loc[0]  # 템플릿 좌상단 x (전체 프레임 기준)
    found_y1 = roi_y1 + max_loc[1]  # 템플릿 좌상단 y (전체 프레임 기준)

    # 중심 좌표 계산
    found_cx = found_x1 + tw / 2
    found_cy = found_y1 + th / 2

    return found_cx, found_cy, float(max_val)


# ══════════════════════════════════════════════════════════════════════════════
# ■ 시각화 함수
# ══════════════════════════════════════════════════════════════════════════════
def get_slot_color(state, slot):
    """슬롯 상태에 따라 표시 색상 반환."""
    if state == STATE_TRACKING:
        return HEAD_COLORS[slot]
    elif state == STATE_SEARCHING:
        return SEARCH_COLOR
    else:  # failed or none
        return FAILED_COLOR


def draw_head_cam(frame, cx, cy, bbox, state, slot):
    """카메라 뷰에 트래킹 박스 + 중심 십자 + 상태 텍스트 표시."""
    color = get_slot_color(state, slot)

    # 트래킹 박스 그리기
    if bbox is not None:
        x, y, w, h = [int(v) for v in bbox]
        # 상태에 따라 선 스타일 변경
        line_type = cv2.LINE_AA
        if state == STATE_SEARCHING:
            # 재탐색 중: 점선 효과를 위해 두 개의 사각형으로 표현
            cv2.rectangle(frame, (x, y), (x+w, y+h), SEARCH_COLOR, 1)
            cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), SEARCH_COLOR, 1)
        elif state == STATE_FAILED:
            cv2.rectangle(frame, (x, y), (x+w, y+h), FAILED_COLOR, 2)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), BOX_COLORS[slot], 2)

    # 중심 십자 마커
    icx, icy = int(cx), int(cy)
    cv2.drawMarker(frame, (icx, icy), color, cv2.MARKER_CROSS, 26, 2, cv2.LINE_AA)
    cv2.circle(frame, (icx, icy), 12, color, 2)

    # 상태 텍스트
    state_txt = {
        STATE_TRACKING:  'TRACKING',
        STATE_SEARCHING: 'SEARCHING...',
        STATE_FAILED:    'FAILED - click',
    }.get(state, '')
    cv2.putText(frame, f'HEAD{slot+1} {state_txt} ({icx},{icy})',
                (icx + 14, icy - 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.46, color, 1)


def draw_search_area(frame, last_x, last_y, last_w, last_h, scale, cam_w, cam_h):
    """재탐색 중일 때 탐색 영역을 반투명 사각형으로 표시."""
    half_sw = int(last_w * scale / 2)
    half_sh = int(last_h * scale / 2)
    x1 = max(0, int(last_x) - half_sw)
    y1 = max(0, int(last_y) - half_sh)
    x2 = min(cam_w, int(last_x) + half_sw)
    y2 = min(cam_h, int(last_y) + half_sh)

    # 반투명 노란 테두리로 탐색 영역 표시
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), -1)
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)  # 8% 불투명도
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 1)
    cv2.putText(frame, 'SEARCH AREA', (x1 + 4, y1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)


def draw_calib_points(frame, pts, done):
    """캘리브레이션 기준점들을 화면에 표시."""
    labels = ['1:TL', '2:TR', '3:BR', '4:BL']
    for i, (px, py) in enumerate(pts):
        cv2.circle(frame, (int(px), int(py)),  8, (0, 0, 255), -1)
        cv2.circle(frame, (int(px), int(py)), 10, (255, 255, 255), 2)
        cv2.putText(frame, labels[i],
                    (int(px) + 10, int(py) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    if done and len(pts) == 4:
        cv2.polylines(frame, [np.int32(pts)], True, (0, 0, 255), 1, cv2.LINE_AA)


def draw_bird_eye(rect_w, rect_h, head_pts):
    """
    Bird's-Eye 뷰 생성.
    head_pts: [(rect_x, rect_y, slot, state), ...]
    """
    canvas = np.zeros((rect_h, rect_w, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)
    for gx in range(0, rect_w, rect_w // 6):
        cv2.line(canvas, (gx, 0), (gx, rect_h), (60, 60, 60), 1)
    for gy in range(0, rect_h, rect_h // 6):
        cv2.line(canvas, (0, gy), (rect_w, gy), (60, 60, 60), 1)
    cv2.rectangle(canvas, (0, 0), (rect_w-1, rect_h-1), (100, 100, 100), 2)

    for rx, ry, slot, state in head_pts:
        cx = int(np.clip(rx, 0, rect_w - 1))
        cy = int(np.clip(ry, 0, rect_h - 1))
        color = get_slot_color(state, slot)
        cv2.drawMarker(canvas, (cx, cy), color, cv2.MARKER_CROSS, 26, 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 15, color, 2)
        state_short = {'tracking':'TRK','searching':'SCH','failed':'FAIL'}.get(state,'')
        cv2.putText(canvas, f'H{slot+1}[{state_short}] ({rx:.0f},{ry:.0f})',
                    (cx + 16, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    cv2.putText(canvas, 'TOP-LEFT',  (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    cv2.putText(canvas, 'TOP-RIGHT', (rect_w - 76, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    cv2.putText(canvas, 'BOT-LEFT',  (4, rect_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    cv2.putText(canvas, "Bird's-Eye VIEW", (8, rect_h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    return canvas


def draw_hud(frame, fps, frame_id, n_pts, h_valid, selected, slot_states,
             search_fail_counts):
    """상단 HUD: FPS, 슬롯 상태, 조작 안내 표시."""
    fh, fw = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (fw, 54), (0, 0, 0), -1)

    if not h_valid:
        if n_pts < 4:
            msg = f'CALIBRATION: Click {4 - n_pts} more point(s)  (TL->TR->BR->BL)'
            col = (0, 200, 255)
        else:
            msg = 'Press ENTER to confirm homography'
            col = (0, 255, 128)
        cv2.putText(frame, msg, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1)
    else:
        cv2.putText(frame,
                    f'FPS:{fps:.1f}  Frame:{frame_id}  | 1/2=slot  LClick=init  C/RC=stop  R=reset',
                    (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

        for s in range(NUM_HEADS):
            st  = slot_states[s]
            hc  = get_slot_color(st, s)
            sel = '>' if s == selected else ' '

            # 상태별 표시 텍스트
            if st == STATE_TRACKING:
                stxt = 'TRACKING'
            elif st == STATE_SEARCHING:
                remain = SEARCH_FAIL_MAX - search_fail_counts[s]
                stxt = f'SEARCHING ({remain}f left)'
            elif st == STATE_FAILED:
                stxt = 'FAILED - LClick to reinit'
            else:
                stxt = 'no init'

            cv2.putText(frame, f'{sel} HEAD{s+1}: {stxt}',
                        (8 + s * 240, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, hc, 1)

    cv2.putText(frame, 'CAMERA VIEW', (fw - 100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1)


# ══════════════════════════════════════════════════════════════════════════════
# ■ 전역 상태 변수 & 마우스 콜백
# ══════════════════════════════════════════════════════════════════════════════
calib_pts      = []
h_valid_flag   = [False]   # 마우스 콜백에서 호모그래피 확정 여부 참조용
selected_slot  = [0]       # 현재 선택된 슬롯 (0=HEAD1, 1=HEAD2)
init_click_pts = [None, None]  # 슬롯별 초기화 클릭 좌표 (None=요청 없음)
stop_flags     = [False, False]


def mouse_cb(event, x, y, flags, param):
    """마우스 이벤트 처리: 캘리브레이션 클릭 / 트래커 초기화 클릭."""
    global calib_pts
    s = selected_slot[0]

    if event == cv2.EVENT_LBUTTONDOWN:
        if not h_valid_flag[0]:
            # 캘리브레이션 단계: 기준점 추가
            if len(calib_pts) < 4:
                calib_pts.append([float(x), float(y)])
                print(f'  Calib point {len(calib_pts)}: ({x}, {y})')
        else:
            # 추적 단계: 선택 슬롯 트래커 초기화 요청
            init_click_pts[s] = (x, y)
            stop_flags[s]     = False
            print(f'  [HEAD{s+1}] init at ({x}, {y})')

    elif event == cv2.EVENT_RBUTTONDOWN:
        if h_valid_flag[0]:
            stop_flags[s] = True
            print(f'  [HEAD{s+1}] stopped')


# ══════════════════════════════════════════════════════════════════════════════
# ■ 메인 함수
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global calib_pts

    os.makedirs(OUT_DIR, exist_ok=True)
    run_ts   = time.strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(OUT_DIR, f'head_retrack_{run_ts}.csv')

    print('=' * 65)
    print('Head Tracking + Auto Re-tracking  (2 People)')
    print('=' * 65)
    print(f'  Camera       : {CAM_ID}')
    print(f'  BirdEye      : {RECT_W}x{RECT_H}')
    print(f'  Search scale : {SEARCH_SCALE}x  |  Conf thr: {SEARCH_CONF_THR}')
    print(f'  Fail max     : {SEARCH_FAIL_MAX} frames before FAILED state')
    print(f'  CSV          : {csv_path}')
    print()
    print('  [Step 1] Click 4 corners (TL->TR->BR->BL), then ENTER')
    print('  [Step 2] Press 1 or 2, then left-click on the head')
    print('           If lost, auto re-search starts automatically')
    print('=' * 65)

    # ── 카메라 열기 ──
    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open camera {CAM_ID}')
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ch = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Camera {CAM_ID}: {cw}x{ch}\n')

    WIN = 'Head Tracking + ReTrack  |  1/2=select  LClick=init  Q=quit'
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, mouse_cb)

    # ── 호모그래피 상태 ──
    H       = None
    h_valid = False

    # ── 슬롯별 상태 변수 (인덱스 0=HEAD1, 1=HEAD2) ──
    trackers          = [None, None]     # CSRT 트래커 객체
    tracker_bbox      = [None, None]     # 현재 트래킹 박스 (x, y, w, h)
    head_raw          = [None, None]     # 마지막 머리 중심 (카메라 픽셀)
    slot_states       = [STATE_NONE,  STATE_NONE]  # 슬롯 상태 문자열
    templates         = [None, None]     # 마지막 성공 프레임의 머리 이미지 패치
    template_boxes    = [None, None]     # 템플릿 저장 시의 박스 크기 (w, h)
    search_fail_count = [0, 0]           # 연속 재탐색 실패 횟수
    tmpl_update_cnt   = [0, 0]           # 템플릿 갱신용 프레임 카운터
    last_retrack_score= [float('nan'), float('nan')]  # 마지막 재탐색 유사도 점수

    # ── CSV ──
    csv_file   = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(CSV_HEADER)

    frame_id   = 0
    save_idx   = 0
    fps        = 0.0
    t_buf      = []
    start_time = time.time()

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                print('[ERROR] Camera read failed.'); break

            frame_id  += 1
            timestamp  = time.time() - start_time
            vis        = frame.copy()

            bird_head_pts = []  # Bird's-Eye 뷰에 표시할 머리 위치 목록

            for s in range(NUM_HEADS):
                retrack_score = float('nan')  # 이 프레임의 재탐색 점수

                # ── [1] 추적 중지 요청 처리 ──
                if stop_flags[s]:
                    trackers[s]          = None
                    tracker_bbox[s]      = None
                    head_raw[s]          = None
                    slot_states[s]       = STATE_NONE
                    templates[s]         = None
                    search_fail_count[s] = 0
                    stop_flags[s]        = False

                # ── [2] 마우스 클릭으로 트래커 초기화 ──
                if init_click_pts[s] is not None:
                    ix, iy = init_click_pts[s]
                    bx = max(0, ix - TRACK_BOX_W // 2)
                    by = max(0, iy - TRACK_BOX_H // 2)
                    bw = min(TRACK_BOX_W, cw - bx)
                    bh = min(TRACK_BOX_H, ch - by)

                    trackers[s] = cv2.TrackerCSRT_create()
                    trackers[s].init(frame, (bx, by, bw, bh))
                    tracker_bbox[s]      = (bx, by, bw, bh)
                    head_raw[s]          = (float(ix), float(iy))
                    slot_states[s]       = STATE_TRACKING
                    search_fail_count[s] = 0
                    tmpl_update_cnt[s]   = 0

                    # 초기 템플릿 즉시 저장
                    crop = frame[by:by+bh, bx:bx+bw]
                    if crop.size > 0:
                        templates[s]      = crop.copy()
                        template_boxes[s] = (bw, bh)

                    init_click_pts[s] = None
                    print(f'  [HEAD{s+1}] Tracker initialized.')

                # ── [3] 현재 슬롯 상태에 따른 처리 ──

                if slot_states[s] == STATE_TRACKING:
                    # ── 정상 추적 중 ──
                    ok, bbox = trackers[s].update(frame)

                    if ok:
                        # 추적 성공
                        tracker_bbox[s] = bbox
                        cx = bbox[0] + bbox[2] / 2
                        cy = bbox[1] + bbox[3] / 2
                        head_raw[s]          = (cx, cy)
                        search_fail_count[s] = 0

                        # 템플릿 주기적 갱신
                        # (추적 중 외형 변화에 대응하기 위해 일정 간격으로 업데이트)
                        tmpl_update_cnt[s] += 1
                        if tmpl_update_cnt[s] >= TEMPLATE_UPDATE_INTERVAL:
                            x, y, w, h = [int(v) for v in bbox]
                            x1 = max(0, x); y1 = max(0, y)
                            x2 = min(cw, x+w); y2 = min(ch, y+h)
                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0 and crop.shape[0] > 5 and crop.shape[1] > 5:
                                templates[s]      = crop.copy()
                                template_boxes[s] = (x2-x1, y2-y1)
                            tmpl_update_cnt[s] = 0

                    else:
                        # 추적 실패 → 재탐색 모드로 전환
                        slot_states[s] = STATE_SEARCHING
                        print(f'  [HEAD{s+1}] Lost! Switching to SEARCHING...')

                elif slot_states[s] == STATE_SEARCHING:
                    # ── 재탐색 중 ──
                    if templates[s] is not None and head_raw[s] is not None:
                        tw_s, th_s = template_boxes[s] if template_boxes[s] else \
                                     (templates[s].shape[1], templates[s].shape[0])

                        result = search_with_template(
                            frame, templates[s],
                            head_raw[s][0], head_raw[s][1],
                            tw_s, th_s,
                            SEARCH_SCALE, SEARCH_CONF_THR, cw, ch
                        )

                        if result is not None:
                            # ── 재탐색 성공 → 트래커 재초기화 ──
                            found_cx, found_cy, score = result
                            retrack_score = score
                            last_retrack_score[s] = score

                            bx = max(0, int(found_cx) - tw_s // 2)
                            by = max(0, int(found_cy) - th_s // 2)
                            bw = min(tw_s, cw - bx)
                            bh = min(th_s, ch - by)

                            trackers[s] = cv2.TrackerCSRT_create()
                            trackers[s].init(frame, (bx, by, bw, bh))
                            tracker_bbox[s]      = (bx, by, bw, bh)
                            head_raw[s]          = (found_cx, found_cy)
                            slot_states[s]       = STATE_TRACKING
                            search_fail_count[s] = 0
                            print(f'  [HEAD{s+1}] Re-acquired! score={score:.3f}  '
                                  f'at ({found_cx:.0f},{found_cy:.0f})')
                        else:
                            # 재탐색 실패
                            search_fail_count[s] += 1
                            if search_fail_count[s] >= SEARCH_FAIL_MAX:
                                slot_states[s] = STATE_FAILED
                                print(f'  [HEAD{s+1}] FAILED after {SEARCH_FAIL_MAX} frames. '
                                      f'Please click to reinitialize.')

                # ── [4] 시각화 ──
                if head_raw[s] is not None and slot_states[s] != STATE_NONE:
                    raw_x, raw_y = head_raw[s]

                    # 재탐색 영역 시각화 (탐색 중일 때만)
                    if slot_states[s] == STATE_SEARCHING and template_boxes[s]:
                        tw_s, th_s = template_boxes[s]
                        draw_search_area(vis, raw_x, raw_y, tw_s, th_s,
                                         SEARCH_SCALE, cw, ch)

                    draw_head_cam(vis, raw_x, raw_y, tracker_bbox[s],
                                  slot_states[s], s)

                # ── [5] 호모그래피 변환 & Bird's-Eye ──
                raw_x_csv = raw_y_csv = rect_x = rect_y = rx_n = ry_n = float('nan')
                h_flag = 0

                if head_raw[s] is not None and slot_states[s] != STATE_NONE:
                    raw_x_csv, raw_y_csv = head_raw[s]
                    if h_valid and H is not None:
                        rect_x, rect_y = apply_homography(H, raw_x_csv, raw_y_csv)
                        rx_n   = rect_x / RECT_W
                        ry_n   = rect_y / RECT_H
                        h_flag = 1
                        bird_head_pts.append((rect_x, rect_y, s, slot_states[s]))

                # ── [6] CSV 기록 ──
                def f(v):
                    return 'nan' if (v != v) else f'{v:.2f}'

                csv_writer.writerow([
                    frame_id, f'{timestamp:.4f}', s + 1,
                    f(raw_x_csv), f(raw_y_csv),
                    f(rect_x), f(rect_y),
                    f'{rx_n:.4f}' if h_flag else 'nan',
                    f'{ry_n:.4f}' if h_flag else 'nan',
                    h_flag,
                    slot_states[s],
                    f'{retrack_score:.4f}' if not (retrack_score != retrack_score) else 'nan',
                ])

            if frame_id % 30 == 0:
                csv_file.flush()

            # ── 캘리브레이션 포인트 & HUD ──
            draw_calib_points(vis, calib_pts, h_valid)
            draw_hud(vis, fps, frame_id, len(calib_pts), h_valid,
                     selected_slot[0], slot_states, search_fail_count)

            # ── Bird's-Eye 합성 ──
            bird         = draw_bird_eye(RECT_W, RECT_H, bird_head_pts)
            bird_resized = cv2.resize(bird, (RECT_W, ch))
            combined     = np.hstack([vis, bird_resized])
            cv2.imshow(WIN, combined)

            # FPS
            t_buf.append(time.time() - t0)
            if len(t_buf) > 20: t_buf.pop(0)
            fps = len(t_buf) / sum(t_buf)

            # ── 키 입력 ──
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):
                print('Quit.'); break

            elif key == 13:   # ENTER: 호모그래피 확정
                if len(calib_pts) == 4:
                    H       = compute_homography(calib_pts, RECT_W, RECT_H)
                    h_valid = True
                    h_valid_flag[0] = True
                    print('\n[OK] Homography computed!')
                    print(f'     H =\n{H}\n')
                    print('  Press 1 or 2, then click on each head.')
                else:
                    print(f'[WARN] Need 4 points, got {len(calib_pts)}')

            elif key == ord('1'):
                selected_slot[0] = 0; print('  [Slot] HEAD 1 selected')
            elif key == ord('2'):
                selected_slot[0] = 1; print('  [Slot] HEAD 2 selected')

            elif key in (ord('c'), ord('C')):
                s = selected_slot[0]
                trackers[s]          = None
                tracker_bbox[s]      = None
                head_raw[s]          = None
                slot_states[s]       = STATE_NONE
                templates[s]         = None
                search_fail_count[s] = 0
                print(f'  [HEAD{s+1}] cleared.')

            elif key in (ord('r'), ord('R')):
                calib_pts = []
                H = None; h_valid = False; h_valid_flag[0] = False
                for s in range(NUM_HEADS):
                    trackers[s]          = None
                    tracker_bbox[s]      = None
                    head_raw[s]          = None
                    slot_states[s]       = STATE_NONE
                    templates[s]         = None
                    search_fail_count[s] = 0
                print('[Reset] All cleared.')

            elif key in (ord('s'), ord('S')):
                snap = os.path.join(OUT_DIR, f'head_snap_{run_ts}_{save_idx:04d}.jpg')
                cv2.imwrite(snap, combined)
                print(f'  Snapshot: {snap}')
                save_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        csv_file.flush(); csv_file.close()
        print(f'\nCSV saved -> {csv_path}')
        print(f'Total frames: {frame_id}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam-id',       type=int,   default=CAM_ID)
    parser.add_argument('--rect-w',       type=int,   default=RECT_W)
    parser.add_argument('--rect-h',       type=int,   default=RECT_H)
    parser.add_argument('--search-scale', type=float, default=SEARCH_SCALE,
                        help='재탐색 영역 확장 배수')
    parser.add_argument('--conf-thr',     type=float, default=SEARCH_CONF_THR,
                        help='템플릿 매칭 유사도 임계값 (0~1)')
    parser.add_argument('--fail-max',     type=int,   default=SEARCH_FAIL_MAX,
                        help='FAILED 전환까지 허용 실패 프레임 수')
    parser.add_argument('--out-dir',      type=str,   default=OUT_DIR)
    args = parser.parse_args()
    CAM_ID          = args.cam_id
    RECT_W          = args.rect_w
    RECT_H          = args.rect_h
    SEARCH_SCALE    = args.search_scale
    SEARCH_CONF_THR = args.conf_thr
    SEARCH_FAIL_MAX = args.fail_max
    OUT_DIR         = args.out_dir
    main()
