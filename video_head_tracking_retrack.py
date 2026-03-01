"""
Video Head Tracking with Auto Re-tracking - 2 People
──────────────────────────────────────────────────────────────────
[프로그램 개요]
  녹화된 영상 파일에서 최대 2명의 머리 위치를 추적합니다.
  webcam_head_tracking_retrack.py 의 영상 파일 버전입니다.

  - CSRT 트래커 + 템플릿 매칭 자동 재탐색
  - 호모그래피로 Bird's-Eye 직각 좌표 추출
  - 일시정지 / 프레임 이동 / 재생 속도 조절 지원

[실행 예시]
  python video_head_tracking_retrack.py --video my_video.mp4
  python video_head_tracking_retrack.py --video my_video.mp4 --speed 0.5
  python video_head_tracking_retrack.py --video my_video.mp4 --start-frame 300

[Controls]
  SPACE       : 일시정지 / 재개
  LEFT arrow  : 10프레임 뒤로 (일시정지 중)
  RIGHT arrow : 10프레임 앞으로 (일시정지 중)
  F           : 1프레임 앞으로 (일시정지 중)
  [ / ]       : 재생 속도 감소 / 증가 (×0.5 / ×2.0)
  Click x4    : 기준점 설정 (캘리브레이션)
  ENTER       : 호모그래피 확정
  1 / 2       : 슬롯 선택 (HEAD 1 / HEAD 2)
  Left-click  : 선택 슬롯 트래커 초기화
  Right-click : 선택 슬롯 추적 중지
  C           : 선택 슬롯 추적 중지
  R           : 전체 초기화 (재캘리브레이션)
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
VIDEO_PATH      = ''                # 입력 영상 파일 경로 (--video 인자로 지정)
OUT_DIR         = 'output'          # 결과 저장 폴더
RECT_W          = 480               # Bird's-Eye 뷰 가로 크기 (픽셀)
RECT_H          = 480               # Bird's-Eye 뷰 세로 크기 (픽셀)
TRACK_BOX_W     = 80                # 트래커 초기 박스 너비 (픽셀)
TRACK_BOX_H     = 80                # 트래커 초기 박스 높이 (픽셀)
NUM_HEADS       = 2                 # 동시 추적 최대 인원 수
SPEED           = 1.0               # 재생 속도 배율 (1.0 = 원본 속도)
START_FRAME     = 0                 # 시작 프레임 번호
LOOP            = False             # True 이면 영상 끝에서 처음으로 반복

# ── 재탐색 관련 설정 ──
SEARCH_SCALE    = 3.5
SEARCH_CONF_THR = 0.40
SEARCH_FAIL_MAX = 90
TEMPLATE_UPDATE_INTERVAL = 5

# ── 슬롯 상태 정의 ──
STATE_NONE      = 'none'
STATE_TRACKING  = 'tracking'
STATE_SEARCHING = 'searching'
STATE_FAILED    = 'failed'

# ── CSV 헤더 ──
CSV_HEADER = [
    'frame',            # 원본 영상의 프레임 번호
    'video_time_sec',   # 원본 영상 기준 시각 (초)
    'head_id',
    'raw_x', 'raw_y',
    'rect_x', 'rect_y',
    'rect_x_norm', 'rect_y_norm',
    'homography_applied',
    'state',
    'retrack_score',
]

# ── 색상 (BGR) ──
HEAD_COLORS   = [(255, 0, 255), (0, 255, 255)]
BOX_COLORS    = [(200, 0, 200), (0, 200, 200)]
SEARCH_COLOR  = (0, 200, 255)
FAILED_COLOR  = (0,  50, 255)


# ══════════════════════════════════════════════════════════════════════════════
# ■ 호모그래피 유틸리티
# ══════════════════════════════════════════════════════════════════════════════
def compute_homography(src_pts, rect_w, rect_h):
    dst_pts = np.float32([
        [0,      0],
        [rect_w, 0],
        [rect_w, rect_h],
        [0,      rect_h],
    ])
    H, _ = cv2.findHomography(np.float32(src_pts), dst_pts, method=0)
    return H


def apply_homography(H, px, py):
    out = cv2.perspectiveTransform(np.float32([[[px, py]]]), H)
    return float(out[0][0][0]), float(out[0][0][1])


# ══════════════════════════════════════════════════════════════════════════════
# ■ 재탐색 (Template Matching)
# ══════════════════════════════════════════════════════════════════════════════
def search_with_template(frame, template, last_x, last_y, last_w, last_h,
                         scale, conf_thr, cam_w, cam_h):
    """
    마지막 위치 주변에서 템플릿 매칭으로 머리를 재탐색합니다.
    성공 시 (found_cx, found_cy, score) 반환, 실패 시 None 반환.
    """
    th, tw = template.shape[:2]
    half_sw = int(last_w * scale / 2)
    half_sh = int(last_h * scale / 2)

    # 탐색 ROI (화면 경계 클램핑)
    roi_x1 = max(0, int(last_x) - half_sw)
    roi_y1 = max(0, int(last_y) - half_sh)
    roi_x2 = min(cam_w, int(last_x) + half_sw)
    roi_y2 = min(cam_h, int(last_y) + half_sh)

    if (roi_y2 - roi_y1) < th or (roi_x2 - roi_x1) < tw:
        return None

    search_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    result     = cv2.matchTemplate(search_roi, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < conf_thr:
        return None

    found_cx = roi_x1 + max_loc[0] + tw / 2
    found_cy = roi_y1 + max_loc[1] + th / 2
    return found_cx, found_cy, float(max_val)


# ══════════════════════════════════════════════════════════════════════════════
# ■ 시각화 함수
# ══════════════════════════════════════════════════════════════════════════════
def get_slot_color(state, slot):
    if state == STATE_TRACKING:  return HEAD_COLORS[slot]
    if state == STATE_SEARCHING: return SEARCH_COLOR
    return FAILED_COLOR


def draw_head_cam(frame, cx, cy, bbox, state, slot):
    color = get_slot_color(state, slot)
    if bbox is not None:
        x, y, w, h = [int(v) for v in bbox]
        if state == STATE_SEARCHING:
            cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), SEARCH_COLOR, 1)
            cv2.rectangle(frame, (x,   y  ), (x+w,   y+h  ), SEARCH_COLOR, 1)
        elif state == STATE_FAILED:
            cv2.rectangle(frame, (x, y), (x+w, y+h), FAILED_COLOR, 2)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), BOX_COLORS[slot], 2)

    icx, icy = int(cx), int(cy)
    cv2.drawMarker(frame, (icx, icy), color, cv2.MARKER_CROSS, 26, 2, cv2.LINE_AA)
    cv2.circle(frame, (icx, icy), 12, color, 2)
    state_txt = {STATE_TRACKING:'TRACKING',
                 STATE_SEARCHING:'SEARCHING...',
                 STATE_FAILED:'FAILED - click'}.get(state, '')
    cv2.putText(frame, f'HEAD{slot+1} {state_txt} ({icx},{icy})',
                (icx+14, icy-10), cv2.FONT_HERSHEY_DUPLEX, 0.46, color, 1)


def draw_search_area(frame, last_x, last_y, last_w, last_h, scale, cam_w, cam_h):
    half_sw = int(last_w * scale / 2)
    half_sh = int(last_h * scale / 2)
    x1 = max(0, int(last_x) - half_sw);  y1 = max(0, int(last_y) - half_sh)
    x2 = min(cam_w, int(last_x) + half_sw); y2 = min(cam_h, int(last_y) + half_sh)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), -1)
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 1)
    cv2.putText(frame, 'SEARCH AREA', (x1+4, y1+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)


def draw_calib_points(frame, pts, done):
    labels = ['1:TL', '2:TR', '3:BR', '4:BL']
    for i, (px, py) in enumerate(pts):
        cv2.circle(frame, (int(px), int(py)),  8, (0, 0, 255), -1)
        cv2.circle(frame, (int(px), int(py)), 10, (255, 255, 255), 2)
        cv2.putText(frame, labels[i], (int(px)+10, int(py)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    if done and len(pts) == 4:
        cv2.polylines(frame, [np.int32(pts)], True, (0, 0, 255), 1, cv2.LINE_AA)


def draw_bird_eye(rect_w, rect_h, head_pts):
    canvas = np.zeros((rect_h, rect_w, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)
    for gx in range(0, rect_w, rect_w // 6):
        cv2.line(canvas, (gx, 0), (gx, rect_h), (60, 60, 60), 1)
    for gy in range(0, rect_h, rect_h // 6):
        cv2.line(canvas, (0, gy), (rect_w, gy), (60, 60, 60), 1)
    cv2.rectangle(canvas, (0, 0), (rect_w-1, rect_h-1), (100, 100, 100), 2)
    for rx, ry, slot, state in head_pts:
        cx = int(np.clip(rx, 0, rect_w-1))
        cy = int(np.clip(ry, 0, rect_h-1))
        color = get_slot_color(state, slot)
        cv2.drawMarker(canvas, (cx, cy), color, cv2.MARKER_CROSS, 26, 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 15, color, 2)
        short = {'tracking':'TRK','searching':'SCH','failed':'FAIL'}.get(state,'')
        cv2.putText(canvas, f'H{slot+1}[{short}] ({rx:.0f},{ry:.0f})',
                    (cx+16, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
    cv2.putText(canvas, 'TOP-LEFT',  (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    cv2.putText(canvas, 'TOP-RIGHT', (rect_w-76, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    cv2.putText(canvas, 'BOT-LEFT',  (4, rect_h-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    cv2.putText(canvas, "Bird's-Eye VIEW", (8, rect_h-16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    return canvas


def draw_hud(frame, vid_fps, proc_fps, cur_frame, total_frames,
             vid_time, n_pts, h_valid, selected, slot_states,
             search_fail_counts, paused, speed):
    """상단 HUD + 하단 진행 바 표시."""
    fh, fw = frame.shape[:2]

    # ── 상단 HUD 배경 ──
    cv2.rectangle(frame, (0, 0), (fw, 54), (0, 0, 0), -1)

    if not h_valid:
        if n_pts < 4:
            msg = f'CALIBRATION: Click {4-n_pts} more point(s)  (TL->TR->BR->BL)  |  SPACE=pause'
            col = (0, 200, 255)
        else:
            msg = 'Press ENTER to confirm homography'
            col = (0, 255, 128)
        cv2.putText(frame, msg, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
    else:
        pause_txt = '[PAUSED]' if paused else f'x{speed:.1f}'
        cv2.putText(frame,
                    f'procFPS:{proc_fps:.1f}  vidFPS:{vid_fps:.1f}  {pause_txt}  '
                    f'Frame:{cur_frame}/{total_frames}  Time:{vid_time:.2f}s',
                    (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
        for s in range(NUM_HEADS):
            st  = slot_states[s]
            hc  = get_slot_color(st, s)
            sel = '>' if s == selected else ' '
            if st == STATE_TRACKING:
                stxt = 'TRACKING'
            elif st == STATE_SEARCHING:
                remain = SEARCH_FAIL_MAX - search_fail_counts[s]
                stxt = f'SEARCHING ({remain}f)'
            elif st == STATE_FAILED:
                stxt = 'FAILED - click'
            else:
                stxt = 'no init'
            cv2.putText(frame, f'{sel} HEAD{s+1}: {stxt}',
                        (8 + s*240, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, hc, 1)

    cv2.putText(frame, 'VIDEO VIEW', (fw-90, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1)

    # ── 하단 진행 바 ──
    bar_h  = 14          # 진행 바 높이
    bar_y  = fh - bar_h  # 진행 바 시작 y 좌표
    cv2.rectangle(frame, (0, bar_y), (fw, fh), (20, 20, 20), -1)  # 배경

    if total_frames > 0:
        # 현재 재생 위치를 비율로 계산하여 진행 바 길이 결정
        ratio  = min(cur_frame / total_frames, 1.0)
        bar_w  = int(fw * ratio)
        cv2.rectangle(frame, (0, bar_y), (bar_w, fh), (0, 180, 80), -1)

    # 프레임 정보 텍스트
    time_str = f'{int(vid_time//60):02d}:{vid_time%60:05.2f}'
    cv2.putText(frame, f'{cur_frame}/{total_frames}  {time_str}',
                (4, fh-2), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 200, 200), 1)


# ══════════════════════════════════════════════════════════════════════════════
# ■ 전역 상태 변수 & 마우스 콜백
# ══════════════════════════════════════════════════════════════════════════════
calib_pts      = []
h_valid_flag   = [False]
selected_slot  = [0]
init_click_pts = [None, None]
stop_flags     = [False, False]


def mouse_cb(event, x, y, flags, param):
    global calib_pts
    s = selected_slot[0]
    if event == cv2.EVENT_LBUTTONDOWN:
        if not h_valid_flag[0]:
            if len(calib_pts) < 4:
                calib_pts.append([float(x), float(y)])
                print(f'  Calib point {len(calib_pts)}: ({x}, {y})')
        else:
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
    global calib_pts, SPEED

    if not VIDEO_PATH:
        print('[ERROR] --video 인자로 영상 파일 경로를 지정하세요.')
        print('  예) python video_head_tracking_retrack.py --video my_video.mp4')
        return

    if not os.path.isfile(VIDEO_PATH):
        print(f'[ERROR] 파일을 찾을 수 없습니다: {VIDEO_PATH}')
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    run_ts   = time.strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(OUT_DIR, f'video_headtrack_{run_ts}.csv')

    # ── 영상 열기 ──
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f'[ERROR] 영상을 열 수 없습니다: {VIDEO_PATH}')
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 전체 프레임 수
    vid_fps      = cap.get(cv2.CAP_PROP_FPS)               # 원본 영상 FPS
    cw           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 영상 너비
    ch           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 영상 높이

    # 시작 프레임 지정
    if START_FRAME > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    # 기본 대기 시간: 원본 FPS 기준 (ms 단위)
    # waitKey에 이 값을 전달하여 영상 속도에 맞게 재생
    base_wait_ms = max(1, int(1000 / vid_fps)) if vid_fps > 0 else 33

    print('=' * 65)
    print('Video Head Tracking + Auto Re-tracking  (2 People)')
    print('=' * 65)
    print(f'  Video    : {os.path.basename(VIDEO_PATH)}')
    print(f'  Size     : {cw}x{ch}  |  FPS: {vid_fps:.1f}  |  Frames: {total_frames}')
    print(f'  BirdEye  : {RECT_W}x{RECT_H}')
    print(f'  Speed    : x{SPEED}')
    print(f'  CSV      : {csv_path}')
    print()
    print('  [Step 1] SPACE로 일시정지 후 기준점 4개 클릭 (TL->TR->BR->BL)')
    print('           ENTER로 확정')
    print('  [Step 2] 1/2 키로 슬롯 선택, 머리 클릭 -> 자동 추적')
    print('  [재생 조작] SPACE=정지/재개  LEFT/RIGHT=10f이동  F=1f앞  [/]=속도')
    print('=' * 65)

    WIN = 'Video Head Tracking  |  SPACE=pause  1/2=slot  LClick=init  Q=quit'
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, mouse_cb)

    # ── 호모그래피 상태 ──
    H       = None
    h_valid = False

    # ── 슬롯별 상태 ──
    trackers          = [None, None]
    tracker_bbox      = [None, None]
    head_raw          = [None, None]
    slot_states       = [STATE_NONE, STATE_NONE]
    templates         = [None, None]
    template_boxes    = [None, None]
    search_fail_count = [0, 0]
    tmpl_update_cnt   = [0, 0]

    # ── 재생 상태 ──
    paused      = True   # 처음엔 일시정지 상태로 시작 (캘리브레이션 편의)

    # ── CSV ──
    csv_file   = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(CSV_HEADER)

    frame_count = START_FRAME  # 처리한 원본 프레임 번호
    save_idx    = 0
    proc_fps    = 0.0
    t_buf       = []
    current_frame_img = None   # 일시정지 시 마지막 프레임 유지용

    print('\n  [시작] 일시정지 상태입니다. SPACE를 눌러 재생하거나,')
    print('         먼저 기준점을 클릭하여 캘리브레이션을 시작하세요.\n')

    try:
        while True:
            t0 = time.time()

            # ── 프레임 읽기 ──
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    # 영상 끝
                    if LOOP:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_count = 0
                        print('[LOOP] 영상을 처음부터 다시 재생합니다.')
                        continue
                    else:
                        print('[END] 영상이 끝났습니다.')
                        break
                frame_count += 1
                current_frame_img = frame.copy()
            else:
                # 일시정지: 마지막 프레임 재사용
                if current_frame_img is None:
                    # 아직 한 프레임도 안 읽은 경우 한 번 읽기
                    ret, frame = cap.read()
                    if not ret:
                        print('[ERROR] 영상을 읽을 수 없습니다.')
                        break
                    frame_count += 1
                    current_frame_img = frame.copy()
                frame = current_frame_img.copy()

            vis = frame.copy()

            # 현재 영상 시각 (초)
            vid_time = frame_count / vid_fps if vid_fps > 0 else 0.0

            bird_head_pts = []

            for s in range(NUM_HEADS):
                retrack_score = float('nan')

                # [1] 정지 요청
                if stop_flags[s]:
                    trackers[s] = None; tracker_bbox[s] = None
                    head_raw[s] = None; slot_states[s]  = STATE_NONE
                    templates[s] = None; search_fail_count[s] = 0
                    stop_flags[s] = False

                # [2] 클릭으로 트래커 초기화
                if init_click_pts[s] is not None:
                    ix, iy = init_click_pts[s]
                    bx = max(0, ix - TRACK_BOX_W//2)
                    by = max(0, iy - TRACK_BOX_H//2)
                    bw = min(TRACK_BOX_W, cw-bx)
                    bh = min(TRACK_BOX_H, ch-by)
                    trackers[s] = cv2.TrackerCSRT_create()
                    trackers[s].init(frame, (bx, by, bw, bh))
                    tracker_bbox[s]      = (bx, by, bw, bh)
                    head_raw[s]          = (float(ix), float(iy))
                    slot_states[s]       = STATE_TRACKING
                    search_fail_count[s] = 0
                    tmpl_update_cnt[s]   = 0
                    crop = frame[by:by+bh, bx:bx+bw]
                    if crop.size > 0:
                        templates[s] = crop.copy(); template_boxes[s] = (bw, bh)
                    init_click_pts[s] = None
                    print(f'  [HEAD{s+1}] Tracker initialized.')

                # [3] 재생 중일 때만 트래커 업데이트
                if not paused:
                    if slot_states[s] == STATE_TRACKING:
                        ok, bbox = trackers[s].update(frame)
                        if ok:
                            tracker_bbox[s] = bbox
                            cx = bbox[0] + bbox[2]/2
                            cy = bbox[1] + bbox[3]/2
                            head_raw[s] = (cx, cy)
                            search_fail_count[s] = 0
                            # 템플릿 주기 갱신
                            tmpl_update_cnt[s] += 1
                            if tmpl_update_cnt[s] >= TEMPLATE_UPDATE_INTERVAL:
                                x,y,w,h = [int(v) for v in bbox]
                                x1=max(0,x); y1=max(0,y)
                                x2=min(cw,x+w); y2=min(ch,y+h)
                                crop = frame[y1:y2, x1:x2]
                                if crop.size>0 and crop.shape[0]>5 and crop.shape[1]>5:
                                    templates[s] = crop.copy()
                                    template_boxes[s] = (x2-x1, y2-y1)
                                tmpl_update_cnt[s] = 0
                        else:
                            slot_states[s] = STATE_SEARCHING
                            print(f'  [HEAD{s+1}] Lost! Switching to SEARCHING...')

                    elif slot_states[s] == STATE_SEARCHING:
                        if templates[s] is not None and head_raw[s] is not None:
                            tw_s, th_s = template_boxes[s] if template_boxes[s] else \
                                         (templates[s].shape[1], templates[s].shape[0])
                            result = search_with_template(
                                frame, templates[s],
                                head_raw[s][0], head_raw[s][1],
                                tw_s, th_s,
                                SEARCH_SCALE, SEARCH_CONF_THR, cw, ch)
                            if result is not None:
                                found_cx, found_cy, score = result
                                retrack_score = score
                                bx = max(0, int(found_cx)-tw_s//2)
                                by = max(0, int(found_cy)-th_s//2)
                                bw = min(tw_s, cw-bx); bh = min(th_s, ch-by)
                                trackers[s] = cv2.TrackerCSRT_create()
                                trackers[s].init(frame, (bx, by, bw, bh))
                                tracker_bbox[s]      = (bx, by, bw, bh)
                                head_raw[s]          = (found_cx, found_cy)
                                slot_states[s]       = STATE_TRACKING
                                search_fail_count[s] = 0
                                print(f'  [HEAD{s+1}] Re-acquired! score={score:.3f}')
                            else:
                                search_fail_count[s] += 1
                                if search_fail_count[s] >= SEARCH_FAIL_MAX:
                                    slot_states[s] = STATE_FAILED
                                    print(f'  [HEAD{s+1}] FAILED. Click to reinit.')

                # [4] 시각화
                if head_raw[s] is not None and slot_states[s] != STATE_NONE:
                    raw_x, raw_y = head_raw[s]
                    if slot_states[s] == STATE_SEARCHING and template_boxes[s]:
                        tw_s, th_s = template_boxes[s]
                        draw_search_area(vis, raw_x, raw_y, tw_s, th_s,
                                         SEARCH_SCALE, cw, ch)
                    draw_head_cam(vis, raw_x, raw_y, tracker_bbox[s], slot_states[s], s)

                # [5] 호모그래피 변환 & Bird's-Eye
                raw_x_csv = raw_y_csv = rect_x = rect_y = rx_n = ry_n = float('nan')
                h_flag = 0
                if head_raw[s] is not None and slot_states[s] != STATE_NONE:
                    raw_x_csv, raw_y_csv = head_raw[s]
                    if h_valid and H is not None:
                        rect_x, rect_y = apply_homography(H, raw_x_csv, raw_y_csv)
                        rx_n = rect_x/RECT_W; ry_n = rect_y/RECT_H
                        h_flag = 1
                        bird_head_pts.append((rect_x, rect_y, s, slot_states[s]))

                # [6] CSV 기록 (재생 중일 때만)
                if not paused:
                    def f(v):
                        return 'nan' if (v != v) else f'{v:.2f}'
                    csv_writer.writerow([
                        frame_count, f'{vid_time:.4f}', s+1,
                        f(raw_x_csv), f(raw_y_csv),
                        f(rect_x), f(rect_y),
                        f'{rx_n:.4f}' if h_flag else 'nan',
                        f'{ry_n:.4f}' if h_flag else 'nan',
                        h_flag, slot_states[s],
                        f'{retrack_score:.4f}' if not (retrack_score != retrack_score) else 'nan',
                    ])

            if not paused and frame_count % 30 == 0:
                csv_file.flush()

            # ── 캘리브레이션 & HUD ──
            draw_calib_points(vis, calib_pts, h_valid)
            draw_hud(vis, vid_fps, proc_fps, frame_count, total_frames,
                     vid_time, len(calib_pts), h_valid, selected_slot[0],
                     slot_states, search_fail_count, paused, SPEED)

            # ── Bird's-Eye 합성 ──
            bird         = draw_bird_eye(RECT_W, RECT_H, bird_head_pts)
            bird_resized = cv2.resize(bird, (RECT_W, ch))
            combined     = np.hstack([vis, bird_resized])
            cv2.imshow(WIN, combined)

            # FPS 계산 (처리 속도)
            t_buf.append(time.time()-t0)
            if len(t_buf)>20: t_buf.pop(0)
            proc_fps = len(t_buf)/sum(t_buf)

            # ── 재생 속도에 맞는 대기 시간 계산 ──
            # 일시정지 중: 30ms (UI 응답 유지) / 재생 중: FPS 기반 대기
            if paused:
                wait_ms = 30
            else:
                wait_ms = max(1, int(base_wait_ms / SPEED))

            key = cv2.waitKey(wait_ms) & 0xFF

            # ── 키 입력 처리 ──
            if key in (ord('q'), ord('Q'), 27):
                print('Quit.'); break

            elif key == ord(' '):  # SPACE: 일시정지 / 재개 토글
                paused = not paused
                print('  [PAUSED]' if paused else '  [PLAYING]')

            elif key == 13:        # ENTER: 호모그래피 확정
                if len(calib_pts) == 4:
                    H = compute_homography(calib_pts, RECT_W, RECT_H)
                    h_valid = True; h_valid_flag[0] = True
                    print('\n[OK] Homography computed!')
                    print(f'     H =\n{H}\n')
                else:
                    print(f'[WARN] Need 4 points, got {len(calib_pts)}')

            elif key == 81 or key == 2424832:  # LEFT arrow: 10프레임 뒤로
                if paused:
                    new_pos = max(0, frame_count - 11)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                    ret, frame = cap.read()
                    if ret:
                        frame_count = new_pos + 1
                        current_frame_img = frame.copy()
                    print(f'  << Frame {frame_count}')

            elif key == 83 or key == 2555904:  # RIGHT arrow: 10프레임 앞으로
                if paused:
                    new_pos = min(total_frames-1, frame_count + 9)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                    ret, frame = cap.read()
                    if ret:
                        frame_count = new_pos + 1
                        current_frame_img = frame.copy()
                    print(f'  >> Frame {frame_count}')

            elif key in (ord('f'), ord('F')):  # F: 1프레임 앞으로 (일시정지 중)
                if paused:
                    ret, frame = cap.read()
                    if ret:
                        frame_count += 1
                        current_frame_img = frame.copy()
                    print(f'  > Frame {frame_count}')

            elif key == ord('['):  # 속도 감소 (절반)
                SPEED = max(0.125, SPEED * 0.5)
                print(f'  Speed: x{SPEED:.3f}')

            elif key == ord(']'):  # 속도 증가 (두 배)
                SPEED = min(8.0, SPEED * 2.0)
                print(f'  Speed: x{SPEED:.3f}')

            elif key == ord('1'):
                selected_slot[0] = 0; print('  [Slot] HEAD 1 selected')
            elif key == ord('2'):
                selected_slot[0] = 1; print('  [Slot] HEAD 2 selected')

            elif key in (ord('c'), ord('C')):
                s = selected_slot[0]
                trackers[s]=None; tracker_bbox[s]=None; head_raw[s]=None
                slot_states[s]=STATE_NONE; templates[s]=None; search_fail_count[s]=0
                print(f'  [HEAD{s+1}] cleared.')

            elif key in (ord('r'), ord('R')):
                calib_pts=[]; H=None; h_valid=False; h_valid_flag[0]=False
                for s in range(NUM_HEADS):
                    trackers[s]=None; tracker_bbox[s]=None; head_raw[s]=None
                    slot_states[s]=STATE_NONE; templates[s]=None; search_fail_count[s]=0
                print('[Reset] All cleared.')

            elif key in (ord('s'), ord('S')):
                snap = os.path.join(OUT_DIR, f'snap_{run_ts}_{save_idx:04d}.jpg')
                cv2.imwrite(snap, combined)
                print(f'  Snapshot: {snap}')
                save_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        csv_file.flush(); csv_file.close()
        print(f'\nCSV saved -> {csv_path}')
        print(f'Processed frames: {frame_count}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Video head tracking with auto re-tracking and homography'
    )
    parser.add_argument('--video',        type=str,   required=True,
                        help='입력 영상 파일 경로 (mp4, avi, mov 등)')
    parser.add_argument('--out-dir',      type=str,   default=OUT_DIR,
                        help=f'결과 저장 폴더 (기본값: {OUT_DIR})')
    parser.add_argument('--rect-w',       type=int,   default=RECT_W)
    parser.add_argument('--rect-h',       type=int,   default=RECT_H)
    parser.add_argument('--speed',        type=float, default=SPEED,
                        help='재생 속도 배율 (기본값: 1.0)')
    parser.add_argument('--start-frame',  type=int,   default=START_FRAME,
                        help='시작 프레임 번호 (기본값: 0)')
    parser.add_argument('--loop',         action='store_true',
                        help='영상 끝에서 처음으로 반복')
    parser.add_argument('--search-scale', type=float, default=SEARCH_SCALE)
    parser.add_argument('--conf-thr',     type=float, default=SEARCH_CONF_THR)
    parser.add_argument('--fail-max',     type=int,   default=SEARCH_FAIL_MAX)
    args = parser.parse_args()

    VIDEO_PATH      = args.video
    OUT_DIR         = args.out_dir
    RECT_W          = args.rect_w
    RECT_H          = args.rect_h
    SPEED           = args.speed
    START_FRAME     = args.start_frame
    LOOP            = args.loop
    SEARCH_SCALE    = args.search_scale
    SEARCH_CONF_THR = args.conf_thr
    SEARCH_FAIL_MAX = args.fail_max

    main()
