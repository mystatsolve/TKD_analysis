"""
Head Tracking with Trajectory Trail - 2 People
──────────────────────────────────────────────────────────────────
[프로그램 개요]
  webcam_head_tracking_retrack.py 에 궤적(Trail) 시각화 기능 추가.
  추적된 좌표를 시간 순서대로 선으로 연결하여 이동 경로를 표시합니다.

  HEAD 1 : 파란색 선 (Blue)
  HEAD 2 : 빨간색 선 (Red)

  - 카메라 뷰 + Bird's-Eye 뷰 양쪽에 궤적 표시
  - 오래된 경로일수록 흐릿하게 페이드 처리
  - CSRT 트래커 + 템플릿 매칭 자동 재탐색 포함

[Controls]
  Click x4    : 기준점 설정 (캘리브레이션)
  ENTER       : 호모그래피 확정
  1 / 2       : 슬롯 선택 (HEAD 1 / HEAD 2)
  Left-click  : 선택 슬롯 트래커 초기화
  Right-click : 선택 슬롯 추적 중지 + 궤적 초기화
  C           : 선택 슬롯 추적 중지 + 궤적 초기화
  T           : 전체 궤적 초기화 (추적은 유지)
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
CAM_ID          = 0
OUT_DIR         = 'ViTPose/output'
RECT_W          = 480
RECT_H          = 480
TRACK_BOX_W     = 80
TRACK_BOX_H     = 80
NUM_HEADS       = 2

# ── 궤적(Trail) 설정 ──────────────────────────────────────────────────────────
MAX_TRAIL       = 600   # 저장할 최대 궤적 점 수 (초과 시 가장 오래된 점 삭제)
                        # 30fps 기준 약 20초 분량
TRAIL_THICKNESS = 2     # 궤적 선 두께 (픽셀)

# HEAD별 궤적 색상 (BGR 형식)
TRAIL_COLORS = [
    (255,  50,  50),   # HEAD 1: 파란색 (Blue)
    ( 50,  50, 255),   # HEAD 2: 빨간색 (Red)
]
# ─────────────────────────────────────────────────────────────────────────────

# ── 재탐색 관련 설정 ──────────────────────────────────────────────────────────
SEARCH_SCALE    = 3.5
SEARCH_CONF_THR = 0.40
SEARCH_FAIL_MAX = 90
TEMPLATE_UPDATE_INTERVAL = 5

# ── 슬롯 상태 ─────────────────────────────────────────────────────────────────
STATE_NONE      = 'none'
STATE_TRACKING  = 'tracking'
STATE_SEARCHING = 'searching'
STATE_FAILED    = 'failed'

# ── CSV 헤더 ──────────────────────────────────────────────────────────────────
CSV_HEADER = [
    'frame', 'timestamp', 'head_id',
    'raw_x', 'raw_y',
    'rect_x', 'rect_y',
    'rect_x_norm', 'rect_y_norm',
    'homography_applied',
    'state',
    'retrack_score',
    'trail_length',   # 이 프레임까지 쌓인 궤적 점 수
]

# ── 표시 색상 (BGR) ───────────────────────────────────────────────────────────
HEAD_COLORS  = [(255, 100, 100), (100, 100, 255)]   # HEAD1=파랑계열, HEAD2=빨강계열
BOX_COLORS   = [(200,  50,  50), ( 50,  50, 200)]
SEARCH_COLOR = (0, 200, 255)
FAILED_COLOR = (0,  50, 255)


# ══════════════════════════════════════════════════════════════════════════════
# ■ 호모그래피 유틸리티
# ══════════════════════════════════════════════════════════════════════════════
def compute_homography(src_pts, rect_w, rect_h):
    dst_pts = np.float32([
        [0, 0], [rect_w, 0], [rect_w, rect_h], [0, rect_h],
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
    """마지막 위치 주변에서 템플릿 매칭으로 재탐색. 성공 시 (cx, cy, score), 실패 시 None."""
    th, tw = template.shape[:2]
    half_sw = int(last_w * scale / 2)
    half_sh = int(last_h * scale / 2)
    roi_x1 = max(0, int(last_x) - half_sw);  roi_y1 = max(0, int(last_y) - half_sh)
    roi_x2 = min(cam_w, int(last_x) + half_sw); roi_y2 = min(cam_h, int(last_y) + half_sh)
    if (roi_y2 - roi_y1) < th or (roi_x2 - roi_x1) < tw:
        return None
    result = cv2.matchTemplate(frame[roi_y1:roi_y2, roi_x1:roi_x2],
                               template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val < conf_thr:
        return None
    return roi_x1 + max_loc[0] + tw/2, roi_y1 + max_loc[1] + th/2, float(max_val)


# ══════════════════════════════════════════════════════════════════════════════
# ■ 궤적(Trail) 그리기
# ══════════════════════════════════════════════════════════════════════════════
def draw_trail(frame, trail_points, base_color, thickness=TRAIL_THICKNESS):
    """
    궤적 좌표들을 선으로 연결하여 그립니다.
    오래된 점일수록 흐리게(페이드) 표시하여 방향감을 줍니다.

    매개변수:
        frame        : 그림을 그릴 이미지
        trail_points : [(x, y), ...] 형태의 좌표 목록 (시간 순서)
        base_color   : 가장 최신 점의 색상 (BGR tuple)
        thickness    : 선 두께
    """
    n = len(trail_points)
    if n < 2:
        return  # 점이 2개 미만이면 선을 그릴 수 없음

    # 각 선분을 오래된 것→새것 순서로 그림
    # 오래된 선분: 어둡게(낮은 알파), 최신 선분: 밝게(높은 알파)
    for i in range(1, n):
        # i/n: 0에 가까울수록 오래됨, 1에 가까울수록 최신
        alpha = i / n

        # 색상 페이드: alpha 비율로 base_color에서 어두운 색으로 보간
        faded_color = tuple(int(ch * alpha) for ch in base_color)

        pt1 = (int(trail_points[i - 1][0]), int(trail_points[i - 1][1]))
        pt2 = (int(trail_points[i    ][0]), int(trail_points[i    ][1]))

        # 최신에 가까울수록 선을 두껍게 표시
        # alpha 0.3 미만은 최소 두께 1, 이상은 thickness 사용
        lw = max(1, int(thickness * alpha))

        cv2.line(frame, pt1, pt2, faded_color, lw, cv2.LINE_AA)

    # 가장 최신 점에 원 표시 (궤적 끝 강조)
    last = trail_points[-1]
    cv2.circle(frame, (int(last[0]), int(last[1])), thickness + 2, base_color, -1)


# ══════════════════════════════════════════════════════════════════════════════
# ■ 기타 시각화 함수
# ══════════════════════════════════════════════════════════════════════════════
def get_slot_color(state, slot):
    if state == STATE_TRACKING:  return HEAD_COLORS[slot]
    if state == STATE_SEARCHING: return SEARCH_COLOR
    return FAILED_COLOR


def draw_head_cam(frame, cx, cy, bbox, state, slot):
    """카메라 뷰에 트래킹 박스 + 중심 마커 표시."""
    color = get_slot_color(state, slot)
    if bbox is not None:
        x, y, w, h = [int(v) for v in bbox]
        if state == STATE_SEARCHING:
            cv2.rectangle(frame, (x-2,y-2),(x+w+2,y+h+2), SEARCH_COLOR, 1)
            cv2.rectangle(frame, (x,  y  ),(x+w,  y+h  ), SEARCH_COLOR, 1)
        elif state == STATE_FAILED:
            cv2.rectangle(frame, (x,y),(x+w,y+h), FAILED_COLOR, 2)
        else:
            cv2.rectangle(frame, (x,y),(x+w,y+h), BOX_COLORS[slot], 2)
    icx, icy = int(cx), int(cy)
    cv2.drawMarker(frame,(icx,icy), color, cv2.MARKER_CROSS, 26, 2, cv2.LINE_AA)
    cv2.circle(frame, (icx,icy), 12, color, 2)
    state_txt = {STATE_TRACKING:'TRACKING', STATE_SEARCHING:'SEARCHING...',
                 STATE_FAILED:'FAILED-click'}.get(state,'')
    cv2.putText(frame, f'HEAD{slot+1} {state_txt} ({icx},{icy})',
                (icx+14, icy-10), cv2.FONT_HERSHEY_DUPLEX, 0.46, color, 1)


def draw_search_area(frame, last_x, last_y, last_w, last_h, scale, cam_w, cam_h):
    half_sw = int(last_w*scale/2); half_sh = int(last_h*scale/2)
    x1=max(0,int(last_x)-half_sw); y1=max(0,int(last_y)-half_sh)
    x2=min(cam_w,int(last_x)+half_sw); y2=min(cam_h,int(last_y)+half_sh)
    overlay = frame.copy()
    cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,200,255),-1)
    cv2.addWeighted(overlay,0.08,frame,0.92,0,frame)
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,200,255),1)
    cv2.putText(frame,'SEARCH AREA',(x1+4,y1+14),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,200,255),1)


def draw_calib_points(frame, pts, done):
    labels = ['1:TL','2:TR','3:BR','4:BL']
    for i,(px,py) in enumerate(pts):
        cv2.circle(frame,(int(px),int(py)), 8,(0,0,255),-1)
        cv2.circle(frame,(int(px),int(py)),10,(255,255,255),2)
        cv2.putText(frame,labels[i],(int(px)+10,int(py)-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),1)
    if done and len(pts)==4:
        cv2.polylines(frame,[np.int32(pts)],True,(0,0,255),1,cv2.LINE_AA)


def draw_bird_eye(rect_w, rect_h, head_pts, trail_rect):
    """
    Bird's-Eye 뷰 생성.
    trail_rect: [(trail_points_slot0, slot0), (trail_points_slot1, slot1)]
    """
    canvas = np.zeros((rect_h, rect_w, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)
    for gx in range(0, rect_w, rect_w//6):
        cv2.line(canvas,(gx,0),(gx,rect_h),(60,60,60),1)
    for gy in range(0, rect_h, rect_h//6):
        cv2.line(canvas,(0,gy),(rect_w,gy),(60,60,60),1)
    cv2.rectangle(canvas,(0,0),(rect_w-1,rect_h-1),(100,100,100),2)

    # ── Bird's-Eye 위의 궤적 그리기 ──
    for trail_pts, slot in trail_rect:
        if len(trail_pts) >= 2:
            # 좌표를 Bird's-Eye 범위 내로 클램핑
            clamped = [(np.clip(x,0,rect_w-1), np.clip(y,0,rect_h-1))
                       for x,y in trail_pts]
            draw_trail(canvas, clamped, TRAIL_COLORS[slot], thickness=2)

    # ── 현재 머리 위치 ──
    for rx, ry, slot, state in head_pts:
        cx = int(np.clip(rx,0,rect_w-1)); cy = int(np.clip(ry,0,rect_h-1))
        color = get_slot_color(state, slot)
        cv2.drawMarker(canvas,(cx,cy),color,cv2.MARKER_CROSS,26,2,cv2.LINE_AA)
        cv2.circle(canvas,(cx,cy),15,color,2)
        short={'tracking':'TRK','searching':'SCH','failed':'FAIL'}.get(state,'')
        cv2.putText(canvas,f'H{slot+1}[{short}] ({rx:.0f},{ry:.0f})',
                    (cx+16,cy+5),cv2.FONT_HERSHEY_SIMPLEX,0.38,color,1)

    cv2.putText(canvas,'TOP-LEFT', (4,14),cv2.FONT_HERSHEY_SIMPLEX,0.35,(150,150,150),1)
    cv2.putText(canvas,'TOP-RIGHT',(rect_w-76,14),cv2.FONT_HERSHEY_SIMPLEX,0.35,(150,150,150),1)
    cv2.putText(canvas,'BOT-LEFT', (4,rect_h-4),cv2.FONT_HERSHEY_SIMPLEX,0.35,(150,150,150),1)
    cv2.putText(canvas,"Bird's-Eye VIEW",(8,rect_h-16),
                cv2.FONT_HERSHEY_SIMPLEX,0.38,(200,200,200),1)
    return canvas


def draw_hud(frame, fps, frame_id, n_pts, h_valid, selected, slot_states,
             search_fail_counts, trail_lengths):
    fh, fw = frame.shape[:2]
    cv2.rectangle(frame,(0,0),(fw,54),(0,0,0),-1)

    if not h_valid:
        if n_pts < 4:
            msg=f'CALIBRATION: Click {4-n_pts} more point(s)  (TL->TR->BR->BL)'
            col=(0,200,255)
        else:
            msg='Press ENTER to confirm homography'; col=(0,255,128)
        cv2.putText(frame,msg,(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.44,col,1)
    else:
        cv2.putText(frame,
            f'FPS:{fps:.1f}  Frame:{frame_id}  | 1/2=slot  LClick=init  T=clear trail  Q=quit',
            (8,16),cv2.FONT_HERSHEY_SIMPLEX,0.40,(200,200,200),1)
        for s in range(NUM_HEADS):
            st=slot_states[s]; hc=get_slot_color(st,s); sel='>' if s==selected else ' '
            if st==STATE_TRACKING:   stxt='TRACKING'
            elif st==STATE_SEARCHING:
                remain=SEARCH_FAIL_MAX-search_fail_counts[s]
                stxt=f'SEARCHING ({remain}f)'
            elif st==STATE_FAILED:   stxt='FAILED - click'
            else:                    stxt='no init'
            tc = TRAIL_COLORS[s]
            cv2.putText(frame,f'{sel} HEAD{s+1}: {stxt}  trail:{trail_lengths[s]}pts',
                        (8+s*280,36),cv2.FONT_HERSHEY_SIMPLEX,0.42,hc,1)
            # 궤적 색상 범례 표시 (작은 선분)
            lx = 8 + s*280 + 7
            cv2.line(frame,(lx,46),(lx+30,46),tc,3,cv2.LINE_AA)

    cv2.putText(frame,'CAMERA VIEW',(fw-100,50),
                cv2.FONT_HERSHEY_SIMPLEX,0.36,(140,140,140),1)


# ══════════════════════════════════════════════════════════════════════════════
# ■ 전역 변수 & 마우스 콜백
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
    global calib_pts

    os.makedirs(OUT_DIR, exist_ok=True)
    run_ts   = time.strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(OUT_DIR, f'head_trail_{run_ts}.csv')

    print('='*65)
    print('Head Tracking + Trail Visualization  (2 People)')
    print('='*65)
    print(f'  Camera     : {CAM_ID}')
    print(f'  BirdEye    : {RECT_W}x{RECT_H}')
    print(f'  Trail max  : {MAX_TRAIL} points  |  HEAD1=Blue  HEAD2=Red')
    print(f'  CSV        : {csv_path}')
    print()
    print('  [Step 1] Click 4 corners (TL->TR->BR->BL), then ENTER')
    print('  [Step 2] Press 1 or 2, then left-click on the head')
    print('  T key : clear all trails')
    print('='*65)

    # ── 카메라 열기 ──
    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open camera {CAM_ID}'); return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ch = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Camera {CAM_ID}: {cw}x{ch}\n')

    WIN = 'Head Tracking + Trail  |  1/2=select  LClick=init  T=clear trail  Q=quit'
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, mouse_cb)

    # ── 호모그래피 ──
    H = None; h_valid = False

    # ── 슬롯별 추적 상태 ──
    trackers          = [None, None]
    tracker_bbox      = [None, None]
    head_raw          = [None, None]
    slot_states       = [STATE_NONE, STATE_NONE]
    templates         = [None, None]
    template_boxes    = [None, None]
    search_fail_count = [0, 0]
    tmpl_update_cnt   = [0, 0]

    # ── 궤적 저장 버퍼 ──
    # trail_raw[s]  : 카메라 뷰 픽셀 좌표 목록 [(x,y), ...]
    # trail_rect[s] : Bird's-Eye 좌표 목록     [(rx,ry), ...]
    trail_raw  = [[], []]
    trail_rect = [[], []]

    # ── CSV ──
    csv_file   = open(csv_path,'w',newline='',encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(CSV_HEADER)

    frame_id   = 0; save_idx = 0
    fps        = 0.0; t_buf = []
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

            bird_head_pts = []   # Bird's-Eye에 표시할 현재 머리 위치

            for s in range(NUM_HEADS):
                retrack_score = float('nan')

                # ── [1] 정지 요청 처리 ──
                if stop_flags[s]:
                    trackers[s]=None; tracker_bbox[s]=None
                    head_raw[s]=None; slot_states[s]=STATE_NONE
                    templates[s]=None; search_fail_count[s]=0
                    # 궤적도 함께 초기화
                    trail_raw[s].clear(); trail_rect[s].clear()
                    stop_flags[s]=False

                # ── [2] 클릭으로 트래커 초기화 ──
                if init_click_pts[s] is not None:
                    ix,iy=init_click_pts[s]
                    bx=max(0,ix-TRACK_BOX_W//2); by=max(0,iy-TRACK_BOX_H//2)
                    bw=min(TRACK_BOX_W,cw-bx);   bh=min(TRACK_BOX_H,ch-by)
                    trackers[s]=cv2.TrackerCSRT_create()
                    trackers[s].init(frame,(bx,by,bw,bh))
                    tracker_bbox[s]=(bx,by,bw,bh)
                    head_raw[s]=(float(ix),float(iy))
                    slot_states[s]=STATE_TRACKING
                    search_fail_count[s]=0; tmpl_update_cnt[s]=0
                    crop=frame[by:by+bh,bx:bx+bw]
                    if crop.size>0:
                        templates[s]=crop.copy(); template_boxes[s]=(bw,bh)
                    # 재초기화 시 궤적 리셋 (새로운 추적 시작)
                    trail_raw[s].clear(); trail_rect[s].clear()
                    init_click_pts[s]=None
                    print(f'  [HEAD{s+1}] Tracker initialized. Trail reset.')

                # ── [3] 트래커 업데이트 ──
                if slot_states[s] == STATE_TRACKING:
                    ok, bbox = trackers[s].update(frame)
                    if ok:
                        tracker_bbox[s]=bbox
                        cx=bbox[0]+bbox[2]/2; cy=bbox[1]+bbox[3]/2
                        head_raw[s]=(cx,cy); search_fail_count[s]=0
                        # 템플릿 갱신
                        tmpl_update_cnt[s]+=1
                        if tmpl_update_cnt[s]>=TEMPLATE_UPDATE_INTERVAL:
                            x,y,w,h=[int(v) for v in bbox]
                            x1=max(0,x);y1=max(0,y);x2=min(cw,x+w);y2=min(ch,y+h)
                            crop=frame[y1:y2,x1:x2]
                            if crop.size>0 and crop.shape[0]>5 and crop.shape[1]>5:
                                templates[s]=crop.copy(); template_boxes[s]=(x2-x1,y2-y1)
                            tmpl_update_cnt[s]=0
                    else:
                        slot_states[s]=STATE_SEARCHING
                        print(f'  [HEAD{s+1}] Lost! SEARCHING...')

                elif slot_states[s] == STATE_SEARCHING:
                    if templates[s] is not None and head_raw[s] is not None:
                        tw_s,th_s=template_boxes[s] if template_boxes[s] else \
                                  (templates[s].shape[1],templates[s].shape[0])
                        result=search_with_template(
                            frame,templates[s],head_raw[s][0],head_raw[s][1],
                            tw_s,th_s,SEARCH_SCALE,SEARCH_CONF_THR,cw,ch)
                        if result is not None:
                            found_cx,found_cy,score=result
                            retrack_score=score
                            bx=max(0,int(found_cx)-tw_s//2)
                            by=max(0,int(found_cy)-th_s//2)
                            bw=min(tw_s,cw-bx); bh=min(th_s,ch-by)
                            trackers[s]=cv2.TrackerCSRT_create()
                            trackers[s].init(frame,(bx,by,bw,bh))
                            tracker_bbox[s]=(bx,by,bw,bh)
                            head_raw[s]=(found_cx,found_cy)
                            slot_states[s]=STATE_TRACKING
                            search_fail_count[s]=0
                            print(f'  [HEAD{s+1}] Re-acquired! score={score:.3f}')
                        else:
                            search_fail_count[s]+=1
                            if search_fail_count[s]>=SEARCH_FAIL_MAX:
                                slot_states[s]=STATE_FAILED
                                print(f'  [HEAD{s+1}] FAILED. Click to reinit.')

                # ── [4] 궤적 업데이트 (정상 추적 중에만 점 추가) ──
                raw_x_csv=raw_y_csv=rect_x=rect_y=rx_n=ry_n=float('nan')
                h_flag=0

                if head_raw[s] is not None and slot_states[s] != STATE_NONE:
                    raw_x_csv,raw_y_csv=head_raw[s]

                    # TRACKING 상태일 때만 궤적에 점 추가
                    if slot_states[s] == STATE_TRACKING:
                        trail_raw[s].append((raw_x_csv, raw_y_csv))
                        # 최대 점 수 초과 시 가장 오래된 점 제거
                        if len(trail_raw[s]) > MAX_TRAIL:
                            trail_raw[s].pop(0)

                    if h_valid and H is not None:
                        rect_x,rect_y=apply_homography(H,raw_x_csv,raw_y_csv)
                        rx_n=rect_x/RECT_W; ry_n=rect_y/RECT_H; h_flag=1
                        bird_head_pts.append((rect_x,rect_y,s,slot_states[s]))

                        # Bird's-Eye 궤적도 TRACKING 중에만 추가
                        if slot_states[s] == STATE_TRACKING:
                            trail_rect[s].append((rect_x, rect_y))
                            if len(trail_rect[s]) > MAX_TRAIL:
                                trail_rect[s].pop(0)

                # ── [5] 시각화: 재탐색 영역 ──
                if slot_states[s]==STATE_SEARCHING and template_boxes[s] and head_raw[s]:
                    tw_s,th_s=template_boxes[s]
                    draw_search_area(vis,head_raw[s][0],head_raw[s][1],
                                     tw_s,th_s,SEARCH_SCALE,cw,ch)

                # ── [6] CSV 기록 ──
                def f(v):
                    return 'nan' if (v!=v) else f'{v:.2f}'
                csv_writer.writerow([
                    frame_id,f'{timestamp:.4f}',s+1,
                    f(raw_x_csv),f(raw_y_csv),f(rect_x),f(rect_y),
                    f'{rx_n:.4f}' if h_flag else 'nan',
                    f'{ry_n:.4f}' if h_flag else 'nan',
                    h_flag, slot_states[s],
                    f'{retrack_score:.4f}' if not (retrack_score!=retrack_score) else 'nan',
                    len(trail_raw[s]),
                ])

            if frame_id % 30 == 0:
                csv_file.flush()

            # ── [7] 현재 머리 위치 마커 그리기 (카메라 뷰에만, 궤적 선 없음) ──
            # 궤적(Trail)은 오른쪽 Bird's-Eye 뷰에서만 표시됩니다.
            for s in range(NUM_HEADS):
                if head_raw[s] is not None and slot_states[s] != STATE_NONE:
                    draw_head_cam(vis, head_raw[s][0], head_raw[s][1],
                                  tracker_bbox[s], slot_states[s], s)

            # ── [9] 캘리브레이션 포인트 & HUD ──
            draw_calib_points(vis, calib_pts, h_valid)
            draw_hud(vis, fps, frame_id, len(calib_pts), h_valid,
                     selected_slot[0], slot_states, search_fail_count,
                     [len(trail_raw[0]), len(trail_raw[1])])

            # ── [10] Bird's-Eye 뷰 생성 ──
            bird = draw_bird_eye(RECT_W, RECT_H, bird_head_pts,
                                 [(trail_rect[s], s) for s in range(NUM_HEADS)])
            bird_resized = cv2.resize(bird, (RECT_W, ch))
            combined     = np.hstack([vis, bird_resized])
            cv2.imshow(WIN, combined)

            # FPS
            t_buf.append(time.time()-t0)
            if len(t_buf)>20: t_buf.pop(0)
            fps=len(t_buf)/sum(t_buf)

            # ── 키 입력 ──
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'),ord('Q'),27):
                print('Quit.'); break

            elif key == 13:   # ENTER: 호모그래피 확정
                if len(calib_pts)==4:
                    H=compute_homography(calib_pts,RECT_W,RECT_H)
                    h_valid=True; h_valid_flag[0]=True
                    print('\n[OK] Homography computed!')
                    print(f'     H =\n{H}\n')
                else:
                    print(f'[WARN] Need 4 points, got {len(calib_pts)}')

            elif key==ord('1'):
                selected_slot[0]=0; print('  [Slot] HEAD 1 selected')
            elif key==ord('2'):
                selected_slot[0]=1; print('  [Slot] HEAD 2 selected')

            elif key in (ord('t'),ord('T')):
                # T: 모든 궤적 초기화 (추적은 계속)
                for s in range(NUM_HEADS):
                    trail_raw[s].clear(); trail_rect[s].clear()
                print('  [Trail] All trails cleared.')

            elif key in (ord('c'),ord('C')):
                s=selected_slot[0]
                trackers[s]=None; tracker_bbox[s]=None; head_raw[s]=None
                slot_states[s]=STATE_NONE; templates[s]=None; search_fail_count[s]=0
                trail_raw[s].clear(); trail_rect[s].clear()
                print(f'  [HEAD{s+1}] cleared.')

            elif key in (ord('r'),ord('R')):
                calib_pts=[]; H=None; h_valid=False; h_valid_flag[0]=False
                for s in range(NUM_HEADS):
                    trackers[s]=None; tracker_bbox[s]=None; head_raw[s]=None
                    slot_states[s]=STATE_NONE; templates[s]=None; search_fail_count[s]=0
                    trail_raw[s].clear(); trail_rect[s].clear()
                print('[Reset] All cleared.')

            elif key in (ord('s'),ord('S')):
                snap=os.path.join(OUT_DIR,f'trail_snap_{run_ts}_{save_idx:04d}.jpg')
                cv2.imwrite(snap,combined)
                print(f'  Snapshot: {snap}')
                save_idx+=1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        csv_file.flush(); csv_file.close()
        print(f'\nCSV saved -> {csv_path}')
        print(f'Total frames: {frame_id}')
        for s in range(NUM_HEADS):
            print(f'  HEAD{s+1} trail points: {len(trail_raw[s])}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam-id',       type=int,   default=CAM_ID)
    parser.add_argument('--rect-w',       type=int,   default=RECT_W)
    parser.add_argument('--rect-h',       type=int,   default=RECT_H)
    parser.add_argument('--max-trail',    type=int,   default=MAX_TRAIL,
                        help=f'최대 궤적 점 수 (기본값: {MAX_TRAIL})')
    parser.add_argument('--search-scale', type=float, default=SEARCH_SCALE)
    parser.add_argument('--conf-thr',     type=float, default=SEARCH_CONF_THR)
    parser.add_argument('--fail-max',     type=int,   default=SEARCH_FAIL_MAX)
    parser.add_argument('--out-dir',      type=str,   default=OUT_DIR)
    args = parser.parse_args()
    CAM_ID          = args.cam_id
    RECT_W          = args.rect_w
    RECT_H          = args.rect_h
    MAX_TRAIL       = args.max_trail
    SEARCH_SCALE    = args.search_scale
    SEARCH_CONF_THR = args.conf_thr
    SEARCH_FAIL_MAX = args.fail_max
    OUT_DIR         = args.out_dir
    main()
