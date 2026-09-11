from maix import camera, display, image, nn, app, gpio, pinmap, http, uart
import time
import math
import threading

# ============================================================
# CONFIGURATION & TUNING PARAMETERS
# ============================================================

# Target Physical Properties & Optics
REAL_DRONE_WIDTH_M = 0.35      # Physical target width (meters)
H_FOV_DEG = 70.0               # Camera Horizontal Field of View (degrees)

# Dual-Thresholding & Acquisition Filters (Kills False Detections)
CONF_ACQUIRE = 0.75            # High confidence required to start initial lock
CONF_TRACK = 0.28              # LOWERED: Catches motion-blurred drones during fast camera pans
IOU_THRESHOLD = 0.40           # NMS IoU threshold
MIN_DRONE_PX = 8               # Ignore noise under 8x8 pixels
MAX_BORDER_MARGIN = 4          # Ignore bounding boxes glued to sensor edges
ACQUIRE_CONSECUTIVE_FRAMES = 5 # Candidate must persist across 3 consecutive frames to lock

# Pixel Movement & Search Tolerances (Tuned for aggressive camera movement)
MAX_ACQUIRE_DRIFT_PX = 150.0   # Max allowed movement during candidate confirmation
TRACK_SEARCH_RADIUS_BASE = 350 # INCREASED: Base radius to survive sudden camera jerks
TRACK_SEARCH_RADIUS_MAX = 700  # INCREASED: Max radius when lost (essentially whole screen)

# Tracking Persistence & Kalman Coasting
MAX_COAST_FRAMES = 15          # INCREASED: Give tracker slightly longer to wait for blur to settle
MAX_LOST_FRAMES = 25           # Hard reset lock after this many frames without detection

# ============================================================
# SERIAL OUTPUT (UART1 on A19 = TX)
# ============================================================
SERIAL_PORT = "/dev/ttyS1"          
SERIAL_BAUDRATE = 115200

from maix import err

err.check_raise(pinmap.set_pin_function("A19", "UART1_TX"), "Failed to set A19 as UART1_TX")
err.check_raise(pinmap.set_pin_function("A18", "UART1_RX"), "Failed to set A18 as UART1_RX")  

try:
    serial_out = uart.UART(SERIAL_PORT, SERIAL_BAUDRATE)
    print("Serial (UART1 / A19) started successfully")
except Exception as e:
    serial_out = None
    print("Serial init failed:", e)

# ============================================================
# 1D CONSTANT-VELOCITY KALMAN FILTER
# ============================================================

class KalmanFilter1D:
    def __init__(self, process_noise=5.0, measurement_noise=2.0):
        # TUNED for camera pans: higher process_noise trusts sudden changes (measurements) more, less lag.
        self.q = process_noise
        self.r = measurement_noise
        self.x = 0.0
        self.v = 0.0
        self.p00 = 1.0
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = 1.0
        self.initialized = False
        self.last_time = None

    def reset(self):
        self.initialized = False
        self.last_time = None

    def predict(self, current_time=None):
        if not self.initialized:
            return self.x
        if current_time is None:
            current_time = time.time()
        
        dt = max(current_time - self.last_time, 0.001)
        self.last_time = current_time

        self.x = self.x + self.v * dt
        self.p00 = self.p00 + dt * (self.p10 + self.p01) + dt * dt * self.p11 + self.q * dt
        self.p01 = self.p01 + dt * self.p11
        self.p10 = self.p10 + dt * self.p11
        self.p11 = self.p11 + self.q * dt
        return self.x

    def update(self, measurement, current_time=None):
        if current_time is None:
            current_time = time.time()

        if not self.initialized:
            self.x = float(measurement)
            self.v = 0.0
            self.p00, self.p01, self.p10, self.p11 = 1.0, 0.0, 0.0, 1.0
            self.last_time = current_time
            self.initialized = True
            return self.x

        dt = max(current_time - self.last_time, 0.001)
        self.last_time = current_time

        x_pred = self.x + self.v * dt
        v_pred = self.v
        p00_pred = self.p00 + dt * (self.p10 + self.p01) + dt * dt * self.p11 + self.q * dt
        p01_pred = self.p01 + dt * self.p11
        p10_pred = self.p10 + dt * self.p11
        p11_pred = self.p11 + self.q * dt

        y = measurement - x_pred
        s = p00_pred + self.r
        k0 = p00_pred / s
        k1 = p10_pred / s

        self.x = x_pred + k0 * y
        self.v = v_pred + k1 * y

        self.p00 = p00_pred - k0 * p00_pred
        self.p01 = p01_pred - k0 * p01_pred
        self.p10 = p10_pred - k1 * p00_pred
        self.p11 = p11_pred - k1 * p01_pred

        return self.x

# Instantiate filter pipeline (Tuned for faster response to camera tracking)
kf_err_x = KalmanFilter1D(process_noise=6.0, measurement_noise=1.5)
kf_err_y = KalmanFilter1D(process_noise=6.0, measurement_noise=1.5)
kf_ang_x = KalmanFilter1D(process_noise=4.0, measurement_noise=2.0)
kf_ang_y = KalmanFilter1D(process_noise=4.0, measurement_noise=2.0)
kf_dist  = KalmanFilter1D(process_noise=0.5, measurement_noise=1.0)

# ============================================================
# HARDWARE & INTERFACE SETUP
# ============================================================

try:
    serial_out = uart.UART(SERIAL_PORT, SERIAL_BAUDRATE)
except Exception:
    serial_out = None

pinmap.set_pin_function("A14", "GPIOA14")
gpio_out = gpio.GPIO("GPIOA14", gpio.Mode.OUT)
gpio_out.value(0)

drone_detector = nn.YOLO26(
    model="/root/models/model_8770.mud",
    dual_buff=True
)

cam = camera.Camera(
    drone_detector.input_width(),
    drone_detector.input_height(),
    drone_detector.input_format()
)

disp = display.Display()
stream = http.JpegStreamer()
stream.start()

# ============================================================
# THREAD-SAFE STATE
# ============================================================

state_lock = threading.Lock()
locked = False
detected_locked = False
is_coasting = False

err_x = 0.0
err_y = 0.0
ang_err_x = 0.0
ang_err_y = 0.0
estimated_dist_m = 0.0

locked_box = None
locked_area = 0.0
locked_aspect = 0.0
lost_frames = 0

latest_jpeg = None
jpeg_lock = threading.Lock()
jpeg_ready = threading.Event()

def stream_thread():
    global latest_jpeg
    while not app.need_exit():
        jpeg_ready.wait(0.2)
        if app.need_exit():
            break
        with jpeg_lock:
            data = latest_jpeg
            jpeg_ready.clear()
        if data is not None:
            try:
                stream.write(data)
            except Exception:
                pass

threading.Thread(target=stream_thread, daemon=True).start()

# ============================================================
# SPATIAL SCORING & CANDIDATE EVALUATION
# ============================================================

def score_candidate(cand_box, target_box, target_area, target_aspect, max_dist, sole_candidate=False):
    x, y, w, h = cand_box
    area = float(w * h)
    aspect = w / max(h, 1)

    cx = x + w / 2.0
    cy = y + h / 2.0
    lx = target_box[0] + target_box[2] / 2.0
    ly = target_box[1] + target_box[3] / 2.0

    center_dist = math.sqrt((cx - lx) ** 2 + (cy - ly) ** 2)
    
    # If it's the only object found, double the allowed search distance
    effective_max_dist = max_dist * 2.5 if sole_candidate else max_dist
    if center_dist > effective_max_dist:
        return None

    is_far = (w < 20 or h < 20)
    
    # RELAXED: Drones stretch out in frames during camera motion blur
    max_size_ratio = 1.3 if is_far else 0.85
    max_aspect_diff = 1.3 if is_far else 1.0

    # If it's the only drone in view, assume it's the right one even if distorted
    if sole_candidate:
        max_size_ratio += 0.5
        max_aspect_diff += 0.5

    if target_area > 1:
        size_ratio = abs(area - target_area) / target_area
        if size_ratio > max_size_ratio:
            return None
    else:
        size_ratio = 0.0

    aspect_diff = abs(aspect - target_aspect)
    if aspect_diff > max_aspect_diff:
        return None

    # Decreased distance penalty so correct size wins over closest-box
    score = (center_dist * 0.5) + (size_ratio * 50.0) + (aspect_diff * 25.0)
    return score, cand_box, area, aspect


# ============================================================
# MAIN DETECTION & TRACKING ENGINE
# ============================================================

candidate_history = []
last_serial_time = time.time()

while not app.need_exit():
    img = cam.read()
    img_w = img.width()
    img_h = img.height()
    now = time.time()

    v_fov_deg = H_FOV_DEG * (img_h / float(img_w))
    focal_length_px = (img_w / 2.0) / math.tan(math.radians(H_FOV_DEG / 2.0))

    active_conf = CONF_TRACK if locked else CONF_ACQUIRE
    objs = drone_detector.detect(img, conf_th=active_conf, iou_th=IOU_THRESHOLD)

    valid_candidates = []
    for obj in objs:
        if (obj.x <= MAX_BORDER_MARGIN or obj.y <= MAX_BORDER_MARGIN or
            (obj.x + obj.w) >= (img_w - MAX_BORDER_MARGIN) or
            (obj.y + obj.h) >= (img_h - MAX_BORDER_MARGIN)):
            continue

        if obj.w >= MIN_DRONE_PX and obj.h >= MIN_DRONE_PX:
            aspect = obj.w / max(obj.h, 1)
            # Relaxed absolute constraints to handle banking drones
            if 0.25 <= aspect <= 3.5:
                valid_candidates.append((obj.x, obj.y, obj.w, obj.h, obj.score))

    # --------------------------------------------------------
    # STATE 1: SCANNING & MULTI-FRAME CONFIRMATION
    # --------------------------------------------------------
    if not locked:
        detected_locked = False
        is_coasting = False

        if len(valid_candidates) > 0:
            best_cand = max(valid_candidates, key=lambda c: c[4])
            cand_cx = best_cand[0] + best_cand[2] / 2.0
            cand_cy = best_cand[1] + best_cand[3] / 2.0

            if len(candidate_history) == 0:
                candidate_history.append(best_cand)
            else:
                last_cx = candidate_history[-1][0] + candidate_history[-1][2] / 2.0
                last_cy = candidate_history[-1][1] + candidate_history[-1][3] / 2.0
                drift = math.sqrt((cand_cx - last_cx) ** 2 + (cand_cy - last_cy) ** 2)

                if drift <= MAX_ACQUIRE_DRIFT_PX:
                    candidate_history.append(best_cand)
                else:
                    candidate_history = [best_cand]

            if len(candidate_history) >= ACQUIRE_CONSECUTIVE_FRAMES:
                locked_box = candidate_history[-1][0:4]
                locked_area = float(locked_box[2] * locked_box[3])
                locked_aspect = locked_box[2] / max(locked_box[3], 1)
                lost_frames = 0
                locked = True
                detected_locked = True
                candidate_history = []
                
                kf_err_x.reset()
                kf_err_y.reset()
                kf_ang_x.reset()
                kf_ang_y.reset()
                kf_dist.reset()
        else:
            candidate_history = []

    # --------------------------------------------------------
    # STATE 2: ACTIVE TRACKING WITH KALMAN COASTING
    # --------------------------------------------------------
    else:
        search_radius = TRACK_SEARCH_RADIUS_BASE if lost_frames < 4 else min(TRACK_SEARCH_RADIUS_MAX, TRACK_SEARCH_RADIUS_BASE + lost_frames * 40)
        best_score = 999999.0
        best_match = None

        # Check if there is only a single drone in view
        sole_candidate = (len(valid_candidates) == 1)

        for cand in valid_candidates:
            res = score_candidate(cand[0:4], locked_box, locked_area, locked_aspect, search_radius, sole_candidate)
            if res and res[0] < best_score:
                best_score = res[0]
                best_match = res

        if best_match is not None:
            _, box, area, aspect = best_match
            locked_box = box
            
            # Smoothly blend area and aspect so it adjusts to camera zoom slowly
            locked_area = (locked_area * 0.7) + (area * 0.3)
            locked_aspect = (locked_aspect * 0.7) + (aspect * 0.3)
            
            lost_frames = 0
            detected_locked = True
            is_coasting = False
        else:
            lost_frames += 1
            detected_locked = False

            if lost_frames <= MAX_COAST_FRAMES:
                is_coasting = True
            else:
                is_coasting = False

            if lost_frames > MAX_LOST_FRAMES:
                locked = False
                locked_box = None
                is_coasting = False
                candidate_history = []

    # --------------------------------------------------------
    # ERROR, DISTANCE, AND KALMAN STATE UPDATE
    # --------------------------------------------------------
    curr_err_x = 0.0
    curr_err_y = 0.0
    curr_ang_x = 0.0
    curr_ang_y = 0.0
    curr_dist = 0.0

    if locked and locked_box is not None:
        cx = img_w / 2.0
        cy = img_h / 2.0

        if detected_locked:
            target_cx = locked_box[0] + locked_box[2] / 2.0
            target_cy = locked_box[1] + locked_box[3] / 2.0

            raw_err_x = target_cx - cx
            raw_err_y = target_cy - cy
            raw_ang_x = (raw_err_x / img_w) * H_FOV_DEG
            raw_ang_y = (raw_err_y / img_h) * v_fov_deg

            pixel_width = max(locked_box[2], 1)
            raw_dist = (focal_length_px * REAL_DRONE_WIDTH_M) / float(pixel_width)

            curr_err_x = kf_err_x.update(raw_err_x, now)
            curr_err_y = kf_err_y.update(raw_err_y, now)
            curr_ang_x = kf_ang_x.update(raw_ang_x, now)
            curr_ang_y = kf_ang_y.update(raw_ang_y, now)
            curr_dist  = kf_dist.update(raw_dist, now)

            locked_box = (
                int(curr_err_x + cx - locked_box[2] / 2.0),
                int(curr_err_y + cy - locked_box[3] / 2.0),
                locked_box[2],
                locked_box[3]
            )

        elif is_coasting:
            # VELOCITY DAMPENING: If camera whips, the object "jumped". We lose it. 
            # We must aggressively slow down the Kalman velocity (friction) so the tracker doesn't fly off screen.
            kf_err_x.v *= 0.85 
            kf_err_y.v *= 0.85
            kf_ang_x.v *= 0.85
            kf_ang_y.v *= 0.85
            
            curr_err_x = kf_err_x.predict(now)
            curr_err_y = kf_err_y.predict(now)
            curr_ang_x = kf_ang_x.predict(now)
            curr_ang_y = kf_ang_y.predict(now)
            curr_dist  = kf_dist.predict(now)

            # CLAMPING: Prevent the coasting box from leaving the camera bounds
            pred_x = int(curr_err_x + cx - locked_box[2] / 2.0)
            pred_y = int(curr_err_y + cy - locked_box[3] / 2.0)
            pred_x = max(0, min(img_w - locked_box[2], pred_x))
            pred_y = max(0, min(img_h - locked_box[3], pred_y))
            
            locked_box = (pred_x, pred_y, locked_box[2], locked_box[3])

        gpio_out.value(1)
    else:
        gpio_out.value(0)

    # Sync state atomically with other threads
    with state_lock:
        err_x = curr_err_x
        err_y = curr_err_y
        ang_err_x = curr_ang_x
        ang_err_y = curr_ang_y
        estimated_dist_m = curr_dist

    # --------------------------------------------------------
    # LABELED TERMINAL & SERIAL TELEMETRY OUTPUT (10 Hz)
    # --------------------------------------------------------
    if now - last_serial_time >= 0.1:
        last_serial_time = now
        
        status_label = "LOCKED" if detected_locked else ("COASTING" if is_coasting else "SEARCHING")
        
        terminal_msg = (
            f"[{status_label}] "
            f"X_Error: {int(curr_err_x):+4d} px | "
            f"Y_Error: {int(curr_err_y):+4d} px | "
            f"X_Angle: {curr_ang_x:+6.2f} deg | "
            f"Y_Angle: {curr_ang_y:+6.2f} deg | "
            f"Distance: {curr_dist:5.2f} m"
        )
        print(terminal_msg, flush=True)

        if serial_out:
            try:
                serial_line = (
                    f"STATUS:{status_label},"
                    f"ERR_X:{int(curr_err_x)},"
                    f"ERR_Y:{int(curr_err_y)},"
                    f"ANG_X:{curr_ang_x:.2f},"
                    f"ANG_Y:{curr_ang_y:.2f},"
                    f"DIST:{curr_dist:.2f}\n"
                )
                serial_out.write(serial_line.encode("utf-8"))
            except Exception:
                pass

    # --------------------------------------------------------
    # OSD & VISUAL OVERLAYS
    # --------------------------------------------------------
    mid_x = img_w // 2
    mid_y = img_h // 2
    img.draw_line(mid_x - 8, mid_y, mid_x + 8, mid_y, color=image.COLOR_YELLOW, thickness=1)
    img.draw_line(mid_x, mid_y - 8, mid_x, mid_y + 8, color=image.COLOR_YELLOW, thickness=1)

    if locked and locked_box is not None:
        bx, by, bw, bh = locked_box
        
        if detected_locked:
            color = image.COLOR_GREEN
            tag = "LOCKED"
            img.draw_line(mid_x, mid_y, int(bx + bw / 2), int(by + bh / 2), color=image.COLOR_GREEN, thickness=1)
        elif is_coasting:
            color = image.COLOR_BLUE
            tag = f"COASTING ({lost_frames})"
        else:
            color = image.COLOR_RED
            tag = f"LOST ({lost_frames})"

        img.draw_rect(bx, by, bw, bh, color=color, thickness=2)
        img.draw_string(bx, max(0, by - 14), tag, color=color)

        img.draw_string(4, 4, f"X Error: {int(curr_err_x)} px | Y Error: {int(curr_err_y)} px", color=image.COLOR_YELLOW)
        img.draw_string(4, 20, f"X Angle: {curr_ang_x:.1f} deg | Y Angle: {curr_ang_y:.1f} deg", color=image.COLOR_YELLOW)
        img.draw_string(4, 36, f"Distance: {curr_dist:.2f} m", color=image.COLOR_YELLOW)
    else:
        acquire_status = f"SCANNING... ({len(candidate_history)}/{ACQUIRE_CONSECUTIVE_FRAMES})"
        img.draw_string(4, 4, acquire_status, color=image.COLOR_YELLOW)

    disp.show(img)

    try:
        jpeg = img.to_jpeg()
        with jpeg_lock:
            latest_jpeg = jpeg
        jpeg_ready.set()
    except Exception:
        pass