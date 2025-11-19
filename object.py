import cv2
import time
import threading
import queue
import numpy as np
import sys
import traceback

# ------------------------
# Configuration
# ------------------------
MODEL_PATH = "yolov8x.pt"   # change if needed
CAM_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

MAX_SPEAK_COUNT = 2
DISTANCE_SCALAR = 5000.0
MIN_BOX_HEIGHT = 5
DISAPPEAR_HOLD_TIME = 0.5
SPEECH_COOLDOWN = 3.0  # seconds between same object announcements

# ------------------------
# Flags & Events
# ------------------------
program_running = threading.Event()
program_running.set()

# ------------------------
# Optional libs
# ------------------------
try:
    import pyttsx3
    TTS_LIB_PRESENT = True
except Exception:
    print("WARNING: pyttsx3 not installed or unavailable. TTS disabled. Install via `pip install pyttsx3`.")
    TTS_LIB_PRESENT = False

try:
    from ultralytics import YOLO
    YOLO_LIB_PRESENT = True
except Exception:
    print("WARNING: ultralytics (YOLO) not installed or unavailable. Detection disabled. Install via `pip install ultralytics`.")
    YOLO_LIB_PRESENT = False

# ------------------------
# Load model (if available)
# ------------------------
if YOLO_LIB_PRESENT:
    try:
        model = YOLO(MODEL_PATH)
        names = model.names
        print("YOLO model loaded.")
    except Exception as e:
        print(f"ERROR loading YOLO model '{MODEL_PATH}': {e}")
        YOLO_LIB_PRESENT = False
        names = {}
else:
    names = {}

# ------------------------
# TTS: queue + worker
# ------------------------
speech_queue = queue.Queue()          # unbounded queue
tts_stop_event = threading.Event()
tts_thread = None
TTS_WORKER_READY = threading.Event()  # set when worker successfully initialized

def safe_print_tts(msg: str):
    """Fallback printing function when pyttsx3 is not available."""
    print("[TTS-PRINT]", msg)

def tts_worker(q: queue.Queue, stop_event: threading.Event):
    """
    Runs in a dedicated thread. Initializes pyttsx3 engine here (in-thread),
    consumes queue, speaks messages using engine.say + runAndWait.
    Robust to engine errors: tries to re-create engine on failures.
    """
    engine = None
    engine_init_attempts = 0
    MAX_INIT_RETRIES = 3

    def try_init_engine():
        nonlocal engine, engine_init_attempts
        engine_init_attempts += 1
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            # Optionally set voice here (platform-specific):
            # voices = engine.getProperty('voices'); engine.setProperty('voice', voices[0].id)
            print(f"[TTS] Engine initialized (attempt {engine_init_attempts}).")
            TTS_WORKER_READY.set()
            return True
        except Exception as e:
            print(f"[TTS] Engine init failed (attempt {engine_init_attempts}): {e}")
            engine = None
            return False

    if not TTS_LIB_PRESENT:
        # Shouldn't come here, but safe-guard
        print("[TTS] pyttsx3 not present in worker. Exiting TTS worker.")
        return

    # Try initializing engine with a few retries
    while engine is None and engine_init_attempts < MAX_INIT_RETRIES and not stop_event.is_set():
        ok = try_init_engine()
        if not ok:
            time.sleep(0.7)

    if engine is None:
        print("[TTS] Failed to initialize TTS engine after retries. Worker exiting.")
        return

    # Main consume loop
    while not stop_event.is_set():
        try:
            # Wait for next message; short timeout so we respond quickly to stop_event
            message = q.get(timeout=0.4)
        except queue.Empty:
            continue

        if message is None:
            # sentinel to break early
            q.task_done()
            break

        try:
            # Speak message
            # IMPORTANT: do not call engine.stop() here.
            engine.say(message)
            engine.runAndWait()  # blocks until speaking finished
        except Exception as e:
            # On runtime error, try to re-create engine once and re-queue message.
            print(f"[TTS] Runtime error while speaking: {e}")
            traceback.print_exc()
            try:
                # attempt to reinitialize engine
                print("[TTS] Attempting to reinitialize engine after runtime error...")
                engine = None
                TTS_WORKER_READY.clear()
                engine_init_attempts = 0
                while engine is None and engine_init_attempts < MAX_INIT_RETRIES and not stop_event.is_set():
                    if try_init_engine():
                        break
                    time.sleep(0.5)
                # If engine restored, re-queue message for speaking (put at front not supported so put normally)
                if engine is not None:
                    q.put(message)
                else:
                    print("[TTS] Could not restore engine. Dropping message.")
            except Exception as e2:
                print("[TTS] Reinit failed:", e2)
        finally:
            try:
                q.task_done()
            except Exception:
                pass

    # End worker: attempt a final run/wait flush (best-effort)
    try:
        if engine is not None:
            # ensure any current speech finished or stop gracefully
            # no engine.stop() to avoid truncation; we simply let runAndWait finish
            pass
    except Exception:
        pass

    print("[TTS] Worker exiting.")

def start_tts_thread():
    global tts_thread
    if not TTS_LIB_PRESENT:
        print("[TTS] pyttsx3 unavailable; not starting TTS thread.")
        return
    tts_thread = threading.Thread(target=tts_worker, args=(speech_queue, tts_stop_event), daemon=True)
    tts_thread.start()
    # Wait briefly for worker to signal readiness (optional)
    if not TTS_WORKER_READY.wait(timeout=3.0):
        print("[TTS] Warning: TTS worker did not become ready within 3s. It may still initialize later.")

def enqueue_speech(msg: str):
    """
    Safe enqueue: if TTS library present we enqueue; otherwise fallback print.
    Using blocking put so messages are preserved.
    """
    if not msg:
        return
    if TTS_LIB_PRESENT:
        try:
            speech_queue.put(msg)  # blocking; queue is unbounded by default
        except Exception as e:
            print(f"[TTS] Failed to enqueue message: {e}")
    else:
        safe_print_tts(msg)

# Start TTS thread if possible
if TTS_LIB_PRESENT:
    start_tts_thread()

# ------------------------
# Camera setup
# ------------------------
cap = cv2.VideoCapture(CAM_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
if not cap.isOpened():
    print(f"ERROR: Could not open camera id {CAM_ID}. Exiting.")
    program_running.clear()
    sys.exit(1)

print("Press 'q' in the window to quit. Starting detection (YOLO if available).")

# ------------------------
# Tracking dictionaries
# ------------------------
spoken_objects = {}      # how many times spoken for each label
tracked_objects = {}     # last seen timestamp for each label
last_speech_time = {}    # last time we spoke about a label

# ------------------------
# Main loop
# ------------------------
try:
    while program_running.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[MAIN] Failed to grab frame; retrying shortly.")
            time.sleep(0.01)
            continue

        now = time.time()
        current_frame_labels = set()

        # status text
        cv2.putText(frame, "Status: DETECTING (YOLO)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # YOLO detection (if available)
        if YOLO_LIB_PRESENT:
            try:
                results = model(frame, device=0, verbose=False)   # Use GPU
                res = results[0] if isinstance(results, (list, tuple)) else results


                boxes = getattr(res, "boxes", None)
                if boxes is not None and len(boxes) > 0:
                    xyxy_arr = boxes.xyxy.cpu().numpy()
                    conf_arr = boxes.conf.cpu().numpy()
                    cls_arr = boxes.cls.cpu().numpy()

                    for i in range(len(xyxy_arr)):
                        x1, y1, x2, y2 = xyxy_arr[i].tolist()
                        conf = float(conf_arr[i])
                        class_id = int(cls_arr[i])
                        label_name = names.get(class_id, str(class_id))

                        current_frame_labels.add(label_name)

                        x1i, y1i, x2i, y2i = map(int, map(round, (x1, y1, x2, y2)))

                        # Draw box and label
                        cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label_name} {conf:.2f}",
                                    (x1i, max(15, y1i-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                        # Distance estimate
                        box_height = float(y2i - y1i)
                        if box_height < MIN_BOX_HEIGHT:
                            continue
                        distance_cm = round(DISTANCE_SCALAR / box_height, 2)

                        # Announce with cooldown + limited repeats
                        speak_count = spoken_objects.get(label_name, 0)
                        last_time = last_speech_time.get(label_name, 0)

                        if speak_count < MAX_SPEAK_COUNT and (now - last_time > SPEECH_COOLDOWN):
                            message = f"{label_name} detected. Distance approximately {distance_cm} centimeters."
                            enqueue_speech(message)
                            spoken_objects[label_name] = speak_count + 1
                            last_speech_time[label_name] = now

            except Exception as e:
                # Catch and continue — don't crash main loop if model has issues
                print(f"[MAIN] YOLO runtime error: {e}")
                traceback.print_exc()

        # update seen timestamps
        for label in current_frame_labels:
            tracked_objects[label] = now

        # disappearance detection
        for label, last_seen in list(tracked_objects.items()):
            if (now - last_seen) > DISAPPEAR_HOLD_TIME:
                enqueue_speech(f"The {label} has disappeared.")
                tracked_objects.pop(label, None)
                spoken_objects.pop(label, None)
                last_speech_time.pop(label, None)

        # Show frame
        cv2.imshow("Detection (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            program_running.clear()
            break

except KeyboardInterrupt:
    print("[MAIN] Interrupted by user (KeyboardInterrupt).")

finally:
    # ------------------------
    # Shutdown & cleanup
    # ------------------------
    print("[MAIN] Shutting down...")
    program_running.clear()

    try:
        cap.release()
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    # Signal TTS worker to stop and let it finish speaking queued items
    if TTS_LIB_PRESENT:
        print("[MAIN] Waiting for TTS queue to drain (up to 5s)...")
        try:
            # Wait some reasonable time for queue to be processed
            # Give up to 5 seconds for queue to finish naturally
            total_waited = 0.0
            wait_step = 0.1
            while not speech_queue.empty() and total_waited < 5.0:
                time.sleep(wait_step)
                total_waited += wait_step

            # Now signal worker to stop
            tts_stop_event.set()
            # Also put a sentinel None so worker can break quicker if blocked waiting
            try:
                speech_queue.put_nowait(None)
            except Exception:
                pass

            if tts_thread is not None:
                tts_thread.join(timeout=3.0)
        except Exception as e:
            print("[MAIN] Error while shutting down TTS:", e)
    else:
        print("[MAIN] TTS not present; nothing to stop.")

    print("[MAIN] Application closed.")