from datetime import datetime
import pytz

def now_tz(tz_str: str):
    tz = pytz.timezone(tz_str)
    return datetime.now(tz)

def log(msg: str):
    t = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{t}] {msg}", flush=True)
