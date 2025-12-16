import os
import pandas as pd
from datetime import datetime

CURRENT_DIR = os.path.join("data", "current")
LOG_PATH = os.path.join(CURRENT_DIR, "live_inputs.csv")

def log_live_input(payload: dict) -> None:
    """
    Append one prediction input payload to data/current/live_inputs.csv
    """
    os.makedirs(CURRENT_DIR, exist_ok=True)

    row = dict(payload)
    row["_logged_at"] = datetime.utcnow().isoformat()

    df = pd.DataFrame([row])

    if os.path.exists(LOG_PATH):
        df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_PATH, index=False)