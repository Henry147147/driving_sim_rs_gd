"""Cooperative cancellation for Python children spawned by a dashboard CLI job."""
import _thread
import os
from pathlib import Path
import sys
import threading
import time

_stop = os.environ.get('RACING_LAB_STOP_FILE')
if _stop:
    sys._racing_lab_watch = True
    def _watch():
        while not Path(_stop).exists():
            time.sleep(.2)
        _thread.interrupt_main()
    threading.Thread(target=_watch, daemon=True).start()
