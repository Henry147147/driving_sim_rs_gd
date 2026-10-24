"""Run one allowlisted CLI with cooperative Ctrl+C cancellation through a file."""
import _thread
import argparse
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stop-file', required=True)
    parser.add_argument('--module')
    parser.add_argument('--script')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments

    def watch():
        while not Path(args.stop_file).exists():
            time.sleep(.2)
        _thread.interrupt_main()

    if not getattr(sys, '_racing_lab_watch', False):
        threading.Thread(target=watch, daemon=True).start()
    try:
        if args.module or args.script:
            sys.argv = [args.module or args.script, *arguments]
            if args.module:
                runpy.run_module(args.module, run_name='__main__')
            else:
                runpy.run_path(args.script, run_name='__main__')
        else:
            # Native command is supplied only by the trusted dashboard catalog.
            flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            child = subprocess.Popen(arguments, creationflags=flags)
            try:
                while child.poll() is None:
                    time.sleep(.2)
                raise SystemExit(child.returncode)
            except KeyboardInterrupt:
                child.terminate()
                try:
                    child.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    child.kill()
                raise
    except KeyboardInterrupt:
        print('\nStopped. Checkpoints already written remain available.', flush=True)
        raise SystemExit(130)


if __name__ == '__main__':
    main()
