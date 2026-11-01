import sys
import os

def enable_global_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_path)
    sys.stderr = Logger(log_path)
    print(f"[LOGGER] Output redirected to: {log_path}")
