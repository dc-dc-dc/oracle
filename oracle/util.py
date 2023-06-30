import platform
import os
import sys

OSX = platform.system() == "Darwin"
WINDOWS = platform.system() == "Windows"


def extract_val(val): return ([x for x in val]
                              if hasattr(val, "__len__") else val)
def getdict(struct): return dict((field, extract_val(getattr(struct, field)))
                                 for field, _ in struct._fields_)


def ascii_str(s): return ''.join(chr(i) for i in s)
def getenv(x, default=0): return type(default)(os.getenv(x, default))


def error_wrap(method: str, c: int):
    if c != 0:
        if LOG_ERROR:
            sys.stderr.write(f"Error: {method} failed with error code {c}\n")
        return None


LOG_ERROR = getenv("DEBUG", 0) > 1
