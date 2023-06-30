import platform
import os
from oracle.util import WINDOWS


def system():
    return {
        "os": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "architecture": platform.architecture(),
        "machine": platform.machine(),
        "user": os.getlogin(),
        "env": dict(os.environ),
        "path": os.environ["PATH"],
        "home": os.environ["HOME"],
        "python": platform.python_version(),
        "hostname": platform.node(),
        "libc": platform.libc_ver() if not WINDOWS else None,
        "cpu": {
            "cores": os.cpu_count(),
            "name": platform.processor(),
        },
        "memory": {
            "total": os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') if not WINDOWS else 0,
        },
    }
