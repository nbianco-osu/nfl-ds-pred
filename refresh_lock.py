"""Process-scoped Windows refresh lock; released automatically after crashes."""
from contextlib import contextmanager
import os
if os.name == 'nt':
    import msvcrt
else:
    import fcntl


@contextmanager
def refresh_lock(root):
    path=root/'audit/refresh.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        if path.stat().st_size==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:raise RuntimeError('Another refresh is in progress') from exc
        try:yield
        finally:
            stream.seek(0)
            if os.name == 'nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream,fcntl.LOCK_UN)
