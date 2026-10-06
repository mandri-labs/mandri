import os
import sys

if sys.platform != "win32":
    import fcntl
    import termios


def main() -> None:
    if sys.platform != "win32":
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)
        os.execvpe(sys.argv[1], sys.argv[1:], os.environ)
    else:
        raise RuntimeError("This entry point requires a POSIX host")


if __name__ == "__main__":
    main()
