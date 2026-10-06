import os
import sys

if sys.platform != "win32":
    import fcntl
    import termios


def main() -> None:
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)
    os.execvpe(sys.argv[1], sys.argv[1:], os.environ)


if __name__ == "__main__":
    main()
