from pathlib import Path

_SOF = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})


def inspect_image(path: Path) -> str | None:
    with path.open("rb") as stream:
        header = stream.read(32)
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            if len(header) < 24 or header[12:16] != b"IHDR":
                raise ValueError("Invalid PNG image")
            _check_pixels(int.from_bytes(header[16:20]), int.from_bytes(header[20:24]))
            return "image/png"
        if not header.startswith(b"\xff\xd8\xff"):
            return None
        stream.seek(2)
        while stream.tell() < path.stat().st_size:
            if stream.read(1) != b"\xff":
                break
            marker = stream.read(1)
            while marker == b"\xff":
                marker = stream.read(1)
            if not marker or marker[0] in {0xDA, 0xD9}:
                break
            length = int.from_bytes(stream.read(2))
            if length < 2:
                break
            if marker[0] in _SOF:
                size = stream.read(5)
                if len(size) != 5:
                    break
                _check_pixels(int.from_bytes(size[3:5]), int.from_bytes(size[1:3]))
                return "image/jpeg"
            stream.seek(length - 2, 1)
    raise ValueError("Invalid JPEG image")


def _check_pixels(width: int, height: int) -> None:
    if not width or not height or width * height > 25_000_000:
        raise ValueError("Image dimensions exceed the supported limit")
