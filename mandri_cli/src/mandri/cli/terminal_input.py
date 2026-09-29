import asyncio
import sys

if sys.platform == "win32":
    import msvcrt
else:
    import select


async def read_terminal_line() -> str:
    characters: list[str] = []
    while True:
        if sys.platform == "win32":
            if msvcrt.kbhit():
                char = msvcrt.getwch()
                if char == "\r":
                    print()
                    return "".join(characters)
                if char == "\x03":
                    raise KeyboardInterrupt
                if char == "\b":
                    if characters:
                        characters.pop()
                        print("\b \b", end="", flush=True)
                elif char in {"\x00", "\xe0"}:
                    msvcrt.getwch()
                else:
                    characters.append(char)
                    print(char, end="", flush=True)
        elif select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.readline()
        await asyncio.sleep(0.05)
