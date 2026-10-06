import base64
import contextlib
import ipaddress
import json
import os
import select
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
from types import FrameType
from typing import cast


def rule(tool: str, *args: str) -> None:
    subprocess.run([tool, "-w", "10", *args], check=True)


class GatewayRelay(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as upstream:
            upstream.connect(cast("GatewayRelayServer", self.server).socket_path)
            peers = {self.request: upstream, upstream: self.request}
            while True:
                readable, _, _ = select.select(list(peers), [], [], 300)
                if not readable:
                    return
                for source in readable:
                    data = source.recv(65536)
                    if not data:
                        return
                    peers[source].sendall(data)


class GatewayRelayServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True
    socket_path: str


def guard(encoded: str) -> None:
    config = json.loads(base64.urlsafe_b64decode(encoded))
    allowed = list(config["exceptions"])
    for address in socket.getaddrinfo(config["ingress_host"], None, type=socket.SOCK_STREAM):
        allowed.append([address[4][0], config["ingress_port"]])
    for version, tool in ((4, "/usr/sbin/iptables"), (6, "/usr/sbin/ip6tables")):
        rule(tool, "-P", "OUTPUT", "DROP")
        rule(tool, "-A", "OUTPUT", "-o", "lo", "-j", "ACCEPT")
        rule(
            tool,
            "-A",
            "OUTPUT",
            "-m",
            "conntrack",
            "--ctstate",
            "ESTABLISHED,RELATED",
            "-j",
            "ACCEPT",
        )
        for network, port in allowed:
            if ipaddress.ip_network(network, strict=False).version == version:
                rule(
                    tool,
                    "-A",
                    "OUTPUT",
                    "-d",
                    network,
                    "-p",
                    "tcp",
                    "--dport",
                    str(port),
                    "-j",
                    "ACCEPT",
                )
        for network in config["denied"]:
            if ipaddress.ip_network(network).version == version:
                rule(tool, "-A", "OUTPUT", "-d", network, "-j", "REJECT")
        rule(tool, "-A", "OUTPUT", "-j", "ACCEPT")
    if config.get("ingress_socket"):
        relay = GatewayRelayServer(("127.0.0.1", config["ingress_port"]), GatewayRelay)
        relay.socket_path = config["ingress_socket"]
        threading.Thread(target=relay.serve_forever, daemon=True).start()
    print("MANDRI_NETWORK_READY", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        signal.pause()


def run(command: list[str]) -> int:
    expected = os.environ.pop("MANDRI_WORKSPACE_IDENTITY", None)
    metadata = os.stat("/workspace")
    actual = f"{metadata.st_dev}:{metadata.st_ino}"
    if expected is None or actual != expected:
        print("MANDRI_WORKSPACE_IDENTITY_MISMATCH", file=sys.stderr, flush=True)
        return 125
    print(f"MANDRI_WORKSPACE_READY {actual}", flush=True)
    if command[0] == "--terminal":
        os.environ.pop("CI", None)
        os.execvp(command[1], command[1:])
    watch_stdin = command[0] == "--watch-stdin"
    if watch_stdin:
        command = command[1:]
    child = subprocess.Popen(command, stdin=subprocess.PIPE, start_new_session=True)
    stdin = child.stdin
    assert stdin is not None
    stopping: float | None = None
    stdin_open = True
    stdin_eof: float | None = None

    def forward(signum: int, _frame: FrameType | None) -> None:
        nonlocal stopping
        if signum == signal.SIGTERM:
            stopping = time.monotonic()
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signum)

    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)
    while child.poll() is None:
        if stdin_open and select.select([sys.stdin.buffer], [], [], 0.1)[0]:
            chunk = os.read(sys.stdin.fileno(), 65536)
            if not chunk:
                stdin_open = False
                stdin_eof = time.monotonic() + (0 if watch_stdin else 2)
                with contextlib.suppress(BrokenPipeError):
                    stdin.close()
            elif not watch_stdin:
                with contextlib.suppress(BrokenPipeError):
                    stdin.write(chunk)
                    stdin.flush()
        else:
            time.sleep(0.05)
        if stdin_eof is not None and time.monotonic() >= stdin_eof:
            stdin_eof = None
            forward(signal.SIGTERM, None)
        if stopping is not None and time.monotonic() - stopping > 10:
            forward(signal.SIGKILL, None)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
    code = child.wait()
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    if sys.argv[1] == "guard":
        guard(sys.argv[2])
    elif sys.argv[1] == "run":
        raise SystemExit(run(sys.argv[2:]))
    else:
        raise SystemExit(2)
