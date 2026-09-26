import argparse

from mandri.cli.command import Command
from mandri.cli.daemon_client import error_message, request, resolve_base_url
from mandri.cli.local import resolve_base_dir
from mandri.cli.run_errors import DaemonUnreachableError


class ResumeSessionCommand(Command):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self._base_dir = resolve_base_dir(args.base_dir)
        self._source_id = args.session_id
        self._action = "resume"
        self._body: dict[str, str] = {}
        if args.mode is not None:
            self._body["mode"] = args.mode

    def run(self) -> int:
        try:
            url = resolve_base_url(self._base_dir)
            response = request(
                "POST", f"{url}/v1/sessions/{self._source_id}/{self._action}", body=self._body
            )
        except DaemonUnreachableError:
            print("daemon unreachable")
            return 3
        if response.status_code >= 400:
            print(error_message(response))
            return 1
        payload = response.json()
        print(payload["id"])
        return 0


class ForkSessionCommand(ResumeSessionCommand):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._action = "fork"
        self._body.update(
            execution_backend=args.execution_backend,
            privacy_mode=args.privacy_mode,
        )
