"""Tests for the shared error base hierarchy across domains."""

import pytest
from mandri.core.errors import MandriError


def test_mandri_error_is_exception() -> None:
    assert issubclass(MandriError, Exception)
    assert str(MandriError("boom")) == "boom"


def test_domain_error_bases_extend_mandri_error() -> None:
    from mandri.core.fs.errors.base import FsError
    from mandri.core.types.approvals import ApprovalError
    from mandri.core.types.sessions import SessionError
    from mandri.gateway.errors.base import GatewayError
    from mandri.runtime.control.errors.base import ControlError
    from mandri.runtime.errors.base import RuntimeDomainError

    assert issubclass(ApprovalError, MandriError)
    assert issubclass(ControlError, MandriError)
    assert issubclass(FsError, MandriError)
    assert issubclass(GatewayError, MandriError)
    assert issubclass(RuntimeDomainError, MandriError)
    assert issubclass(SessionError, MandriError)


def test_domain_errors_catchable_as_mandri_error() -> None:
    from mandri.core.fs.errors.listing import FsNotFoundError
    from mandri.sessions.errors.mutations import SessionNotFoundError

    assert issubclass(FsNotFoundError, MandriError)
    assert issubclass(SessionNotFoundError, MandriError)

    with pytest.raises(MandriError):
        raise FsNotFoundError("missing")


def test_session_state_error_catchable_as_session_and_mandri_error() -> None:
    from mandri.core.types.sessions import SessionError, SessionStateError

    with pytest.raises(SessionError):
        raise SessionStateError("bad state")
    with pytest.raises(MandriError):
        raise SessionStateError("bad state")
