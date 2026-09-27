"""Typed error codes and exception for the websocket request protocol."""

import enum


class ProtocolErrorCode(enum.StrEnum):
    AGENT_UNSUPPORTED = "agent_unsupported"
    INTERNAL_ERROR = "internal_error"
    SESSION_CONFLICT = "session_conflict"
    HARNESS_STORE_UNAVAILABLE = "harness_store_unavailable"
    UNKNOWN_ACTION = "unknown_action"
    SESSION_NOT_RUNNING = "session_not_running"
    APPROVAL_NOT_PENDING = "approval_not_pending"
    APPROVAL_ALREADY_ANSWERED = "approval_already_answered"
    APPROVAL_EXPIRED = "approval_expired"
    STEER_UNSUPPORTED = "steer_unsupported"
    STEER_NO_ACTIVE_TURN = "steer_no_active_turn"
    MODE_REQUIRES_RESTART = "mode_requires_restart"
    MODE_REJECTED = "mode_rejected"
    PROMPT_DELIVERY_FAILED = "prompt_delivery_failed"
    CONTROL_DELIVERY_FAILED = "control_delivery_failed"
    INVALID_PARAMS = "invalid_params"
    ATTACHMENT_STORAGE_UNAVAILABLE = "attachment_storage_unavailable"
    DUPLICATE_OP_ID = "duplicate_op_id"


class ProtocolError(Exception):
    def __init__(self, code: ProtocolErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
