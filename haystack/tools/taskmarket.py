# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""TaskMarket requester tools for Haystack agents."""

import hashlib
import json
import math
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from typing import Annotated, Any, Literal

import httpx

from haystack.core.serialization import generate_qualified_class_name

from .from_function import create_tool_from_function
from .toolset import Toolset

DEFAULT_API_URL = "https://api.taskmarket.dev"
DEFAULT_CLI = "taskmarket"
BASE_CHAIN_ID = 8453
USDC_CONTRACT = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
PLATFORM_FEE_BPS = Decimal("750")
RELAY_FEE_USDC = Decimal("0.001")
USDC_QUANTUM = Decimal("0.000001")
PREVIEW_TTL = timedelta(minutes=15)
TaskMode = Literal["bounty", "claim", "pitch", "benchmark"]
SUPPORTED_MODES = frozenset({"bounty", "claim", "pitch", "benchmark"})
JsonTransport = Callable[[str, str, dict[str, str], dict[str, Any] | None], dict[str, Any]]


@dataclass(frozen=True)
class _PendingPreview:
    request: dict[str, Any]
    deadline: datetime
    expires_at: datetime
    maximum_spend: Decimal


@dataclass(frozen=True)
class _CliResult:
    succeeded: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    ambiguous: bool = False


@dataclass(frozen=True)
class _CreationContext:
    pending: _PendingPreview
    request: dict[str, Any]
    confirmation_token: str


class TaskMarketToolset(Toolset):
    """
    A confirmation-gated TaskMarket requester toolset.

    Read tools use TaskMarket's public HTTP API. The write path requires a
    matching preview, ``confirm=True``, and an application-provided approval
    callback. Before calling the first-party CLI once, it checks Base/USDC
    configuration and balance. The toolset never receives wallet keys and has
    no operation for accepting or rejecting submissions.

    :param api_url: TaskMarket API origin.
    :param cli_path: First-party ``taskmarket`` executable name or path.
    :param timeout: HTTP and CLI timeout in seconds.
    :param approval: Callback that presents the exact preview to a human and
        returns whether task creation is authorized.
    """

    def __init__(
        self,
        *,
        api_url: str = DEFAULT_API_URL,
        cli_path: str = DEFAULT_CLI,
        timeout: float = 15.0,
        approval: Callable[[dict[str, Any]], bool] | None = None,
        transport: JsonTransport | None = None,
        cli_runner: Callable[[list[str], bool], _CliResult] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not api_url.strip():
            raise ValueError("`api_url` must not be empty")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("`timeout` must be a positive finite number")

        self.api_url = api_url.rstrip("/")
        self.cli_path = cli_path
        self.timeout = timeout
        self._approval = approval
        self._transport = transport
        self._cli_runner = cli_runner
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._pending_previews: dict[str, _PendingPreview] = {}

        super().__init__(
            tools=[
                create_tool_from_function(
                    self.list_tasks,
                    name="taskmarket_list_tasks",
                    description="List live TaskMarket tasks without spending funds.",
                ),
                create_tool_from_function(
                    self.get_task, name="taskmarket_get_task", description="Retrieve live status for a TaskMarket task."
                ),
                create_tool_from_function(
                    self.list_submissions,
                    name="taskmarket_list_submissions",
                    description="List TaskMarket submissions for human review only.",
                ),
                create_tool_from_function(
                    self.preview_task,
                    name="taskmarket_preview_task",
                    description="Prepare an exact TaskMarket request for review without spending funds.",
                ),
                create_tool_from_function(
                    self.create_task,
                    name="taskmarket_create_task",
                    description=(
                        "Create a TaskMarket task only after a matching preview and human approval. "
                        "Never accept or reject submissions automatically."
                    ),
                ),
            ]
        )

    def list_tasks(
        self,
        status: Annotated[str, "Task status to list"] = "open",
        mode: Annotated[str | None, "Optional task mode"] = None,
        tags: Annotated[list[str] | None, "Optional tags that must match"] = None,
        limit: Annotated[int, "Maximum number of tasks to return"] = 20,
    ) -> dict[str, Any]:
        """List live public tasks."""
        if not 1 <= limit <= 100:
            return {"error": "`limit` must be between 1 and 100", "retry": False}
        params = {"status": status, "limit": str(limit)}
        if mode:
            params["mode"] = mode
        if tags:
            params["tags"] = ",".join(tag.strip() for tag in tags if tag.strip())
        return self._request_json("GET", "/api/tasks", params=params)

    def get_task(self, task_id: Annotated[str, "0x-prefixed TaskMarket task ID"]) -> dict[str, Any]:
        """Retrieve a live task without changing it."""
        error = _validate_task_id(task_id)
        if error:
            return {"error": error, "retry": False}
        return self._request_json("GET", f"/api/tasks/{task_id}")

    def list_submissions(self, task_id: Annotated[str, "0x-prefixed TaskMarket task ID"]) -> dict[str, Any]:
        """Return submissions for human review without accepting work."""
        error = _validate_task_id(task_id)
        if error:
            return {"error": error, "retry": False}
        return self._request_json("GET", f"/api/tasks/{task_id}/submissions")

    def preview_task(
        self,
        description: Annotated[str, "Complete task deliverable description"],
        reward_usdc: Annotated[str, "Positive USDC reward with at most 6 decimals"],
        duration_hours: Annotated[float, "How long the task should remain open"],
        mode: Annotated[TaskMode, "TaskMarket task mode"] = "bounty",
        tags: Annotated[list[str] | None, "Optional task tags"] = None,
    ) -> dict[str, Any]:
        """Prepare an exact, reviewable request without spending funds."""
        try:
            request = self._normalise_request(
                description=description, reward_usdc=reward_usdc, duration_hours=duration_hours, mode=mode, tags=tags
            )
        except ValueError as exc:
            return {"error": str(exc), "retry": False}

        now = self._utc_now()
        try:
            deadline = now + timedelta(hours=float(request["durationHours"]))
        except (OverflowError, ValueError) as exc:
            return {"error": f"`duration_hours` is too large: {exc}", "retry": False}
        if deadline <= now:
            return {"error": "`duration_hours` must produce a future deadline", "retry": False}

        request["deadline"] = _format_datetime(deadline)
        maximum_spend = _maximum_spend(Decimal(request["rewardUsdc"]))
        request["maximumSpendUsdc"] = _format_usdc(maximum_spend)
        token = _confirmation_digest(request)
        expires_at = min(deadline, now + PREVIEW_TTL)
        self._pending_previews[token] = _PendingPreview(
            request=request, deadline=deadline, expires_at=expires_at, maximum_spend=maximum_spend
        )
        return {
            **request,
            "network": "Base",
            "chainId": BASE_CHAIN_ID,
            "currency": "USDC",
            "usdcContract": USDC_CONTRACT,
            "confirmationToken": token,
            "expiresAt": _format_datetime(expires_at),
        }

    def create_task(
        self,
        description: Annotated[str, "Complete task deliverable description"],
        reward_usdc: Annotated[str, "Positive USDC reward with at most 6 decimals"],
        duration_hours: Annotated[float, "How long the task should remain open"],
        confirmation_token: Annotated[str, "Unchanged token from taskmarket_preview_task"],
        confirm: Annotated[bool, "Must be true in addition to the approval callback"] = False,
        mode: Annotated[TaskMode, "TaskMarket task mode"] = "bounty",
        tags: Annotated[list[str] | None, "Optional task tags"] = None,
    ) -> dict[str, Any]:
        """Create one task after matching preview, approval, and wallet checks."""
        if not confirm:
            return {
                "error": (
                    "Creation is confirmation-gated. Review `taskmarket_preview_task` first, "
                    "then call `taskmarket_create_task` with `confirm=True`."
                ),
                "retry": False,
            }
        context = self._prepare_creation_context(
            description=description,
            reward_usdc=reward_usdc,
            duration_hours=duration_hours,
            confirmation_token=confirmation_token,
            mode=mode,
            tags=tags,
        )
        if isinstance(context, dict):
            return context

        approval_error = self._request_approval(context)
        if approval_error is not None:
            return approval_error

        preflight = self._preflight_cli(context.pending.maximum_spend)
        if not preflight.succeeded:
            return {
                "error": preflight.error or "TaskMarket wallet preflight failed.",
                "retry": False,
                "status": "blocked",
            }

        cli_args = self._build_create_args(context)
        if isinstance(cli_args, dict):
            return cli_args

        result = self._run_cli(cli_args, True)
        task_id = _task_id_from_cli_result(result.data)
        if not result.succeeded or task_id is None:
            if not result.succeeded:
                error = result.error or "Task creation failed; inspect live status before retrying."
                status = "unknown" if result.ambiguous else "failed"
            else:
                error = "The CLI returned no task ID. Inspect live status before retrying."
                status = "unknown"
            return {"error": error, "retry": False, "status": status}

        self._pending_previews.pop(context.confirmation_token, None)
        return {
            "taskId": task_id,
            "taskUrl": f"{self.api_url}/api/tasks/{task_id}",
            "status": "created",
            "retry": False,
        }

    def _prepare_creation_context(
        self,
        *,
        description: str,
        reward_usdc: str,
        duration_hours: float,
        confirmation_token: str,
        mode: TaskMode,
        tags: list[str] | None,
    ) -> dict[str, Any] | _CreationContext:
        if not confirmation_token:
            return {"error": "A `confirmation_token` from `taskmarket_preview_task` is required.", "retry": False}

        pending = self._pending_previews.get(confirmation_token)
        if pending is None:
            return {"error": "The preview is missing or has already been used.", "retry": False}
        now = self._utc_now()
        if now >= pending.expires_at or now >= pending.deadline:
            self._pending_previews.pop(confirmation_token, None)
            return {"error": "The preview expired. Run `taskmarket_preview_task` again.", "retry": False}

        try:
            request = self._normalise_request(
                description=description, reward_usdc=reward_usdc, duration_hours=duration_hours, mode=mode, tags=tags
            )
        except ValueError as exc:
            return {"error": str(exc), "retry": False}
        reviewed_request = {key: pending.request[key] for key in request}
        if request != reviewed_request:
            return {
                "error": (
                    "The creation arguments differ from the reviewed preview. "
                    "Run `taskmarket_preview_task` again and review the new record."
                ),
                "retry": False,
            }
        return _CreationContext(pending=pending, request=request, confirmation_token=confirmation_token)

    def _request_approval(self, context: _CreationContext) -> dict[str, Any] | None:
        if self._approval is None:
            return {
                "error": "No approval callback is configured; task creation is blocked.",
                "retry": False,
                "status": "blocked",
            }
        preview = {
            **context.pending.request,
            "network": "Base",
            "chainId": BASE_CHAIN_ID,
            "currency": "USDC",
            "usdcContract": USDC_CONTRACT,
            "confirmationToken": context.confirmation_token,
            "expiresAt": _format_datetime(context.pending.expires_at),
        }
        try:
            approved = self._approval(preview)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"The approval callback failed: {exc}", "retry": False, "status": "blocked"}
        if not approved:
            return {
                "error": "The approval callback did not authorize task creation.",
                "retry": False,
                "status": "denied",
            }
        return None

    def _build_create_args(self, context: _CreationContext) -> list[str] | dict[str, Any]:
        remaining_hours = (context.pending.deadline - self._utc_now()).total_seconds() / 3600
        if remaining_hours <= 0:
            self._pending_previews.pop(context.confirmation_token, None)
            return {"error": "The reviewed deadline has passed. Run `taskmarket_preview_task` again.", "retry": False}

        cli_args = [
            "task",
            "create",
            "--description",
            context.request["description"],
            "--reward",
            context.request["rewardUsdc"],
            "--duration",
            f"{remaining_hours:.9f}",
            "--mode",
            context.request["mode"],
        ]
        if context.request["tags"]:
            cli_args.extend(["--tags", ",".join(context.request["tags"])])
        return cli_args

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration, not runtime callbacks or wallet state."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"api_url": self.api_url, "cli_path": self.cli_path, "timeout": self.timeout},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskMarketToolset":
        """Recreate a safe, read-capable toolset from serialized configuration."""
        config = data["data"]
        return cls(api_url=config["api_url"], cli_path=config["cli_path"], timeout=config["timeout"])

    def _preflight_cli(self, maximum_spend: Decimal) -> _CliResult:
        if self._cli_runner is None and shutil.which(self.cli_path) is None:
            return _CliResult(
                succeeded=False,
                error=("The first-party `taskmarket` CLI was not found. Install `@lucid-agents/taskmarket` and retry."),
            )
        deposit = self._run_cli(["deposit"], False)
        if not deposit.succeeded or deposit.data is None:
            return _CliResult(succeeded=False, error=deposit.error or "Could not verify TaskMarket wallet network.")
        expected = {"network": "Base", "chainId": BASE_CHAIN_ID, "currency": "USDC", "usdcContract": USDC_CONTRACT}
        if any(deposit.data.get(key) != value for key, value in expected.items()):
            return _CliResult(succeeded=False, error="TaskMarket wallet is not configured for Base USDC.")

        stats = self._run_cli(["stats"], False)
        if not stats.succeeded or stats.data is None:
            return _CliResult(succeeded=False, error=stats.error or "Could not verify available USDC balance.")
        try:
            balance = Decimal(str(stats.data["balanceUsdc"]))
        except (KeyError, InvalidOperation, TypeError, ValueError):
            return _CliResult(succeeded=False, error="TaskMarket CLI returned an unreadable USDC balance.")
        if not balance.is_finite() or balance < maximum_spend:
            return _CliResult(
                succeeded=False,
                error=f"Insufficient USDC balance for the reviewed maximum spend ({_format_usdc(maximum_spend)} USDC).",
            )
        return _CliResult(succeeded=True, data=stats.data)

    def _run_cli(self, args: list[str], is_write: bool) -> _CliResult:
        if self._cli_runner is not None:
            return self._cli_runner(args, is_write)
        try:
            completed = subprocess.run(
                [self.cli_path, *args], capture_output=True, check=False, text=True, timeout=self.timeout
            )
        except FileNotFoundError:
            return _CliResult(succeeded=False, error="The first-party `taskmarket` CLI was not found.")
        except subprocess.TimeoutExpired:
            return _CliResult(
                succeeded=False,
                error=("TaskMarket CLI timed out. Inspect live status before retrying; the command was not retried."),
                ambiguous=is_write,
            )
        parsed: Any = None
        if completed.stdout.strip():
            try:
                parsed = json.loads(completed.stdout)
            except json.JSONDecodeError:
                parsed = None
        if completed.returncode == 0 and isinstance(parsed, dict):
            if parsed.get("ok") is False:
                return _CliResult(succeeded=False, error="TaskMarket CLI rejected the command.", ambiguous=is_write)
            data = parsed.get("data", parsed)
            return _CliResult(succeeded=isinstance(data, dict), data=data if isinstance(data, dict) else None)
        return _CliResult(succeeded=False, error="TaskMarket CLI rejected the command.", ambiguous=is_write)

    def _request_json(
        self, method: str, path: str, *, params: dict[str, str] | None = None, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if self._transport is not None:
            return self._transport(method, path, params or {}, body)
        try:
            with httpx.Client(base_url=self.api_url, timeout=self.timeout, follow_redirects=False) as client:
                response = client.request(method, path, params=params, json=body)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return {"error": f"TaskMarket request failed: {exc}", "retry": True}
        if not isinstance(payload, dict):
            return {"error": "TaskMarket returned a non-object JSON response.", "retry": True}
        return payload

    def _utc_now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None:
            return now.replace(tzinfo=timezone.utc)
        return now.astimezone(timezone.utc)

    @staticmethod
    def _normalise_request(
        *, description: str, reward_usdc: str, duration_hours: float, mode: TaskMode, tags: list[str] | None
    ) -> dict[str, Any]:
        if not isinstance(description, str) or not description.strip():
            raise ValueError("`description` must not be empty")
        try:
            reward = Decimal(str(reward_usdc).strip())
        except (InvalidOperation, ValueError):
            raise ValueError("`reward_usdc` must be a positive USDC amount") from None
        if not reward.is_finite() or reward <= 0:
            raise ValueError("`reward_usdc` must be a positive USDC amount")
        exponent = reward.as_tuple().exponent
        if isinstance(exponent, str) or exponent < -6:
            raise ValueError("`reward_usdc` supports at most 6 decimal places")
        reward = reward.quantize(USDC_QUANTUM)
        try:
            duration = Decimal(str(duration_hours))
        except (InvalidOperation, ValueError):
            raise ValueError("`duration_hours` must be a positive finite number") from None
        if not duration.is_finite() or duration <= 0:
            raise ValueError("`duration_hours` must be a positive finite number")
        if not isinstance(mode, str) or mode not in SUPPORTED_MODES:
            raise ValueError("`mode` must be one of: bounty, claim, pitch, benchmark")
        return {
            "description": description.strip(),
            "rewardUsdc": _format_usdc(reward),
            "durationHours": _format_decimal(duration),
            "mode": mode,
            "tags": [tag.strip() for tag in tags or [] if tag.strip()],
        }


def _validate_task_id(task_id: str) -> str | None:
    if not isinstance(task_id, str) or re.fullmatch(r"0x[0-9a-fA-F]{64}", task_id) is None:
        return "`task_id` must be a 0x-prefixed 32-byte TaskMarket task ID"
    return None


def _format_usdc(value: Decimal) -> str:
    return f"{value.quantize(USDC_QUANTUM):.6f}"


def _format_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _format_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _maximum_spend(reward: Decimal) -> Decimal:
    fee = reward * PLATFORM_FEE_BPS / Decimal("10000")
    return (reward + fee + RELAY_FEE_USDC).quantize(USDC_QUANTUM, rounding=ROUND_CEILING)


def _confirmation_digest(request: dict[str, Any]) -> str:
    encoded = json.dumps(request, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _task_id_from_cli_result(data: dict[str, Any] | None) -> str | None:
    if data is None:
        return None
    task_id = data.get("taskId")
    return task_id if isinstance(task_id, str) and re.fullmatch(r"0x[0-9a-fA-F]{64}", task_id) else None
