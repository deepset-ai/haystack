from datetime import datetime, timezone
from typing import Any

from haystack.tools import TaskMarketToolset
from haystack.tools.taskmarket import USDC_CONTRACT, _CliResult

TASK_ID = "0x" + "a" * 64


def _tool(toolset: TaskMarketToolset, name: str) -> Any:
    return next(tool for tool in toolset.get_selectable_tools() if tool.name == name)


def test_taskmarket_toolset_lists_live_tasks() -> None:
    calls: list[tuple[str, str, dict[str, str], Any]] = []

    def transport(method: str, path: str, params: dict[str, str], body: Any) -> dict[str, Any]:
        calls.append((method, path, params, body))
        return {"tasks": [{"id": "0xabc", "status": "open"}]}

    toolset = TaskMarketToolset(transport=transport)
    list_tasks = _tool(toolset, "taskmarket_list_tasks")

    result = list_tasks.invoke(status="open", limit=1)

    assert result == {"tasks": [{"id": "0xabc", "status": "open"}]}
    assert calls == [("GET", "/api/tasks", {"status": "open", "limit": "1"}, None)]


def test_taskmarket_toolset_reads_task_and_submissions() -> None:
    calls: list[tuple[str, str, dict[str, str], Any]] = []

    def transport(method: str, path: str, params: dict[str, str], body: Any) -> dict[str, Any]:
        calls.append((method, path, params, body))
        return {"path": path}

    toolset = TaskMarketToolset(transport=transport)

    assert _tool(toolset, "taskmarket_get_task").invoke(task_id=TASK_ID) == {"path": f"/api/tasks/{TASK_ID}"}
    assert _tool(toolset, "taskmarket_list_submissions").invoke(task_id=TASK_ID) == {
        "path": f"/api/tasks/{TASK_ID}/submissions"
    }
    assert calls == [("GET", f"/api/tasks/{TASK_ID}", {}, None), ("GET", f"/api/tasks/{TASK_ID}/submissions", {}, None)]


def test_taskmarket_toolset_rejects_noncanonical_task_ids() -> None:
    calls: list[tuple[str, str, dict[str, str], Any]] = []

    def transport(method: str, path: str, params: dict[str, str], body: Any) -> dict[str, Any]:
        calls.append((method, path, params, body))
        return {}

    result = _tool(TaskMarketToolset(transport=transport), "taskmarket_get_task").invoke(task_id="0x../secret")

    assert "32-byte" in result["error"]
    assert result["retry"] is False
    assert calls == []


def test_taskmarket_preview_contains_exact_base_usdc_budget() -> None:
    now = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)
    toolset = TaskMarketToolset(clock=lambda: now)

    result = _tool(toolset, "taskmarket_preview_task").invoke(
        description="Deliver a tested data connector",
        reward_usdc="1.00",
        duration_hours=4,
        mode="bounty",
        tags=["python", "  ", "agents"],
    )

    assert result["rewardUsdc"] == "1.000000"
    assert result["durationHours"] == "4"
    assert result["deadline"] == "2026-08-13T16:00:00Z"
    assert result["maximumSpendUsdc"] == "1.076000"
    assert result["expiresAt"] == "2026-08-13T12:15:00Z"
    assert result["network"] == "Base"
    assert result["chainId"] == 8453
    assert result["currency"] == "USDC"
    assert result["usdcContract"] == USDC_CONTRACT
    assert result["tags"] == ["python", "agents"]
    assert len(result["confirmationToken"]) == 64


def test_taskmarket_create_requires_confirmation_before_any_side_effect() -> None:
    calls: list[tuple[list[str], bool]] = []

    def cli_runner(args: list[str], is_write: bool) -> _CliResult:
        calls.append((args, is_write))
        return _CliResult(succeeded=True, data={"taskId": TASK_ID})

    approvals: list[dict[str, Any]] = []

    def approve(preview: dict[str, Any]) -> bool:
        approvals.append(preview)
        return True

    toolset = TaskMarketToolset(cli_runner=cli_runner, approval=approve)

    result = _tool(toolset, "taskmarket_create_task").invoke(
        description="Deliver a tested data connector",
        reward_usdc="1",
        duration_hours=4,
        confirmation_token="not-previewed",
        confirm=False,
    )

    assert result["retry"] is False
    assert "confirmation-gated" in result["error"]
    assert calls == []
    assert approvals == []


def test_taskmarket_create_runs_one_approved_cli_flow() -> None:
    now = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)
    calls: list[tuple[list[str], bool]] = []

    def cli_runner(args: list[str], is_write: bool) -> _CliResult:
        calls.append((args, is_write))
        if args == ["deposit"]:
            return _CliResult(
                succeeded=True,
                data={"network": "Base", "chainId": 8453, "currency": "USDC", "usdcContract": USDC_CONTRACT},
            )
        if args == ["stats"]:
            return _CliResult(succeeded=True, data={"balanceUsdc": "5"})
        return _CliResult(succeeded=True, data={"taskId": TASK_ID})

    approvals: list[dict[str, Any]] = []

    def approve(preview: dict[str, Any]) -> bool:
        approvals.append(preview)
        return True

    toolset = TaskMarketToolset(clock=lambda: now, cli_runner=cli_runner, approval=approve)
    preview = _tool(toolset, "taskmarket_preview_task").invoke(
        description="Deliver a tested data connector", reward_usdc="1", duration_hours=4
    )

    result = _tool(toolset, "taskmarket_create_task").invoke(
        description="Deliver a tested data connector",
        reward_usdc="1",
        duration_hours=4,
        confirmation_token=preview["confirmationToken"],
        confirm=True,
    )

    assert result == {
        "taskId": TASK_ID,
        "taskUrl": f"https://api.taskmarket.dev/api/tasks/{TASK_ID}",
        "status": "created",
        "retry": False,
    }
    assert len(approvals) == 1
    assert calls[0] == (["deposit"], False)
    assert calls[1] == (["stats"], False)
    assert calls[2][0][:2] == ["task", "create"]
    assert calls[2][1] is True


def test_taskmarket_create_does_not_retry_ambiguous_cli_write() -> None:
    calls: list[tuple[list[str], bool]] = []

    def cli_runner(args: list[str], is_write: bool) -> _CliResult:
        calls.append((args, is_write))
        if args == ["deposit"]:
            return _CliResult(
                succeeded=True,
                data={"network": "Base", "chainId": 8453, "currency": "USDC", "usdcContract": USDC_CONTRACT},
            )
        if args == ["stats"]:
            return _CliResult(succeeded=True, data={"balanceUsdc": "5"})
        return _CliResult(succeeded=False, error="timed out", ambiguous=True)

    toolset = TaskMarketToolset(cli_runner=cli_runner, approval=lambda _preview: True)
    preview = _tool(toolset, "taskmarket_preview_task").invoke(
        description="Deliver a tested data connector", reward_usdc="1", duration_hours=4
    )

    result = _tool(toolset, "taskmarket_create_task").invoke(
        description="Deliver a tested data connector",
        reward_usdc="1",
        duration_hours=4,
        confirmation_token=preview["confirmationToken"],
        confirm=True,
    )

    assert result["status"] == "unknown"
    assert result["retry"] is False
    assert len(calls) == 3
    assert calls[-1][1] is True


def test_taskmarket_rejects_unknown_mode() -> None:
    toolset = TaskMarketToolset()

    result = _tool(toolset, "taskmarket_preview_task").invoke(
        description="Deliver a tested data connector", reward_usdc="1", duration_hours=4, mode="unknown"
    )

    assert "mode" in result["error"]
    assert result["retry"] is False
