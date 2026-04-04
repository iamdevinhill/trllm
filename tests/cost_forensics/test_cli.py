"""Tests for the CLI entry point."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from pyrapide import Computation, Event

from trllm.cost_forensics.cli import main


def _write_comp(path: Path, comp: Computation) -> None:
    path.write_text(json.dumps(comp.to_dict()))


def _make_comp() -> Computation:
    comp = Computation()
    root = Event(name="user_request", payload={})
    llm = Event(
        name="llm_call",
        payload={
            "model": "gpt-4o",
            "usage": {"input_tokens": 5000, "output_tokens": 500},
        },
    )
    comp.record(root)
    comp.record(llm, caused_by=[root])
    return comp


class TestCLIAnalyze:
    def test_analyze_text_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        comp_file = tmp_path / "comp.json"
        _write_comp(comp_file, _make_comp())
        monkeypatch.setattr(
            "sys.argv",
            ["trllm-cost-forensics", "analyze", str(comp_file)],
        )
        main()
        captured = capsys.readouterr()
        assert "user_request" in captured.out
        assert "$" in captured.out

    def test_analyze_json_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        comp_file = tmp_path / "comp.json"
        _write_comp(comp_file, _make_comp())
        monkeypatch.setattr(
            "sys.argv",
            ["trllm-cost-forensics", "analyze", str(comp_file), "--json"],
        )
        main()
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "total_cost" in data
        assert "waste" in data

    def test_analyze_with_anthropic_provider(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        comp_file = tmp_path / "comp.json"
        _write_comp(comp_file, _make_comp())
        monkeypatch.setattr(
            "sys.argv",
            [
                "trllm-cost-forensics",
                "analyze",
                str(comp_file),
                "--provider",
                "anthropic",
            ],
        )
        main()
        captured = capsys.readouterr()
        assert "user_request" in captured.out

    def test_analyze_budget_pass(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        comp_file = tmp_path / "comp.json"
        _write_comp(comp_file, _make_comp())
        monkeypatch.setattr(
            "sys.argv",
            [
                "trllm-cost-forensics",
                "analyze",
                str(comp_file),
                "--budget",
                "10.0",
            ],
        )
        # Should not raise or exit
        main()

    def test_analyze_budget_exceeded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        comp_file = tmp_path / "comp.json"
        _write_comp(comp_file, _make_comp())
        monkeypatch.setattr(
            "sys.argv",
            [
                "trllm-cost-forensics",
                "analyze",
                str(comp_file),
                "--budget",
                "0.0001",
            ],
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


class TestCLIDiff:
    def test_diff_text_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        before_file = tmp_path / "before.json"
        after_file = tmp_path / "after.json"
        _write_comp(before_file, _make_comp())
        _write_comp(after_file, _make_comp())
        monkeypatch.setattr(
            "sys.argv",
            [
                "trllm-cost-forensics",
                "diff",
                str(before_file),
                str(after_file),
            ],
        )
        main()
        captured = capsys.readouterr()
        assert "Cost diff" in captured.out

    def test_diff_json_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        before_file = tmp_path / "before.json"
        after_file = tmp_path / "after.json"
        _write_comp(before_file, _make_comp())
        _write_comp(after_file, _make_comp())
        monkeypatch.setattr(
            "sys.argv",
            [
                "trllm-cost-forensics",
                "diff",
                str(before_file),
                str(after_file),
                "--json",
            ],
        )
        main()
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "before_total" in data
        assert "after_total" in data
