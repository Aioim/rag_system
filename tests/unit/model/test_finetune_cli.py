"""CLI 单元测试 — 参数解析 + 命令路由"""

import tempfile
import json
from pathlib import Path

import pytest

from model.finetune.cli import _build_parser


class TestCliParser:
    def test_embedding_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["embedding", "--data", "data/triplets.jsonl"])
        assert args.command == "embedding"
        assert args.data == Path("data/triplets.jsonl")

    def test_embedding_with_name(self):
        parser = _build_parser()
        args = parser.parse_args([
            "embedding", "--data", "data/triplets.jsonl",
            "--name", "my-emb", "--epochs", "5",
        ])
        assert args.name == "my-emb"
        assert args.epochs == 5

    def test_llm_with_teacher(self):
        parser = _build_parser()
        args = parser.parse_args([
            "llm", "--data", "data/instructions.jsonl",
            "--teacher", "claude-sonnet-5", "--alpha", "0.3",
        ])
        assert args.command == "llm"
        assert args.teacher == "claude-sonnet-5"
        assert args.alpha == 0.3

    def test_llm_generate_only(self):
        parser = _build_parser()
        args = parser.parse_args([
            "llm", "--data", "data/instructions.jsonl",
            "--teacher", "claude-sonnet-5", "--generate-only",
        ])
        assert args.generate_only is True

    def test_remove_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["remove", "--name", "my-adapter"])
        assert args.command == "remove"
        assert args.name == "my-adapter"

    def test_info_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["info", "--name", "my-adapter"])
        assert args.command == "info"
        assert args.name == "my-adapter"

    def test_list_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["list"])
        assert args.command == "list"

    def test_missing_command_raises(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_reranker_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["reranker", "--data", "data/rerank.jsonl"])
        assert args.command == "reranker"


class TestApplyOverrides:
    def test_overrides_apply(self):
        from model.finetune.config import FinetuneConfig
        from model.finetune.cli import _apply_overrides
        import argparse

        config = FinetuneConfig()
        parser = _build_parser()
        args = parser.parse_args([
            "embedding", "--data", "test.jsonl",
            "--epochs", "10", "--batch-size", "4",
        ])
        config = _apply_overrides(config, args)

        assert config.training.epochs == 10
        assert config.training.batch_size == 4
