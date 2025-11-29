"""Basic import tests to ensure package bootstrap succeeds."""

from llm_lab import ExperimentPaths, __version__
from llm_lab.posttrain_llm import PostTrainingConfig
from llm_lab.pretrain_llm import PretrainingConfig


def test_version_semver() -> None:
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_configs_instantiation(tmp_path) -> None:
    pre_cfg = PretrainingConfig()
    post_cfg = PostTrainingConfig()
    paths = ExperimentPaths(project_root=tmp_path)
    assert pre_cfg.output_dir.exists()
    assert post_cfg.output_dir.exists()
    assert paths.project_root == tmp_path
