from pathlib import Path

from scripts.project_size import list_project_files, measure_file, summarize


def test_project_size_ignores_ref_outputs_and_caches(tmp_path: Path) -> None:
    (tmp_path / "wavelet").mkdir()
    (tmp_path / "wavelet" / "app.py").write_text("def run():\n    return 1\n")
    for ignored in ("ref", "outputs", ".venv", "__pycache__"):
        directory = tmp_path / ignored
        directory.mkdir()
        (directory / "ignored.py").write_text("def ignored():\n    return 2\n")

    files = list_project_files(tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in files] == [
        "wavelet/app.py"
    ]


def test_project_size_summarizes_python_complexity(tmp_path: Path) -> None:
    source_dir = tmp_path / "wavelet" / "trainer"
    source_dir.mkdir(parents=True)
    source = source_dir / "module.py"
    source.write_text(
        "\n".join(
            [
                "# comment",
                "",
                "class Runner:",
                "    def run(self, flag):",
                "        if flag:",
                "            return 1",
                "        return 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = measure_file(tmp_path, source)
    summary = summarize([metrics], timestamp="2026-06-04T00:00:00+00:00", top=1)

    assert summary["files"] == 1
    assert summary["lines"] == 7
    assert summary["source_lines"] == 5
    assert summary["comment_lines"] == 1
    assert summary["blank_lines"] == 1
    assert summary["python_functions"] == 1
    assert summary["python_classes"] == 1
    assert summary["python_complexity_points"] == 1
    assert summary["categories"] == {
        "trainer": {
            "files": 1,
            "lines": 7,
            "source_lines": 5,
            "python_functions": 1,
            "python_classes": 1,
            "python_complexity_points": 1,
        }
    }
    assert summary["core"] == {
        "files": 1,
        "lines": 7,
        "source_lines": 5,
        "python_functions": 1,
        "python_classes": 1,
        "python_complexity_points": 1,
    }
    assert summary["largest_files"] == [
        {"path": "wavelet/trainer/module.py", "lines": 7, "source_lines": 5}
    ]
