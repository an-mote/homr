"""用临时目录和模拟 Docker 验证版本固定、失败记录、原始输出保留。"""
import hashlib
import importlib.util
import json
import pathlib
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = pathlib.Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("homr_enhanced_runner", "运行.py")
prepare = load("homr_enhanced_prepare", "应用补丁.py")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def valid_xml(duration="4", divisions="4", grace=False, namespace=False):
    duration_tag = "" if duration is None else f"<duration>{duration}</duration>"
    grace_tag = "<grace/>" if grace else ""
    divisions_tag = "" if divisions is None else f"<divisions>{divisions}</divisions>"
    namespace_tag = ' xmlns="http://www.musicxml.org/ns/musicxml"' if namespace else ""
    return (
        f"<score-partwise{namespace_tag}><part-list/><part id='P1'><measure number='1'>"
        f"<attributes>{divisions_tag}</attributes><note>{grace_tag}<rest/>"
        f"{duration_tag}</note></measure></part></score-partwise>"
    )


class ToolFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="homr-tool-tests-")
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name).resolve()
        self.tool = self.root / "tools"
        self.tool.mkdir()
        self.baseline = self.root / "baseline"
        self.baseline.mkdir()
        self.models = self.root / "models"
        self.models.mkdir()
        self.source = self.root / "prepared"
        self.source.mkdir()
        self.input = self.root / "original.PNG"
        self.input.write_bytes(b"original input bytes")
        original = b"VALUE = 1\n"
        enhanced = b"VALUE = 2\n"
        model = b"offline model bytes"
        (self.baseline / "pkg.py").write_bytes(original)
        (self.source / "pkg.py").write_bytes(enhanced)
        (self.models / "model.onnx").write_bytes(model)
        (self.source / "model.onnx").write_bytes(model)
        patch_data = (
            "--- a/pkg.py\n+++ b/pkg.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n"
        ).encode()
        (self.tool / "修复.patch").write_bytes(patch_data)
        self.lock = {
            "upstream_commit": "fixed-upstream-commit",
            "image_id": "sha256:" + "a" * 64,
            "patch_sha256": digest(patch_data),
            "baseline_files": [{"path": "pkg.py", "sha256": digest(original)}],
            "patched_files": [{"path": "pkg.py", "sha256": digest(enhanced)}],
            "models": [{"path": "model.onnx", "sha256": digest(model)}],
            "prepared_files": [
                {"path": "model.onnx", "sha256": digest(model)},
                {"path": "pkg.py", "sha256": digest(enhanced)},
            ],
        }
        self.record = {key: self.lock[key] for key in ("upstream_commit", "patch_sha256", "image_id")}
        self.record["files"] = list(self.lock["prepared_files"])
        self.write_lock()
        self.write_record()
        self.addCleanup(patch.stopall)
        patch.object(runner, "HERE", self.tool).start()
        patch.object(prepare, "HERE", self.tool).start()
        self.inspect = patch.object(
            runner.subprocess, "check_output", return_value=self.lock["image_id"] + "\n"
        ).start()

    def write_lock(self):
        (self.tool / "版本锁.json").write_text(json.dumps(self.lock))

    def write_record(self):
        (self.source / "增强版本.json").write_text(json.dumps(self.record))

    def run_fake(self, exit_code=0, outputs=None, runtime=None, failure=None, cleanup_failure=None):
        output = self.root / "run-output"
        calls = []

        def execute(command, **kwargs):
            calls.append(command)
            if command[:2] == ["docker", "run"]:
                kwargs["stdout"].write(b"raw engine log\n")
                for relative, content in (outputs or {}).items():
                    target = output / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content)
                if runtime is not None:
                    (output / "runtime.json").write_text(runtime)
                if failure is not None:
                    raise failure
                return subprocess.CompletedProcess(command, exit_code)
            if command[:3] == ["docker", "rm", "--force"]:
                if cleanup_failure is not None:
                    raise cleanup_failure
                return subprocess.CompletedProcess(command, 0)
            self.fail("意外调用外部命令：" + str(command))

        with patch.object(runner.subprocess, "run", side_effect=execute):
            result = runner.run(self.source, self.input, output, timeout=3)
        saved = json.loads((output / "result.json").read_text())
        self.assertEqual(result, saved)
        self.assertEqual(b"original input bytes", (output / "input.png").read_bytes())
        self.assertEqual(b"raw engine log\n", (output / "engine.log").read_bytes())
        self.assertTrue(result["needs_musical_review"])
        self.assertGreaterEqual(result["elapsed_seconds"], 0)
        return result, output, calls


class VersionTests(ToolFixture):
    def test_valid_source_matches_locked_manifest(self):
        self.assertEqual(self.record, runner.validate_source(self.source))

    def test_locked_version_fields_cannot_change(self):
        for key in ("upstream_commit", "patch_sha256", "image_id"):
            with self.subTest(key=key):
                original = self.record[key]
                self.record[key] = "other version"
                self.write_record()
                with self.assertRaises(ValueError):
                    runner.validate_source(self.source)
                self.record[key] = original

    def test_changed_source_or_model_is_rejected(self):
        for filename in ("pkg.py", "model.onnx"):
            with self.subTest(filename=filename):
                path = self.source / filename
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                with self.assertRaises(ValueError):
                    runner.validate_source(self.source)
                path.write_bytes(original)

    def test_manifest_cannot_hide_changed_source(self):
        (self.source / "pkg.py").write_text("tampered")
        self.record["files"] = [item for item in self.record["files"] if item["path"] != "pkg.py"]
        self.write_record()
        with self.assertRaises(ValueError):
            runner.validate_source(self.source)

    def test_extra_source_is_rejected(self):
        (self.source / "new.py").write_text("unexpected code")
        with self.assertRaises(ValueError):
            runner.validate_source(self.source)

    def test_generated_python_cache_is_allowed(self):
        cache = self.source / "__pycache__"
        cache.mkdir()
        (cache / "pkg.cpython-312.pyc").write_bytes(b"bytecode")
        runner.validate_source(self.source)

    def test_prepare_applies_patch_and_records_all_files(self):
        output = self.root / "fresh-prepared"
        result = prepare.prepare(self.baseline, self.models, output)
        self.assertEqual(output, result)
        self.assertEqual(b"VALUE = 2\n", (output / "pkg.py").read_bytes())
        self.assertEqual(b"offline model bytes", (output / "model.onnx").read_bytes())
        self.assertEqual(b"VALUE = 1\n", (self.baseline / "pkg.py").read_bytes())
        self.assertEqual(self.lock["prepared_files"], json.loads((output / "增强版本.json").read_text())["files"])

    def test_prepare_rejects_changed_baseline_patch_or_model_before_output(self):
        for target in (self.baseline / "pkg.py", self.tool / "修复.patch", self.models / "model.onnx"):
            with self.subTest(target=target.name):
                original = target.read_bytes()
                target.write_bytes(original + b"changed")
                output = self.root / "fresh-prepared"
                with self.assertRaises(ValueError):
                    prepare.prepare(self.baseline, self.models, output)
                self.assertFalse(output.exists())
                target.write_bytes(original)

    def test_prepare_never_overwrites_existing_output(self):
        output = self.root / "already-prepared"
        output.mkdir()
        (output / "kept.txt").write_text("keep")
        with self.assertRaises(ValueError):
            prepare.prepare(self.baseline, self.models, output)
        self.assertEqual("keep", (output / "kept.txt").read_text())


class OutputTests(ToolFixture):
    def test_valid_xml_success_uses_fixed_image_and_no_network(self):
        result, _, calls = self.run_fake(outputs={"input.musicxml": valid_xml()})
        self.assertEqual("success", result["status"])
        self.assertEqual(0, result["exit_code"])
        command = calls[0]
        self.assertEqual("none", command[command.index("--network") + 1])
        self.assertIn(self.lock["image_id"], command)
        self.assertIn("--read-only", command)
        self.assertIn(str(self.source) + ":/opt/homr:ro", command)
        self.assertEqual(digest(valid_xml().encode()), result["output_sha256"]["input.musicxml"])

    def test_exit_nonzero_preserves_partial_xml_and_failure(self):
        result, _, _ = self.run_fake(exit_code=1, outputs={"input.musicxml": valid_xml()})
        self.assertEqual("failed", result["status"])
        self.assertEqual(1, result["exit_code"])
        self.assertEqual(1, len(result["musicxml_paths"]))

    def test_nested_partial_xml_is_reported(self):
        result, output, _ = self.run_fake(exit_code=1, outputs={"pages/page-1.musicxml": valid_xml()})
        self.assertEqual("failed", result["status"])
        self.assertIn(str(output / "pages/page-1.musicxml"), result["musicxml_paths"])
        self.assertIn("pages/page-1.musicxml", result["output_sha256"])

    def test_exit_zero_without_xml_is_invalid_output(self):
        result, _, _ = self.run_fake()
        self.assertEqual("invalid_output", result["status"])
        self.assertIn("MusicXML", result["reason"])

    def test_malformed_xml_is_preserved_and_reported(self):
        result, output, _ = self.run_fake(outputs={"input.musicxml": "<score-partwise>"})
        self.assertEqual("invalid_output", result["status"])
        self.assertEqual("<score-partwise>", (output / "input.musicxml").read_text())

    def test_nonpositive_divisions_and_duration_are_rejected(self):
        for divisions, duration in ((None, "4"), ("0", "4"), ("-1", "4"), ("4", None), ("4", "0"), ("4", "-1")):
            with self.subTest(divisions=divisions, duration=duration):
                path = self.root / "check.musicxml"
                path.write_text(valid_xml(duration=duration, divisions=divisions))
                with self.assertRaises(ValueError):
                    runner.validate_xml(path)

    def test_grace_without_duration_and_namespaced_musicxml_are_valid(self):
        for xml in (valid_xml(grace=True, duration=None), valid_xml(namespace=True)):
            path = self.root / "check.musicxml"
            path.write_text(xml)
            runner.validate_xml(path)

    def test_timeout_removes_container_and_saves_result(self):
        result, _, calls = self.run_fake(failure=subprocess.TimeoutExpired("docker", 3))
        self.assertEqual("timeout", result["status"])
        self.assertIsNone(result["exit_code"])
        self.assertEqual(["docker", "rm", "--force"], calls[1][:3])
        self.assertEqual(calls[0][calls[0].index("--name") + 1], calls[1][3])

    def test_timeout_cleanup_failure_does_not_lose_result(self):
        result, _, _ = self.run_fake(
            failure=subprocess.TimeoutExpired("docker", 3),
            cleanup_failure=subprocess.TimeoutExpired("docker rm", 15),
        )
        self.assertEqual("timeout", result["status"])
        self.assertIn("TimeoutExpired", result["cleanup_error"])

    def test_docker_spawn_failure_does_not_lose_result(self):
        result, _, _ = self.run_fake(failure=OSError("Docker unavailable"))
        self.assertEqual("failed", result["status"])
        self.assertIn("Docker unavailable", result["reason"])

    def test_invalid_runtime_json_keeps_music_result_and_records_error(self):
        result, _, _ = self.run_fake(outputs={"input.musicxml": valid_xml()}, runtime="{broken")
        self.assertEqual("success", result["status"])
        self.assertIn("JSONDecodeError", result["runtime_error"])

    def test_linux_runtime_measurements_retain_exact_scope(self):
        runtime = {"elapsed_seconds": 1.25, "peak_rss_kib": 2048,
                   "peak_rss_scope": "Linux homr process including native allocations", "exit_code": 0}
        result, _, _ = self.run_fake(outputs={"input.musicxml": valid_xml()}, runtime=json.dumps(runtime))
        self.assertEqual(runtime, result["linux_runtime"])

    def test_run_never_overwrites_existing_output(self):
        output = self.root / "existing-run"
        output.mkdir()
        (output / "keep.txt").write_text("keep")
        with self.assertRaises(ValueError):
            runner.run(self.source, self.input, output)
        self.assertEqual("keep", (output / "keep.txt").read_text())
        self.inspect.assert_not_called()


class ContainerTests(ToolFixture):
    def run_container(self, error=None, expected_exception=SystemExit):
        output = self.root / "container-work"
        output.mkdir()
        arguments = [str(HERE / "容器推理.py"), "/work/input.png"]
        with patch.object(sys, "argv", arguments), patch.object(sys, "path", list(sys.path)), patch(
            "pathlib.Path", return_value=output
        ), patch("os.sched_getaffinity", return_value={0, 1}, create=True), patch(
            "runpy.run_module", side_effect=error
        ) as inference:
            with self.assertRaises(expected_exception) as exit_context:
                runpy.run_path(str(HERE / "容器推理.py"), run_name="__main__")
        inference.assert_called_once_with("homr.main", run_name="__main__")
        return getattr(exit_context.exception, "code", None), json.loads((output / "runtime.json").read_text())

    def test_container_success_records_same_process_runtime(self):
        code, runtime = self.run_container()
        self.assertEqual(0, code)
        self.assertEqual(0, runtime["exit_code"])
        self.assertIn("including native allocations", runtime["peak_rss_scope"])
        self.assertGreaterEqual(runtime["peak_rss_kib"], 0)
        self.assertEqual([0, 1], runtime["cpu_affinity"])

    def test_container_preserves_nonzero_engine_exit(self):
        code, runtime = self.run_container(SystemExit(2))
        self.assertEqual(2, code)
        self.assertEqual(2, runtime["exit_code"])

    def test_container_exception_keeps_runtime_before_propagating_failure(self):
        _, runtime = self.run_container(RuntimeError("inference failed"), RuntimeError)
        self.assertEqual(1, runtime["exit_code"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
