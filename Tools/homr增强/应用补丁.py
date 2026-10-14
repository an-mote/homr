"""从固定上游创建独立增强源码；拒绝改变原目录或覆盖已有输出。"""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import tempfile

HERE = pathlib.Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(baseline, models, output):
    baseline, models, output = (pathlib.Path(p).resolve() for p in (baseline, models, output))
    if output.exists():
        raise ValueError("输出目录已存在，请使用新的目录，原结果不会被覆盖")
    lock = json.loads((HERE / "版本锁.json").read_text())
    for item in lock["baseline_files"]:
        path = baseline / item["path"]
        if not path.is_file() or sha(path) != item["sha256"]:
            raise ValueError("上游源码与固定版本不一致：" + item["path"])
    patch_file = HERE / "修复.patch"
    if sha(patch_file) != lock["patch_sha256"]:
        raise ValueError("补丁哈希不一致")
    for item in lock["models"]:
        path = models / item["path"]
        if not path.is_file() or sha(path) != item["sha256"]:
            raise ValueError("模型缺失或哈希不一致：" + item["path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="homr-prepare-", dir=output.parent) as temporary:
        source = pathlib.Path(temporary) / "source"
        shutil.copytree(baseline, source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.onnx", ".git"))
        # The upstream staff parser uses CRLF. Normalize only patched files for
        # both macOS BSD patch and Linux patch, then restore the locked endings.
        for item in lock["patched_files"]:
            target = source / item["path"]
            if target.exists():
                target.write_text(target.read_text(), newline="\n")
        applied = subprocess.run(["patch", "-p1", "-i", str(patch_file)], cwd=source,
                                 capture_output=True, text=True)
        if applied.returncode != 0:
            raise RuntimeError("补丁应用失败：\n" + applied.stdout + applied.stderr)
        for item in lock["patched_files"]:
            target = source / item["path"]
            if item.get("newline") == "CRLF":
                target.write_text(target.read_text(), newline="\r\n")
            if sha(target) != item["sha256"]:
                raise ValueError("应用后的源码哈希不一致：" + item["path"])
        for item in lock["models"]:
            destination = source / item["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(models / item["path"], destination)
        files = [{"path": p.relative_to(source).as_posix(), "sha256": sha(p)}
                 for p in sorted(source.rglob("*")) if p.is_file()]
        if files != lock["prepared_files"]:
            raise ValueError("构建后的完整源码清单与版本锁不一致")
        record = {"upstream_commit": lock["upstream_commit"], "patch_sha256": lock["patch_sha256"],
                  "image_id": lock["image_id"], "files": files}
        (source / "增强版本.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        shutil.move(str(source), output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--models", required=True, help="包含 homr/segmentation 和 homr/transformer 的模型根目录")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(prepare(args.baseline, args.models, args.output))


if __name__ == "__main__":
    main()
