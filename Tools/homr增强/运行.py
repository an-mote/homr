"""离线运行固定 homr 增强版，保留输入、原始 XML、日志与失败状态。"""
import argparse
import datetime
import hashlib
import json
import pathlib
import shutil
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET

HERE = pathlib.Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_source(source):
    lock = json.loads((HERE / "版本锁.json").read_text())
    record = json.loads((source / "增强版本.json").read_text())
    if (record["upstream_commit"], record["patch_sha256"], record["image_id"]) != (
        lock["upstream_commit"], lock["patch_sha256"], lock["image_id"]
    ):
        raise ValueError("增强源码的补丁或镜像版本不一致")
    if record["files"] != lock["prepared_files"]:
        raise ValueError("增强源码清单与版本锁不一致")
    actual_paths = {p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()
                    and "__pycache__" not in p.parts and p.name != "增强版本.json"}
    if actual_paths != {item["path"] for item in lock["prepared_files"]}:
        raise ValueError("增强源码存在额外或缺失文件")
    for item in lock["prepared_files"]:
        path = source / item["path"]
        if not path.is_file() or sha(path) != item["sha256"]:
            raise ValueError("增强源码或模型已变化：" + item["path"])
    return record


def validate_xml(path):
    root = ET.parse(path).getroot()
    def local(element):
        return element.tag.rsplit("}", 1)[-1]
    if local(root) != "score-partwise":
        raise ValueError("MusicXML 根节点无效")
    notes = [e for e in root.iter() if local(e) == "note"]
    if not notes:
        raise ValueError("MusicXML 没有音符或休止符")
    divisions = [e for e in root.iter() if local(e) == "divisions"]
    if not divisions or any(int(e.text or "0") <= 0 for e in divisions):
        raise ValueError("MusicXML divisions 缺失或无效")
    for note in notes:
        if any(local(e) == "grace" for e in note):
            continue
        durations = [e for e in note if local(e) == "duration"]
        if not durations or any(int(e.text or "0") <= 0 for e in durations):
            raise ValueError("MusicXML 含缺失或非正时值")


def run(source, input_path, output, timeout=600):
    source, input_path, output = (pathlib.Path(p).resolve() for p in (source, input_path, output))
    if not input_path.is_file():
        raise ValueError("输入文件不存在")
    if output.exists():
        raise ValueError("输出目录已存在，请为本次运行选择新目录")
    record = validate_source(source)
    image_id = subprocess.check_output(["docker", "image", "inspect", record["image_id"],
                                        "--format", "{{.Id}}"], text=True).strip()
    if image_id != record["image_id"]:
        raise ValueError("固定的 Docker 镜像不存在")
    output.mkdir(parents=True)
    copied = output / ("input" + input_path.suffix.lower())
    shutil.copyfile(input_path, copied)
    name = "fengpiano-homr-enhanced-" + uuid.uuid4().hex[:12]
    command = ["docker", "run", "--rm", "--name", name, "--platform", "linux/amd64",
               "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=1g",
               "--cpuset-cpus", "0-7", "--cpus", "8", "--memory", "7g", "--pids-limit", "512",
               "-e", "OMP_NUM_THREADS=4", "-e", "OPENBLAS_NUM_THREADS=4", "-e", "MKL_NUM_THREADS=4",
               "-v", str(source) + ":/opt/homr:ro", "-v", str(HERE) + ":/enhancement:ro",
               "-v", str(output) + ":/work:rw", "--workdir", "/work", "--entrypoint", "python",
               image_id, "/enhancement/容器推理.py", "/work/" + copied.name]
    started = time.monotonic()
    result = {"input_sha256": sha(input_path), "patch_sha256": record["patch_sha256"],
              "image_id": image_id, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "timeout_seconds": timeout, "command": command, "musicxml_paths": []}
    with (output / "engine.log").open("wb") as log:
        try:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            result["exit_code"] = process.returncode
            result["status"] = "success" if process.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            result.update(exit_code=None, status="timeout")
            try:
                cleanup = subprocess.run(["docker", "rm", "--force", name], capture_output=True, timeout=15)
                if cleanup.returncode != 0:
                    result["cleanup_error"] = "Docker did not confirm timed-out container removal"
            except (subprocess.TimeoutExpired, OSError) as error:
                result["cleanup_error"] = f"{type(error).__name__}: {error}"
        except OSError as error:
            result.update(exit_code=None, status="failed", reason=str(error))
    result["elapsed_seconds"] = time.monotonic() - started
    paths = sorted(output.rglob("*.musicxml"))
    result["musicxml_paths"] = [str(p) for p in paths]
    if result["status"] == "success":
        try:
            if not paths:
                raise ValueError("引擎没有生成 MusicXML")
            for path in paths:
                validate_xml(path)
        except (ValueError, ET.ParseError) as error:
            result.update(status="invalid_output", reason=str(error))
    result["output_sha256"] = {p.relative_to(output).as_posix(): sha(p) for p in paths}
    result["needs_musical_review"] = True
    runtime = output / "runtime.json"
    if runtime.is_file():
        try:
            result["linux_runtime"] = json.loads(runtime.read_text())
        except (json.JSONDecodeError, OSError) as error:
            result["runtime_error"] = f"{type(error).__name__}: {error}"
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout 必须大于零")
    result = run(args.source, args.input, args.output, args.timeout)
    print(json.dumps({k: result[k] for k in ("status", "elapsed_seconds", "musicxml_paths")}, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()
