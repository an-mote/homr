"""容器内记录同一识谱进程的退出码、耗时、峰值内存。"""
import json
import os
import pathlib
import resource
import runpy
import sys
import time

sys.path.insert(0, "/opt/homr")
output = pathlib.Path("/work")
sys.argv = ["homr", sys.argv[1], "--gpu", "no"]
start = time.monotonic()
code = 0
try:
    runpy.run_module("homr.main", run_name="__main__")
except SystemExit as error:
    code = error.code or 0
except BaseException:
    code = 1
    raise
finally:
    (output / "runtime.json").write_text(json.dumps({
        "elapsed_seconds": time.monotonic() - start,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_rss_scope": "Linux homr process including native allocations",
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "exit_code": code,
    }, indent=2) + "\n")
sys.exit(code)
