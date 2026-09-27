#!/usr/bin/env python3
"""Run one isolated CLI, AgentTurnBench or server evaluation with GPU telemetry.

Distributed workers must already be started on the peers with identical model
and placement arguments. Pass extra TensorSharp arguments after --.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def stop(process):
    if process is None:
        return {"started": False}
    if process.poll() is not None:
        return {"started": True, "requested": False, "forced": False, "exit_code": process.returncode}
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return {"started": True, "requested": False, "forced": False, "exit_code": process.returncode}
    forced = False
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        forced = True
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    return {"started": True, "requested": True, "forced": forced, "exit_code": process.returncode}


def shutdown_failures(shutdown, log, allow_sigterm=False):
    """A successful HTTP response cannot hide a crash while disposing the model."""
    failures = []
    if shutdown.get("started"):
        # A worker may have no SIGTERM handler; that is an intentional stop.
        expected = (0, -signal.SIGTERM) if allow_sigterm and shutdown.get("requested") else (0,)
        if shutdown.get("forced") or shutdown.get("exit_code") not in expected:
            failures.append("Model process did not shut down cleanly: " + json.dumps(shutdown))
    for line in log.splitlines():
        if (("GGML_ASSERT(" in line and "failed" in line) or "GGML_ABORT" in line
                or "Unhandled exception." in line):
            failures.append(line)
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--dotnet", default="/workspace/dotnet/dotnet")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--model-verification-report", type=Path, help="Completed full-shard download verification report")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--server", action="store_true")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--agent-turn", action="store_true", help="Run the scheduler benchmark; extra arguments select scenarios and drafter")
    p.add_argument("--port", type=int, default=5100)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--timeout", type=int, default=1200)
    p.add_argument("--startup-timeout", type=int, default=180)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--warmup-prefill", type=int, default=512)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--eval-max-tokens", type=int, default=48)
    p.add_argument("--reasoning-effort", choices=("low", "medium", "high"), help="Optional effort sent to text and media HTTP requests")
    p.add_argument("--eval-cases", help="Comma-separated server probe names; omitted runs all eight")
    p.add_argument("--media-fixtures", type=Path, help="Run the existing image/video evaluator after text, in the same server")
    p.add_argument("--media-projector", type=Path, help="Projector passed to the server's --mmproj and hashed in evidence")
    p.add_argument("--media-scenarios", default="image_ocr,multi_image,image_follow_up")
    p.add_argument("--media-max-tokens", type=int, default=256)
    p.add_argument("--dump-logits", action="store_true", help="Dump the first non-warmup forward's F32 logits")
    p.add_argument("--initial-kv-tokens", type=int, help="Explicit initial GPU KV capacity, for growth validation")
    p.add_argument("args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if sum((a.server, a.worker, a.agent_turn)) > 1:
        p.error("--server, --worker and --agent-turn are mutually exclusive")
    if bool(a.media_fixtures) != bool(a.media_projector) or (a.media_fixtures and not a.server):
        p.error("media-fixtures and media-projector must be supplied together with --server")
    a.repo = a.repo.resolve()
    a.output = a.output.resolve()
    if a.baseline:
        a.baseline = a.baseline.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    extra = a.args[1:] if a.args[:1] == ["--"] else a.args
    if a.media_projector and any(arg.split('=', 1)[0] == '--mmproj' for arg in extra):
        p.error("--media-projector supplies --mmproj; do not specify both")
    project = "AgentTurnBench" if a.agent_turn else "TensorSharp.Server.Host" if a.server else "TensorSharp.Cli"
    binary = (a.repo / "benchmarks/AgentTurnBench/bin/Release/net10.0") if a.agent_turn else a.repo / project / "bin"
    command = [a.dotnet, str(binary / (project + ".dll")),
               "--model", str(a.model), "--backend", "ggml_cuda"]
    command += ["--out", str(a.output / "rows.json")] if a.agent_turn else ["--host", "127.0.0.1", "--port", str(a.port)] if a.server else [] if a.worker else [
        "--benchmark", "--bench-prefill", "512", "--bench-decode", "128", "--bench-runs", "3"]
    command += extra
    if a.media_projector:
        a.media_projector = a.media_projector.resolve()
        a.media_fixtures = a.media_fixtures.resolve()
        command += ['--mmproj', str(a.media_projector)]
    env = dict(os.environ, DOTNET_ROOT=str(Path(a.dotnet).parent), TENSORSHARP_GGML_NO_UPDATE="1",
               MAX_CONTEXT=str(a.context), TS_PREFILL_WARMUP_LEN=str(a.warmup_prefill),
               OMP_NUM_THREADS=str(a.cpu_threads), OPENBLAS_NUM_THREADS=str(a.cpu_threads))
    if a.dump_logits:
        env["TS_DUMP_LOGITS"] = str((a.output / "prefill-logits.bin").resolve())
    if a.initial_kv_tokens is not None:
        if a.initial_kv_tokens < 1:
            p.error("--initial-kv-tokens must be positive")
        env["TS_KV_INITIAL_TOKENS"] = str(a.initial_kv_tokens)
    upstream = a.repo / "ExternalProjects/ggml"
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=all"], text=True)
    if dirty:
        raise RuntimeError("Upstream ggml source must be unchanged: " + dirty)
    report = {"status": "running", "command": command, "started_unix": time.time(),
              "harness_sha256": sha(Path(__file__)),
              "native_sha256": sha(binary / "libGgmlOps.so"),
              "ggml_revision": subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip(),
              "ggml_clean": True}
    if a.model_verification_report:
        verification = json.loads(a.model_verification_report.read_text())
        if verification.get("status") != "verified":
            raise ValueError("Model shard verification is incomplete")
        if str(a.model.resolve()) not in {str(Path(s["path"]).resolve()) for s in verification["shards"]}:
            raise ValueError("Selected model is absent from shard verification")
        for shard in verification["shards"]:
            if Path(shard["path"]).stat().st_size != shard["bytes"]:
                raise ValueError("Verified model shard size changed: " + shard["path"])
        report["model_verification_report"] = str(a.model_verification_report)
        report["model_verification_report_sha256"] = sha(a.model_verification_report)
        report["model_shards"] = verification["shards"]
        report["model_sha256"] = next(s["sha256"] for s in verification["shards"]
            if Path(s["path"]).resolve() == a.model.resolve())
        report["model_hash_source"] = "completed full-shard verification; file sizes rechecked"
    else:
        report["model_sha256"] = sha(a.model)
        report["model_hash_source"] = "full hash at launch"
    report["managed_sha256"] = {name: sha(binary / name) for name in
        (project + ".dll", "TensorSharp.Models.dll", "TensorSharp.Runtime.dll", "TensorSharp.Backends.GGML.dll", "TensorSharp.Distributed.dll")}
    for name in ("TensorSharp.Chat.dll", "TensorSharp.Server.dll"):
        if (binary / name).is_file():
            report["managed_sha256"][name] = sha(binary / name)
    if a.media_projector:
        report['media_projector'] = {'path': str(a.media_projector), 'sha256': sha(a.media_projector)}
    report["environment"] = {k: env[k] for k in ("MAX_CONTEXT", "TS_PREFILL_WARMUP_LEN", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                               "CUDA_VISIBLE_DEVICES", "NCCL_P2P_DISABLE", "TS_DUMP_LOGITS", "TS_KV_INITIAL_TOKENS",
                                               "TENSORSHARP_TP_DEGREE", "TENSORSHARP_LAYER_SPLIT_DEGREE",
                                               "TENSORSHARP_TP_DEVICES", "TENSORSHARP_LAYER_SPLIT_DEVICES",
                                               "TENSORSHARP_TP_NODE_ID", "TENSORSHARP_TP_PEERS",
                                               "TS_GLM_NGPU", "TS_GLM_NATIVE", "TS_GLM_TP_SHARD",
                                               "TS_DSV4_NGPU", "TS_DSV41_TP", "TS_DSV4_UBATCH", "TS_DSV4_THREADS",
                                               "TS_DSV4_VRAM_RESERVE_MB", "TS_DSV4_LOAD_THREADS", "TS_DSV4_LOAD_CHUNK_MB",
                                               "TS_DSV41_ENGRAM_DEVICE", "TS_DSV41_ENGRAM_WARM",
                                               "TS_DSV41_ENGRAM_THREADS", "TS_DSV41_ENGRAM_RANDOM",
                                               "TS_SCHED_MAX_BATCHED_TOKENS", "TS_SCHED_SOLO_PREFILL_CHUNK", "TS_SCHED_PREFIX_CACHE",
                                               "TS_CPU_MOE_THREADS", "TS_HOST_MOE_PIN", "TS_N_CPU_MOE", "TS_CPU_MOE",
                                               "TS_SPEC", "TS_SPEC_TYPE", "TS_SPEC_DRAFT", "TS_SPEC_PMIN", "TS_SPEC_DRAFT_MODEL",
                                               "TS_GLM_MTP", "TS_GLM_UBATCH", "TS_GLM_THREADS", "TS_GLM_VRAM_RESERVE_MB",
                                               "TS_GLM_LOAD_THREADS", "TS_GLM_LOAD_CHUNK_MB", "TS_Q4E_LAYER_SPLIT") if k in env}
    # Preserve command and binary/model identity even if the parent is killed
    # while a large checkpoint is loading. A surviving "running" record is an
    # interrupted attempt, never a completed validation result.
    (a.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    telemetry = process = None
    monitor_stop = threading.Event()
    def memory_monitor():
        base = Path("/sys/fs/cgroup")
        if not (base / "memory.current").exists():
            return
        with (a.output / "memory.csv").open("w", buffering=1) as log, (a.output / "process.csv").open("w", buffering=1) as proc_log:
            log.write("unix_time,cgroup_current_bytes,anon_bytes,file_bytes\n")
            proc_log.write("unix_time,pid,rss_kib,threads,minor_faults,major_faults,read_bytes,write_bytes,rchar,syscr\n")
            while not monitor_stop.is_set():
                values = dict(line.split() for line in (base / "memory.stat").read_text().splitlines())
                stamp = time.time()
                log.write(f"{stamp},{(base / 'memory.current').read_text().strip()},{values.get('anon', '0')},{values.get('file', '0')}\n")
                if process is not None:
                    try:
                        proc = Path('/proc') / str(process.pid)
                        # The executable name in field 2 can contain spaces or
                        # parentheses; fields after its final ')' start at 3.
                        stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
                        status = dict(line.split(':', 1) for line in (proc / 'status').read_text().splitlines())
                        io = dict(line.split(':', 1) for line in (proc / 'io').read_text().splitlines())
                        row = (stamp, process.pid, status.get('VmRSS', '0').split()[0],
                               status.get('Threads', '0').strip(), stat[7], stat[9],
                               io.get('read_bytes', '0').strip(), io.get('write_bytes', '0').strip(),
                               io.get('rchar', '0').strip(), io.get('syscr', '0').strip())
                        proc_log.write(','.join(map(str, row)) + '\n')
                    except (OSError, IndexError):
                        pass  # The child may exit between the /proc reads.
                monitor_stop.wait(1)
    monitor = threading.Thread(target=memory_monitor, daemon=True)
    monitor.start()
    try:
        with (a.output / "gpu.csv").open("w") as gpu_log, (a.output / "process.log").open("w") as log:
            telemetry = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,name,memory.used,utilization.gpu,power.draw,temperature.gpu", "--format=csv", "-lms", "500"],
                stdout=gpu_log, stderr=subprocess.STDOUT, start_new_session=True)
            process = subprocess.Popen(command, cwd=a.repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            if a.server:
                deadline = time.monotonic() + a.startup_timeout
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError("Server exited: " + str(process.returncode))
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2) as response:
                            if response.status == 200:
                                break
                    except Exception:
                        time.sleep(1)
                else:
                    raise TimeoutError("Server did not become ready")
                evaluate = [sys.executable, str(Path(__file__).with_name("parallel-server-eval.py")),
                    "--url", f"http://127.0.0.1:{a.port}", "--model", a.model.name,
                    "--output", str(a.output / "evaluation.json"), "--timeout", str(a.timeout),
                    "--concurrency", str(a.concurrency), "--repeats", str(a.repeats),
                    "--max-tokens", str(a.eval_max_tokens)]
                if a.eval_cases:
                    evaluate += ["--cases", a.eval_cases]
                if a.reasoning_effort:
                    evaluate += ["--reasoning-effort", a.reasoning_effort]
                if a.baseline:
                    evaluate += ["--baseline", str(a.baseline)]
                report["evaluation_command"] = evaluate
                report["text_exit_code"] = subprocess.call(evaluate, cwd=a.repo, env=env)
                report["exit_code"] = report["text_exit_code"]
                if a.media_fixtures:
                    media = [sys.executable, str(a.repo / 'benchmarks/engine_comparison/validate_deepseek41_media.py'),
                        '--url', f'http://127.0.0.1:{a.port}', '--model', a.model.name,
                        '--fixtures', str(a.media_fixtures), '--output', str(a.output / 'media.json'),
                        '--weights-id', report.get('model_verification_report_sha256', report['model_sha256']),
                        '--companion-sha256', report['media_projector']['sha256'],
                        '--profile', ' '.join(extra), '--concurrency', str(a.concurrency),
                        '--scenarios', a.media_scenarios, '--max-tokens', str(a.media_max_tokens), '--blocking']
                    if a.reasoning_effort:
                        media += ['--reasoning-effort', a.reasoning_effort]
                    report['media_command'] = media
                    report['media_exit_code'] = subprocess.call(media, cwd=a.repo, env=env)
                    report['exit_code'] = int(bool(report['text_exit_code'] or report['media_exit_code']))
            else:
                report["exit_code"] = process.wait(timeout=a.timeout)
            report["status"] = "completed" if report["exit_code"] == 0 else "failed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
    finally:
        report["shutdown"] = stop(process)
        stop(telemetry)
        monitor_stop.set()
        monitor.join(timeout=2)
        process_log = a.output / "process.log"
        report["shutdown_failures"] = shutdown_failures(report["shutdown"],
            process_log.read_text(errors="replace") if process_log.exists() else "", allow_sigterm=a.worker)
        if report["shutdown_failures"]:
            report["status"] = "failed"
        report["wall_seconds"] = time.time() - report["started_unix"]
        (a.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
