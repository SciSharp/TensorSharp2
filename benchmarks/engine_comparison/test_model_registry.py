"""Coverage for the benchmark config files themselves — the model / scenario /
engine / backend registries every recorded cell is labelled from.

Nothing here launches an engine, reads a GGUF or touches the network: the
assertions are about what each `benchmark_config*.json` RESOLVES to once
`config.py` has applied its substitutions and defaults. That is the part with no
other guard — a typo in a registry surfaces at run time, after a multi-hundred-
GiB cold load, as a skipped or mislabelled row rather than as an error.

Two kinds of test live here. `ConfigRegistryTests` runs over EVERY config file,
so a new variant is covered the moment it is dropped in the directory.
`Glm53Qwen38ConfigTests` pins the handful of facts that make the 8xA40
GLM-5.3 / Qwen3.8-Flash-Next matrix mean what it says — the ones an innocent
"cleanup" would otherwise silently invert.
"""
import importlib.util
import json
import os
import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_FILES = sorted(HERE.glob("benchmark_config*.json"))

# An environment variable name as the launched servers receive it. Every key of
# a backend's `env` block is exported verbatim to the engine process, so a note
# or comment parked in there becomes a bogus variable in the server's
# environment instead of documentation.
ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def load_config(path):
    """`config.py` freshly evaluated against one config file.

    config.py reads its file at import time, so each config needs its own module
    instance. The BENCH_CONFIG override and the temporary module name are both
    undone afterwards, leaving the caller's environment as it was.
    """
    name = "config_under_test_" + re.sub(r"\W", "_", path.stem)
    previous = os.environ.get("BENCH_CONFIG")
    os.environ["BENCH_CONFIG"] = str(path)
    try:
        spec = importlib.util.spec_from_file_location(name, HERE / "config.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module          # dataclasses resolve types through sys.modules
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(name, None)
        return module
    finally:
        if previous is None:
            os.environ.pop("BENCH_CONFIG", None)
        else:
            os.environ["BENCH_CONFIG"] = previous


def raw(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class ConfigRegistryTests(unittest.TestCase):
    """Invariants that must hold for every config file in this directory."""

    def test_config_files_are_discovered(self):
        # A glob that matches nothing would make every test below vacuous.
        self.assertTrue(CONFIG_FILES, "no benchmark_config*.json files found")

    def test_registries_build(self):
        for path in CONFIG_FILES:
            with self.subTest(config=path.name):
                cfg = load_config(path)
                self.assertTrue(cfg.MODELS, "no models registered")
                self.assertTrue(cfg.ENGINES, "no engines registered")
                self.assertTrue(cfg.BACKENDS, "no backends registered")

    def test_defaults_resolve(self):
        for path in CONFIG_FILES:
            cfg = load_config(path)
            with self.subTest(config=path.name):
                for model_id in cfg.DEFAULT_MODELS:
                    self.assertIn(model_id, cfg.MODELS)
                for engine_id in cfg.DEFAULT_ENGINES:
                    self.assertIn(engine_id, cfg.ENGINES)
                for backend_id in cfg.DEFAULT_BACKENDS:
                    self.assertIn(backend_id, cfg.BACKENDS)
                for scenario_id in cfg.DEFAULT_SCENARIOS:
                    # `prefill_<N>` is synthesized rather than declared; the
                    # registry's __contains__ is what decides, not the JSON.
                    self.assertIn(scenario_id, cfg.SCENARIOS)

    def test_every_model_file_has_a_download_url(self):
        # A model whose files resolve to no URL cannot be fetched by
        # download_models.py, and the gap only shows up on a fresh host.
        for path in CONFIG_FILES:
            cfg = load_config(path)
            for model_id, model in cfg.MODELS.items():
                for role, _file, url in model.files():
                    with self.subTest(config=path.name, model=model_id, role=role):
                        self.assertTrue(url, "no download URL resolved")

    def test_default_backends_are_launchable_by_a_default_engine(self):
        for path in CONFIG_FILES:
            cfg = load_config(path)
            for backend_id in cfg.DEFAULT_BACKENDS:
                spec = cfg.BACKENDS[backend_id]
                launchable = [
                    engine_id for engine_id in cfg.DEFAULT_ENGINES
                    if (engine_id == "tensorsharp" and spec.ts_backend)
                    or (engine_id == "llamacpp" and spec.llama_ngl is not None)
                    or (engine_id == "vllm" and spec.vllm)]
                with self.subTest(config=path.name, backend=backend_id):
                    self.assertTrue(launchable,
                                    "no default engine has a launch mapping for this backend")

    def test_env_blocks_declare_only_environment_variables(self):
        # `ts_env` / `llama_env` are copied into the child process as-is, so a
        # "_note" key here would be exported as a variable. Notes belong beside
        # the mapping, where the loader ignores them.
        for path in CONFIG_FILES:
            for backend_id, backend in (raw(path).get("backends") or {}).items():
                if backend_id.startswith("_") or not isinstance(backend, dict):
                    continue
                for engine_id in ("tensorsharp", "llamacpp"):
                    mapping = backend.get(engine_id)
                    if not isinstance(mapping, dict):
                        continue
                    for key in (mapping.get("env") or {}):
                        with self.subTest(config=path.name, backend=backend_id,
                                          engine=engine_id, key=key):
                            self.assertRegex(key, ENV_NAME_RE)

    def test_scenario_and_timeout_blocks_carry_no_stray_notes(self):
        # Both blocks are consumed wholesale: every `scenarios` entry becomes a
        # ScenarioSpec (and needs a "kind"), and every ready_timeout_s value is
        # floated. A comment parked in either one is a crash or a phantom
        # scenario, not documentation.
        for path in CONFIG_FILES:
            config_json = raw(path)
            for scenario_id, scenario in (config_json.get("scenarios") or {}).items():
                with self.subTest(config=path.name, scenario=scenario_id):
                    self.assertFalse(scenario_id.startswith("_"))
                    self.assertIn("kind", scenario)
            for key, value in (config_json.get("ready_timeout_s") or {}).items():
                with self.subTest(config=path.name, timeout=key):
                    self.assertIsInstance(value, (int, float))

    def test_default_models_run_or_are_gated_by_a_declared_tp_range(self):
        # A default model with no runnable cell is legitimate — a checkpoint that
        # only fits across N GPUs declares `min_tp` and is driven with an
        # explicit `--tp N` — but ONLY that. Any other reason (a backend with no
        # mapping, a modality typo) means the matrix silently drops the model.
        for path in CONFIG_FILES:
            cfg = load_config(path)
            for model_id in cfg.DEFAULT_MODELS:
                model = cfg.MODELS[model_id]
                runnable = any(
                    cfg.applies(engine_id, backend_id, model, cfg.SCENARIOS[scenario_id],
                                mtp=False, tp=tp)[0]
                    for engine_id in cfg.DEFAULT_ENGINES
                    for backend_id in cfg.DEFAULT_BACKENDS
                    for scenario_id in cfg.DEFAULT_SCENARIOS
                    for tp in cfg.DEFAULT_TP_DEGREES)
                with self.subTest(config=path.name, model=model_id):
                    self.assertTrue(runnable or model.min_tp > 1 or model.max_tp,
                                    "model is in defaults.models but no default cell runs and "
                                    "it declares no tp range that would explain the gate")


class ExplicitPlacementConfigTests(unittest.TestCase):
    def test_deepseek4_is_separate_from_tensor_only_matrix(self):
        tensor = load_config(HERE / "benchmark_config_multigpu.json")
        layer = load_config(HERE / "benchmark_config_deepseek4_layer.json")
        self.assertNotIn("dsv4-flash", tensor.MODELS)
        self.assertIn("dsv4-flash", layer.MODELS)
        self.assertEqual(layer.DEFAULT_TP_DEGREES, [4])
        for spec in layer.BACKENDS.values():
            self.assertEqual(spec.ts_tp_arg, "--layer-split")
            if spec.llama_ngl is not None:
                self.assertEqual(tuple(spec.llama_tp_extra_args), ("--split-mode", "layer"))

    def test_deepseek41_distinguishes_whole_layers_and_expert_shards(self):
        cfg = load_config(HERE / "benchmark_config_deepseek41.json")
        for name in ("ggml_cuda_layer", "ggml_cuda_layer_cpu_moe4"):
            self.assertEqual(cfg.BACKENDS[name].ts_tp_arg, "--layer-split")
            self.assertEqual(cfg.BACKENDS[name].ts_env["TS_DSV41_TP"], "0")
        tensor = cfg.BACKENDS["ggml_cuda_true_tp"]
        self.assertEqual(tensor.ts_tp_arg, "--tp")
        self.assertEqual(tensor.ts_env["TS_DSV41_TP"], "{tp}")


class Glm53Qwen38ConfigTests(unittest.TestCase):
    """Drift guards for the 8xA40 GLM-5.3 / Qwen3.8-Flash-Next matrix.

    Each assertion here stands for a fact about the engines, not a preference:
    they are the ones that decide whether a cell in that matrix can run at all.
    """

    CONFIG = HERE / "benchmark_config_glm53_qwen38.json"

    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(cls.CONFIG)
        cls.json = raw(cls.CONFIG)

    def test_tp_device_pool_stays_empty(self):
        # Filling defaults.tp_devices in pins a tp=1 cell to GPU 0, where none of
        # these three checkpoints fit — and tp=1 is exactly how both GLM models
        # are served (their native executor claims every visible GPU itself).
        # Asserted against the file, since BENCH_TP_DEVICES may override the
        # loaded value on a developer's box.
        self.assertEqual(self.json["defaults"]["tp_devices"], [])

    def test_qwen38_declares_the_tp_it_cannot_run_without(self):
        # qwen4exp is spread by the SHARED loader, which reads the degree from
        # `--layer-split N`. Without one it is a
        # single-device load of 175.3 GiB. `min_tp` turns that into a recorded
        # skip instead of an OOM.
        self.assertGreater(self.cfg.MODELS["qwen38-flash-next"].min_tp, 1)

    def test_glm_models_are_hosted_at_tp1(self):
        # The glm native executor takes every visible GPU when no degree is
        # given, so tp=1 is their placement — and `--tp N` would switch them to
        # tensor parallelism, which also disables the NextN draft head.
        for model_id in ("glm53", "glm53-flash"):
            with self.subTest(model=model_id):
                self.assertEqual(self.cfg.MODELS[model_id].min_tp, 1)

    def test_the_two_columns_differ_in_whether_a_degree_is_passed(self):
        layer = self.cfg.BACKENDS["ggml_cuda_layer"]
        tp = self.cfg.BACKENDS["ggml_cuda_split"]
        self.assertEqual(layer.ts_env["TS_GLM_NGPU"], "0")
        self.assertFalse(self.cfg.ts_tp_supported(layer),
                         "the layer column must pass no --tp: it is the GLM placement")
        self.assertTrue(self.cfg.ts_tp_supported(tp),
                        "the tp column is the only one qwen38-flash-next can run on")
        # Layer placement on both engines is what makes the qwen column
        # comparable: `--split-mode layer` is the only multi-GPU mode llama.cpp
        # has for qwen4exp, and TensorSharp explicitly selects `--layer-split N`.
        self.assertEqual(tp.ts_tp_arg, "--layer-split")
        self.assertEqual(tuple(tp.llama_tp_extra_args), ("--split-mode", "layer"))

    def test_qwen38_runs_on_the_explicit_layer_split_column_and_nowhere_else(self):
        model = self.cfg.MODELS["qwen38-flash-next"]
        scenario = self.cfg.SCENARIOS["text_short"]
        live = [(backend_id, tp)
                for backend_id in self.cfg.BACKENDS
                for tp in self.cfg.DEFAULT_TP_DEGREES
                if self.cfg.applies("tensorsharp", backend_id, model, scenario, tp=tp)[0]]
        self.assertEqual(live, [("ggml_cuda_split", 8)])

    def test_llama_context_matches_the_tensorsharp_window_per_slot(self):
        # llama-server divides `-c` across its `--parallel` slots, so the two
        # engines only size the same per-sequence window when the configured
        # context is MAX_CONTEXT x parallel.
        extra = self.cfg.LLAMA_EXTRA_ARGS
        parallel = int(extra[extra.index("--parallel") + 1])
        window = int(self.cfg.BACKENDS["ggml_cuda_layer"].ts_env["MAX_CONTEXT"])
        self.assertEqual(self.cfg.LLAMA_CONTEXT_SIZE, window * parallel)

    def test_mtp_flags_match_the_checkpoints(self):
        # glm53 ships a complete NextN block (blk.78) and TensorSharp loads it;
        # glm53-flash ships one the executor explicitly refuses to run; the
        # Qwen Q8_0 shards carry no nextn tensor at all.
        self.assertTrue(self.cfg.MODELS["glm53"].mtp_supported)
        self.assertFalse(self.cfg.MODELS["glm53-flash"].mtp_supported)
        self.assertFalse(self.cfg.MODELS["qwen38-flash-next"].mtp_supported)

    def test_modalities_match_the_published_projectors(self):
        # glm-dsa publishes no mmproj and GlmDsaModel refuses one, so an `image`
        # cell for glm53 must be a recorded skip, not an attempted run.
        self.assertIsNone(self.cfg.MODELS["glm53"].mmproj)
        ok, why = self.cfg.applies("tensorsharp", "ggml_cuda_layer", self.cfg.MODELS["glm53"],
                                   self.cfg.SCENARIOS["image"], tp=1)
        self.assertFalse(ok)
        self.assertIn("image", why)
        for model_id in ("qwen38-flash-next", "glm53-flash"):
            with self.subTest(model=model_id):
                self.assertIn("image", self.cfg.MODELS[model_id].modalities)
                self.assertIsNotNone(self.cfg.MODELS[model_id].mmproj)


if __name__ == "__main__":
    unittest.main()
