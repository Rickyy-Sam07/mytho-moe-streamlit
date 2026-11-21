"""Run the paper's two Top-1 checkpoints and the shared Qwen baseline."""

import hashlib
import io
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from time import perf_counter
from zipfile import ZipFile

from evidence import APP_DIR, DOMAINS, ROOT, Result

MODELS = ROOT / "models"
PACKAGED = APP_DIR / "checkpoints"
CACHE = APP_DIR / ".cache"
BASE_MODEL = "Qwen/Qwen3-4B"
BACKBONE_FILES = ["*.json", "*.safetensors", "merges.txt", "vocab.json", "chat_template.jinja"]
ROUTER_FILES = {
    "mixed": ("Final MoE v14.zip", "run/router_best.pt"),
    "adapted": ("Mytho MoE Router V2.zip", "router_best.pt"),
}
ADAPTER_FILES = {
    "hvm": ("hvm-v1.zip", "hvm_qwen3_4b/checkpoint-9500"),
    "rama": ("RAMA Ultimate Training Backup.zip", "rama_ultimate_qwen3_4b/final_adapter"),
}
BASE_SYSTEM = "Return only the requested answer or structured query without extra commentary."
RAMA_SYSTEM = (
    "You are RaMO-QA, an expert assistant for the Valmiki Ramayana Mythology Ontology (RaMO). "
    "Answer strictly from the ontology facts. Return only the requested answer or structured query without extra commentary."
)
PROMPT_LIMIT = 1024
GENERATION_LIMIT = 256


def load_router_states():
    import torch

    states, hashes = {}, {}
    for mode, (archive_name, member) in ROUTER_FILES.items():
        if PACKAGED.is_dir():
            payload = (PACKAGED / f"{mode}.pt").read_bytes()
        else:
            with ZipFile(MODELS / archive_name) as archive:
                payload = archive.read(member)
        hashes[mode] = hashlib.sha256(payload).hexdigest()
        states[mode] = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)

    mixed, adapted = states["mixed"], states["adapted"]
    for key in ("base_model", "domain_order", "dim", "architecture", "router_max_length"):
        if mixed[key] != adapted[key]:
            raise ValueError(f"Router checkpoints disagree on {key}.")
    if mixed["base_model"] != BASE_MODEL or mixed["domain_order"] != list(DOMAINS):
        raise ValueError("The router checkpoints have unexpected backbone or domain labels.")
    revision = mixed["feature_metadata"]["resolved_base_revision"]
    if len(revision) != 40 or any(
        state["feature_metadata"]["resolved_base_revision"] != revision
        or not state["feature_metadata"]["quantize_4bit"]
        for state in states.values()
    ):
        raise ValueError("The routers must share a pinned 4-bit Qwen revision.")
    if (mixed["identity"]["expert_hashes"] != adapted["identity"]["expert_hashes"]
            or adapted["training_provenance"]["reference_sha256"] != hashes["mixed"]):
        raise ValueError("The adapted checkpoint does not match the original router and expert pair.")
    return states


def adapter_directory(domain, expected):
    if PACKAGED.is_dir():
        directory = PACKAGED / domain
        for name, digest in expected.items():
            with (directory / name).open("rb") as source:
                if hashlib.file_digest(source, "sha256").hexdigest() != digest:
                    raise ValueError(f"{domain} {name} differs from the paper's checkpoint hash.")
        return directory
    archive_name, prefix = ADAPTER_FILES[domain]
    directory = CACHE / "adapters" / domain
    directory.mkdir(parents=True, exist_ok=True)
    with ZipFile(MODELS / archive_name) as archive:
        for name, digest in expected.items():
            path = directory / name
            if path.is_file():
                with path.open("rb") as cached:
                    if hashlib.file_digest(cached, "sha256").hexdigest() == digest:
                        continue
            tmp_path = None
            try:
                with archive.open(f"{prefix}/{name}") as source, NamedTemporaryFile(
                    dir=directory, delete=False, mode="w+b"
                ) as temporary:
                    tmp_path = Path(temporary.name)
                    shutil.copyfileobj(source, temporary)
                    temporary.seek(0)
                    if hashlib.file_digest(temporary, "sha256").hexdigest() != digest:
                        raise ValueError(f"{domain} {name} differs from the paper's checkpoint hash.")
                tmp_path.replace(path)
            finally:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)
    return directory


def prepared_backbone(revision):
    """Recognize a completed Hugging Face local-dir download of the pinned model."""
    directory = CACHE / "qwen3-4b"
    index = directory / "model.safetensors.index.json"
    if not index.is_file():
        return None
    with index.open(encoding="utf-8") as source:
        shards = set(json.load(source)["weight_map"].values())
    for name in shards | {"config.json", "generation_config.json", "tokenizer.json",
                          "tokenizer_config.json", index.name}:
        metadata = directory / ".cache" / "huggingface" / "download" / f"{name}.metadata"
        if not (directory / name).is_file() or not metadata.is_file():
            return None
        with metadata.open(encoding="utf-8") as source:
            if source.readline().strip() != revision:
                return None
    return directory


def format_prompt(query, style, tokenizer):
    body = f"### Task: Question Answering\nQuestion: {query}\n### Answer:"
    if style == "hvm":
        return body + "\n"
    if style == "rama":
        return (f"<|im_start|>system\n{RAMA_SYSTEM}<|im_end|>\n"
                f"<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n")
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": BASE_SYSTEM}, {"role": "user", "content": body}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


class MythoEngine:
    def __init__(self, zero_gpu=False):
        import torch

        if not zero_gpu and not torch.cuda.is_available():
            raise RuntimeError("Live inference requires a CUDA-enabled NVIDIA GPU and PyTorch build.")

        from huggingface_hub import snapshot_download
        from peft import PeftModel
        from torch import nn
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        states = load_router_states()
        mixed = states["mixed"]
        self.router_limit = mixed["router_max_length"]
        self.revision = mixed["feature_metadata"]["resolved_base_revision"]
        self.device = torch.device("cuda:0")
        self.lock = Lock()

        local = prepared_backbone(self.revision)
        if local is None:
            snapshot_download(BASE_MODEL, revision=self.revision, allow_patterns=BACKBONE_FILES,
                              local_dir=CACHE / "qwen3-4b", cache_dir=CACHE / "hub")
            local = prepared_backbone(self.revision)
        if local is None:
            raise RuntimeError("The pinned Qwen3-4B download is incomplete.")
        self.tokenizer = AutoTokenizer.from_pretrained(local, use_fast=True, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            local, local_files_only=True, torch_dtype=torch.float16, device_map="cuda" if zero_gpu else {"": 0},
            attn_implementation="sdpa", quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
        )
        identities = mixed["identity"]["expert_hashes"]
        # Read adapter tensors on CPU during ZeroGPU's startup CUDA emulation.
        adapter_options = {"torch_device": "cpu"} if zero_gpu else {}
        self.model = PeftModel.from_pretrained(
            base, adapter_directory("hvm", identities["hvm"]), adapter_name="hvm",
            is_trainable=False, **adapter_options,
        )
        self.model.load_adapter(adapter_directory("rama", identities["rama"]), adapter_name="rama",
                                is_trainable=False, **adapter_options)
        self.model.requires_grad_(False).eval()
        self.routers = {}
        for mode, state in states.items():
            dim = state["dim"]
            width = state["architecture"]["router_hidden"]
            router = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width), nn.GELU(),
                                   nn.Dropout(state["architecture"]["router_dropout"]), nn.Linear(width, 2))
            router.load_state_dict(state["state_dict"])
            self.routers[mode] = router.eval()

    def route(self, query, modes):
        import torch

        started = perf_counter()
        tokens = self.tokenizer(query, return_tensors="pt", add_special_tokens=False,
                                return_token_type_ids=False).to(self.device)
        truncated = tokens.input_ids.shape[1] > self.router_limit
        tokens = {key: value[:, :self.router_limit] for key, value in tokens.items()}
        with self.model.disable_adapter():
            hidden = self.model.get_base_model().model(**tokens, use_cache=False).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            features = (hidden.float() * mask).sum(1) / mask.sum(1).clamp_min(1)
            # Training cached pooled features as float16 before running the float32 classifier.
            features = features.to(torch.float16).cpu().float()
        if not torch.isfinite(features).all():
            raise RuntimeError("The backbone returned non-finite router features.")
        probabilities = {
            mode: dict(zip(DOMAINS, self.routers[mode](features)[0].softmax(-1).tolist()))
            for mode in modes
        }
        return probabilities, truncated, perf_counter() - started

    def generate(self, query, selected):
        import torch

        if selected:
            self.model.set_adapter(selected)
            self.model.requires_grad_(False)
        prompt = format_prompt(query, selected or "base", self.tokenizer)
        tokens = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False,
                                return_token_type_ids=False).to(self.device)
        prompt_length = tokens.input_ids.shape[1]
        if prompt_length > PROMPT_LIMIT:
            raise ValueError(f"The formatted prompt exceeds {PROMPT_LIMIT} tokens. Shorten the question.")
        stop_ids = {self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|im_end|>")} - {None}
        started = perf_counter()
        output = self.model.generate(
            **tokens, do_sample=False, max_new_tokens=GENERATION_LIMIT, eos_token_id=sorted(stop_ids),
            pad_token_id=self.tokenizer.pad_token_id, use_cache=True, return_dict_in_generate=True,
            output_logits=True,
        )
        generated = output.sequences[0, prompt_length:]
        ids = generated.tolist()
        seconds = perf_counter() - started
        raw = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        answer = raw.split("</think>", 1)[-1].split("###", 1)[0].strip()
        if "<think>" in answer:
            answer = answer.split("<think>", 1)[0].strip()
        log_probs = [logits[0].float().log_softmax(-1)[token_id]
                     for logits, token_id in zip(output.logits, ids) if token_id not in stop_ids]
        likelihood = math.exp(float(torch.stack(log_probs).mean())) if log_probs and answer else None
        return Result(
            answer=answer, expert=selected, generation_likelihood=likelihood,
            generated_tokens=len(ids), generation_seconds=seconds,
            hit_generation_limit=len(ids) == GENERATION_LIMIT and ids[-1] not in stop_ids,
        )

    def answer(self, query, modes):
        import torch

        routed = [mode for mode in modes if mode != "base"]
        with self.lock, torch.inference_mode():
            scores, truncated, elapsed = self.route(query, routed) if routed else ({}, False, None)
            results = {}
            expert_answers = {}
            for mode in modes:
                if mode == "base":
                    with self.model.disable_adapter():
                        results[mode] = self.generate(query, None)
                else:
                    probabilities = scores[mode]
                    selected = max(probabilities, key=probabilities.get)
                    if selected not in expert_answers:
                        expert_answers[selected] = self.generate(query, selected)
                    results[mode] = replace(expert_answers[selected], probabilities=probabilities,
                                            router_seconds=elapsed, router_truncated=truncated)
            return results
