"""服务端 SDXS CLIP-L 文本编码，浏览器无需加载完整 Text Encoder。"""

from collections import OrderedDict
from pathlib import Path
import html
import json
import math
import struct
import threading

import numpy as np


MODEL_DIR = Path(__file__).resolve().parent / "models" / "sdxs-512-dreamshaper"
TOKENIZER_DIR = MODEL_DIR / "tokenizer"
TEXT_ENCODER_PATH = MODEL_DIR / "text_encoder" / "model.safetensors"
SEQUENCE_LENGTH = 77
HIDDEN_SIZE = 768
NUM_LAYERS = 12
NUM_HEADS = 12
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
BOS_TOKEN_ID = 49406
EOS_TOKEN_ID = 49407


def _bytes_to_unicode():
    values = list(range(ord("!"), ord("~") + 1))
    values += list(range(ord("¡"), ord("¬") + 1))
    values += list(range(ord("®"), ord("ÿ") + 1))
    characters = values[:]
    extra = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            characters.append(256 + extra)
            extra += 1
    return dict(zip(values, map(chr, characters)))


def _pairs(word):
    return set(zip(word, word[1:]))


class ClipTokenizer:
    def __init__(self, vocab_path, merges_path):
        import regex

        self.regex = regex
        self.encoder = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        merges = Path(merges_path).read_text(encoding="utf-8").strip().splitlines()[1:48895]
        self.ranks = {tuple(item.split()): index for index, item in enumerate(merges)}
        self.byte_encoder = _bytes_to_unicode()
        self.cache = {}
        self.pattern = regex.compile(
            r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
            regex.IGNORECASE,
        )

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _pairs(word)
        while pairs:
            pair = min(pairs, key=lambda item: self.ranks.get(item, float("inf")))
            if pair not in self.ranks:
                break
            first, second = pair
            merged = []
            index = 0
            while index < len(word):
                try:
                    next_index = word.index(first, index)
                except ValueError:
                    merged.extend(word[index:])
                    break
                merged.extend(word[index:next_index])
                index = next_index
                if index < len(word) - 1 and word[index] == first and word[index + 1] == second:
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
            if len(word) == 1:
                break
            pairs = _pairs(word)
        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text):
        text = html.unescape(html.unescape(text)).strip().lower()
        token_ids = []
        for token in self.regex.findall(self.pattern, text):
            encoded = "".join(self.byte_encoder[value] for value in token.encode("utf-8"))
            token_ids.extend(self.encoder[item] for item in self._bpe(encoded).split(" "))
        result = [BOS_TOKEN_ID, *token_ids[: SEQUENCE_LENGTH - 2], EOS_TOKEN_ID]
        return result + [EOS_TOKEN_ID] * (SEQUENCE_LENGTH - len(result))


class StreamingSafeTensor:
    DTYPES = {"F16": np.float16, "F32": np.float32}

    def __init__(self, path):
        self.file = open(path, "rb")
        header_size = struct.unpack("<Q", self.file.read(8))[0]
        self.header = json.loads(self.file.read(header_size))
        self.data_start = 8 + header_size

    def _descriptor(self, name):
        descriptor = self.header.get(name)
        if not descriptor:
            raise KeyError(f"Text Encoder缺少权重：{name}")
        return descriptor

    def get_tensor(self, name):
        descriptor = self._descriptor(name)
        start, end = descriptor["data_offsets"]
        self.file.seek(self.data_start + start)
        raw = self.file.read(end - start)
        return np.frombuffer(raw, dtype=self.DTYPES[descriptor["dtype"]]).reshape(descriptor["shape"])

    def get_rows(self, name, row_indices):
        descriptor = self._descriptor(name)
        shape = descriptor["shape"]
        if len(shape) != 2:
            raise ValueError(f"{name}不是二维权重")
        dtype = self.DTYPES[descriptor["dtype"]]
        row_size = shape[1]
        row_bytes = row_size * np.dtype(dtype).itemsize
        start = descriptor["data_offsets"][0]
        output = np.empty((len(row_indices), row_size), dtype=np.float32)
        for output_index, row_index in enumerate(row_indices):
            self.file.seek(self.data_start + start + int(row_index) * row_bytes)
            output[output_index] = np.frombuffer(self.file.read(row_bytes), dtype=dtype)
        return output

    def close(self):
        self.file.close()


def _tensor(handle, name):
    return handle.get_tensor(name).astype(np.float32, copy=False)


def _linear(handle, name, values):
    weight = _tensor(handle, f"{name}.weight")
    bias = _tensor(handle, f"{name}.bias")
    return values @ weight.T + bias


def _layer_norm(values, weight, bias, epsilon=1e-5):
    mean = values.mean(axis=-1, keepdims=True)
    variance = ((values - mean) ** 2).mean(axis=-1, keepdims=True)
    return (values - mean) / np.sqrt(variance + epsilon) * weight + bias


def _encode_prompt(handle, input_ids):
    hidden = handle.get_rows("text_model.embeddings.token_embedding.weight", input_ids)
    hidden += _tensor(handle, "text_model.embeddings.position_embedding.weight")[:SEQUENCE_LENGTH]
    causal_mask = np.triu(np.full((SEQUENCE_LENGTH, SEQUENCE_LENGTH), -np.inf, dtype=np.float32), 1)
    scale = 1.0 / math.sqrt(HEAD_DIM)

    for layer_index in range(NUM_LAYERS):
        prefix = f"text_model.encoder.layers.{layer_index}"
        residual = hidden
        normalized = _layer_norm(
            hidden,
            _tensor(handle, f"{prefix}.layer_norm1.weight"),
            _tensor(handle, f"{prefix}.layer_norm1.bias"),
        )
        query = _linear(handle, f"{prefix}.self_attn.q_proj", normalized)
        key = _linear(handle, f"{prefix}.self_attn.k_proj", normalized)
        value = _linear(handle, f"{prefix}.self_attn.v_proj", normalized)
        query = query.reshape(SEQUENCE_LENGTH, NUM_HEADS, HEAD_DIM).transpose(1, 0, 2) * scale
        key = key.reshape(SEQUENCE_LENGTH, NUM_HEADS, HEAD_DIM).transpose(1, 0, 2)
        value = value.reshape(SEQUENCE_LENGTH, NUM_HEADS, HEAD_DIM).transpose(1, 0, 2)
        scores = query @ key.transpose(0, 2, 1) + causal_mask
        scores -= scores.max(axis=-1, keepdims=True)
        attention = np.exp(scores)
        attention /= attention.sum(axis=-1, keepdims=True)
        attended = (attention @ value).transpose(1, 0, 2).reshape(SEQUENCE_LENGTH, HIDDEN_SIZE)
        hidden = residual + _linear(handle, f"{prefix}.self_attn.out_proj", attended)

        residual = hidden
        normalized = _layer_norm(
            hidden,
            _tensor(handle, f"{prefix}.layer_norm2.weight"),
            _tensor(handle, f"{prefix}.layer_norm2.bias"),
        )
        activated = _linear(handle, f"{prefix}.mlp.fc1", normalized)
        activated = activated / (1.0 + np.exp(np.clip(-1.702 * activated, -80, 80)))
        hidden = residual + _linear(handle, f"{prefix}.mlp.fc2", activated)

    hidden = _layer_norm(
        hidden,
        _tensor(handle, "text_model.final_layer_norm.weight"),
        _tensor(handle, "text_model.final_layer_norm.bias"),
    )
    return hidden[np.newaxis, ...].astype(np.float32, copy=False)


class SDXSTextEmbeddingService:
    def __init__(self, model_dir=MODEL_DIR, cache_size=16):
        self.model_dir = Path(model_dir)
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.tokenizer = None

    def _ensure_ready(self):
        vocab = self.model_dir / "tokenizer" / "vocab.json"
        merges = self.model_dir / "tokenizer" / "merges.txt"
        encoder = self.model_dir / "text_encoder" / "model.safetensors"
        missing = [str(path) for path in (vocab, merges, encoder) if not path.exists()]
        if missing:
            script = "prepare_lcm_tiny_sd.py" if self.model_dir.name == "lcm-tiny-sd" else "convert_sdxs_to_onnx.py"
            raise FileNotFoundError(f"缺少{self.model_dir.name}文本编码资产，请先运行 scripts/{script}：" + ", ".join(missing))
        if self.tokenizer is None:
            self.tokenizer = ClipTokenizer(vocab, merges)
        return encoder

    def encode(self, prompt):
        clean_prompt = str(prompt or "").strip()
        if not clean_prompt:
            raise ValueError("Prompt不能为空")
        if len(clean_prompt) > 500:
            raise ValueError("Prompt不能超过500个字符")

        with self.lock:
            cached = self.cache.get(clean_prompt)
            if cached is not None:
                self.cache.move_to_end(clean_prompt)
                return cached.copy()

            encoder_path = self._ensure_ready()
            input_ids = self.tokenizer.encode(clean_prompt)
            handle = StreamingSafeTensor(encoder_path)
            try:
                embedding = _encode_prompt(handle, input_ids)
            finally:
                handle.close()

            self.cache[clean_prompt] = embedding.copy()
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
            return embedding


_SERVICE = SDXSTextEmbeddingService()
_MODEL_SERVICES = {
    "sdxs-512-dreamshaper": _SERVICE,
    "lcm-tiny-sd": SDXSTextEmbeddingService(MODEL_DIR.parent / "lcm-tiny-sd"),
}


def get_text_to_image_prompt_embedding(prompt, model_id):
    if model_id not in _MODEL_SERVICES:
        raise ValueError("不支持的文生图模型")
    return _MODEL_SERVICES[model_id].encode(prompt)


def get_sdxs_prompt_embedding(prompt):
    return _SERVICE.encode(prompt)
