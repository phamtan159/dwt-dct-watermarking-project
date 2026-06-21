from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel, AutoModelForCTC, Wav2Vec2FeatureExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_ROOT = Path(r"D:\A Project YTB\L2artic")
DEFAULT_OUTPUT_DIR = DEFAULT_CORPUS_ROOT / "regenerated_cache_v1"
SCHEMA_VERSION = "l2arctic_raw_feature_cache_v1"
TARGET_SR = 16000
SILENCE = {"", "sil", "sp", "spn", "<sil>", "<eps>"}
PROTOCOL_FEATURE_SETS = [
    "posterior_canonical",
    "posterior_canonical_duration_energy",
    "posterior_canonical_duration_energy_wavlm",
    "posterior_canonical_duration_energy_wavlm_sa",
]

SPEAKERS_BY_L1 = {
    "Arabic": {"male": ["ABA", "YBAA"], "female": ["SKA", "ZHAA"]},
    "Chinese": {"male": ["BWC", "TXHC"], "female": ["LXC", "NCC"]},
    "Hindi": {"male": ["ASI", "RRBI"], "female": ["SVBI", "TNI"]},
    "Korean": {"male": ["HKK", "YKWK"], "female": ["HJK", "YDCK"]},
    "Spanish": {"male": ["EBVS", "ERMS"], "female": ["MBMPS", "NJS"]},
    "Vietnamese": {"male": ["HQTV", "TLV"], "female": ["PNV", "THV"]},
}

VIETNAMESE_FOLDS = [
    {"fold_id": "test_TLV", "validation": ["THV"], "test": ["TLV"]},
    {"fold_id": "test_HQTV", "validation": ["PNV"], "test": ["HQTV"]},
    {"fold_id": "test_THV", "validation": ["TLV"], "test": ["THV"]},
    {"fold_id": "test_PNV", "validation": ["HQTV"], "test": ["PNV"]},
]

PHONE_INVENTORY = [
    "AX",
    "ERR",
    "aj",
    "aw",
    "b",
    "d",
    "d\u0292",
    "ej",
    "err",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "ow",
    "p",
    "r",
    "s",
    "t",
    "t\u0283",
    "u",
    "v",
    "w",
    "z",
    "\u00e6",
    "\u00f0",
    "\u014b",
    "\u0251",
    "\u0254",
    "\u0254j",
    "\u0259",
    "\u025b",
    "\u026a",
    "\u027e",
    "\u0283",
    "\u028a",
    "\u0292",
    "\u03b8",
]

ARPABET_TO_COMPARE = {
    "AA": ["\u0251"],
    "AE": ["\u00e6"],
    "AH": ["\u0259"],
    "AO": ["\u0254"],
    "AW": ["aw"],
    "AY": ["aj"],
    "B": ["b"],
    "CH": ["t\u0283"],
    "D": ["d"],
    "DH": ["\u00f0"],
    "EH": ["\u025b"],
    "ER": ["\u0259", "r"],
    "EY": ["ej"],
    "F": ["f"],
    "G": ["g"],
    "HH": ["h"],
    "IH": ["\u026a"],
    "IY": ["i"],
    "JH": ["d\u0292"],
    "K": ["k"],
    "L": ["l"],
    "M": ["m"],
    "N": ["n"],
    "NG": ["\u014b"],
    "OW": ["ow"],
    "OY": ["\u0254j"],
    "P": ["p"],
    "R": ["r"],
    "S": ["s"],
    "SH": ["\u0283"],
    "T": ["t"],
    "TH": ["\u03b8"],
    "UH": ["\u028a"],
    "UW": ["u"],
    "V": ["v"],
    "W": ["w"],
    "Y": ["j"],
    "Z": ["z"],
    "ZH": ["\u0292"],
}

RAW_PHONE_MAP = {
    "\u0261": "g",
    "\u0279": "r",
    "\u027b": "r",
    "\u025a": "\u0259 r",
    "\u025d": "\u0259 r",
    "\u0251\u02d0": "\u0251",
    "\u0254\u02d0": "\u0254",
    "a\u026a": "aj",
    "a\u028a": "aw",
    "e\u026a": "ej",
    "o\u028a": "ow",
    "\u0254\u026a": "\u0254j",
    "t\u0361\u0283": "t\u0283",
    "d\u0361\u0292": "d\u0292",
    "\u02a7": "t\u0283",
    "\u02a4": "d\u0292",
    "th": "\u03b8",
    "dh": "\u00f0",
    "ax": "\u0259",
    "er": "\u0259 r",
}

EXPECTED_FEATURES = {
    "p": {"consonant", "bilabial", "labial", "plosive"},
    "b": {"consonant", "bilabial", "labial", "plosive", "voiced"},
    "m": {"consonant", "bilabial", "labial", "nasal", "voiced"},
    "f": {"consonant", "labiodental", "labial", "fricative"},
    "v": {"consonant", "labiodental", "labial", "fricative", "voiced"},
    "\u03b8": {"consonant", "dental", "fricative"},
    "\u00f0": {"consonant", "dental", "fricative", "voiced"},
    "t": {"consonant", "alveolar", "plosive"},
    "d": {"consonant", "alveolar", "plosive", "voiced"},
    "s": {"consonant", "alveolar", "fricative"},
    "z": {"consonant", "alveolar", "fricative", "voiced"},
    "n": {"consonant", "alveolar", "nasal", "voiced"},
    "l": {"consonant", "alveolar", "approximant", "voiced"},
    "r": {"consonant", "alveolar", "approximant", "rhotic", "voiced"},
    "\u027e": {"consonant", "alveolar", "voiced"},
    "\u0283": {"consonant", "postalveolar", "fricative"},
    "\u0292": {"consonant", "postalveolar", "fricative", "voiced"},
    "t\u0283": {"consonant", "postalveolar", "affricates"},
    "d\u0292": {"consonant", "postalveolar", "affricates", "voiced"},
    "k": {"consonant", "velar", "plosive"},
    "g": {"consonant", "velar", "plosive", "voiced"},
    "\u014b": {"consonant", "velar", "nasal", "voiced"},
    "h": {"consonant", "glotal", "fricative"},
    "w": {"consonant", "approximant", "semivowel", "labial", "voiced", "round"},
    "j": {"consonant", "approximant", "semivowel", "palatal", "voiced"},
    "i": {"vowel", "high", "front", "monophthong", "voiced"},
    "\u026a": {"vowel", "high", "front", "monophthong", "short", "voiced"},
    "\u025b": {"vowel", "mid", "front", "monophthong", "voiced"},
    "\u00e6": {"vowel", "low", "front", "monophthong", "voiced"},
    "\u0259": {"vowel", "mid", "central", "monophthong", "voiced"},
    "\u0251": {"vowel", "low", "back", "monophthong", "voiced"},
    "\u0254": {"vowel", "mid", "back", "round", "monophthong", "voiced"},
    "\u028a": {"vowel", "high", "back", "round", "monophthong", "short", "voiced"},
    "u": {"vowel", "high", "back", "round", "monophthong", "voiced"},
    "ej": {"vowel", "front", "diphthong", "voiced"},
    "aj": {"vowel", "low", "front", "diphthong", "voiced"},
    "aw": {"vowel", "low", "back", "diphthong", "voiced"},
    "ow": {"vowel", "back", "round", "diphthong", "voiced"},
    "\u0254j": {"vowel", "back", "round", "diphthong", "voiced"},
}


@dataclass
class Segment:
    speaker_id: str
    sample_id: str
    index: int
    expected: str
    human_observed: str
    human_op: str
    start: float
    end: float
    raw_text: str

    @property
    def key(self) -> str:
        return f"{self.speaker_id}/{self.sample_id}/{self.index}/{self.expected}/{self.start:.3f}-{self.end:.3f}"


def eprint(message: str) -> None:
    print(message, flush=True)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def normalize_phone_token(raw: str | None) -> list[str]:
    token = str(raw or "").strip()
    token = token.strip('"').strip()
    token = token.replace("*", "")
    token = re.sub(r"\d+", "", token)
    if not token:
        return []
    if token.lower() in SILENCE:
        return []
    upper = token.upper()
    if upper in ARPABET_TO_COMPARE:
        return ARPABET_TO_COMPARE[upper]
    token = token.lower()
    mapped = RAW_PHONE_MAP.get(token, token)
    parts = [part for part in mapped.split() if part and part not in SILENCE]
    return parts


def normalize_model_token(raw: str | None) -> str | None:
    token = str(raw or "").strip()
    if not token or token.startswith("<"):
        return None
    token = re.sub(r"\d+$", "", token)
    mapped = RAW_PHONE_MAP.get(token, RAW_PHONE_MAP.get(token.lower(), token))
    if mapped in PHONE_INVENTORY:
        return mapped
    if " " in mapped:
        return None
    return mapped if mapped in PHONE_INVENTORY else None


def parse_textgrid_intervals(path: Path) -> dict[str, list[tuple[float, float, str]]]:
    tiers: dict[str, list[tuple[float, float, str]]] = {}
    current_name: str | None = None
    current_start: float | None = None
    current_end: float | None = None
    in_tier = False
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if line.startswith("item ["):
            current_name = None
            in_tier = True
            continue
        if in_tier and line.startswith("name ="):
            current_name = line.split("=", 1)[1].strip().strip('"')
            tiers.setdefault(current_name, [])
            continue
        if current_name and line.startswith("xmin ="):
            try:
                current_start = float(line.split("=", 1)[1].strip())
            except ValueError:
                current_start = None
            continue
        if current_name and line.startswith("xmax ="):
            try:
                current_end = float(line.split("=", 1)[1].strip())
            except ValueError:
                current_end = None
            continue
        if current_name and line.startswith("text ="):
            text = line.split("=", 1)[1].strip().strip('"')
            if current_start is not None and current_end is not None:
                tiers[current_name].append((current_start, current_end, text))
            current_start = None
            current_end = None
    return tiers


def segments_from_annotation(speaker_id: str, path: Path) -> list[Segment]:
    tiers = parse_textgrid_intervals(path)
    phone_intervals = tiers.get("phones") or []
    rows: list[Segment] = []
    for start, end, text in phone_intervals:
        raw_parts = [part.strip() for part in str(text or "").split(",")]
        if not raw_parts or raw_parts[0].lower() in SILENCE:
            continue
        expected_phones = normalize_phone_token(raw_parts[0])
        if not expected_phones:
            continue
        if len(raw_parts) >= 3:
            observed_phones = normalize_phone_token(raw_parts[1])
            op_code = raw_parts[2].strip().lower()
            if op_code == "d":
                human_op = "delete"
                observed_phones = []
            elif op_code == "s":
                human_op = "substitute"
            elif op_code == "a":
                continue
            else:
                human_op = "substitute"
        else:
            observed_phones = list(expected_phones)
            human_op = "match"
        for local_idx, expected in enumerate(expected_phones):
            if human_op == "match":
                observed = expected
            elif human_op == "delete":
                observed = ""
            else:
                observed = observed_phones[min(local_idx, len(observed_phones) - 1)] if observed_phones else ""
            rows.append(
                Segment(
                    speaker_id=speaker_id,
                    sample_id=path.stem,
                    index=len(rows),
                    expected=expected,
                    human_observed=observed,
                    human_op=human_op,
                    start=float(start),
                    end=float(end),
                    raw_text=text,
                )
            )
    return rows


def read_audio16(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32, copy=False)
    if sr == TARGET_SR:
        return audio, sr
    duration = len(audio) / float(sr)
    target_len = max(1, int(round(duration * TARGET_SR)))
    old_x = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    new_x = np.linspace(0.0, duration, num=target_len, endpoint=False)
    resampled = np.interp(new_x, old_x, audio).astype(np.float32)
    return resampled, TARGET_SR


def clip_audio(audio: np.ndarray, start: float, end: float, pad: float = 0.0) -> np.ndarray:
    s = max(0, int(round((start - pad) * TARGET_SR)))
    e = min(len(audio), int(round((end + pad) * TARGET_SR)))
    if e <= s:
        return np.zeros(max(1, int(0.02 * TARGET_SR)), dtype=np.float32)
    return audio[s:e].astype(np.float32, copy=False)


def rms(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x.astype(np.float32))) + 1e-12))


class Wav2Vec2Posterior:
    def __init__(self, model_path: Path, device: str):
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModelForCTC.from_pretrained(str(model_path), local_files_only=True).to(device)
        self.model.eval()
        self.device = device
        with (model_path / "vocab.json").open("r", encoding="utf-8") as f:
            vocab = json.load(f)
        self.id_to_token = {idx: token for token, idx in vocab.items()}

    @torch.inference_mode()
    def utterance_probs(self, audio: np.ndarray) -> np.ndarray:
        inputs = self.extractor(audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=False)
        input_values = inputs.input_values.to(self.device)
        logits = self.model(input_values).logits[0]
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def segment_posterior(self, frame_probs: np.ndarray, start: float, end: float, duration: float) -> dict[str, float]:
        if frame_probs.shape[0] == 0:
            return {phone: 0.0 for phone in PHONE_INVENTORY}
        frame_count = frame_probs.shape[0]
        s = int(math.floor(max(0.0, start) / max(duration, 1e-6) * frame_count))
        e = int(math.ceil(max(0.0, end) / max(duration, 1e-6) * frame_count))
        s = min(max(s, 0), frame_count - 1)
        e = min(max(e, s + 1), frame_count)
        avg = frame_probs[s:e].mean(axis=0)
        posterior = {phone: 0.0 for phone in PHONE_INVENTORY}
        for idx, prob in enumerate(avg):
            token = normalize_model_token(self.id_to_token.get(idx))
            if token in posterior:
                posterior[token] += float(prob)
        total = sum(posterior.values())
        if total > 1e-12:
            posterior = {phone: value / total for phone, value in posterior.items()}
        return posterior


class SpeechAttributeModel:
    def __init__(self, model_path: Path, device: str):
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModelForCTC.from_pretrained(str(model_path), local_files_only=True).to(device)
        self.model.eval()
        self.device = device
        with (model_path / "vocab.json").open("r", encoding="utf-8") as f:
            vocab = json.load(f)
        self.id_by_token = {token: int(idx) for token, idx in vocab.items()}
        attrs = []
        for token in vocab:
            if token.startswith("p_"):
                attr = token[2:]
                if f"n_{attr}" in vocab:
                    attrs.append(attr)
        self.attrs = sorted(attrs)

    @torch.inference_mode()
    def utterance_probs(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) < 8:
            audio = np.pad(audio, (0, 8 - len(audio)))
        inputs = self.extractor(audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=False)
        logits = self.model(inputs.input_values.to(self.device)).logits[0]
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def positive_probs_from_frames(self, frame_probs: np.ndarray, start: float, end: float, duration: float) -> dict[str, float]:
        if frame_probs.shape[0] == 0:
            avg = np.zeros(max(self.id_by_token.values()) + 1, dtype=np.float32)
        else:
            frame_count = frame_probs.shape[0]
            s = int(math.floor(max(0.0, start) / max(duration, 1e-6) * frame_count))
            e = int(math.ceil(max(0.0, end) / max(duration, 1e-6) * frame_count))
            s = min(max(s, 0), frame_count - 1)
            e = min(max(e, s + 1), frame_count)
            avg = frame_probs[s:e].mean(axis=0)
        out: dict[str, float] = {}
        for attr in self.attrs:
            p_id = self.id_by_token.get(f"p_{attr}")
            n_id = self.id_by_token.get(f"n_{attr}")
            if p_id is None or n_id is None:
                continue
            denom = float(avg[p_id] + avg[n_id])
            out[attr] = float(avg[p_id] / denom) if denom > 1e-12 else 0.5
        return out


class WavLMSummary:
    def __init__(self, model_path: Path, device: str):
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True).to(device)
        self.model.eval()
        self.device = device

    @torch.inference_mode()
    def utterance_hidden(self, audio: np.ndarray) -> np.ndarray:
        inputs = self.extractor(audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=False)
        hidden = self.model(inputs.input_values.to(self.device)).last_hidden_state[0]
        return hidden.cpu().numpy()

    @staticmethod
    def segment_score(hidden: np.ndarray, start: float, end: float, duration: float) -> float:
        if hidden.shape[0] == 0:
            return 0.0
        frame_count = hidden.shape[0]
        s = int(math.floor(max(0.0, start) / max(duration, 1e-6) * frame_count))
        e = int(math.ceil(max(0.0, end) / max(duration, 1e-6) * frame_count))
        s = min(max(s, 0), frame_count - 1)
        e = min(max(e, s + 1), frame_count)
        seg_norm = float(np.linalg.norm(hidden[s:e], axis=1).mean())
        utt_norm = float(np.linalg.norm(hidden, axis=1).mean()) + 1e-8
        return seg_norm / utt_norm


def expected_match_score(expected: str, probs: dict[str, float]) -> float:
    attrs = EXPECTED_FEATURES.get(expected)
    if not attrs:
        return 0.5
    values = [float(probs.get(attr, 0.5)) for attr in attrs]
    return float(sum(values) / len(values)) if values else 0.5


def entropy(probabilities: dict[str, float]) -> float:
    total = 0.0
    for value in probabilities.values():
        if value > 1e-12:
            total -= value * math.log(value)
    return float(total)


def verdict_from_ops(human_op: str, predicted_op: str) -> str:
    truth = human_op != "match"
    pred = predicted_op != "match"
    if truth and pred:
        return "TP"
    if truth and not pred:
        return "FN"
    if not truth and pred:
        return "FP"
    return "TN"


def flatten_source_speakers() -> list[str]:
    return [
        speaker
        for l1, group in SPEAKERS_BY_L1.items()
        if l1 != "Vietnamese"
        for gender in ["male", "female"]
        for speaker in group[gender]
    ]


def all_l2_speakers() -> list[str]:
    return [
        speaker
        for group in SPEAKERS_BY_L1.values()
        for gender in ["male", "female"]
        for speaker in group[gender]
    ]


def select_speakers(value: str | None) -> list[str]:
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    return all_l2_speakers()


def make_protocol_manifest(output_dir: Path, details_csv: Path, feature_cache: Path) -> dict:
    source = flatten_source_speakers()
    rows = []
    for fold in VIETNAMESE_FOLDS:
        for feature_set in PROTOCOL_FEATURE_SETS:
            rows.append(
                {
                    "fold_id": fold["fold_id"],
                    "variant": "source_only",
                    "feature_set": feature_set,
                    "train_speakers": ",".join(source),
                    "validation_speakers": ",".join(fold["validation"]),
                    "test_speakers": ",".join(fold["test"]),
                    "details_csv": str(details_csv),
                    "feature_cache": str(feature_cache),
                    "output_dir": str(output_dir / "models" / fold["fold_id"] / "source_only" / feature_set),
                }
            )
    return {
        "schema_version": "l2arctic_frozen_source_vietnamese_raw_cache_v1",
        "feature_cache_schema": SCHEMA_VERSION,
        "details_csv": str(details_csv),
        "feature_cache": str(feature_cache),
        "speaker_inventory": SPEAKERS_BY_L1,
        "run_table": rows,
        "leakage_rules": [
            "Feature cache is generated from raw audio and annotation timing only.",
            "No human_op, human_observed, verdict, or y_error is stored in the feature cache.",
            "The source_only model trains on non-Vietnamese speakers only.",
            "Thresholds are selected on the validation speaker only.",
            "The Vietnamese test speaker is used only for final metrics.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate L2-ARCTIC MDD feature cache from raw wav + TextGrid annotations.")
    parser.add_argument("--corpus-root", default=str(DEFAULT_CORPUS_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--speakers", default="", help="Comma-separated speaker IDs; default is all 24 L2-ARCTIC speakers.")
    parser.add_argument("--limit-files-per-speaker", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wav2vec2-model", default=str(PROJECT_ROOT / "pretrained" / "facebook-wav2vec2-lv-60-espeak-cv-ft"))
    parser.add_argument("--sa-model", default=str(PROJECT_ROOT / "pretrained" / "mostafaashahin-SA_US_Adult"))
    parser.add_argument("--wavlm-model", default=str(PROJECT_ROOT / "pretrained" / "microsoft-wavlm-large"))
    parser.add_argument("--skip-sa", action="store_true")
    parser.add_argument("--skip-wavlm", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=25)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    corpus_root = Path(args.corpus_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details_csv = output_dir / "eval_l2arctic_raw_regenerated_details.csv"
    feature_cache_path = output_dir / "meta_classifier_feature_cache.raw_v1.json"
    manifest_path = output_dir / "manifest.json"
    speakers = select_speakers(args.speakers)
    requested_all_speakers = set(speakers) == set(all_l2_speakers())
    if requested_all_speakers:
        protocol_path = PROJECT_ROOT / "data" / "protocols" / "l2arctic_frozen_source_vietnamese_raw_cache_v1.json"
    else:
        protocol_path = output_dir / "l2arctic_frozen_source_vietnamese_raw_cache_v1.partial_protocol.json"
    protocol_csv_path = protocol_path.with_suffix(".runs.csv")

    device = args.device
    eprint(f"Regenerating raw L2-ARCTIC cache: speakers={len(speakers)} device={device}")
    eprint(f"Output: {output_dir}")

    cache = {}
    if args.resume and feature_cache_path.exists():
        cache = json.loads(feature_cache_path.read_text(encoding="utf-8"))
        eprint(f"Resume: loaded {len(cache)} cached segment features.")

    wav2vec2 = Wav2Vec2Posterior(Path(args.wav2vec2_model), device)
    sa_model = None if args.skip_sa else SpeechAttributeModel(Path(args.sa_model), device)
    wavlm = None if args.skip_wavlm else WavLMSummary(Path(args.wavlm_model), device)

    all_segments: list[Segment] = []
    segment_counts = Counter()
    for speaker in speakers:
        ann_dir = corpus_root / speaker / "annotation"
        files = sorted(ann_dir.glob("*.TextGrid"))
        if args.limit_files_per_speaker:
            files = files[: args.limit_files_per_speaker]
        for tg in files:
            segs = segments_from_annotation(speaker, tg)
            all_segments.extend(segs)
            segment_counts[speaker] += len(segs)

    by_utterance: dict[tuple[str, str], list[Segment]] = {}
    for seg in all_segments:
        by_utterance.setdefault((seg.speaker_id, seg.sample_id), []).append(seg)

    processed = 0
    skipped_existing = 0
    started = time.time()
    for (speaker, sample), segs in sorted(by_utterance.items()):
        if args.resume and all(seg.key in cache for seg in segs):
            skipped_existing += len(segs)
            continue
        wav_path = corpus_root / speaker / "wav" / f"{sample}.wav"
        if not wav_path.exists():
            eprint(f"Missing wav: {wav_path}")
            continue
        audio, sr = read_audio16(wav_path)
        duration = len(audio) / float(sr)
        utt_rms = rms(audio)
        try:
            wav2vec2_probs = wav2vec2.utterance_probs(audio)
            sa_frame_probs = sa_model.utterance_probs(audio) if sa_model is not None else None
            wavlm_hidden = wavlm.utterance_hidden(audio) if wavlm is not None else None
        except Exception as exc:
            eprint(f"Model failure on {speaker}/{sample}: {exc}")
            continue

        for seg in segs:
            seg_duration = max(0.0, seg.end - seg.start)
            posterior = wav2vec2.segment_posterior(wav2vec2_probs, seg.start, seg.end, duration)
            sorted_post = sorted(posterior.items(), key=lambda item: item[1], reverse=True)
            top_phone, top_prob = sorted_post[0]
            second_prob = sorted_post[1][1] if len(sorted_post) > 1 else 0.0
            predicted = top_phone if top_phone else "<delete>"
            predicted_op = "match" if predicted == seg.expected else "substitute"
            clip = clip_audio(audio, seg.start, seg.end)
            energy_ratio = rms(clip) / max(utt_rms, 1e-8)
            sa_probs = sa_model.positive_probs_from_frames(sa_frame_probs, seg.start, seg.end, duration) if sa_frame_probs is not None else {}
            sa_score = expected_match_score(seg.expected, sa_probs) if sa_model is not None else 0.5
            wavlm_score = wavlm.segment_score(wavlm_hidden, seg.start, seg.end, duration) if wavlm_hidden is not None else 0.0
            cache[seg.key] = {
                "posterior": {k: round(float(v), 8) for k, v in posterior.items()},
                "top_phone": predicted,
                "top_prob": round(float(top_prob), 8),
                "second_prob": round(float(second_prob), 8),
                "entropy": round(entropy(posterior), 8),
                "margin": round(float(top_prob - second_prob), 8),
                "sa_expected_match_score": round(float(sa_score), 8),
                "sa_positive_probs": {k: round(float(v), 8) for k, v in sorted(sa_probs.items())},
                "wavlm_presence_score": round(float(wavlm_score), 8),
                "energy_ratio": round(float(energy_ratio), 8),
                "duration_sec": round(float(seg_duration), 8),
                "log_duration_sec": round(math.log(max(seg_duration, 1e-6)), 8),
                "duration_utterance_ratio": round(float(seg_duration / max(duration, 1e-6)), 8),
            }
            processed += 1
        if processed and processed % max(args.save_every, 1) == 0:
            write_json(feature_cache_path, cache)
            eprint(f"  cached={len(cache)} processed_new={processed} elapsed={time.time() - started:.1f}s")

    write_json(feature_cache_path, cache)

    detail_rows = []
    missing_cache = 0
    for seg in all_segments:
        item = cache.get(seg.key)
        if not item:
            missing_cache += 1
            continue
        predicted = item["top_phone"] or "<delete>"
        predicted_op = "match" if predicted == seg.expected else "substitute"
        detail_rows.append(
            {
                "speaker_id": seg.speaker_id,
                "sample_id": seg.sample_id,
                "index": seg.index,
                "expected": seg.expected,
                "human_observed": seg.human_observed,
                "human_op": seg.human_op,
                "predicted": predicted,
                "predicted_op": predicted_op,
                "verdict": verdict_from_ops(seg.human_op, predicted_op),
                "start": f"{seg.start:.6f}",
                "end": f"{seg.end:.6f}",
                "raw_text": seg.raw_text,
            }
        )
    with details_csv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "speaker_id",
            "sample_id",
            "index",
            "expected",
            "human_observed",
            "human_op",
            "predicted",
            "predicted_op",
            "verdict",
            "start",
            "end",
            "raw_text",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(detail_rows)

    protocol = make_protocol_manifest(output_dir, details_csv, feature_cache_path)
    write_json(protocol_path, protocol)
    protocol_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with protocol_csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["fold_id", "variant", "feature_set", "train_speakers", "validation_speakers", "test_speakers", "details_csv", "feature_cache", "output_dir"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(protocol["run_table"])

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "corpus_root": str(corpus_root),
        "output_dir": str(output_dir),
        "details_csv": str(details_csv),
        "feature_cache": str(feature_cache_path),
        "protocol_json": str(protocol_path),
        "protocol_runs_csv": str(protocol_csv_path),
        "global_protocol_written": requested_all_speakers,
        "speakers": speakers,
        "segment_counts": dict(sorted(segment_counts.items())),
        "num_segments_from_annotation": len(all_segments),
        "num_feature_cache_entries": len(cache),
        "num_detail_rows": len(detail_rows),
        "missing_cache_rows": missing_cache,
        "models": {
            "wav2vec2": str(Path(args.wav2vec2_model)),
            "speech_attribute": None if args.skip_sa else str(Path(args.sa_model)),
            "wavlm": None if args.skip_wavlm else str(Path(args.wavlm_model)),
        },
        "leakage_controls": {
            "feature_inputs": ["raw wav audio", "TextGrid segment start/end", "canonical expected phone"],
            "feature_cache_excludes": ["human_op", "human_observed", "verdict", "y_error"],
            "split": "frozen source-to-Vietnamese protocol written separately",
        },
    }
    write_json(manifest_path, manifest)
    eprint(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
