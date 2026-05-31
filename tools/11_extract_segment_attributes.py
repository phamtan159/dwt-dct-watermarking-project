"""
Build phoneme-level speech/visual attribute records for feedback and classifiers.

This tool runs after compare/dataset creation. It maps each target and observed
phoneme to phonetic features from Speech-Attribute-Transcription-main and, when
a local speech-attribute model is available, can also predict observed features
directly from the raw phoneme audio span.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_P2ATT_MAP = (
    WORKSPACE_ROOT
    / "Speech-Attribute-Transcription-main"
    / "data"
    / "Phoneme2att_ipa_att_Diph_v2.csv"
)
SA_MODEL_NAME = "mostafaashahin-SA_US_Adult"
DEFAULT_SA_MODEL = PROJECT_ROOT / "pretrained" / SA_MODEL_NAME
SILENCE_PHONES = {"", "sil", "sp", "spn", "<eps>", "<sil>", "SIL", None}
NUMERIC_AUDIO_FIELDS = (
    "num_frames",
    "hidden_size",
    "embedding_mean",
    "embedding_std",
    "embedding_norm",
    "activation_min",
    "activation_max",
)
FEATURE_CATEGORIES = {
    "voiced": "voicing",
    "plosive": "manner",
    "fricative": "manner",
    "affricates": "manner",
    "approximant": "manner",
    "nasal": "manner",
    "semivowel": "manner",
    "vowel": "manner",
    "consonant": "manner",
    "bilabial": "place",
    "labiodental": "place",
    "dental": "place",
    "alveolar": "place",
    "postalveolar": "place",
    "velar": "place",
    "glotal": "place",
    "palatal": "place",
    "labial": "place",
    "high": "vowel_height",
    "nearhigh": "vowel_height",
    "highmid": "vowel_height",
    "mid": "vowel_height",
    "lowmid": "vowel_height",
    "nearlow": "vowel_height",
    "low": "vowel_height",
    "front": "vowel_backness",
    "central": "vowel_backness",
    "back": "vowel_backness",
    "round": "rounding",
    "rhotic": "rhoticity",
    "diphthong": "vowel_quality",
    "long": "vowel_length",
}
PHONE_ALIASES = {
    "th": "theta",
    "theta": "θ",
    "dh": "ð",
    "sh": "ʃ",
    "zh": "ʒ",
    "ch": "tʃ",
    "jh": "dʒ",
    "ng": "ŋ",
    "r": "ɹ",
    "ɾ": "ɹ",
    "y": "j",
    "hh": "h",
    "ɡ": "g",
    "ay": "aɪ",
    "aj": "aɪ",
    "ey": "eɪ",
    "ej": "eɪ",
    "ow": "oʊ",
    "aw": "aʊ",
    "oy": "ɔɪ",
    "uw": "u",
    "iy": "i",
    "ih": "ɪ",
    "uh": "ʊ",
    "eh": "ɛ",
    "ae": "æ",
    "aa": "ɑ",
    "ao": "ɔ",
    "ah": "ʌ",
    "er": "ɝ",
    "ɚ": "ɝ",
    "ə": "ʌ",
}


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def to_posix(path: Path | str | None) -> str | None:
    if path is None:
        return None
    return str(path).replace("\\", "/")


def first_present(*values):
    for value in values:
        if value is not None and str(value).strip() != "":
            return value
    return None


def normalize_label(value) -> str:
    if value is None or str(value).strip() == "":
        return "OK"
    label = str(value).strip()
    if label.upper() == "OK" or label in {"0", "no_error"}:
        return "OK"
    return label


def normalize_phone(phone) -> str:
    if phone is None:
        return ""
    value = str(phone).strip().lower()
    value = value.replace("ː", "")
    value = value.replace(" ", "")
    value = value.replace("ɡ", "g")
    return value


def phone_lookup_keys(phone) -> list[str]:
    value = normalize_phone(phone)
    if not value:
        return []

    keys = [value]
    alias = PHONE_ALIASES.get(value)
    if alias:
        keys.append(PHONE_ALIASES.get(alias, alias))
    if value == "theta":
        keys.append("θ")
    return list(dict.fromkeys(keys))


def is_silence(phone) -> bool:
    return phone in SILENCE_PHONES or normalize_phone(phone) in SILENCE_PHONES


class PhoneticFeatureMap:
    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Phoneme-to-attribute map not found: {path}")
        self.path = path
        self.lookup: dict[str, dict[str, int]] = {}
        self.display_phone: dict[str, str] = {}
        self.attribute_names: list[str] = []
        self._load()

    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"Empty phoneme-to-attribute map: {self.path}")

        excluded = {"Phoneme_arpa", "Phoneme_ipa", "Phoneme_other", "Language"}
        self.attribute_names = [
            name for name in rows[0].keys() if name and name not in excluded
        ]

        for row in rows:
            features = {}
            for name in self.attribute_names:
                try:
                    features[name] = int(float(row.get(name, 0) or 0))
                except ValueError:
                    features[name] = 0

            phones = [
                row.get("Phoneme_arpa"),
                row.get("Phoneme_ipa"),
                row.get("Phoneme_other"),
            ]
            display = row.get("Phoneme_ipa") or row.get("Phoneme_arpa") or ""
            for phone in phones:
                for key in phone_lookup_keys(phone):
                    self.lookup[key] = features
                    self.display_phone[key] = display

    def features_for(self, phone) -> tuple[dict[str, int] | None, str]:
        for key in phone_lookup_keys(phone):
            if key in self.lookup:
                return dict(self.lookup[key]), "ok"
        if is_silence(phone):
            return None, "silence"
        return None, "unknown_phoneme"


class SpeechAttributePredictor:
    def __init__(self, model_path: Path | str, device: str | None = None):
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        self.model_path = str(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForCTC.from_pretrained(self.model_path, local_files_only=True).to(self.device)
        self.model.eval()
        self.sampling_rate = int(self.processor.feature_extractor.sampling_rate)
        vocab = self.processor.tokenizer.get_vocab()
        attributes = []
        for token in vocab:
            if token.startswith("p_") and f"n_{token[2:]}" in vocab:
                attributes.append(token[2:])
        self.attributes = sorted(set(attributes))
        self.token_ids = {
            attr: {
                "p": self.processor.tokenizer.convert_tokens_to_ids(f"p_{attr}"),
                "n": self.processor.tokenizer.convert_tokens_to_ids(f"n_{attr}"),
            }
            for attr in self.attributes
        }
        self._audio_cache = {}

    def _load_audio(self, audio_path: Path):
        import numpy as np
        import soundfile as sf

        key = str(audio_path.resolve())
        if key not in self._audio_cache:
            data, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
            if getattr(data, "ndim", 1) > 1:
                data = np.mean(data, axis=1)
            self._audio_cache[key] = (data, int(sample_rate))
        return self._audio_cache[key]

    def predict(
        self,
        audio_path: Path,
        start: float,
        end: float,
        context_seconds: float = 0.04,
        min_duration_seconds: float = 0.08,
    ) -> dict:
        audio, sample_rate = self._load_audio(audio_path)
        if sample_rate != self.sampling_rate:
            raise ValueError(
                f"Expected {self.sampling_rate} Hz audio, got {sample_rate} Hz: {audio_path}"
            )

        original_start = float(start)
        original_end = float(end)
        audio_duration = len(audio) / sample_rate
        start = max(0.0, original_start - max(0.0, float(context_seconds)))
        end = min(audio_duration, original_end + max(0.0, float(context_seconds)))
        duration = end - start
        min_duration = max(0.0, float(min_duration_seconds))
        if duration < min_duration:
            center = (original_start + original_end) / 2.0
            half = min_duration / 2.0
            start = max(0.0, center - half)
            end = min(audio_duration, center + half)
            if end - start < min_duration:
                start = max(0.0, end - min_duration)
                end = min(audio_duration, start + min_duration)

        start_index = max(0, int(round(float(start) * sample_rate)))
        end_index = min(len(audio), int(round(float(end) * sample_rate)))
        if end_index <= start_index:
            raise ValueError(f"Invalid segment window {start}-{end} in {audio_path}")

        clip = audio[start_index:end_index]
        inputs = self.processor(
            audio=clip,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            logits = self.model(**inputs).logits

        features = {}
        confidences = {}
        for attr, ids in self.token_ids.items():
            pair_logits = logits[:, :, [ids["n"], ids["p"]]]
            probs = self.torch.softmax(pair_logits, dim=-1)[0, :, 1]
            confidence = float(probs.mean().detach().cpu())
            features[attr] = 1 if confidence >= 0.5 else 0
            confidences[attr] = round(confidence if features[attr] else 1.0 - confidence, 4)

        return {
            "features": features,
            "feature_confidence": confidences,
            "num_frames": int(logits.shape[1]),
            "sampling_rate": sample_rate,
            "original_start": original_start,
            "original_end": original_end,
            "effective_start": round(start_index / sample_rate, 6),
            "effective_end": round(end_index / sample_rate, 6),
        }


def numeric(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def audio_side(seg: dict, side: str) -> dict | None:
    audio = seg.get("audio_attributes") or {}
    if side in audio and isinstance(audio[side], dict):
        return audio[side]
    key = "wavlm_standard_attributes" if side == "standard" else "wavlm_real_attributes"
    value = seg.get(key)
    return value if isinstance(value, dict) else None


def compact_audio_attributes(seg: dict) -> dict | None:
    standard = audio_side(seg, "standard")
    real = audio_side(seg, "real")
    if not standard and not real:
        return None

    deltas = {}
    if standard and real:
        for field in NUMERIC_AUDIO_FIELDS:
            left = numeric(standard.get(field))
            right = numeric(real.get(field))
            if left is not None and right is not None:
                deltas[field] = round(right - left, 6)

    return {
        "model": first_present(
            (real or {}).get("model"),
            (standard or {}).get("model"),
            (seg.get("audio_attributes") or {}).get("model"),
        ),
        "standard": standard,
        "real": real,
        "delta_real_minus_standard": deltas,
    }


def compare_features(expected: dict | None, observed: dict | None) -> tuple[list[dict], list[str]]:
    if not expected or not observed:
        return [], []

    errors = []
    categories = []
    for name in sorted(expected):
        if name not in observed:
            continue
        if int(expected[name]) == int(observed[name]):
            continue
        category = FEATURE_CATEGORIES.get(name, "other")
        errors.append(
            {
                "attribute": name,
                "category": category,
                "expected": int(expected[name]),
                "observed": int(observed[name]),
            }
        )
        if category not in categories:
            categories.append(category)
    return errors, categories


def label_for_segment(seg: dict, fallback=None) -> str:
    for key in ("final_error_label", "label", "error_code", "error_id", "error"):
        value = seg.get(key)
        if value is not None and str(value).strip() != "":
            return normalize_label(value)
    return normalize_label(fallback)


def resolve_path(raw_path, project_root: Path) -> Path | None:
    if raw_path is None or str(raw_path).strip() == "":
        return None
    raw = Path(str(raw_path))
    if raw.is_absolute():
        return raw
    candidates = [
        project_root / raw,
        project_root / "models" / raw,
        Path.cwd() / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return project_root / raw


def infer_audio_path(sample: dict, rel_id: str | None, audio_dir: Path) -> Path | None:
    for key in ("audio_path", "audio"):
        path = resolve_path(sample.get(key), PROJECT_ROOT)
        if path and path.exists():
            return path
    if rel_id:
        candidate = audio_dir / Path(rel_id).with_suffix(".wav")
        if candidate.exists():
            return candidate
    sample_id = first_present(sample.get("sample_id"), sample.get("audio_id"), sample.get("id"))
    if sample_id:
        candidate = audio_dir / f"{Path(str(sample_id)).name}.wav"
        if candidate.exists():
            return candidate
    return None


def segment_window(seg: dict, mode: str) -> tuple[float | None, float | None, str]:
    if mode == "observed":
        start = numeric(seg.get("raw_start"))
        end = numeric(seg.get("raw_end"))
        if start is not None and end is not None and end > start:
            return start, end, "wav2vec2_raw_span"

    start = numeric(seg.get("start"))
    end = numeric(seg.get("end"))
    if start is not None and end is not None and end > start:
        return start, end, "mfa_aligned_span"
    return None, None, "missing_span"


def predict_segment_features(
    predictor: SpeechAttributePredictor | None,
    audio_path: Path | None,
    seg: dict,
    window_mode: str,
    context_seconds: float,
    min_duration_seconds: float,
) -> tuple[dict | None, str, dict | None]:
    if predictor is None:
        return None, "model_unavailable", None
    if audio_path is None or not audio_path.exists():
        return None, "audio_missing", None

    start, end, source = segment_window(seg, window_mode)
    if start is None or end is None:
        return None, "missing_span", None

    try:
        result = predictor.predict(
            audio_path,
            start,
            end,
            context_seconds=context_seconds,
            min_duration_seconds=min_duration_seconds,
        )
    except Exception as exc:  # Keep export robust even if one clip fails.
        return None, f"prediction_failed: {exc}", None

    metadata = {
        "window_source": source,
        "start": start,
        "end": end,
        "effective_start": result.get("effective_start"),
        "effective_end": result.get("effective_end"),
        "context_seconds": context_seconds,
        "min_duration_seconds": min_duration_seconds,
        "num_frames": result.get("num_frames"),
        "sampling_rate": result.get("sampling_rate"),
        "feature_confidence": result.get("feature_confidence"),
    }
    return result["features"], "ok", metadata


def llm_feedback_input(sample: dict, seg: dict, feature_errors: list[dict], categories: list[str]) -> dict:
    return {
        "word": seg.get("word"),
        "target_phoneme": seg.get("phoneme_standard") or seg.get("standard_phone"),
        "observed_phoneme": seg.get("phoneme_real") or seg.get("phone") or seg.get("phoneme"),
        "final_error_label": label_for_segment(seg),
        "feature_error_categories": categories,
        "feature_errors": feature_errors,
        "audio_attributes": compact_audio_attributes(seg),
        "visual_attributes": seg.get("visual_attributes"),
        "instruction": "Generate concise Vietnamese teacher-style feedback and one concrete correction drill.",
        "transcript": sample.get("transcript"),
    }


def enrich_segment(
    sample: dict,
    seg: dict,
    fallback_label,
    feature_map: PhoneticFeatureMap,
    predictor: SpeechAttributePredictor | None,
    audio_path: Path | None,
    sa_window: str,
    context_seconds: float,
    min_duration_seconds: float,
) -> dict | None:
    standard_phone = first_present(
        seg.get("phoneme_standard"),
        seg.get("standard_phone"),
        seg.get("target_phoneme"),
    )
    observed_phone = first_present(
        seg.get("phoneme_real"),
        seg.get("phone"),
        seg.get("phoneme"),
        seg.get("observed_phoneme"),
    )
    if is_silence(standard_phone):
        return None

    expected, expected_status = feature_map.features_for(standard_phone)
    observed_symbolic, observed_symbolic_status = feature_map.features_for(observed_phone)
    predicted, predicted_status, prediction_metadata = predict_segment_features(
        predictor,
        audio_path,
        seg,
        sa_window,
        context_seconds,
        min_duration_seconds,
    )
    observed = predicted or observed_symbolic
    observed_source = "speech_attribute_model" if predicted else "phoneme_symbolic"
    feature_errors, categories = compare_features(expected, observed)

    output = {
        **seg,
        "id": seg.get("id"),
        "word": seg.get("word"),
        "phoneme_standard": standard_phone,
        "phoneme_real": observed_phone,
        "target_phoneme": standard_phone,
        "observed_phoneme": observed_phone,
        "label": label_for_segment(seg, fallback_label),
        "expected_features": expected,
        "expected_features_status": expected_status,
        "observed_features": observed,
        "observed_feature_source": observed_source,
        "observed_features_symbolic": observed_symbolic,
        "observed_features_symbolic_status": observed_symbolic_status,
        "predicted_features": predicted,
        "predicted_features_status": predicted_status,
        "speech_attribute_prediction": prediction_metadata,
        "feature_errors": feature_errors,
        "feature_error_categories": categories,
        "audio_attributes": compact_audio_attributes(seg),
        "visual_attributes": seg.get("visual_attributes"),
    }
    output["llm_feedback_input"] = llm_feedback_input(sample, output, feature_errors, categories)
    return output


def sample_from_annotation(path: Path, root: Path, annotation: dict, audio_dir: Path) -> dict:
    rel_id = path.relative_to(root).with_suffix("").as_posix()
    sample_id = first_present(annotation.get("sample_id"), annotation.get("audio_id"), Path(rel_id).name)
    parts = Path(rel_id).parts
    speaker_id = first_present(
        annotation.get("speaker_id"),
        parts[0] if len(parts) > 1 else None,
        "unknown_speaker",
    )
    audio_path = audio_dir / Path(rel_id).with_suffix(".wav")
    return {
        "id": rel_id,
        "sample_id": sample_id,
        "audio_id": annotation.get("audio_id", sample_id),
        "video_id": annotation.get("video_id"),
        "speaker_id": speaker_id,
        "transcript": annotation.get("transcript"),
        "annotation_path": to_posix(path),
        "audio_path": to_posix(audio_path),
        "alignment_method": annotation.get("alignment_method"),
        "alignment_score": annotation.get("alignment_score"),
        "phoneme_model": annotation.get("phoneme_model"),
        "phoneme_decoder": annotation.get("phoneme_decoder"),
        "audio_attribute_model": annotation.get("attribute_model"),
        "segments": annotation.get("segments", []),
    }


def samples_from_legacy_audio_dataset(payload: list[dict]) -> list[dict]:
    samples = []
    for item in payload:
        phonemes = item.get("phonemes", [])
        labels = item.get("labels", [])
        segments = []
        for index, phone_item in enumerate(phonemes):
            label = labels[index] if index < len(labels) else "OK"
            segment_id = phone_item.get("segment_id") or f"{index:03d}_{phone_item.get('phone', '')}"
            segments.append(
                {
                    "id": segment_id,
                    "start": phone_item.get("s", phone_item.get("start")),
                    "end": phone_item.get("e", phone_item.get("end")),
                    "phoneme_real": phone_item.get("phone"),
                    "phoneme_standard": phone_item.get("standard_phone"),
                    "word": phone_item.get("word"),
                    "word_index": phone_item.get("word_index"),
                    "phoneme_index_in_word": phone_item.get("phoneme_index_in_word"),
                    "label": label,
                }
            )
        samples.append({**item, "segments": segments})
    return samples


def resolve_auto_input() -> Path:
    final_dataset = PROJECT_ROOT / "data" / "final" / "dataset.json"
    compare_dir = PROJECT_ROOT / "data" / "annotations" / "compare"
    if final_dataset.exists():
        try:
            payload = load_json(final_dataset)
            if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
                return final_dataset
        except Exception:
            pass
    if compare_dir.exists():
        return compare_dir
    return final_dataset


def load_samples(input_path: Path, audio_dir: Path) -> tuple[list[dict], dict]:
    if input_path.is_dir():
        paths = sorted(path for path in input_path.rglob("*.json") if path.is_file())
        samples = [
            sample_from_annotation(path, input_path, load_json(path), audio_dir)
            for path in paths
        ]
        return samples, {"kind": "compare_dir", "path": to_posix(input_path)}

    payload = load_json(input_path)
    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        return payload["samples"], {
            "kind": payload.get("schema_version", "dataset"),
            "path": to_posix(input_path),
            "label_map": payload.get("label_map"),
        }
    if isinstance(payload, dict) and isinstance(payload.get("segments"), list):
        sample = sample_from_annotation(input_path, input_path.parent, payload, audio_dir)
        return [sample], {"kind": "compare_file", "path": to_posix(input_path)}
    if isinstance(payload, list):
        return samples_from_legacy_audio_dataset(payload), {
            "kind": "legacy_audio_dataset",
            "path": to_posix(input_path),
        }
    raise ValueError(f"Unsupported input JSON format: {input_path}")


def resolve_sa_model_path(raw_model: str) -> Path:
    model_path = Path(raw_model)
    return model_path


def init_predictor(args) -> tuple[SpeechAttributePredictor | None, dict]:
    requested_model_path = Path(args.sa_model)
    model_path = resolve_sa_model_path(args.sa_model)
    status = {
        "requested_path": to_posix(requested_model_path),
        "path": to_posix(model_path),
        "window": args.sa_window,
        "status": "disabled" if args.disable_sa_model else "missing",
        "attributes": [],
    }
    if args.disable_sa_model:
        return None, status
    if not model_path.exists():
        status["status"] = f"missing: copy the full model into pretrained/{SA_MODEL_NAME}"
        return None, status
    try:
        predictor = SpeechAttributePredictor(model_path, device=args.device)
    except Exception as exc:
        status["status"] = f"load_failed: {exc}"
        return None, status
    status.update(
        {
            "status": "ok",
            "device": predictor.device,
            "sampling_rate": predictor.sampling_rate,
            "attributes": predictor.attributes,
        }
    )
    return predictor, status


def enrich_samples(samples: list[dict], args, feature_map: PhoneticFeatureMap, predictor) -> list[dict]:
    audio_dir = Path(args.audio_dir)
    enriched_samples = []
    for sample in samples:
        rel_id = sample.get("id") or sample.get("video_id") or sample.get("audio_id")
        audio_path = infer_audio_path(sample, rel_id, audio_dir)
        output_sample = {
            key: value
            for key, value in sample.items()
            if key not in {"segments", "phonemes", "labels"}
        }
        output_sample.setdefault("id", rel_id)
        output_sample.setdefault("speaker_id", sample.get("speaker_id") or "unknown_speaker")
        output_sample["audio_path"] = to_posix(audio_path or sample.get("audio_path"))

        labels = sample.get("labels", [])
        segments = []
        for index, seg in enumerate(sample.get("segments", [])):
            fallback_label = labels[index] if index < len(labels) else None
            enriched = enrich_segment(
                sample,
                seg,
                fallback_label,
                feature_map,
                predictor,
                audio_path,
                args.sa_window,
                args.sa_context_seconds,
                args.sa_min_duration_seconds,
            )
            if enriched is not None or args.include_silence:
                segments.append(enriched or seg)

        output_sample["segments"] = segments
        enriched_samples.append(output_sample)
    return enriched_samples


def write_speaker_views(output_path: Path, output: dict) -> None:
    speakers: dict[str, list[dict]] = {}
    for sample in output.get("samples", []):
        speaker_id = sample.get("speaker_id") or "unknown_speaker"
        speakers.setdefault(str(speaker_id), []).append(sample)

    for speaker_id, samples in sorted(speakers.items()):
        speaker_output = {
            **{key: value for key, value in output.items() if key != "samples"},
            "speaker_id": speaker_id,
            "num_samples": len(samples),
            "num_segments": sum(len(sample.get("segments", [])) for sample in samples),
            "samples": samples,
        }
        write_json(output_path.parent / "speakers" / speaker_id / output_path.name, speaker_output)


def main() -> None:
    configure_stdout()
    parser = argparse.ArgumentParser(description="Extract phoneme-level feature attributes.")
    parser.add_argument("--input", default="auto", help="Dataset JSON, compare JSON, compare dir, or auto.")
    parser.add_argument("--output", default="data/final/segment_attributes.json")
    parser.add_argument("--p2att-map", default=str(DEFAULT_P2ATT_MAP))
    parser.add_argument("--audio-dir", default="data/audio")
    parser.add_argument("--sa-model", default=str(DEFAULT_SA_MODEL))
    parser.add_argument("--sa-window", choices=["observed", "standard"], default="observed")
    parser.add_argument("--sa-context-seconds", type=float, default=0.04)
    parser.add_argument("--sa-min-duration-seconds", type=float, default=0.08)
    parser.add_argument("--device", default=None)
    parser.add_argument("--disable-sa-model", action="store_true")
    parser.add_argument("--include-silence", action="store_true")
    parser.add_argument("--no-speaker-views", action="store_true")
    args = parser.parse_args()

    input_path = resolve_auto_input() if args.input == "auto" else Path(args.input)
    output_path = Path(args.output)
    feature_map = PhoneticFeatureMap(Path(args.p2att_map))
    predictor, model_status = init_predictor(args)
    samples, source = load_samples(input_path, Path(args.audio_dir))
    enriched_samples = enrich_samples(samples, args, feature_map, predictor)

    output = {
        "schema_version": "segment_attributes_v1",
        "source": source,
        "phoneme_to_attribute_map": to_posix(feature_map.path),
        "speech_attribute_model": model_status,
        "feature_categories": FEATURE_CATEGORIES,
        "num_samples": len(enriched_samples),
        "num_segments": sum(len(sample.get("segments", [])) for sample in enriched_samples),
        "samples": enriched_samples,
    }
    if source.get("label_map") is not None:
        output["label_map"] = source.get("label_map")

    write_json(output_path, output)
    if not args.no_speaker_views:
        write_speaker_views(output_path, output)

    print(f"Wrote {output_path} ({output['num_samples']} samples, {output['num_segments']} segments)")
    print(f"Input: {input_path}")
    print(f"Speech attribute model: {model_status['status']}")


if __name__ == "__main__":
    main()
