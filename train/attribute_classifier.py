from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.append(str(TOOLS_DIR))

try:
    from rule_engine import primary_evidence_policy, rule_segment
except Exception:  # pragma: no cover - keep feature extraction usable in isolation.
    primary_evidence_policy = None
    rule_segment = None


OK_LABEL = "OK"

AUDIO_NUMERIC_FIELDS = (
    "num_frames",
    "embedding_mean",
    "embedding_std",
    "embedding_norm",
    "activation_min",
    "activation_max",
)

MEDIAPIPE_FEATURES = (
    "lip_width",
    "mouth_opening",
    "mouth_opening_ratio",
    "jaw_opening_proxy",
    "lip_rounding_proxy",
    "labiodental_contact_proxy",
)

MEDIAPIPE_STATS = ("mean", "std", "min", "max")

MOUTH_CLIP_FIELDS = (
    "num_frames",
    "readable_frames",
    "brightness_mean",
    "brightness_std",
    "contrast_mean",
    "motion_mean",
    "motion_std",
    "motion_max",
)

REQUESTED_AUDIO_SUMMARY_FIELDS = (
    "num_frames",
    "embedding_std",
    "embedding_norm",
    "activation_min",
    "activation_max",
)

REQUESTED_VISUAL_STATS = (
    "mouth_opening",
    "mouth_opening_ratio",
    "labiodental_contact_proxy",
)

REQUESTED_MOUTH_CLIP_FIELDS = (
    "motion_mean",
    "motion_std",
    "motion_max",
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_label(value) -> str:
    if value is None or str(value).strip() == "":
        return OK_LABEL
    label = str(value).strip()
    if label.upper() == OK_LABEL or label.lower() in {"0", "no_error", "ok/correct", "correct"}:
        return OK_LABEL
    return label


def target_label(seg: dict, target: str = "label") -> str:
    if target == "severity":
        return normalize_label(seg.get("final_error_severity"))
    if target == "primary_evidence":
        return normalize_label(seg.get("final_error_primary_evidence"))
    if target == "category":
        return normalize_label(seg.get("final_error_category"))

    for key in ("final_error_label", "error_code", "label", "error_id", "error"):
        value = seg.get(key)
        if value is not None and str(value).strip() != "":
            return normalize_label(value)
    return OK_LABEL


def safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def add_numeric(features: dict[str, float], name: str, value) -> None:
    value = safe_float(value)
    if value is not None:
        features[name] = float(value)


def add_bool(features: dict[str, float], name: str, value) -> None:
    features[name] = 1.0 if value else 0.0


def add_category(features: dict[str, float], prefix: str, value) -> None:
    value = "" if value is None else str(value).strip().lower()
    if value:
        features[f"{prefix}={value}"] = 1.0


def add_binary_feature_dict(features: dict[str, float], prefix: str, payload) -> None:
    if not isinstance(payload, dict):
        return
    for name, value in payload.items():
        add_numeric(features, f"{prefix}.{name}", value)


def add_vector_head(features: dict[str, float], prefix: str, values) -> None:
    if not isinstance(values, list):
        return
    for index, value in enumerate(values[:16]):
        add_numeric(features, f"{prefix}.vector_head_{index:02d}", value)


def add_audio_side(features: dict[str, float], prefix: str, payload: dict | None) -> None:
    if not isinstance(payload, dict):
        return
    for field in AUDIO_NUMERIC_FIELDS:
        add_numeric(features, f"{prefix}.{field}", payload.get(field))
    add_vector_head(features, prefix, payload.get("vector_head"))


def add_audio_delta(features: dict[str, float], standard: dict | None, real: dict | None) -> None:
    if not isinstance(standard, dict) or not isinstance(real, dict):
        return

    for field in AUDIO_NUMERIC_FIELDS:
        left = safe_float(standard.get(field))
        right = safe_float(real.get(field))
        if left is not None and right is not None:
            features[f"audio.delta.{field}"] = right - left
            features[f"audio.abs_delta.{field}"] = abs(right - left)

    std_head = standard.get("vector_head")
    real_head = real.get("vector_head")
    if isinstance(std_head, list) and isinstance(real_head, list):
        for index, (left_raw, right_raw) in enumerate(zip(std_head[:16], real_head[:16])):
            left = safe_float(left_raw)
            right = safe_float(right_raw)
            if left is not None and right is not None:
                features[f"audio.delta.vector_head_{index:02d}"] = right - left


def add_audio_attributes(features: dict[str, float], seg: dict) -> None:
    audio = seg.get("audio_attributes") or {}
    standard = audio.get("standard") or seg.get("wavlm_standard_attributes")
    real = audio.get("real") or seg.get("wavlm_real_attributes")
    add_bool(features, "has_audio_attributes", bool(standard or real))
    add_audio_side(features, "audio.standard", standard)
    add_audio_side(features, "audio.real", real)
    add_audio_delta(features, standard, real)


def add_visual_attributes(features: dict[str, float], seg: dict) -> None:
    visual = seg.get("visual_attributes") or {}
    add_bool(features, "has_visual_attributes", bool(visual))

    mediapipe = visual.get("mediapipe") or {}
    add_numeric(features, "visual.mediapipe.frames_selected", mediapipe.get("frames_selected"))
    add_numeric(features, "visual.mediapipe.frames_detected", mediapipe.get("frames_detected"))
    add_numeric(features, "visual.mediapipe.face_detection_rate", mediapipe.get("face_detection_rate"))
    add_bool(features, "visual.mediapipe.tongue_landmarks_available", mediapipe.get("tongue_landmarks_available"))

    stats = mediapipe.get("stats") or {}
    for feature_name in MEDIAPIPE_FEATURES:
        payload = stats.get(feature_name) or {}
        for stat_name in MEDIAPIPE_STATS:
            add_numeric(features, f"visual.mediapipe.{feature_name}.{stat_name}", payload.get(stat_name))

    mouth_clip = visual.get("mouth_clip") or {}
    for field in MOUTH_CLIP_FIELDS:
        add_numeric(features, f"visual.mouth_clip.{field}", mouth_clip.get(field))
    add_vector_head(features, "visual.mouth_clip", mouth_clip.get("vector_head"))


def add_phonetic_attributes(features: dict[str, float], seg: dict) -> None:
    expected = seg.get("expected_features")
    observed = seg.get("observed_features")
    observed_symbolic = seg.get("observed_features_symbolic")
    predicted = seg.get("predicted_features")

    add_bool(features, "has_expected_phonetic_features", isinstance(expected, dict))
    add_bool(features, "has_observed_phonetic_features", isinstance(observed, dict))
    add_bool(features, "has_predicted_sa_features", isinstance(predicted, dict))
    add_category(features, "phonetic.observed_source", seg.get("observed_feature_source"))
    add_category(features, "phonetic.expected_status", seg.get("expected_features_status"))
    add_category(features, "phonetic.observed_symbolic_status", seg.get("observed_features_symbolic_status"))
    add_category(features, "phonetic.predicted_status", seg.get("predicted_features_status"))

    add_binary_feature_dict(features, "phonetic.expected", expected)
    add_binary_feature_dict(features, "phonetic.observed", observed)
    add_binary_feature_dict(features, "phonetic.observed_symbolic", observed_symbolic)
    add_binary_feature_dict(features, "phonetic.predicted_sa", predicted)

    if isinstance(expected, dict) and isinstance(observed, dict):
        for name, expected_value in expected.items():
            if name not in observed:
                continue
            try:
                delta = float(observed[name]) - float(expected_value)
            except (TypeError, ValueError):
                continue
            features[f"phonetic.delta.{name}"] = delta
            features[f"phonetic.abs_delta.{name}"] = abs(delta)

    feature_errors = seg.get("feature_errors") or []
    add_numeric(features, "phonetic.feature_error_count", len(feature_errors))
    for item in feature_errors:
        add_category(features, "phonetic.error_attribute", item.get("attribute"))
        add_category(features, "phonetic.error_category", item.get("category"))

    for category in seg.get("feature_error_categories") or []:
        add_category(features, "phonetic.error_category_present", category)


def add_speech_attribute_prediction(features: dict[str, float], seg: dict) -> None:
    prediction = seg.get("speech_attribute_prediction") or {}
    add_bool(features, "has_speech_attribute_prediction", bool(prediction))
    if not isinstance(prediction, dict):
        return

    for field in (
        "start",
        "end",
        "effective_start",
        "effective_end",
        "context_seconds",
        "min_duration_seconds",
        "num_frames",
        "sampling_rate",
    ):
        add_numeric(features, f"speech_attr.{field}", prediction.get(field))

    start = safe_float(prediction.get("effective_start"))
    end = safe_float(prediction.get("effective_end"))
    if start is not None and end is not None:
        add_numeric(features, "speech_attr.effective_duration", max(0.0, end - start))

    add_category(features, "speech_attr.window_source", prediction.get("window_source"))
    confidence = prediction.get("feature_confidence") or {}
    if isinstance(confidence, dict):
        for name, value in confidence.items():
            add_numeric(features, f"speech_attr.confidence.{name}", value)


def add_requested_wavlm_summary(features: dict[str, float], seg: dict) -> None:
    audio = seg.get("audio_attributes") or {}
    standard = audio.get("standard") or seg.get("wavlm_standard_attributes")
    real = audio.get("real") or seg.get("wavlm_real_attributes")
    add_bool(features, "requested.has_wavlm_real", isinstance(real, dict))
    add_bool(features, "requested.has_wavlm_standard", isinstance(standard, dict))

    for side_name, payload in (("standard", standard), ("real", real)):
        if not isinstance(payload, dict):
            continue
        for field in REQUESTED_AUDIO_SUMMARY_FIELDS:
            add_numeric(features, f"requested.wavlm.{side_name}.{field}", payload.get(field))

    if isinstance(standard, dict) and isinstance(real, dict):
        for field in REQUESTED_AUDIO_SUMMARY_FIELDS:
            left = safe_float(standard.get(field))
            right = safe_float(real.get(field))
            if left is not None and right is not None:
                features[f"requested.wavlm.delta.{field}"] = right - left
                features[f"requested.wavlm.abs_delta.{field}"] = abs(right - left)


def add_requested_speech_confidence(features: dict[str, float], seg: dict) -> None:
    prediction = seg.get("speech_attribute_prediction") or {}
    add_bool(features, "requested.has_speech_attribute_prediction", isinstance(prediction, dict) and bool(prediction))
    if not isinstance(prediction, dict):
        return

    for field in ("start", "end", "effective_start", "effective_end", "context_seconds", "min_duration_seconds", "num_frames"):
        add_numeric(features, f"requested.speech_attr.{field}", prediction.get(field))

    start = safe_float(prediction.get("effective_start"))
    end = safe_float(prediction.get("effective_end"))
    if start is not None and end is not None:
        add_numeric(features, "requested.duration.effective_seconds", max(0.0, end - start))

    confidence = prediction.get("feature_confidence") or {}
    if isinstance(confidence, dict):
        for name, value in confidence.items():
            add_numeric(features, f"requested.speech_attr.confidence.{name}", value)
        add_numeric(features, "requested.aspiration_proxy.confidence_aspiration", confidence.get("aspiration"))
        add_numeric(features, "requested.aspiration_proxy.confidence_aspirated", confidence.get("aspirated"))

        fricative = safe_float(confidence.get("fricative"))
        plosive = safe_float(confidence.get("plosive"))
        if fricative is not None:
            features["requested.frication_vs_stop.fricative_confidence"] = fricative
        if plosive is not None:
            features["requested.frication_vs_stop.stop_confidence"] = plosive
        if fricative is not None and plosive is not None:
            features["requested.frication_vs_stop.fricative_minus_stop"] = fricative - plosive
            features["requested.frication_vs_stop.stop_minus_fricative"] = plosive - fricative

        for name in ("vowel", "front", "back", "central", "high", "mid", "low", "long", "short", "round", "diphthong", "monophthong"):
            add_numeric(features, f"requested.vowel_quality.{name}", confidence.get(name))


def add_requested_visual_summary(features: dict[str, float], seg: dict) -> None:
    visual = seg.get("visual_attributes") or {}
    add_bool(features, "requested.has_visual_attributes", bool(visual))

    mediapipe = visual.get("mediapipe") or {}
    add_numeric(features, "requested.visual.face_detection_rate", mediapipe.get("face_detection_rate"))
    add_bool(features, "requested.visual.tongue_landmarks_available", mediapipe.get("tongue_landmarks_available"))

    stats = mediapipe.get("stats") or {}
    for feature_name in REQUESTED_VISUAL_STATS:
        payload = stats.get(feature_name) or {}
        for stat_name in MEDIAPIPE_STATS:
            add_numeric(features, f"requested.visual.{feature_name}.{stat_name}", payload.get(stat_name))

    mouth_clip = visual.get("mouth_clip") or {}
    for field in REQUESTED_MOUTH_CLIP_FIELDS:
        add_numeric(features, f"requested.visual.{field}", mouth_clip.get(field))


def add_requested_duration_and_transition(
    features: dict[str, float],
    seg: dict,
    include_wavlm: bool = True,
) -> None:
    start = safe_float(seg.get("start"))
    end = safe_float(seg.get("end"))
    raw_start = safe_float(seg.get("raw_start"))
    raw_end = safe_float(seg.get("raw_end"))
    if start is not None and end is not None:
        aligned_duration = max(0.0, end - start)
        features["requested.duration.aligned_seconds"] = aligned_duration
        features["requested.aspiration_proxy.aligned_duration_seconds"] = aligned_duration
    if raw_start is not None and raw_end is not None:
        features["requested.duration.raw_seconds"] = max(0.0, raw_end - raw_start)
    if not include_wavlm:
        return

    audio = seg.get("audio_attributes") or {}
    standard = audio.get("standard") or seg.get("wavlm_standard_attributes")
    real = audio.get("real") or seg.get("wavlm_real_attributes")
    if isinstance(standard, dict) and isinstance(real, dict):
        standard_frames = safe_float(standard.get("num_frames"))
        real_frames = safe_float(real.get("num_frames"))
        if standard_frames is not None and real_frames is not None:
            features["requested.transition.frame_delta"] = real_frames - standard_frames
            features["requested.transition.abs_frame_delta"] = abs(real_frames - standard_frames)

        for field in ("activation_min", "activation_max", "embedding_norm", "embedding_std"):
            left = safe_float(standard.get(field))
            right = safe_float(real.get(field))
            if left is not None and right is not None:
                features[f"requested.energy_spectral.delta.{field}"] = right - left
                features[f"requested.energy_spectral.abs_delta.{field}"] = abs(right - left)

        for field in ("activation_max", "embedding_std", "num_frames"):
            left = safe_float(standard.get(field))
            right = safe_float(real.get(field))
            if right is not None:
                features[f"requested.aspiration_proxy.real_{field}"] = right
            if left is not None and right is not None:
                features[f"requested.aspiration_proxy.delta_{field}"] = right - left


def add_requested_feature_set(features: dict[str, float], seg: dict, include_wavlm: bool = True) -> None:
    add_requested_duration_and_transition(features, seg, include_wavlm=include_wavlm)
    add_requested_speech_confidence(features, seg)
    add_requested_visual_summary(features, seg)
    if include_wavlm:
        add_requested_wavlm_summary(features, seg)


def add_recommended_target_and_position(features: dict[str, float], seg: dict) -> None:
    target = (
        seg.get("target_phoneme")
        or seg.get("phoneme_standard")
        or seg.get("standard_phone")
    )
    add_category(features, "recommended.target_phoneme", target)
    add_numeric(features, "recommended.position.word_index", seg.get("word_index"))
    add_numeric(features, "recommended.position.phoneme_index_in_word", seg.get("phoneme_index_in_word"))
    index_in_word = safe_float(seg.get("phoneme_index_in_word"))
    if index_in_word is not None:
        add_bool(features, "recommended.position.is_word_initial", index_in_word == 0)


def add_recommended_feature_set(features: dict[str, float], seg: dict, include_wavlm: bool = True) -> None:
    add_recommended_target_and_position(features, seg)
    add_requested_duration_and_transition(features, seg, include_wavlm=include_wavlm)
    add_requested_speech_confidence(features, seg)

    visual_policy = visual_policy_for_segment(seg)
    add_category(features, "recommended.primary_evidence_policy", visual_policy)
    if visual_policy in {"visual", "audio_visual"}:
        add_requested_visual_summary(features, seg)
    else:
        add_bool(features, "recommended.visual_skipped_by_policy", True)
        add_bool(features, "recommended.has_visual_attributes", bool(seg.get("visual_attributes")))

    if include_wavlm:
        add_requested_wavlm_summary(features, seg)


def add_rule_features(features: dict[str, float], seg: dict) -> None:
    if rule_segment is None:
        return
    result = rule_segment(seg)
    add_category(features, "rule.label", result.get("label"))
    add_numeric(features, "rule.confidence", result.get("confidence"))
    add_category(features, "rule.primary_evidence", result.get("primary_evidence"))
    evidence = result.get("evidence") or []
    add_numeric(features, "rule.evidence_count", len(evidence))
    for item in evidence:
        source = item.get("source")
        reason = item.get("reason")
        feature = item.get("feature")
        if source:
            add_category(features, "rule.evidence_source", source)
        if reason:
            add_category(features, "rule.reason", reason)
        if feature:
            add_category(features, "rule.feature", feature)


def visual_policy_for_segment(seg: dict) -> str:
    if primary_evidence_policy is None:
        return "audio_visual"
    return primary_evidence_policy(seg)


def segment_features(
    seg: dict,
    include_rule: bool = True,
    include_wavlm: bool = True,
    include_phoneme_identity: bool = True,
    include_phonetic_features: bool = True,
    feature_set: str = "full",
) -> dict[str, float]:
    features: dict[str, float] = {}
    if feature_set == "recommended":
        add_recommended_feature_set(features, seg, include_wavlm=include_wavlm)
        return features
    if feature_set == "requested":
        add_requested_feature_set(features, seg, include_wavlm=include_wavlm)
        return features

    standard = seg.get("phoneme_standard") or seg.get("standard_phone")
    real = seg.get("phoneme_real") or seg.get("phone") or seg.get("phoneme")

    start = safe_float(seg.get("start"))
    end = safe_float(seg.get("end"))
    if start is not None:
        features["time.start"] = start
    if end is not None:
        features["time.end"] = end
    if start is not None and end is not None:
        features["time.duration"] = max(0.0, end - start)

    if include_phoneme_identity:
        add_category(features, "phoneme.standard", standard)
        add_category(features, "phoneme.real", real)
        add_bool(features, "phoneme.is_deleted", real in {None, ""})
        add_bool(
            features,
            "phoneme.same_as_standard",
            str(standard or "").strip().lower() == str(real or "").strip().lower(),
        )
    add_category(features, "alignment.op", seg.get("alignment_op"))
    add_numeric(features, "alignment.cost", seg.get("alignment_cost"))
    add_numeric(features, "alignment.acoustic_support", seg.get("acoustic_support"))
    add_numeric(features, "position.word_index", seg.get("word_index"))
    add_numeric(features, "position.phoneme_index_in_word", seg.get("phoneme_index_in_word"))
    add_bool(features, "clip.visual_exists", seg.get("visual_clip_exists"))
    add_bool(features, "clip.audio_exists", seg.get("audio_clip_exists"))

    if include_wavlm:
        add_audio_attributes(features, seg)

    visual_policy = visual_policy_for_segment(seg)
    add_category(features, "primary_evidence.policy", visual_policy)
    if visual_policy in {"visual", "audio_visual"}:
        add_visual_attributes(features, seg)
    else:
        add_bool(features, "has_visual_attributes", bool(seg.get("visual_attributes")))
        add_bool(features, "visual.skipped_by_primary_evidence_policy", True)

    if include_phonetic_features:
        add_phonetic_attributes(features, seg)
    add_speech_attribute_prediction(features, seg)
    if include_rule:
        add_rule_features(features, seg)
    return features


def iter_segments(dataset: dict):
    for sample in dataset.get("samples", []):
        sample_id = sample.get("id") or sample.get("video_id") or sample.get("audio_id")
        for seg in sample.get("segments", []):
            yield sample_id, sample, seg


def labels_from_dataset(dataset: dict, observed_labels: list[str]) -> list[str]:
    observed = []
    for label in observed_labels:
        label = normalize_label(label)
        if label not in observed:
            observed.append(label)
    observed_set = set(observed)

    labels: list[str] = []
    raw_map = dataset.get("label_map") or {}
    if isinstance(raw_map, dict):
        indexed = []
        fallback = []
        for key, value in raw_map.items():
            label = normalize_label(key)
            if isinstance(value, int):
                indexed.append((value, label))
            elif isinstance(value, dict):
                raw_index = value.get("index", value.get("id"))
                label = normalize_label(value.get("code", key))
                try:
                    indexed.append((int(raw_index), label))
                except (TypeError, ValueError):
                    fallback.append(label)
            else:
                fallback.append(label)
        for _, label in sorted(indexed):
            if label in observed_set and label not in labels:
                labels.append(label)
        for label in fallback:
            if label in observed_set and label not in labels:
                labels.append(label)

    for label in observed:
        if label not in labels:
            labels.append(label)

    if OK_LABEL in labels:
        labels.remove(OK_LABEL)
    return [OK_LABEL] + labels


def build_examples(
    dataset: dict,
    include_rule: bool = True,
    include_wavlm: bool = True,
    include_phoneme_identity: bool = True,
    include_phonetic_features: bool = True,
    target: str = "label",
    feature_set: str = "full",
    feature_names: list[str] | None = None,
    label_names: list[str] | None = None,
):
    rows = []
    observed_labels = []
    feature_space = set(feature_names or [])

    for sample_id, sample, seg in iter_segments(dataset):
        features = segment_features(
            seg,
            include_rule=include_rule,
            include_wavlm=include_wavlm,
            include_phoneme_identity=include_phoneme_identity,
            include_phonetic_features=include_phonetic_features,
            feature_set=feature_set,
        )
        label = target_label(seg, target=target)
        rows.append(
            {
                "sample_id": sample_id,
                "speaker_id": sample.get("speaker_id") or "unknown_speaker",
                "segment_id": seg.get("id"),
                "sample": sample,
                "segment": seg,
                "features": features,
                "label": label,
            }
        )
        observed_labels.append(label)
        if feature_names is None:
            feature_space.update(features.keys())

    if feature_names is None:
        feature_names = sorted(feature_space)
    if label_names is None:
        label_names = labels_from_dataset(dataset, observed_labels)
    return rows, feature_names, label_names


def vectorize(features: dict[str, float], feature_names: list[str]) -> np.ndarray:
    return np.array([float(features.get(name, 0.0)) for name in feature_names], dtype=np.float32)


def matrix_from_rows(rows, feature_names: list[str]) -> np.ndarray:
    if not rows:
        return np.zeros((0, len(feature_names)), dtype=np.float32)
    return np.stack([vectorize(row["features"], feature_names) for row in rows], axis=0)


def labels_to_indices(rows, label_names: list[str]) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(label_names)}
    return np.array([label_to_index.get(row["label"], label_to_index[OK_LABEL]) for row in rows], dtype=np.int64)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


class AttributeSoftmaxClassifier:
    def __init__(self, input_dim: int, num_labels: int, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, 0.01, size=(input_dim, num_labels)).astype(np.float32)
        self.bias = np.zeros((num_labels,), dtype=np.float32)
        self.mean = np.zeros((input_dim,), dtype=np.float32)
        self.std = np.ones((input_dim,), dtype=np.float32)

    def fit_normalizer(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        self.mean = x.mean(axis=0).astype(np.float32)
        self.std = x.std(axis=0).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def logits(self, x: np.ndarray) -> np.ndarray:
        x_norm = self.normalize(x)
        return x_norm @ self.weights + self.bias

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.logits(x))

    def save(self, path: Path, metadata: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            weights=self.weights,
            bias=self.bias,
            mean=self.mean,
            std=self.std,
            metadata=json.dumps(metadata, ensure_ascii=False),
        )

    @classmethod
    def load(cls, path: Path):
        payload = np.load(path, allow_pickle=False)
        metadata = json.loads(str(payload["metadata"]))
        model = cls(
            input_dim=len(metadata["feature_names"]),
            num_labels=len(metadata["label_names"]),
        )
        model.weights = payload["weights"].astype(np.float32)
        model.bias = payload["bias"].astype(np.float32)
        model.mean = payload["mean"].astype(np.float32)
        model.std = payload["std"].astype(np.float32)
        return model, metadata


def feature_evidence(features: dict[str, float], limit: int = 8) -> list[dict]:
    ranked = sorted(
        ((name, value) for name, value in features.items() if abs(value) > 0),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    return [
        {"source": "attribute_classifier", "feature": name, "value": round(float(value), 6)}
        for name, value in ranked[:limit]
    ]


def predict_payload(dataset: dict, model: AttributeSoftmaxClassifier, metadata: dict) -> dict:
    include_rule = bool(metadata.get("include_rule_features", True))
    include_wavlm = bool(metadata.get("include_wavlm_features", True))
    include_phoneme_identity = bool(metadata.get("include_phoneme_identity_features", True))
    include_phonetic_features = bool(metadata.get("include_phonetic_features", True))
    target = str(metadata.get("target", "label"))
    feature_set = str(metadata.get("feature_set", "full"))
    rows, _, _ = build_examples(
        dataset,
        include_rule=include_rule,
        include_wavlm=include_wavlm,
        include_phoneme_identity=include_phoneme_identity,
        include_phonetic_features=include_phonetic_features,
        target=target,
        feature_set=feature_set,
        feature_names=metadata["feature_names"],
        label_names=metadata["label_names"],
    )
    by_sample: dict[str, dict] = {}

    for row in rows:
        sample_id = row["sample_id"]
        sample_payload = by_sample.setdefault(
            sample_id,
            {
                "id": sample_id,
                "audio_id": row["sample"].get("audio_id"),
                "video_id": row["sample"].get("video_id"),
                "sample_id": row["sample"].get("sample_id"),
                "speaker_id": row["sample"].get("speaker_id") or row.get("speaker_id"),
                "segments": [],
            },
        )
        x = vectorize(row["features"], metadata["feature_names"]).reshape(1, -1)
        probs = model.predict_proba(x).reshape(-1)
        pred_index = int(np.argmax(probs))
        top_indices = np.argsort(probs)[::-1][: min(3, len(metadata["label_names"]))]

        sample_payload["segments"].append(
            {
                "id": row["segment_id"],
                "sample_id": sample_id,
                "segment_id": row["segment_id"],
                "predicted_label": metadata["label_names"][pred_index],
                "confidence": round(float(probs[pred_index]), 4),
                "topk": [
                    {
                        "label": metadata["label_names"][int(index)],
                        "confidence": round(float(probs[int(index)]), 4),
                    }
                    for index in top_indices
                ],
                "evidence": feature_evidence(row["features"]),
            }
        )

    return {
        "schema_version": "attribute_classifier_predictions_v1",
        "model": "attribute_softmax_classifier",
        "source_dataset": dataset.get("annotation_dir"),
        "feature_count": len(metadata["feature_names"]),
        "labels": metadata["label_names"],
        "target": target,
        "samples": list(by_sample.values()),
    }
