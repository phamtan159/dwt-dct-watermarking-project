SILENCE_PHONES = {"", "sil", "sp", "spn", "<eps>", "<sil>", None}
TH_PHONES = {"theta", "th", "θ", "Î¸"}
DH_PHONES = {"dh", "ð", "Ã°"}
V_LIKE_ERRORS = {"j", "y", "w", "d", "z", "i"}
FINAL_CONSONANTS = {"s", "z", "f", "v", "p", "b", "m", "t", "d", "k", "g"}
TARGET_FINAL_LABELS = {
    "d": "final_d_weak_or_omitted",
    "t": "final_t_weak_or_omitted",
    "p": "final_p_weak_or_omitted",
}
TH_PHONES.update({"\u03b8"})
DH_PHONES.update({"\u00f0"})
VOICING_PAIRS = {("z", "s"), ("v", "f"), ("d", "t"), ("g", "k")}


def normalize_phone(phone):
    if phone is None:
        return ""
    return str(phone).strip().lower()


def primary_evidence_policy(seg):
    std_phone = normalize_phone(
        seg.get("target_phoneme")
        or seg.get("phoneme_standard")
        or seg.get("standard_phone")
        or seg.get("phone")
    )
    return "audio" if std_phone else "none"


def default_primary_evidence(seg):
    label = normalize_phone(
        seg.get("final_error_label")
        or seg.get("error_code")
        or seg.get("error_id")
        or seg.get("label")
    )
    if label in {"ok", "no_error", "0"}:
        return "none"
    return primary_evidence_policy(seg)


def numeric(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def audio_stat(seg, side, name):
    audio = seg.get("audio_attributes") or {}
    payload = audio.get(side) or {}
    return numeric(payload.get(name))


def audio_evidence(seg):
    evidence = []
    support = numeric(seg.get("acoustic_support"))
    if support is not None:
        evidence.append(
            {
                "source": "Wav2Vec2 alignment",
                "feature": "acoustic_support",
                "value": support,
            }
        )

    standard_norm = audio_stat(seg, "standard", "embedding_norm")
    real_norm = audio_stat(seg, "real", "embedding_norm")
    if standard_norm is not None and real_norm is not None:
        evidence.append(
            {
                "source": "WavLM",
                "feature": "embedding_norm_delta",
                "value": round(abs(standard_norm - real_norm), 6),
            }
        )

    for item in (seg.get("feature_errors") or [])[:6]:
        evidence.append(
            {
                "source": "phonetic_features",
                "feature": item.get("attribute"),
                "category": item.get("category"),
                "expected": item.get("expected"),
                "observed": item.get("observed"),
            }
        )
    return evidence


def infer_label(std_phone, real_phone, seg):
    if std_phone == real_phone:
        return "OK"
    if std_phone in {"theta", "th", "Î¸", "ÃŽÂ¸", "θ"} and real_phone in {"t", "th", ""}:
        return "th_weak_or_omitted" if real_phone == "" else "th_to_t"
    if std_phone in {"dh", "Ã°", "ÃƒÂ°", "ð"} and real_phone in {"d", ""}:
        return "dh_weak_or_omitted" if real_phone == "" else "dh_to_d"
    if std_phone in TH_PHONES and real_phone in {"t", "th", ""}:
        return "th_to_t"
    if std_phone in DH_PHONES and real_phone in {"d", ""}:
        return "dh_to_d"
    if std_phone == "v" and real_phone in V_LIKE_ERRORS:
        return "v_to_y_w"
    if std_phone in TARGET_FINAL_LABELS and real_phone == "":
        return TARGET_FINAL_LABELS[std_phone]
    if std_phone == "r" and real_phone == "" and str(seg.get("word", "")).lower() in {"mother", "brother", "weather"}:
        return "final_er_r_weak_or_omitted"
    if std_phone in FINAL_CONSONANTS and real_phone == "":
        return "final_consonant_omitted"
    if (std_phone, real_phone) in VOICING_PAIRS:
        return "final_voicing_error"
    return seg.get("error_code") or "OTHER"


def confidence_for(label, seg):
    if label == "OK":
        return 0.9

    support = numeric(seg.get("acoustic_support"), 0.0)
    align_cost = numeric(seg.get("alignment_cost"), 0.5)
    confidence = 0.55

    if support is not None:
        confidence += min(0.2, max(0.0, support) * 0.2)
    if align_cost is not None:
        confidence += min(0.2, max(0.0, align_cost) * 0.12)

    if seg.get("audio_attributes"):
        confidence += 0.08
    if seg.get("feature_errors"):
        confidence += 0.06

    return round(min(confidence, 0.92), 3)


def rule_segment(seg):
    std_phone = normalize_phone(seg.get("phoneme_standard") or seg.get("standard_phone"))
    real_phone = normalize_phone(seg.get("phoneme_real") or seg.get("phone") or seg.get("phoneme"))

    if std_phone in SILENCE_PHONES:
        return {
            "label": "OK",
            "confidence": 0.0,
            "primary_evidence": "none",
            "evidence": [{"source": "rule_engine", "reason": "silence segment skipped"}],
        }

    label = infer_label(std_phone, real_phone, seg)
    evidence_segment = dict(seg)
    evidence_segment["final_error_label"] = label
    evidence = [
        {
            "source": "Wav2Vec2 phoneme comparison",
            "target_phoneme": std_phone,
            "observed_phoneme": real_phone,
            "alignment_op": seg.get("alignment_op"),
        }
    ]
    evidence.extend(audio_evidence(seg))

    return {
        "label": label,
        "confidence": confidence_for(label, seg),
        "primary_evidence": default_primary_evidence(evidence_segment),
        "evidence": evidence,
    }
