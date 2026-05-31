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


def numeric(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def visual_stat(seg, name, stat="mean"):
    visual = seg.get("visual_attributes") or {}
    mediapipe = visual.get("mediapipe") or {}
    stats = mediapipe.get("stats") or {}
    return numeric((stats.get(name) or {}).get(stat))


def audio_stat(seg, side, name):
    audio = seg.get("audio_attributes") or {}
    payload = audio.get(side) or {}
    return numeric(payload.get(name))


def visual_evidence(seg, target_label):
    evidence = []
    if target_label.startswith("v_"):
        labiodental = visual_stat(seg, "labiodental_contact_proxy")
        rounding = visual_stat(seg, "lip_rounding_proxy")
        if labiodental is not None:
            evidence.append(
                {
                    "source": "MediaPipe",
                    "feature": "labiodental_contact_proxy",
                    "value": labiodental,
                    "interpretation": "lower value can support weak /v/ labiodental contact",
                }
            )
        if rounding is not None:
            evidence.append(
                {
                    "source": "MediaPipe",
                    "feature": "lip_rounding_proxy",
                    "value": rounding,
                    "interpretation": "higher rounding can support /v/ drifting toward /w/",
                }
            )

    if target_label.startswith("final_") or target_label == "final_consonant_omitted":
        motion = ((seg.get("visual_attributes") or {}).get("mouth_clip") or {}).get("motion_mean")
        if motion is not None:
            evidence.append(
                {
                    "source": "MediaPipe/mouth_clip",
                    "feature": "motion_mean",
                    "value": motion,
                    "interpretation": "low final mouth motion can support weak final closure",
                }
            )
    return evidence


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
    if seg.get("visual_attributes"):
        confidence += 0.07
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
            "evidence": [{"source": "rule_engine", "reason": "silence segment skipped"}],
        }

    label = infer_label(std_phone, real_phone, seg)
    evidence = [
        {
            "source": "Wav2Vec2 phoneme comparison",
            "target_phoneme": std_phone,
            "observed_phoneme": real_phone,
            "alignment_op": seg.get("alignment_op"),
        }
    ]
    evidence.extend(audio_evidence(seg))
    evidence.extend(visual_evidence(seg, label))

    return {
        "label": label,
        "confidence": confidence_for(label, seg),
        "evidence": evidence,
    }
