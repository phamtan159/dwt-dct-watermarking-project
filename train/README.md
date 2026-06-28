# Train Artifacts

Thu muc nay chi giu artifact can cho pipeline hien tai.

```text
current/stage1_meta_mdd_classifier.pt
current/stage2_observed_phone_classifier.pt
```

Y nghia:

- `stage1_meta_mdd_classifier.pt`: Meta-classifier phat hien segment dung/sai.
- `stage2_observed_phone_classifier.pt`: Classifier du doan observed phoneme khi Stage 1 bao loi.

Nhung benchmark cu nhu L2-ARCTIC split thu nghiem, Speechocean split, SNR audit, threshold sweep, test predictions khong nam trong runtime pipeline. Neu can viet paper, tao lai bang cac tool trong `tools/25_*`, `tools/26_*`, `tools/30_*`, `tools/31_*` va luu ket qua benchmark o ngoai repo hoac trong thu muc artifact rieng.
