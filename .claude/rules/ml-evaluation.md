# ML EVALUATION & THRESHOLDING

## Core Principle
ML outputs are probabilistic — never binary truth.

## Required Metrics

Every model MUST track:
- precision
- recall
- false positive rate
- false negative rate

## Thresholding

- Similarity score ≠ decision
- MUST define threshold bands:

LOW_CONFIDENCE   → ignore
MEDIUM_CONFIDENCE → human review
HIGH_CONFIDENCE  → auto-flag

## Calibration

Thresholds MUST be:
- dataset-specific
- periodically recalibrated

## Feedback Loop

System MUST support:
- human review outcomes
- false positive corrections
- model re-training inputs

## Adversarial Considerations

System MUST handle:
- cropping
- watermark removal
- re-encoding
- slight transformations

## Anti-Patterns (FORBIDDEN)

- Hardcoded thresholds without validation
- No feedback loop
- Blind trust in embeddings