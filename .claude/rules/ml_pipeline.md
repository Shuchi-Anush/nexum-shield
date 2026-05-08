# ML Pipeline Rules

## Components
- Fingerprinting (perceptual hashing)
- Embeddings (CLIP-like)
- Similarity search

## Constraints
- Embeddings must be deterministic
- Similarity thresholds must be configurable

## Risks
- Adversarial evasion (cropping, noise)
- Threshold miscalibration

## DO NOT
- Hardcode thresholds
- Mix fingerprint + embedding logic blindly