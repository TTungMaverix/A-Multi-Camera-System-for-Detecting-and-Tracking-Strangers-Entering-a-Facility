# Phase F Body Fallback

## Why This Was Necessary

The current dataset is not face-centric.

In practice:

- many events have no usable face crop
- many overlap cases are still valuable for re-identification because the body/person crop is strong

So unknown-to-unknown association must not collapse when face quality is poor.

## What Changed

### 1. Stronger body descriptor

The body feature extractor now uses a lightweight descriptor built from:

- HOG on a normalized body crop
- HS color histograms on full / upper / lower body regions

This is still a lightweight, interpretable baseline.

No heavy new body model was introduced.

### 2. Representative-aware body evidence

Association now compares a new body observation against:

- top body references in the gallery
- the representative body embedding

instead of only a single weak reference effect.

### 3. Real body-primary fallback

The policy now supports:

- `face-primary` when face is reliable
- `body-primary` when face is missing or unreliable

Crucially, low-quality face is no longer treated as mandatory secondary evidence just because an embedding exists.

This was a major source of false splits before Phase F.

## New Logging Fields

Association logs now expose:

- `modality_primary`
- `modality_secondary`
- `body_fallback_used`
- `face_unusable_reason`

This makes it explicit when the system reuses an unknown ID by body-first reasoning.

## Observed Effect

On the post-fix run before re-tuning:

- `body_fallback_used_count = 156`
- `face_unusable_event_count = 156`

This confirms that the new flow is solving the right problem:

- unknown-to-unknown cross-camera matching
- not just known-face matching
