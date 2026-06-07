# Current Model Structure

## Overview

The current model is a phase-aware, body-part-aware, kinematic reasoning network for student action error recognition.
It follows a simple design principle:

1. RGB is the primary modality for semantic understanding.
2. Soft temporal phases are predicted from RGB features.
3. Human-centric body parts are discovered from RGB spatial features.
4. Skeleton is used only as late-stage motion evidence.
5. Video-level prediction is aggregated from phase-level predictions.

This keeps the model interpretable and avoids treating skeleton as a second semantic backbone.

## End-to-End Pipeline

### 1. RGB Backbone

Input video is sampled as a clip of RGB frames and passed through a temporal backbone.

- Supported backbones:
  - `x3d`
  - `simple_cnn`
  - `action_slot`

The backbone outputs:

- `temporal_features`: temporal sequence features of shape `B x T' x C`
- `spatial_features`: spatiotemporal feature map of shape `B x T' x H' x W' x C`

RGB is the only modality used to build the main semantic representation.

### 2. Soft Phase Assignment

`SoftPhaseAssignment` receives `temporal_features` and predicts continuous phase durations and boundaries.

It produces:

- `phase_weights`: soft continuous masks over time, shape `B x K x T'`
- `phase_features`: phase-level RGB features, shape `B x K x C`
- `phase_boundaries`
- `phase_durations`

This module uses boundary-based continuous soft segmentation, so each phase still corresponds to a continuous time segment rather than scattered frames.

### 3. Phase-Aware Body-Part Discovery

`PhaseAwarePartPrototypeAggregator` receives:

- `spatial_features`
- `phase_weights`

It performs human-centric body-part discovery with fixed semantic part priors.

Current default preset:

- `full_body_7`
  - head
  - torso
  - left_arm
  - right_arm
  - left_leg
  - right_leg
  - feet_contact

It outputs:

- `part_attn`: phase-aware part attention maps
- `part_tokens`: phase-aware RGB part tokens, shape `B x K x P x C`
- `phase_part_features`: pooled RGB part representation, shape `B x K x C`

This is the main human-centric semantic branch.

### 4. Late Skeleton Motion Aggregation

Skeleton is not used in early feature extraction and does not participate in phase prediction.

Instead, after RGB phases are produced, skeleton is aligned to these RGB phases using the same `phase_weights`.

`PhaseAwareSkeletonAggregator` performs:

1. skeleton normalization
2. joint velocity computation
3. joint-to-body-part pooling
4. phase-wise pooling with RGB phase masks

It outputs:

- `phase_skeleton_part_coords`: phase-wise part coordinates, shape `B x K x P x D`
- `phase_skeleton_part_velocity`: phase-wise part velocities, shape `B x K x P x D`

Important: the skeleton branch is now parameter-free and no longer provides an independent semantic feature stream. It only supplies normalized kinematic quantities.

### 5. Kinematic Chain Reasoning

`KinematicChainReasoner` receives:

- `rgb_part_tokens`
- `phase_skeleton_part_coords`
- `phase_skeleton_part_velocity`

It constructs an explicit body-part graph according to the selected part preset.

For `full_body_7`, the graph models connections such as:

- head <-> torso
- torso <-> arms
- torso <-> legs
- legs <-> feet contact

For each phase, the module reasons over:

- RGB part semantics
- part displacement
- part velocity
- edge length
- edge speed

It outputs:

- `phase_kinematic_features`: phase-level kinematic representation, shape `B x K x C`

This is the core “motion correctness evidence” module.

### 6. Phase-Level Error Prediction

The final phase representation is intentionally compact.

Current phase context is:

- `phase_part_features`
- `phase_kinematic_features`

These are concatenated into a tensor of shape `B x K x 2C`, then passed to `phase_error_head`.

The model outputs:

- `phase_logits`: phase-wise error logits, shape `B x K x E`

### 7. Video-Level Aggregation

There is no separate video classification head anymore.

Video-level logits are directly aggregated from phase logits using max-MIL:

- `logits = max_k(phase_logits)`

This means the model assumes:

- if one phase strongly exhibits an error, the whole video should be considered erroneous for that class.

This keeps the decision logic simple and consistent with weak video-level supervision.

## Loss Design

### Main Loss

- `error loss`
  - binary cross-entropy on video-level logits

### Optional Auxiliary Losses

- `phase_duration`
  - regularizes phase durations to avoid collapse
- `part_diversity`
  - encourages different part slots to be distinct
- `part_entropy`
  - regularizes part attention sharpness
- `part_consistency`
  - encourages slot semantics to remain stable across phases
- `skeleton_smoothness`
  - penalizes noisy joint acceleration
- `kinematic_length`
  - regularizes phase-wise body-chain edge length consistency
- `kinematic_symmetry`
  - regularizes left-right motion symmetry when appropriate

## Design Motivation

### Why RGB first?

RGB contains the strongest semantic cues for:

- action identity
- phase transitions
- visible body appearance
- contextual execution clues

So RGB should determine where the suspicious phase and suspicious body part are.

### Why skeleton late?

Skeleton is strong at geometry and motion consistency, but weak at full semantic understanding.

Using skeleton only in the late stage makes its role very clear:

- RGB discovers possible error locations
- skeleton verifies whether the motion is mechanically abnormal

### Why no separate video head?

The task is weakly supervised at video level, but the model is explicitly phase-aware.

So a separate video head is unnecessary.
Using phase logits plus MIL aggregation is simpler, more interpretable, and better aligned with the architecture.

## Current Practical Summary

The current model can be summarized as:

`RGB Backbone -> Soft Phase Segmentation -> Phase-Aware Body-Part Discovery -> Late Skeleton Kinematic Pooling -> Kinematic Chain Reasoning -> Phase Error Prediction -> Max-MIL Video Prediction`

This is the current code-aligned architecture and should be treated as the canonical model description for writing, reporting, and figure generation.
