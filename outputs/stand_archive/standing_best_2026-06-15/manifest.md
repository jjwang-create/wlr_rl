# Standing Policy Archive - 2026-06-15

This archive contains the two usable strict-standing policies after fixing PyBullet base-frame handling.

## model_602

- Source checkpoint: `logs/rsl_rl/cartpole_direct/2026-06-15_19-31-44_stand_strict_t4_urdf_contactfix_k64/model_602.pt`
- Archived checkpoint: `model_602/model_602.pt`
- Video: `model_602/stand_10s.mp4`
- Preview: `model_602/stand_t5.jpg`
- Isaac eval: `model_602/eval_isaac.log`
- PyBullet diagnosis: `model_602/diagnose_pybullet.log`

Isaac strict eval:

```text
steps=600 num_envs=64
min_height=0.3346
min_front_height=0.1931
min_rear_height=0.2062
max_abs_roll_deg=3.84
max_abs_pitch_deg=3.28
base_contact_fraction=0.0000
non_wheel_contact_fraction=0.0000
first_done_step=600
timeout_count=64
```

PyBullet diagnosis after coordinate/dynamics fix:

```text
min_front_height=0.2279
min_rear_height=0.2302
max_abs_roll_deg=0.64
max_abs_pitch_deg=0.85
first_non_wheel_contact_time=None
```

Behavior note: nearly static standing, minimal wheel motion.

## model_681

- Source checkpoint: `logs/rsl_rl/cartpole_direct/2026-06-15_21-41-30_stand_strict_from602_conservative_k64/model_681.pt`
- Archived checkpoint: `model_681/model_681.pt`
- Video: `model_681/stand_10s.mp4`
- Preview: `model_681/stand_t5.jpg`
- Isaac eval: `model_681/eval_isaac.log`
- PyBullet diagnosis: `model_681/diagnose_pybullet.log`

Isaac strict eval:

```text
steps=600 num_envs=64
min_height=0.3407
min_front_height=0.2000
min_rear_height=0.2110
max_abs_roll_deg=2.46
max_abs_pitch_deg=3.18
base_contact_fraction=0.0000
non_wheel_contact_fraction=0.0000
first_done_step=600
terminated_count=0
timeout_count=64
```

PyBullet diagnosis after coordinate/dynamics fix:

```text
min_front_height=0.2318
min_rear_height=0.2305
max_abs_roll_deg=0.69
max_abs_pitch_deg=0.78
first_non_wheel_contact_time=None
```

Behavior note: stable standing with small wheel correction and slow drift.

## PyBullet Replay Fix

The old replay script treated PyBullet's inertial/COM base pose as the URDF `base_link` pose. With `URDF_USE_INERTIA_FROM_FILE`, `getBasePositionAndOrientation()` returns the inertial frame. The replay and diagnosis scripts now transform this pose back to URDF `base_link` before building observations, camera targets, and front/rear clearance metrics.
