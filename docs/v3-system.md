# V3 system and evidence boundaries

V3 is a modular manipulation agent in a MuJoCo scene. The default application configuration is V3. `AgentConfig()` keeps its V2 defaults for regression compatibility. The original V2 compatibility package is retained because the test suite exercises it.

## Execution loop

1. `agent/task_interpreter.py` extracts a goal from the instruction. `agent/task_planner.py` checks the required skill sequence.
2. `simulation/mujoco_camera.py` supplies calibrated wrist RGB-D observations. Normal captures are 640 × 480, with 1280 × 960 detail captures available on demand.
3. `perception/visual_identity.py` and the identity resolver use multimodal image evidence to bind a requested object to a tracked instance. Candidate comparison and side observations supplement full-scene grounding. The complete instruction remains part of identity assessment.
4. `perception/active_perception.py`, depth processing, geometry analysis, and `world/world_model.py` maintain measured instances and uncertainty. Local observations support tracking and geometry. Explicit attribute conflicts can veto a proposed identity.
5. `manipulation/` and `spatial/` propose grasps, transport routes, and placement regions from those measurements. `simulation/planning_backend.py` builds the robot's collision representation from the observed world.
6. `skills/manipulation.py` coordinates approach, contact validation, probe lift, lift, transport, release, and retreat. `robot/mujoco_robot.py` exposes robot proprioception and actuation, with contact and motion checks in `validation/`.
7. `validation/task_validator.py` evaluates fresh final evidence. `recovery/` selects bounded retries and replanning. `studio/` presents these events alongside the observer scene.

All module paths above are relative to [`src/`](../src). The simulator constructs the scene separately. Runtime perception and task packages cannot import scene registries or simulator-truth object APIs; the boundary is checked by `tests/test_agent_ground_truth_guard.py`. Physical tests may inspect simulator state as a test oracle.

## Identity, geometry, and uncertainty

The multimodal model supplies evidence about which visible object matches a description. RGB-D measurement supplies locations and geometry. Robot calibration supplies kinematics and the camera transform. Confidence, current visibility, and identity authorization are distinct fields. Temporary occlusion can preserve bounded memory, while instance association conflicts require renewed model confirmation. Final placement uses fresh observations of the destination.

The current segmentation and tracking methods are engineering baselines. They do not guarantee separation of touching objects, recognition of arbitrary novel instances, or robust behavior on transparent and reflective objects. Model confidence is not a calibrated probability of correctness.

## Current release regression

A recorded cup-on-cylinder task exposed a repeatable contact failure. With the wrist tilted and the cup still above the support, a one-second jaw opening allowed the body to rotate while the upper finger pads remained in contact. The V3 loaded-release command now clears that interval with a smooth bounded opening. Physical friction remains continuous throughout loaded release. A stopped release preserves the contact mode, and stop/safety callbacks remain active.

The regression fixture `tests/fixtures/v3_tilted_release.json` contains numerical simulation state reconstructed from that task. It contains no credentials or camera uploads. Tests execute the actual MuJoCo controller and dynamics, compare stable releases across small support offsets, reproduce tipping under the preceding opening behavior, and check interruption and V2 compatibility. Production release control does not read fixture or object-registry values.

## Evidence included in this snapshot

- `docs/media/v3-studio.png` is the final Studio screenshot from a complete online task.
- `docs/demo-validation.json` is a selected summary of that task's result and verification. Full local task archives are not published.
- `docs/media/release-replay.mp4` is a separate, reconstructed physics experiment shown at 0.2× speed.
- 347 automated tests passed in the verified macOS environment. Some tests use model stubs, some use actual dynamics or rendered images. These results are distinct from online model accuracy and aggregate task success.

Online recognition sends scene images to the model endpoint configured locally. Application logs and captures are written under the ignored `outputs/` directory. Service credentials are loaded from the local environment or local `.env`, both outside the published source snapshot.

## Source map

| Path | Responsibility |
| --- | --- |
| `src/agent/` | Goal interpretation, planning, state machine and records |
| `src/perception/` | Visual identity, active observation, image/depth processing |
| `src/world/` | Observed instances, tracking memory and scene relations |
| `src/manipulation/`, `src/spatial/` | Grasp/motion candidates and placement constraints |
| `src/skills/` | Feedback-driven execution and recovery observation skills |
| `src/robot/`, `src/simulation/` | Robot boundary, MuJoCo adapter, camera and collision backend |
| `src/validation/`, `src/recovery/` | Checks, task acceptance and bounded recovery |
| `src/studio/` | Local native interface, event views and 3D observer scene |
| `tests/` | Contract, perception, geometry, physics and UI regressions |

The older V2 architecture notes in this repository provide historical context. This page and the root README describe the current V3 execution boundary.
