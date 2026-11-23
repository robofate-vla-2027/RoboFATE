"""Offline annotation and restored-state validation (Appendix B.1)."""
import numpy as np
from .data import query_history

BOUNDARY_PROMPT = """Review the supplied failed robot episode retrospectively. Select one candidate boundary at which a future task-threatening error is already supported by the recorded observations, but the recorded adverse event has not occurred. Use only the supplied evidence. Do not select a fixed action index across episodes. A boundary matches an eligible policy query after its indexed applied action. The corrective continuation starts with the next action. Require all eight consecutive causal policy-query observations and positive remaining budget. Return the earliest eligible boundary at which the risk is visually or kinematically supported and correction remains plausible. If the record does not support such a boundary, reject the source. Future observations may establish what failed, but distinguish them from evidence visible at the selected boundary. Do not claim hidden simulator state equivalence or guaranteed recoverability. Return one JSON object with fields: accept (boolean), boundary_action (integer or null), history_indices (eight integers or an empty list), risk_description (string), causal_evidence_indices (integer list), retrospective_evidence_indices (integer list), remaining_actions (integer), and decision_reason (string). A rejection uses a null boundary."""


def validate_annotation(annotation, candidates, observed_indices, first_adverse, action_budget):
    expected = {"accept", "boundary_action", "history_indices", "risk_description",
                "causal_evidence_indices", "retrospective_evidence_indices", "remaining_actions", "decision_reason"}
    if set(annotation) != expected or type(annotation["accept"]) is not bool:
        raise ValueError("Annotation fields do not match the boundary-review schema")
    for key in ("risk_description", "decision_reason"):
        if not isinstance(annotation[key], str):
            raise ValueError("Annotation explanation must be text")
    for key in ("history_indices", "causal_evidence_indices", "retrospective_evidence_indices"):
        if not isinstance(annotation[key], list) or any(type(i) is not int for i in annotation[key]):
            raise ValueError("Evidence indices must be integer lists")
    if not annotation["accept"]:
        if annotation["boundary_action"] is not None or annotation["history_indices"]:
            raise ValueError("Rejected annotations have no boundary or history")
        return False
    boundary = annotation["boundary_action"]
    if type(boundary) is not int or boundary not in candidates or boundary >= first_adverse:
        raise ValueError("Boundary must be a supplied pre-adverse candidate")
    history = query_history(boundary)
    if annotation["history_indices"] != history or not set(history) <= set(observed_indices):
        raise ValueError("Annotation must retain the exact causal query history")
    if annotation["remaining_actions"] != action_budget - boundary or action_budget <= boundary:
        raise ValueError("Remaining action budget does not match the boundary")
    if not annotation["causal_evidence_indices"]:
        raise ValueError("Accepted boundary requires causal evidence")
    if any(i > boundary for i in annotation["causal_evidence_indices"]):
        raise ValueError("Causal evidence contains future observations")
    for key in ("causal_evidence_indices", "retrospective_evidence_indices"):
        if not set(annotation[key]) <= set(observed_indices):
            raise ValueError("Annotation cites absent observations")
    return True


def compare_restored_state(reference, restored):
    """All required fields must pass; unavailable state cannot qualify a replay."""
    numerical = ("robot_joint_positions", "robot_joint_velocities", "object_positions",
                 "object_linear_velocities", "object_angular_velocities", "articulation_positions",
                 "articulation_velocities", "controller_targets", "interpolation_state", "gripper_state")
    exact = ("attachments", "task_flags", "task_progress", "termination_flags", "rng_states",
             "simulator_time", "action_counter", "remaining_actions", "remaining_time",
             "contact_solver_state", "task_dynamic_state")
    required = set(numerical + exact + ("object_quaternions", "rgb"))
    missing = sorted(required - set(reference) | required - set(restored))
    results = {name: False for name in required}
    for name in numerical:
        if name not in reference or name not in restored:
            continue
        a, b = np.asarray(reference[name], float), np.asarray(restored[name], float)
        empty_articulations = name in ("articulation_positions", "articulation_velocities") and not a.size
        results[name] = bool(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
                             and (empty_articulations or (a.size and np.max(np.abs(a - b)) <= 1e-6)))
    for name in exact:
        if name in reference and name in restored:
            results[name] = reference[name] is not None and restored[name] is not None and reference[name] == restored[name]
    if "object_quaternions" in reference and "object_quaternions" in restored:
        a, b = np.asarray(reference["object_quaternions"], float), np.asarray(restored["object_quaternions"], float)
        if a.shape == b.shape and a.ndim == 2 and a.shape[-1] == 4 and len(a):
            norm_a, norm_b = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
            if np.all(norm_a > 0) and np.all(norm_b > 0):
                dots = np.abs(((a / norm_a[:, None]) * (b / norm_b[:, None])).sum(-1))
                angle = 2 * np.arccos(np.clip(dots, 0, 1))
                results["object_quaternions"] = bool(np.isfinite(angle).all() and np.all(angle <= 1e-6))
    if "rgb" in reference and "rgb" in restored:
        cameras = {"head", "left_wrist", "right_wrist"}
        a, b = reference["rgb"], restored["rgb"]
        def same_rgb(left, right):
            left, right = np.asarray(left), np.asarray(right)
            return bool(left.ndim == 3 and left.shape[-1] == 3 and left.size
                        and left.dtype == np.uint8 and right.dtype == np.uint8
                        and np.array_equal(left, right))
        results["rgb"] = (set(a) == cameras and set(b) == cameras
                          and all(same_rgb(a[k], b[k]) for k in cameras))
    return {"qualified": all(results.values()), "missing_fields": missing, "comparisons": results}
