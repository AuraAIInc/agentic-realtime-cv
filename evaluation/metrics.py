"""Core matching + metric computation for detection-stage evaluation.

Compares one prediction set (raw / revised / validated detections from a
tracking run) against YOLO ground-truth labels, frame by frame.

Matching: per frame, predictions are processed in descending confidence
order and greedily assigned to the highest-IoU unmatched ground-truth box
of the same class (class-aware) or any class (class-agnostic).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Box:
    """Axis-aligned box in absolute pixels with an integer class id."""

    x1: float
    y1: float
    x2: float
    y2: float
    cls: int
    conf: float = 1.0

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes in [0, 1]."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area() + b.area() - inter
    return inter / union if union > 0 else 0.0


@dataclass
class MatchResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    matched_ious: list = field(default_factory=list)
    # (pred_cls, gt_cls) for every class-agnostic IoU match; used for confusion.
    agnostic_pairs: list = field(default_factory=list)


def match_frame(
    preds: list,
    gts: list,
    iou_thr: float = 0.5,
    class_aware: bool = True,
) -> MatchResult:
    """Greedily match predictions to ground truth within one frame.

    Predictions are visited in descending confidence order; each takes the
    highest-IoU still-unmatched ground-truth box at IoU >= iou_thr (and of
    the same class when class_aware). Unmatched predictions are FPs,
    unmatched ground truth are FNs.
    """
    res = MatchResult()
    used = [False] * len(gts)
    for pred in sorted(preds, key=lambda b: b.conf, reverse=True):
        best, best_i = -1.0, -1
        for i, gt in enumerate(gts):
            if used[i]:
                continue
            if class_aware and pred.cls != gt.cls:
                continue
            v = iou(pred, gt)
            if v >= iou_thr and v > best:
                best, best_i = v, i
        if best_i >= 0:
            used[best_i] = True
            res.tp += 1
            res.matched_ious.append(best)
            if not class_aware:
                res.agnostic_pairs.append((pred.cls, gts[best_i].cls))
        else:
            res.fp += 1
    res.fn = sum(1 for u in used if not u)
    return res


def prf(tp: int, fp: int, fn: int) -> tuple:
    """Return (precision, recall, f1) with zero-division guarded as 0.0."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def evaluate_frames(
    frames: dict,
    iou_thr: float = 0.5,
    classes: list | None = None,
) -> dict:
    """Aggregate metrics over frames: {frame_id: (preds, gts)}.

    Returns overall class-aware P/R/F1, class-agnostic (localization-only)
    P/R/F1, per-class P/R/F1, mean matched IoU, and a confusion summary of
    predicted-vs-true classes on spatially matched pairs.
    """
    tot = MatchResult()
    agn = MatchResult()
    per_cls: dict = {}
    for preds, gts in frames.values():
        m = match_frame(preds, gts, iou_thr, class_aware=True)
        tot.tp += m.tp
        tot.fp += m.fp
        tot.fn += m.fn
        tot.matched_ious += m.matched_ious
        a = match_frame(preds, gts, iou_thr, class_aware=False)
        agn.tp += a.tp
        agn.fp += a.fp
        agn.fn += a.fn
        agn.agnostic_pairs += a.agnostic_pairs

    p, r, f1 = prf(tot.tp, tot.fp, tot.fn)
    ap, ar, af1 = prf(agn.tp, agn.fp, agn.fn)
    out = {
        "iou_thr": iou_thr,
        "frames": len(frames),
        "n_gt": sum(len(g) for _, g in frames.values()),
        "n_pred": sum(len(p_) for p_, _ in frames.values()),
        "tp": tot.tp,
        "fp": tot.fp,
        "fn": tot.fn,
        "precision": p,
        "recall": r,
        "f1": f1,
        "agnostic": {
            "tp": agn.tp,
            "fp": agn.fp,
            "fn": agn.fn,
            "precision": ap,
            "recall": ar,
            "f1": af1,
        },
        "mean_matched_iou": (
            sum(tot.matched_ious) / len(tot.matched_ious)
            if tot.matched_ious
            else 0.0
        ),
    }

    cls_ids = classes
    if cls_ids is None:
        cls_ids = sorted(
            {b.cls for preds, gts in frames.values() for b in list(preds) + list(gts)}
        )
    per = {}
    for c in cls_ids:
        ctp = cfp = cfn = 0
        for preds, gts in frames.values():
            m = match_frame(
                [b for b in preds if b.cls == c],
                [b for b in gts if b.cls == c],
                iou_thr,
                class_aware=True,
            )
            ctp += m.tp
            cfp += m.fp
            cfn += m.fn
        cp, cr, cf1 = prf(ctp, cfp, cfn)
        per[c] = {
            "tp": ctp,
            "fp": cfp,
            "fn": cfn,
            "precision": cp,
            "recall": cr,
            "f1": cf1,
        }
    out["per_class"] = per

    # Confusion on spatially matched pairs: how often was the class right?
    confusion: dict = {}
    for pc, gc in agn.agnostic_pairs:
        confusion.setdefault(str(gc), {}).setdefault(str(pc), 0)
        confusion[str(gc)][str(pc)] += 1
    n_match = len(agn.agnostic_pairs)
    n_right = sum(1 for pc, gc in agn.agnostic_pairs if pc == gc)
    out["confusion_on_matched"] = confusion
    out["matched_pairs"] = n_match
    out["class_accuracy_on_matched"] = (n_right / n_match) if n_match else 0.0
    return out
