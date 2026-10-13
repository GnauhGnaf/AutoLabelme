"""Cross-frame identity: one name per physical entity, for the whole clip.

Tracks are kept per class and matched to each frame's detections by Hungarian assignment
(scipy.optimize.linear_sum_assignment) over a cost mixing mask IoU, colour appearance,
centroid distance and area ratio. The cost is what does the work where a greedy IoU match
is not enough: two people walking past each other are a high-IoU ambiguity that only
appearance breaks.

Names are assigned only AFTER the clip has been tracked, and only to tracks seen MIN_HITS
times. So a single-frame false positive can never burn a number, and person1 stays person1
for the whole video instead of being renumbered every frame the way the still-image path
(masks_to_labelme.order_shapes with labels=None) does.

Everything here is deterministic: no RNG, no wall clock, no dict-iteration-order
dependence (classes are always processed in sorted order).
"""
import collections
import math

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

import entities as E

# --- lifecycle ---------------------------------------------------------------
MIN_HITS = 3        # frames a track must be seen before it earns a global name
LOST_MAX = 24       # frames unmatched before a confirmed track goes dormant (still re-IDable)
                    # and before an unconfirmed one is deleted outright

# --- association gate --------------------------------------------------------
# A pair is admissible when it clears EITHER bar: high mask overlap, OR centroids close
# enough. And-of-both would strand a fast-moving or briefly-occluded entity; or-of-both
# still rejects "the person on the left is now the person on the right".
IOU_GATE = 0.30
DIST_GATE = 0.10    # fraction of the image diagonal, per sampled frame

# --- re-identification -------------------------------------------------------
REID_SIM = 0.75     # appearance cosine needed to reclaim a dormant track's number

# --- post-hoc merge (strict on purpose; see Tracker.merge_similar) -----------
MERGE_SIM = 0.85
MERGE_GAP_MAX = 48
MERGE_DIST = 0.40   # fraction of the diagonal

# Cross-class merging: the same physical object the detector labels differently in
# different frames (COCO 'book' in one frame, 'laptop' in the next). Tighter on every
# axis than the same-class case, because confusing two different objects is worse than
# leaving one object with two names.
CROSS_SIM = 0.80
CROSS_GAP_MAX = 24
CROSS_REACH = 1.0       # centroid distance <= this many sqrt(area), i.e. same footprint
CROSS_ABS_MIN = 40      # ...but always permit at least this many pixels

# --- cost weights ------------------------------------------------------------
W_IOU, W_APP, W_DIST, W_AREA = 1.0, 0.6, 0.4, 0.2
APP_EMA = 0.30      # how fast a track's appearance descriptor follows new evidence
VEL_EMA = 0.40      # how fast a track's velocity estimate follows new evidence
BIG = 1e6           # cost of a forbidden pair

HUE_BINS, SAT_BINS = 30, 32


def appearance(bgr, mask):
    """L1-normalised hue/saturation histogram of the pixels inside ``mask``.

    This is the descriptor the re-ID and the cost matrix compare. Hue x saturation rather
    than a colour average because a person's silhouette is a mix of hair, skin and clothing;
    the distribution distinguishes two similarly-dressed people better than its mean does.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], mask.astype(np.uint8), [HUE_BINS, SAT_BINS],
                     [0, 180, 0, 256]).ravel()
    s = float(h.sum())
    return (h / s).astype(np.float32) if s else h.astype(np.float32)


def cosine(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na and nb else 0.0


def _bbox(mask):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _iou_cached(a, ba, aa, b, bb, ab):
    """IoU of two masks clipped to their overlapping bbox.

    The full-frame version costs ~1.5 MB of work per pair, and a 124-frame clip builds
    ~20k pairs; cropping first is the difference between a few seconds and a minute.
    """
    x0, y0 = max(ba[0], bb[0]), max(ba[1], bb[1])
    x1, y1 = min(ba[2], bb[2]), min(ba[3], bb[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = int(np.logical_and(a[y0:y1, x0:x1], b[y0:y1, x0:x1]).sum())
    return inter / (aa + ab - inter) if inter else 0.0


def _translate(mask, dx, dy, shape):
    """Shift a mask by a sub-pixel amount, nearest neighbour (masks are binary)."""
    if abs(dx) < 0.05 and abs(dy) < 0.05:
        return mask
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], np.float32)
    return cv2.warpAffine(mask.astype(np.uint8), M, (shape[1], shape[0]),
                          flags=cv2.INTER_NEAREST, borderValue=0).astype(bool)


class Track:
    """One physical entity. ``last_mask`` is kept in its own frame's coordinates.

    All timekeeping is in SAMPLE ORDINALS (0, 1, 2, ... position in the extracted
    sequence), never source frame numbers. Under --stride the two differ, and a tracker
    that thought in source frames would read every 8-frame step as an 8-frame absence:
    LOST_MAX would expire instantly and every track would report a gap at every step.
    ``Tracker.summary`` maps back to source frames on the way out.
    """

    def __init__(self, tid, name, k, mask, cent, app):
        self.tid = tid
        self.name = name
        self.label = None          # global name, set by Tracker.finalize
        self.present = [k]
        self.first_x = float(cent[0])
        self.last_k = k
        self.last_mask = mask
        self.last_box = _bbox(mask)
        self.last_area = int(mask.sum())
        self.last_cent = np.asarray(cent, float)
        self.v = np.zeros(2)       # centroid velocity, px per sampled frame
        self.app = app

    def update(self, k, mask, cent, app):
        gap = k - self.last_k
        if gap > 0:
            self.v = (1 - VEL_EMA) * self.v + VEL_EMA * ((np.asarray(cent, float) - self.last_cent) / gap)
        self.app = (1 - APP_EMA) * self.app + APP_EMA * app
        self.present.append(k)
        self.last_k = k
        self.last_mask = mask
        self.last_box = _bbox(mask)
        self.last_area = int(mask.sum())
        self.last_cent = np.asarray(cent, float)

    @property
    def hits(self):
        return len(self.present)


class Tracker:
    """Per-class multi-object tracker. Feed it frames in order, name things at the end."""

    def __init__(self, width, height):
        self.W, self.H = width, height
        self.diag = float(np.hypot(width, height))
        self.tracks = []
        self.alias = {}            # absorbed tid -> surviving tid, filled by merge_similar
        self._next_tid = 0
        self._prev_gray = None
        self._shifts = collections.deque(maxlen=2 * LOST_MAX + 4)
        self.events = []           # (frame, kind, message) - every decision that changes identity

    # --- camera motion ---
    def _camera_shift(self, gray):
        """Frame-to-frame translation, from phase correlation. ~0.1 px on a static camera."""
        shift = np.zeros(2)
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            try:
                (dx, dy), _ = cv2.phaseCorrelate(self._prev_gray, gray)
                if np.isfinite(dx) and np.isfinite(dy):
                    shift = np.array([dx, dy])
            except cv2.error:
                pass          # a degenerate frame pair; treat as no motion
        self._prev_gray = gray
        self._shifts.append(shift)
        return shift

    def _cam_shift_over(self, gap):
        """Accumulated camera translation over the last ``gap`` sampled frames."""
        n = min(gap, len(self._shifts))
        if n <= 0:
            return np.zeros(2)
        return np.sum(list(self._shifts)[-n:], axis=0)

    def _predict(self, t, k):
        """Where ``t`` should be now: its own velocity, plus camera motion over the gap."""
        gap = k - t.last_k
        d = t.v * gap + self._cam_shift_over(gap)
        return _translate(t.last_mask, float(d[0]), float(d[1]), (self.H, self.W)), t.last_cent + d

    # --- main loop ---
    def step(self, k, items, bgr):
        """Associate one sample. ``k`` is the sample ordinal, not the source frame number.

        Returns [(tid, item)] for every detection, matched or new. The caller stores the
        polygons against those tids; the labels do not exist yet.
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self._camera_shift(gray)

        by_class = collections.defaultdict(list)
        for it in items:
            by_class[it["name"]].append(it)

        out = []
        for cls in sorted(by_class):
            dets = by_class[cls]
            app = [appearance(bgr, d["mask"]) for d in dets]
            boxes = [_bbox(d["mask"]) for d in dets]
            areas = [int(d["mask"].sum()) for d in dets]
            cents = [E.centroid(d["points"]) for d in dets]

            live = [t for t in self.tracks if t.name == cls and k - t.last_k <= LOST_MAX]
            owner = {}          # detection index -> Track

            if live:
                cost = np.full((len(live), len(dets)), BIG, np.float32)
                for i, t in enumerate(live):
                    pred, pcent = self._predict(t, k)
                    pbox = _bbox(pred)
                    if pbox is None:
                        continue
                    parea = int(pred.sum())
                    for j, d in enumerate(dets):
                        ov = _iou_cached(pred, pbox, parea, d["mask"], boxes[j], areas[j])
                        dist = float(np.linalg.norm(pcent - np.asarray(cents[j], float))) / self.diag
                        if ov < IOU_GATE and dist > DIST_GATE:
                            continue
                        ratio = areas[j] / t.last_area if t.last_area else 1.0
                        cost[i, j] = (W_IOU * (1 - ov)
                                      + W_APP * (1 - cosine(t.app, app[j]))
                                      + W_DIST * min(dist / DIST_GATE, 2.0)
                                      + W_AREA * abs(1 - ratio))
                for i, j in zip(*linear_sum_assignment(cost)):
                    if cost[i, j] < BIG:
                        t = live[i]
                        t.update(k, dets[j]["mask"], cents[j], app[j])
                        owner[j] = t

            for j, d in enumerate(dets):
                if j in owner:
                    out.append((owner[j].tid, d))
                    continue
                t = self._reid(cls, k, d, cents[j], app[j])
                if t is None:
                    t = Track(self._next_tid, cls, k, d["mask"], cents[j], app[j])
                    self._next_tid += 1
                    self.tracks.append(t)
                out.append((t.tid, d))
        return out

    def _reid(self, cls, k, det, cent, app):
        """Reclaim a dormant track of the same class when the appearance matches.

        Only CONFIRMED dormants are eligible - 'confirmed' here being hits >= MIN_HITS,
        NOT label != None: labels do not exist yet while the clip is still being tracked.
        An unconfirmed dormant has no number worth keeping, and reviving a 1-frame flicker
        would hand a real entity that track's early frames. This is what keeps a person who
        walks out of shot and back in as person1.
        """
        best, best_sim = None, REID_SIM
        for t in self.tracks:
            if t.name != cls or t.hits < MIN_HITS or k - t.last_k <= LOST_MAX:
                continue
            sim = cosine(t.app, app)
            if sim > best_sim:
                best, best_sim = t, sim
        if best is None:
            return None
        self.events.append((k, "reid",
                            f"track#{best.tid} ({best.name}) reappears after "
                            f"{k - best.last_k} samples (appearance {best_sim:.2f})"))
        # Pin last_k to k first: that makes the gap 0, which suppresses the velocity
        # update - extrapolating a velocity across a 20-sample absence would be noise.
        best.last_k = k
        best.update(k, det["mask"], cent, app)
        return best

    # --- naming ---
    def _mergeable(self, ta, tb):
        """Could these two confirmed tracks be one physical entity? Reason, or None.

        Two entities that were EVER seen in the same frame are two entities, however alike,
        so that is the first gate.
        """
        if not set(ta.present).isdisjoint(tb.present):
            return None
        same = ta.name == tb.name
        if not same and "person" in (ta.name, tb.name):
            return None       # never dissolve a person's identity into an object's
        sim = cosine(ta.app, tb.app)
        if same:
            if sim < MERGE_SIM:
                return None
            gap_max = MERGE_GAP_MAX
            reach = max(MERGE_DIST * self.diag, CROSS_ABS_MIN)
        else:
            if sim < CROSS_SIM:
                return None
            gap_max = CROSS_GAP_MAX
            # Scale-aware: "same place" has to mean overlapping footprints, not a small
            # absolute distance that would be huge for a phone and tiny for a sofa.
            reach = max(CROSS_REACH * math.sqrt(min(ta.last_area, tb.last_area)), CROSS_ABS_MIN)

        # How far apart are the two SPANS? Normally one yields to the other (a handover,
        # so this is positive), but a detector that alternates between two labels for one
        # object - 'book' at samples 5-8, 'laptop' at 2 and 9-14 - produces interleaved
        # spans instead. Both mean the same thing, so compare spans rather than requiring a
        # handover; requiring gap >= 0 would silently reject every interleaved case.
        head = max(min(ta.present), min(tb.present))
        tail = min(max(ta.present), max(tb.present))
        gap = head - tail
        if gap > gap_max:
            return None
        # Centroids from two different samples, compared without correcting for camera
        # motion - legitimate because the shift is ~0.1 px/sample here.
        d = float(np.linalg.norm(tb.last_cent - ta.last_cent))
        if d > reach:
            return None
        return (f"appearance {sim:.2f}, spans {gap:+d} samples apart, {d:.0f}px apart"
                + ("" if same else f", classes {ta.name}/{tb.name}"))

    def merge_similar(self):
        """Conservative post-hoc merge of tracks that are one physical entity.

        Two cases, neither reachable during tracking:

        - Same class: a reappearance whose gap or position failed the association gate
          while the appearance is near-identical. Stricter than the per-frame re-ID
          (MERGE_SIM > REID_SIM), because wrongly merging two people is worse than leaving
          one of them with two names.
        - Different class: the tracker matches WITHIN a class, so one object that the
          detector calls 'book' in one frame and 'laptop' in the next becomes two entities
          with two names and can never be linked during tracking. Left alone this is a
          plain violation of one-entity-one-name, so it is merged here.

        Every merge is logged with its evidence.
        """
        confirmed = [t for t in self.tracks if t.hits >= MIN_HITS]
        parent = {t.tid: t.tid for t in confirmed}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a in range(len(confirmed)):
            for b in range(a + 1, len(confirmed)):
                ta, tb = confirmed[a], confirmed[b]
                why = self._mergeable(ta, tb)
                if why is None:
                    continue
                ra, rb = find(ta.tid), find(tb.tid)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
                    # Names do not exist yet at this point, so identify by tid.
                    self.events.append((min(min(ta.present), min(tb.present)), "merge",
                                        f"{ta.name}#{ta.tid} + {tb.name}#{tb.tid} "
                                        f"-> one entity ({why})"))

        groups = collections.defaultdict(list)
        for t in confirmed:
            groups[find(t.tid)].append(t)
        absorbed = []
        for grp in groups.values():
            if len(grp) == 1:
                continue
            # Most evidence wins the name; ties go to whichever appeared first, so the
            # result does not depend on track-creation order.
            keep = max(grp, key=lambda t: (t.hits, -min(t.present)))
            for t in grp:
                if t is not keep:
                    # The absorbed track stops being a track: its frames become the
                    # survivor's, and alias lets the driver's (tid -> points) records
                    # resolve to the survivor's label rather than vanishing.
                    self.alias[t.tid] = keep.tid
                    keep.present = sorted(set(keep.present) | set(t.present))
                    absorbed.append(t)
        for t in absorbed:
            self.tracks.remove(t)

    def finalize(self):
        """Give every confirmed track its global name. Returns {tid: label}.

        Numbering seeds from (first frame, centroid x within that frame), so the first frame
        comes out left-to-right exactly as the still-image convention orders it, and is
        frozen from then on.

        The returned map includes merged tracks' ids, pointing at their survivor's label,
        so a caller can look up any tid it saw during step().
        """
        self.merge_similar()
        confirmed = [t for t in self.tracks if t.hits >= MIN_HITS]

        by_class = collections.defaultdict(list)
        for t in confirmed:
            by_class[t.name].append(t)
        for cls in sorted(by_class):
            grp = sorted(by_class[cls], key=lambda t: (min(t.present), t.first_x))
            for n, t in enumerate(grp, 1):
                t.label = f"{cls}{n}"

        labels = {t.tid: t.label for t in confirmed}
        for absorbed, keep in self.alias.items():
            if keep in labels:
                labels[absorbed] = labels[keep]
        return labels

    def summary(self, frame_of=None):
        """Per-track record for tracks.json: name, class, span, gaps.

        ``frame_of`` maps a sample ordinal to its source frame number, which is what a
        reader wants in the file (seek there in the clip). Defaults to identity, so the
        numbers are ordinals when the caller has no mapping.
        """
        idx = (lambda k: k) if frame_of is None else frame_of.__getitem__
        out = []
        for t in [t for t in self.tracks if t.label]:
            p = sorted(t.present)
            gaps = [[idx(a), idx(b)] for a, b in zip(p, p[1:]) if b - a > 1]
            cls, _ = E.split_label(t.label)
            out.append({"label": t.label, "class": cls,
                        "first_frame": idx(p[0]), "last_frame": idx(p[-1]),
                        "n_samples": len(p), "gaps": gaps})
        out.sort(key=lambda d: (d["class"], E.split_label(d["label"])[1]))
        return out
