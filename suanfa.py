import math
import random

DATA_MIN = 0.0
DATA_MAX = 100.0
EPS = 0.15
MAX_ITER = 30
DEFAULT_N = 150
GAUSS_CENTERS = [(25, 35), (70, 30), (50, 70)]
GAUSS_SIGMA = 8.0

CLUSTER_COLORS = ["#2563EB", "#E11D48", "#059669",
                  "#D97706", "#7C3AED", "#0891B2",
                  "#DB2777", "#65A30D", "#EA580C", "#4F46E5"]
UNLABELED_COLOR = "#94A3B8"


def generate_points(mode="gauss", n=DEFAULT_N, seed=None):

    rng = random.Random(seed)
    pts = []

    if mode == "gauss":
        for i in range(n):
            cx, cy = GAUSS_CENTERS[i % 3]
            x = _clamp(rng.gauss(cx, GAUSS_SIGMA))
            y = _clamp(rng.gauss(cy, GAUSS_SIGMA))
            pts.append((x, y))
    else:
        for _ in range(n):
            pts.append((rng.uniform(2, 98), rng.uniform(2, 98)))

    return pts


def _clamp(v):
    return max(DATA_MIN, min(DATA_MAX, v))


def init_centroids(points, k, seed=None, method="random"):

    if method == "plusplus":
        return init_centroids_plusplus(points, k, seed)
    if not points:
        return []
    rng = random.Random(seed)
    if len(points) < k:
        k = len(points)
    return [tuple(p) for p in rng.sample(points, k)]


def init_centroids_plusplus(points, k, seed=None):

    if not points:
        return []
    rng = random.Random(seed)
    if len(points) < k:
        k = len(points)

    first = rng.randrange(len(points))
    chosen_idx = [first]
    chosen = [tuple(points[first])]
    while len(chosen) < k:
        weights = []
        for (x, y) in points:
            nearest = min((x - cx) ** 2 + (y - cy) ** 2
                          for (cx, cy) in chosen)
            weights.append(nearest)

        total = sum(weights)
        if total <= 0.0:
            rest = [i for i in range(len(points)) if i not in chosen_idx]
            if not rest:
                break
            pick_i = rng.choice(rest)
        else:
            target = rng.uniform(0.0, total)
            acc = 0.0
            pick_i = len(points) - 1
            for i, w in enumerate(weights):
                acc += w
                if acc >= target:
                    pick_i = i
                    break
        chosen_idx.append(pick_i)
        chosen.append(tuple(points[pick_i]))

    return chosen


def assign(points, centroids):


    labels = []
    for (x, y) in points:
        best_cluster = 0
        best_d2 = float("inf")
        for i, (cx, cy) in enumerate(centroids):
            d2 = (x - cx) ** 2 + (y - cy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_cluster = i
        labels.append(best_cluster)
    return labels


def update(points, labels, k, old_centroids):


    sums = [[0.0, 0.0, 0] for _ in range(k)]
    for (x, y), lab in zip(points, labels):
        if 0 <= lab < k:
            sums[lab][0] += x
            sums[lab][1] += y
            sums[lab][2] += 1

    new_centroids = []
    max_shift = 0.0
    for i in range(k):
        sx, sy, count = sums[i]
        if count == 0:
            fallback = old_centroids[i] if i < len(old_centroids) else (DATA_MAX / 2, DATA_MAX / 2)
            new_centroids.append(tuple(fallback))
            continue

        nx, ny = sx / count, sy / count
        new_centroids.append((nx, ny))
        if i < len(old_centroids):
            ox, oy = old_centroids[i]
            max_shift = max(max_shift, math.hypot(nx - ox, ny - oy))

    return new_centroids, max_shift


def calc_sse(points, labels, centroids):


    total = 0.0
    for (x, y), lab in zip(points, labels):
        if 0 <= lab < len(centroids):
            cx, cy = centroids[lab]
            total += (x - cx) ** 2 + (y - cy) ** 2
    return total


def run_until_converged(points, centroids, k,
                        eps=EPS, max_iter=MAX_ITER):

    history = []
    cur = [tuple(c) for c in centroids]
    labels = [-1] * len(points)
    converged = False
    iters = 0

    for it in range(1, max_iter + 1):
        labels = assign(points, cur)
        sse_assign = calc_sse(points, labels, cur)
        history.append({"iter": it, "phase": "assign",
                        "sse": sse_assign, "shift": None,
                        "centroids": list(cur)})

        new_c, shift = update(points, labels, k, cur)
        sse_update = calc_sse(points, labels, new_c)
        history.append({"iter": it, "phase": "update",
                        "sse": sse_update, "shift": shift,
                        "centroids": list(new_c)})

        cur = new_c
        iters = it

        if shift < eps:
            converged = True
            break

    return {
        "history": history,
        "converged": converged,
        "iters": iters,
        "final_sse": history[-1]["sse"] if history else 0.0,
    }


def sse_of_k(points, k, seed=None, method="plusplus", repeats=2):

    best = float("inf")
    for r in range(max(1, repeats)):
        s = None if seed is None else seed + r * 101
        cs = init_centroids(points, k, s, method)
        if len(cs) < k:
            continue
        result = run_until_converged(points, cs, k)
        if not result["history"]:
            continue
        final_c = result["history"][-1]["centroids"]
        value = calc_sse(points, assign(points, final_c), final_c)
        best = min(best, value)
    return best if best < float("inf") else 0.0


def elbow_curve(points, k_min=1, k_max=10, seed=None,
                method="plusplus", repeats=2):

    k_min = max(1, k_min)
    k_max = min(k_max, len(points))
    curve = []
    for k in range(k_min, k_max + 1):
        curve.append((k, sse_of_k(points, k, seed, method, repeats)))
    return curve


def find_elbow(curve):

    if len(curve) < 3:
        return curve[-1][0] if curve else 1
    vals = [v for _, v in curve]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return None

    n = len(curve) - 1
    x0, y0 = 0.0, (vals[0] - lo) / (hi - lo)
    x1, y1 = 1.0, (vals[-1] - lo) / (hi - lo)
    denom = math.hypot(y1 - y0, x1 - x0)
    best_k = None
    best_d = -1.0
    for i in range(1, n):
        xn = i / n
        yn = (vals[i] - lo) / (hi - lo)
        d = abs((y1 - y0) * xn - (x1 - x0) * yn + x1 * y0 - y1 * x0) / denom
        if d > best_d:
            best_d = d
            best_k = curve[i][0]
    return best_k


if __name__ == "__main__":
    pts = generate_points("gauss", 150, seed=2026)
    cs = init_centroids(pts, 3, seed=7)
    result = run_until_converged(pts, cs, 3)
    print("轮次  阶段      SSE          质心最大位移")
    for h in result["history"]:
        shift = "   —   " if h["shift"] is None else "%8.4f" % h["shift"]
        print("%3d   %-7s  %10.2f  %s" % (h["iter"], h["phase"], h["sse"], shift))
    print("收敛：%s，共 %d 轮，最终 SSE = %.2f"
          % (result["converged"], result["iters"], result["final_sse"]))
    curve = elbow_curve(pts, 1, 10, seed=7)
    print("\n肘部法则（同一批点，K=1~10）：")
    for k, v in curve:
        print("  K=%2d  SSE = %10.2f" % (k, v))
    print("推荐 K = %s" % find_elbow(curve))
