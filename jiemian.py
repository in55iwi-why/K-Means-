import tkinter as tk
from tkinter import messagebox

import suanfa as core

CANVAS_W = 600
CANVAS_H = 420
PAD = 20
POINT_R = 4
CENTROID_R = 9
AUTO_DELAY = 900
ELBOW_K_MAX = 10
ELBOW_REPEATS = 2
ELBOW_CANVAS_W = 560
ELBOW_CANVAS_H = 300

STEP_LABELS = ["① 选择 k", "② 设初始质心", "③ 按距离分配",
               "④ 重算中心", "⑤ 重复至收敛"]


def data_to_pixel(x, y):

    inner_w = CANVAS_W - PAD * 2
    inner_h = CANVAS_H - PAD * 2
    span = core.DATA_MAX - core.DATA_MIN
    px = PAD + (x - core.DATA_MIN) / span * inner_w
    py = PAD + (y - core.DATA_MIN) / span * inner_h
    return px, py


def draw_all(canvas, points, labels, centroids, track=None):
    canvas.delete("all")

    for gx in range(0, CANVAS_W + 1, 50):
        canvas.create_line(gx, 0, gx, CANVAS_H, fill="#EDF0F5")
    for gy in range(0, CANVAS_H + 1, 50):
        canvas.create_line(0, gy, CANVAS_W, gy, fill="#EDF0F5")

    for i, (x, y) in enumerate(points):
        lab = labels[i] if i < len(labels) else -1
        if 0 <= lab < len(centroids):
            px, py = data_to_pixel(x, y)
            cx, cy = data_to_pixel(*centroids[lab])
            canvas.create_line(px, py, cx, cy, fill="#DDE3EC")

    if track:
        for i, path in enumerate(track):
            if len(path) < 2:
                continue
            color = core.CLUSTER_COLORS[i % len(core.CLUSTER_COLORS)]
            pix = [data_to_pixel(*p) for p in path]
            for a, b in zip(pix[:-1], pix[1:]):
                canvas.create_line(a[0], a[1], b[0], b[1],
                                   fill=color, dash=(4, 3), width=1)

    for i, (x, y) in enumerate(points):
        lab = labels[i] if i < len(labels) else -1
        color = (core.CLUSTER_COLORS[lab % len(core.CLUSTER_COLORS)]
                 if 0 <= lab else core.UNLABELED_COLOR)
        px, py = data_to_pixel(x, y)
        canvas.create_oval(px - POINT_R, py - POINT_R,
                           px + POINT_R, py + POINT_R,
                           fill=color, outline="")

    for i, (cx, cy) in enumerate(centroids):
        px, py = data_to_pixel(cx, cy)
        r = CENTROID_R
        canvas.create_polygon(px, py - r, px + r, py,
                              px, py + r, px - r, py,
                              fill="#0F172A", outline="#FFFFFF", width=2)
        canvas.create_text(px, py - 20, text="C%d" % (i + 1),
                           font=("Microsoft YaHei", 10, "bold"),
                           fill="#0F172A")


def draw_elbow_curve(canvas, curve, elbow_k=None):
    canvas.delete("all")
    if not curve:
        return

    w = int(canvas.cget("width"))
    h = int(canvas.cget("height"))
    left, right, top, bottom = 62, 22, 24, 40
    ks = [k for k, _ in curve]
    vals = [v for _, v in curve]
    vmax = max(vals) if max(vals) > 0 else 1.0
    span = vmax if vmax > 0 else 1.0

    def pos(k, v):
        x = left + (k - ks[0]) / max(1, ks[-1] - ks[0]) * (w - left - right)
        y = top + (1 - v / span) * (h - top - bottom)
        return x, y

    for i in range(5):
        gy = top + i * (h - top - bottom) / 4
        canvas.create_line(left, gy, w - right, gy, fill="#EDF0F5")
        canvas.create_text(left - 8, gy, text="%.0f" % (vmax - i * span / 4),
                           anchor=tk.E, font=("Consolas", 8), fill="#64748B")

    for k in ks:
        x, _ = pos(k, 0)
        canvas.create_text(x, h - bottom + 14, text=str(k),
                           font=("Consolas", 9), fill="#475569")

    canvas.create_line(left, top, left, h - bottom, fill="#94A3B8")
    canvas.create_line(left, h - bottom, w - right, h - bottom, fill="#94A3B8")
    canvas.create_text((left + w - right) / 2, h - 8,
                       text="簇数 K", anchor=tk.CENTER,
                       font=("Microsoft YaHei", 9), fill="#475569")

    pix = [pos(k, v) for k, v in curve]
    for a, b in zip(pix[:-1], pix[1:]):
        canvas.create_line(a[0], a[1], b[0], b[1], fill="#2563EB", width=2)
    for (k, v), (x, y) in zip(curve, pix):
        hit = k == elbow_k
        r = 5 if hit else 3
        color = "#DC2626" if hit else "#2563EB"
        canvas.create_oval(x - r, y - r, x + r, y + r,
                           fill=color, outline="white", width=1)
        canvas.create_text(x, y - 13, text="%.0f" % v,
                           font=("Consolas", 8), fill="#475569")

    if elbow_k is not None:
        for (k, v), (x, y) in zip(curve, pix):
            if k == elbow_k:
                canvas.create_line(x, y + 8, x, h - bottom,
                                   fill="#DC2626", dash=(4, 3))
                canvas.create_text(x + 8, top + 4, text="肘部 K=%d" % elbow_k,
                                   anchor=tk.NW,
                                   font=("Microsoft YaHei", 9, "bold"),
                                   fill="#DC2626")
                break


class App(object):
    def __init__(self, root):
        self.root = root
        self.root.title("K-Means 聚类迭代模拟器 · 第06章 人工智能")
        self.root.geometry("1000x860")
        self.root.resizable(False, False)

        self.points = []
        self.points_mode = None
        self.labels = []
        self.centroids = []
        self.track = []
        self.k = 3
        self.iter = 0
        self.phase = "idle"
        self.sse_history = []
        self.last_shift = 0.0
        self.auto_running = False
        self.elbow_win = None
        self.elbow_curve_data = []

        top = tk.Frame(root)
        top.pack(fill=tk.X, padx=14, pady=(10, 6))
        tk.Label(top, text="K-Means 聚类迭代模拟器",
                 font=("Microsoft YaHei", 16, "bold")).pack(side=tk.LEFT)
        self.status = tk.StringVar(value="未初始化")
        tk.Label(top, textvariable=self.status,
                 font=("Microsoft YaHei", 10), fg="#1D4ED8").pack(side=tk.RIGHT)

        body = tk.Frame(root)
        body.pack(fill=tk.X, padx=14)
        self.canvas = tk.Canvas(body, width=CANVAS_W, height=CANVAS_H,
                                bg="#FAFAFA", highlightthickness=1,
                                highlightbackground="#DDE3EA")
        self.canvas.pack(side=tk.LEFT, padx=(0, 14))

        right = tk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(right, text="聚类个数 k", font=("Microsoft YaHei", 10, "bold")).pack(anchor=tk.W)
        self.k_var = tk.IntVar(value=3)
        kf = tk.Frame(right)
        kf.pack(anchor=tk.W, pady=(4, 10))
        for idx, v in enumerate(range(2, ELBOW_K_MAX + 1)):
            tk.Radiobutton(kf, text=str(v), value=v, variable=self.k_var,
                           font=("Microsoft YaHei", 9)).grid(row=idx // 5,
                                                             column=idx % 5,
                                                             sticky=tk.W)

        tk.Label(right, text="数据集", font=("Microsoft YaHei", 10, "bold")).pack(anchor=tk.W)
        self.mode_var = tk.StringVar(value="gauss")
        mf = tk.Frame(right)
        mf.pack(anchor=tk.W, pady=(4, 10))
        tk.Radiobutton(mf, text="高斯三簇", value="gauss", variable=self.mode_var,
                       font=("Microsoft YaHei", 10)).pack(side=tk.LEFT)
        tk.Radiobutton(mf, text="均匀随机", value="uniform", variable=self.mode_var,
                       font=("Microsoft YaHei", 10)).pack(side=tk.LEFT)

        tk.Label(right, text="初始质心方式",
                 font=("Microsoft YaHei", 10, "bold")).pack(anchor=tk.W)
        self.init_var = tk.StringVar(value="plusplus")
        itf = tk.Frame(right)
        itf.pack(anchor=tk.W, pady=(4, 10))
        tk.Radiobutton(itf, text="K-Means++", value="plusplus",
                       variable=self.init_var,
                       font=("Microsoft YaHei", 10)).pack(side=tk.LEFT)
        tk.Radiobutton(itf, text="随机", value="random",
                       variable=self.init_var,
                       font=("Microsoft YaHei", 10)).pack(side=tk.LEFT)

        bf = tk.Frame(right)
        bf.pack(anchor=tk.W, pady=(2, 4))
        for idx, (text, cmd, color) in enumerate([
            ("初始化质心", self.on_init, "#1D4ED8"),
            ("单步执行", self.on_step, "#F59E0B"),
            ("自动播放", self.on_auto, "#059669"),
            ("肘部法则找 K", self.on_elbow, "#7C3AED"),
            ("重置", self.on_reset, "#DC2626"),
        ]):
            btn = tk.Button(bf, text=text, width=14, height=1, command=cmd,
                            bg=color, fg="white", relief=tk.FLAT,
                            font=("Microsoft YaHei", 10, "bold"))
            if idx == 4:
                btn.grid(row=idx // 2, column=0, columnspan=2,
                         sticky=tk.EW, padx=3, pady=3)
            else:
                btn.grid(row=idx // 2, column=idx % 2, padx=3, pady=3)

        info = tk.LabelFrame(right, text="运行指标", padx=8, pady=6,
                             font=("Microsoft YaHei", 9))
        info.pack(anchor=tk.W, fill=tk.X, pady=(8, 0))
        self.iter_var = tk.StringVar(value="迭代轮次：0")
        self.sse_var = tk.StringVar(value="SSE：—")
        self.shift_var = tk.StringVar(value="质心最大位移：—")
        for var in (self.iter_var, self.sse_var, self.shift_var):
            tk.Label(info, textvariable=var, anchor=tk.W,
                     font=("Microsoft YaHei", 10)).pack(fill=tk.X)

        steps = tk.Frame(root)
        steps.pack(fill=tk.X, padx=14, pady=(12, 4))
        self.step_labels = []
        for text in STEP_LABELS:
            lb = tk.Label(steps, text=text, width=17, relief=tk.GROOVE,
                          font=("Microsoft YaHei", 9), bg="#F1F5F9", fg="#64748B")
            lb.pack(side=tk.LEFT, padx=3)
            self.step_labels.append(lb)

        bottom = tk.Frame(root)
        bottom.pack(fill=tk.BOTH, expand=True, padx=14, pady=(4, 12))
        tk.Label(bottom, text="迭代日志",
                 font=("Microsoft YaHei", 10, "bold")).pack(anchor=tk.W)
        self.log_box = tk.Text(bottom, height=9, font=("Consolas", 10),
                               bg="#F8FAFC", relief=tk.GROOVE)
        self.log_box.pack(fill=tk.BOTH, expand=True)
        self.log_box.tag_config("warn", foreground="#DC2626")
        self.log_box.tag_config("ok", foreground="#047857")

        self.refresh()

    def log(self, msg, tag=None):
        self.log_box.insert(tk.END, msg + "\n", tag)
        self.log_box.see(tk.END)

    def set_step(self, idx):
        for i, lb in enumerate(self.step_labels):
            if i == idx:
                lb.config(bg="#1D4ED8", fg="white")
            elif i < idx:
                lb.config(bg="#DBEAFE", fg="#1D4ED8")
            else:
                lb.config(bg="#F1F5F9", fg="#64748B")

    def refresh(self):
        draw_all(self.canvas, self.points, self.labels, self.centroids, self.track)
        self.iter_var.set("迭代轮次：%d / %d" % (self.iter, core.MAX_ITER))
        self.sse_var.set("SSE：%.2f" % self.sse_history[-1] if self.sse_history else "SSE：—")
        self.shift_var.set("质心最大位移：%.4f" % self.last_shift
                           if self.centroids else "质心最大位移：—")

    def record_sse(self, value, stage):
        if self.sse_history and value > self.sse_history[-1] + 1e-6:
            self.log("  ⚠ 自检告警：SSE 由 %.2f 上升到 %.2f，请检查 %s 实现"
                     % (self.sse_history[-1], value, stage), "warn")
        self.sse_history.append(value)

    def ensure_points(self):
        mode = self.mode_var.get()
        if not self.points or self.points_mode != mode:
            self.points = core.generate_points(mode, core.DEFAULT_N)
            self.points_mode = mode
            self.labels = [-1] * len(self.points)
            self.centroids = []
            self.track = []
            self.phase = "idle"
            self.iter = 0
            self.sse_history = []
            self.last_shift = 0.0
            self.refresh()

    def on_init(self):
        self.auto_running = False
        self.ensure_points()
        self.k = self.k_var.get()
        method = self.init_var.get()
        self.centroids = core.init_centroids(self.points, self.k, None, method)
        self.labels = [-1] * len(self.points)
        self.track = [[c] for c in self.centroids]
        self.iter = 0
        self.phase = "updated"
        self.sse_history = []
        self.last_shift = 0.0
        self.status.set("已就绪，待分配样本")
        self.log_box.delete("1.0", tk.END)
        self.log("初始化：%d 个样本（%s），k=%d，%s 选出 %d 个初始质心"
                 % (len(self.points), self.points_mode, self.k,
                    "K-Means++" if method == "plusplus" else "随机",
                    len(self.centroids)))
        if method == "plusplus":
            self.log("  K-Means++：首个质心随机，其余按「到已有质心的距离平方」加权抽取，越远越容易被选中")
        else:
            self.log("  随机初始化：完全等概率抽 k 个样本，可能挤在同一簇里，收敛结果不稳定")
        self.log("  切换「初始质心方式」再点一次初始化，可在同一批点上对比两种起点的效果")
        self.set_step(1)
        self.refresh()

    def on_step(self):
        if not self.centroids:
            messagebox.showinfo("提示", "请先点「初始化质心」")
            return
        if self.phase == "converged":
            self.status.set("已收敛，算法停止")
            return

        if self.phase in ("idle", "updated"):
            self.labels = core.assign(self.points, self.centroids)
            sse = core.calc_sse(self.points, self.labels, self.centroids)
            self.record_sse(sse, "assign")
            self.phase = "assigned"
            self.log("第 %d 轮 · 分配：%d 个样本划入 %d 簇  |  SSE = %.2f"
                     % (self.iter + 1, len(self.points), self.k, sse))
            self.set_step(2)
            self.status.set("已分配，待重算质心")
        else:
            new_c, shift = core.update(self.points, self.labels, self.k, self.centroids)
            self.centroids = new_c
            self.last_shift = shift
            self.iter += 1
            for i, c in enumerate(new_c):
                if i < len(self.track):
                    self.track[i].append(c)
            sse = core.calc_sse(self.points, self.labels, self.centroids)
            self.record_sse(sse, "update")
            self.log("第 %d 轮 · 更新：质心最大位移 = %.4f  |  SSE = %.2f"
                     % (self.iter, shift, sse))

            if shift < core.EPS or self.iter >= core.MAX_ITER:
                self.phase = "converged"
                self.auto_running = False
                reason = "质心位移 < 阈值 %.2f" % core.EPS if shift < core.EPS else "达到最大迭代 %d 轮" % core.MAX_ITER
                self.log("✓ 已收敛（%s），最终 SSE = %.2f，共 %d 轮"
                         % (reason, sse, self.iter), "ok")
                self.status.set("已收敛 · 自动停止")
                self.set_step(4)
            else:
                self.phase = "updated"
                self.set_step(3)
                self.status.set("质心已更新，继续迭代")

        self.refresh()

    def on_auto(self):
        if self.auto_running:
            self.auto_running = False
            self.status.set("已暂停")
            return
        if not self.centroids:
            messagebox.showinfo("提示", "请先点「初始化质心」")
            return
        if self.phase == "converged":
            self.status.set("已收敛，无需播放")
            return

        self.auto_running = True
        self.status.set("自动播放中…（再次点击可暂停）")

        def tick():
            if not self.auto_running or self.phase == "converged":
                self.auto_running = False
                return
            self.on_step()
            if self.auto_running:
                self.root.after(AUTO_DELAY, tick)

        tick()

    def on_elbow(self):
        self.auto_running = False
        self.ensure_points()
        self.status.set("肘部法则计算中…")
        self.root.update_idletasks()
        curve = core.elbow_curve(self.points, 1, ELBOW_K_MAX,
                                 seed=None, method="plusplus",
                                 repeats=ELBOW_REPEATS)
        elbow_k = core.find_elbow(curve)
        self.elbow_curve_data = curve
        self.log("肘部法则：对同一批 %d 个点依次取 K=1~%d，每个 K 用 K-Means++ 跑 %d 次取最小 SSE"
                 % (len(self.points), ELBOW_K_MAX, ELBOW_REPEATS))
        for k, v in curve:
            self.log("  K=%2d  SSE = %10.2f" % (k, v))
        if elbow_k is None:
            self.log("曲线几乎是平的，没找到明显肘部（数据可能没有清晰的簇结构）", "warn")
        else:
            self.log("✓ 推荐 K = %d（曲线由陡转平的拐点，之后 K 再增大 SSE 下降很小）"
                     % elbow_k, "ok")
        self.status.set("肘部法则完成 · 推荐 K = %s" % (elbow_k if elbow_k else "无"))
        self.show_elbow_window(curve, elbow_k)

    def show_elbow_window(self, curve, elbow_k):
        if self.elbow_win is not None and self.elbow_win.winfo_exists():
            self.elbow_win.destroy()
        win = tk.Toplevel(self.root)
        self.elbow_win = win
        win.title("肘部法则 · SSE 随 K 的变化")
        win.resizable(False, False)
        win.transient(self.root)
        tk.Label(win, text="K 越大 SSE 越小；拐点之后曲线变平，那个 K 就是性价比最优的 k",
                 font=("Microsoft YaHei", 10), fg="#334155").pack(padx=12, pady=(10, 4))
        cv = tk.Canvas(win, width=ELBOW_CANVAS_W, height=ELBOW_CANVAS_H,
                       bg="white", highlightthickness=1,
                       highlightbackground="#DDE3EA")
        cv.pack(padx=12, pady=4)
        draw_elbow_curve(cv, curve, elbow_k)
        tk.Label(win, text="  ".join("K%d:%.0f" % (k, v) for k, v in curve),
                 font=("Consolas", 9), fg="#475569",
                 wraplength=ELBOW_CANVAS_W).pack(padx=12, pady=(0, 4))
        tk.Label(win, text=("推荐 K = %d" % elbow_k) if elbow_k else "没有明显肘部",
                 font=("Microsoft YaHei", 11, "bold"),
                 fg="#DC2626" if elbow_k else "#64748B").pack(pady=(0, 6))
        bf = tk.Frame(win)
        bf.pack(pady=(0, 10))
        if elbow_k:
            tk.Button(bf, text="用推荐 K 重新初始化", width=18,
                      command=lambda: self.apply_elbow_k(elbow_k),
                      bg="#1D4ED8", fg="white", relief=tk.FLAT,
                      font=("Microsoft YaHei", 10, "bold")).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="关闭", width=8, command=win.destroy,
                  bg="#E2E8F0", relief=tk.FLAT,
                  font=("Microsoft YaHei", 10)).pack(side=tk.LEFT, padx=4)

    def apply_elbow_k(self, k):
        if 2 <= k <= ELBOW_K_MAX:
            self.k_var.set(k)
        self.on_init()
        self.log("已按肘部法则推荐 K = %d 重新初始化（沿用同一批样本，方便对比）"
                 % k, "ok")
        if self.elbow_win is not None and self.elbow_win.winfo_exists():
            self.elbow_win.destroy()

    def on_reset(self):
        self.auto_running = False
        if self.elbow_win is not None and self.elbow_win.winfo_exists():
            self.elbow_win.destroy()
        self.points, self.labels, self.centroids = [], [], []
        self.points_mode = None
        self.track = []
        self.sse_history = []
        self.elbow_curve_data = []
        self.iter = 0
        self.last_shift = 0.0
        self.phase = "idle"
        self.status.set("未初始化")
        self.log_box.delete("1.0", tk.END)
        self.set_step(0)
        self.refresh()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
