import numpy as np
from scipy.spatial import cKDTree

def extract_border_from_partial(pcd,           # 已归一化的剩余点云 ndarray(N,3)
                                r_list=(0.1, 0.2, 0.3),   # 归一化空间下的半径
                                tau2=0.35):
    """
    返回边界点在原点云中的索引（int ndarray）
    """
    tree = cKDTree(pcd)
    scores = np.zeros(len(pcd), dtype=np.float32)

    # ---------- 只计算一次的全局密度 ----------
    d, _ = tree.query(pcd, k=2)
    mean_nn = d[:, 1].mean()
    # 用邻居球体积的倒数作参考密度，防止量纲问题
    rho_global = 1.0 / (4 / 3 * np.pi * mean_nn ** 3)

    # ---------- 多尺度投票 ----------
    for r in r_list:
        idx_list = tree.query_ball_point(pcd, r)
        for i, idx in enumerate(idx_list):
            if len(idx) < 6:
                continue
            pts = pcd[idx]
            mu = pts.mean(axis=0)
            cov = (pts - mu).T @ (pts - mu) / len(idx)
            l = np.linalg.eigvalsh(cov)[::-1]
            theta = l[1] / (l[0] + 1e-8)

            # 正确的 delta，保证 <= 1
            v_ball = 4 / 3 * np.pi * r ** 3
            delta = len(idx) / (v_ball * rho_global + 1e-8)

            w = max(0.0, np.exp(-theta) * (1 - delta))  # 强制截断到 [0,1]
            scores[i] += w
            print(scores)
    scores /= len(r_list)

    # ---------- 候选点 ----------
    cand_idx = np.where(scores > tau2)[0]
    if cand_idx.size == 0:
        return np.array([], dtype=np.int64)

    # ---------- 欧式聚类 ----------
    cand_pts = pcd[cand_idx]
    tree2 = cKDTree(cand_pts)
    clusters = []
    used = np.zeros(len(cand_pts), bool)
    for i in range(len(cand_pts)):
        if used[i]: continue
        seed = [i]
        queue = [i]
        used[i] = True
        while queue:
            q = queue.pop()
            neigh = tree2.query_ball_point(cand_pts[q], r=0.03)
            for n in neigh:
                if not used[n]:
                    used[n] = True
                    queue.append(n)
                    seed.append(n)
        if len(seed) >= 10:
            clusters.append(cand_idx[np.array(seed)])

    # ---------- 主方向裁剪 ----------
    border_idx = []
    for cl in clusters:
        pts = pcd[cl]
        if len(pts) < 8:
            continue
        mu = pts.mean(axis=0)
        _, eigvecs = np.linalg.eigh(np.cov(pts.T))
        dir_vec = eigvecs[:, 0]            # 最长轴
        proj = np.dot(pts - mu, dir_vec)
        k = max(1, int(0.1 * len(proj)))
        order = np.argsort(proj)
        tips = np.concatenate([order[:k], order[-k:]])
        border_idx.extend(cl[tips])

    return np.unique(border_idx)


# ---------------- demo ----------------
if __name__ == "__main__":
    # 假设 pcd 已是缺失后、归一化的 ndarray
    pcd = np.load("crown_partial.npy")   # 请替换为真实文件
    idx = extract_border_from_partial(pcd)
    print("检测到边界点数量：", idx.shape[0])

    # 可视化（可选）
    try:
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pcd)
        pc.colors = o3d.utility.Vector3dVector([[0.5,0.5,0.5]]*len(pcd))
        np.asarray(pc.colors)[idx] = [1,0,0]
        o3d.visualization.draw_geometries([pc])
    except ImportError:
        pass