import numpy as np

# ==========================================
# 1. Softmax：把 logit → 概率
# ==========================================
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# ==========================================
# 2. MLogloss 损失函数
# ==========================================
def mlogloss(y_true, y_pred_proba):
    eps = 1e-10
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    loss = 0
    n = len(y_true)
    for i in range(n):
        true_cls = int(y_true[i])
        loss -= np.log(y_pred_proba[i, true_cls])
    return loss / n

# ==========================================
# 🌟 XGBoost 核心：MLogloss 的 一阶导 + 二阶导
# ==========================================
def xgb_mlogloss_gradient(y_true, y_pred_proba, num_classes):
    n = len(y_true)
    grad = np.zeros((n, num_classes))
    hess = np.zeros((n, num_classes))  # 👈 XGBoost 必须有二阶导！

    for i in range(n):
        t = int(y_true[i])
        for k in range(num_classes):
            p = y_pred_proba[i, k]
            g = p - 1.0 if k == t else p
            h = p * (1.0 - p)  # 二阶导 Hessian
            grad[i, k] = g
            hess[i, k] = h
    return grad, hess

# ==========================================
# 🌟 真正 XGBoost CART 树（多层、带二阶导、带正则）
# ==========================================
class XGBTree:
    def __init__(self, max_depth=3, reg_lambda=1.0):
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda  # XGB 正则
        self.split_feat = None
        self.split_val = None
        self.left = None
        self.right = None
        self.weight = None  # 叶子权重

    # 🌟 XGBoost 分裂增益公式（真正的 XGB 分裂！）
    def xgb_gain(self, G, H):
        return 0.5 * (G ** 2) / (H + self.reg_lambda)

    # 寻找最优分裂
    def find_split(self, X, G, H):
        best_gain = -float('inf')
        best_feat = None
        best_val = None
        n_samples, n_feats = X.shape

        G_total = G.sum()
        H_total = H.sum()

        for f in range(n_feats):
            sorted_idx = np.argsort(X[:, f])
            G_left, H_left = 0.0, 0.0

            for i in range(n_samples - 1):
                idx = sorted_idx[i]
                G_left += G[idx]
                H_left += H[idx]
                G_right = G_total - G_left
                H_right = H_total - H_left

                gain = self.xgb_gain(G_total, H_total) - \
                       self.xgb_gain(G_left, H_left) - \
                       self.xgb_gain(G_right, H_right)

                if gain > best_gain and X[sorted_idx[i], f] != X[sorted_idx[i+1], f]:
                    best_gain = gain
                    best_feat = f
                    best_val = X[idx, f]

        return best_feat, best_val

    # 建树（递归多层）
    def build(self, X, G, H, depth):
        # 叶子节点：计算 XGB 权重
        if depth >= self.max_depth:
            self.weight = -G.sum() / (H.sum() + self.reg_lambda)
            return

        best_feat, best_val = self.find_split(X, G, H)

        if best_feat is None:
            self.weight = -G.sum() / (H.sum() + self.reg_lambda)
            return

        # 分裂
        self.split_feat = best_feat
        self.split_val = best_val
        left_mask = X[:, best_feat] <= best_val
        right_mask = ~left_mask

        # 递归建左右子树 → 多层！
        self.left = XGBTree(self.max_depth, self.reg_lambda)
        self.right = XGBTree(self.max_depth, self.reg_lambda)
        self.left.build(X[left_mask], G[left_mask], H[left_mask], depth + 1)
        self.right.build(X[right_mask], G[right_mask], H[right_mask], depth + 1)

    def fit(self, X, grad, hess):
        self.build(X, grad, hess, depth=0)

    # 预测
    def predict(self, X):
        if self.split_feat is None:
            return np.full(len(X), self.weight)
        mask = X[:, self.split_feat] <= self.split_val
        pred = np.zeros(len(X))
        pred[mask] = self.left.predict(X[mask])
        pred[~mask] = self.right.predict(X[~mask])
        return pred

# ==========================================
# 🌟 手写 XGBoost 多分类（MLogloss）
# ==========================================
class XGBoostMLogloss:
    def __init__(self, num_classes, n_estimators=6, max_depth=3, lr=0.3, reg_lambda=1.0):
        self.num_classes = num_classes
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.trees = [[] for _ in range(num_classes)]

    def fit(self, X, y):
        n_samples = X.shape[0]
        logits = np.zeros((n_samples, self.num_classes))

        for iter in range(self.n_estimators):
            y_proba = softmax(logits)
            loss = mlogloss(y, y_proba)
            print(f"Iter {iter+1} | MLogloss = {loss:.4f}")

            grad, hess = xgb_mlogloss_gradient(y, y_proba, self.num_classes)

            for k in range(self.num_classes):
                tree = XGBTree(max_depth=self.max_depth, reg_lambda=self.reg_lambda)
                tree.fit(X, grad[:, k], hess[:, k])
                self.trees[k].append(tree)
                logits[:, k] += self.lr * tree.predict(X)

    def predict_proba(self, X):
        logits = np.zeros((len(X), self.num_classes))
        for k in range(self.num_classes):
            for tree in self.trees[k]:
                logits[:, k] += self.lr * tree.predict(X)
        return softmax(logits)

# ==========================================
# 测试：物流时效 0~20 天概率预测
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)
    N = 1000
    X = np.random.randint(0, 100, (N, 3))
    y = np.random.randint(0, 21, N)
    num_classes = 21

    # 🌟 这是真正的 XGBoost！
    model = XGBoostMLogloss(num_classes=21, n_estimators=6, max_depth=3, lr=0.3)
    model.fit(X, y)

    print("\n预测 0~20 天概率：")
    prob = model.predict_proba(X[:5])
    print(np.round(prob, 3))