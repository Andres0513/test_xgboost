import numpy as np

# ===================== 1. 修复版单棵 CART 回归树 =====================
class DecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        # 终止条件：样本太少 或 树太深 → 返回叶子节点
        if len(y) < self.min_samples_split or depth >= self.max_depth:
            return {"value": np.mean(y)}

        best_feature = None
        best_threshold = None
        best_gain = -float("inf")
        best_left_idx = None
        best_right_idx = None

        # 遍历所有特征找最优分裂
        for feature in range(self.n_features):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left_idx = X[:, feature] <= t
                right_idx = X[:, feature] > t

                # 如果分裂后一边没样本，跳过
                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue

                left_y = y[left_idx]
                right_y = y[right_idx]

                # 平方误差增益（越小越好）
                gain = np.var(left_y) * len(left_y) + np.var(right_y) * len(right_y)
                if -gain > best_gain:
                    best_gain = -gain
                    best_feature = feature
                    best_threshold = t
                    best_left_idx = left_idx
                    best_right_idx = right_idx

        # 如果找不到有效分裂，直接返回叶子节点（修复BUG关键！）
        if best_feature is None:
            return {"value": np.mean(y)}

        # 切分数据
        left_X = X[best_left_idx]
        left_y = y[best_left_idx]
        right_X = X[best_right_idx]
        right_y = y[best_right_idx]

        # 递归构建子树
        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(left_X, left_y, depth + 1),
            "right": self._build_tree(right_X, right_y, depth + 1)
        }

    def predict(self, X):
        preds = []
        for x in X:
            node = self.tree
            while "value" not in node:
                if x[node["feature"]] <= node["threshold"]:
                    node = node["left"]
                else:
                    node = node["right"]
            preds.append(node["value"])
        return np.array(preds)

# ===================== 2. 极简 XGBoost =====================
class SimpleXGBoost:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.base_pred = None

    def fit(self, X, y):
        # 初始预测值：均值
        self.base_pred = np.full_like(y, np.mean(y), dtype=float)
        y_pred = self.base_pred.copy()

        # 逐棵训练
        for _ in range(self.n_estimators):
            # 计算残差（梯度）
            grad = y - y_pred

            # 训练树拟合残差
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, grad)

            # 更新预测
            y_pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        pred = self.base_pred[0]
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred

# ===================== 测试 =====================
if __name__ == "__main__":
    # 简单数据集 y = 2x
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

    model = SimpleXGBoost(n_estimators=30, learning_rate=0.3, max_depth=2)
    model.fit(X, y)

    y_pred = model.predict(X)
    print("真实值：", y)
    print("预测值：", np.round(y_pred, 2))