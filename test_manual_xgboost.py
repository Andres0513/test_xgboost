import numpy as np

# ==============================================
# 手写 XGBoost 单棵树
# 完全复刻XGB核心：二阶泰勒、L2正则、gamma、最小分裂样本
# ==============================================
class XGBTree:
    def __init__(self, max_depth=3, min_samples_split=2, reg_lambda=1.0, gamma=0.1):
        self.max_depth = max_depth          # 树最大深度，限制复杂度
        self.min_samples_split = min_samples_split  # 节点最少样本数，不足则不分裂
        self.reg_lambda = reg_lambda        # L2正则系数 λ，防止叶子权重过大
        self.gamma = gamma                  # 分裂最小增益阈值，剪枝用
        self.tree = None                    # 存储整棵树结构

    # XGBoost 叶子节点最优权重公式
    # w = -sum(g) / (sum(h) + λ)
    def calc_leaf_weight(self, g, h):
        sum_g = np.sum(g)   # 一阶导总和G
        sum_h = np.sum(h)   # 二阶导总和H
        return -sum_g / (sum_h + self.reg_lambda)

    # 计算节点分数（用于分裂增益计算）
    def calc_score(self, g, h):
        sum_g = np.sum(g)
        sum_h = np.sum(h)
        return sum_g ** 2 / (sum_h + self.reg_lambda)

    # 递归构建树
    def _build_tree(self, X, g, h, depth):
        # 终止条件1：当前节点样本数 < 最小分裂样本数，停止分裂
        if len(g) < self.min_samples_split:
            return {"weight": self.calc_leaf_weight(g, h)}
        # 终止条件2：达到最大树深，停止分裂
        if depth >= self.max_depth:
            return {"weight": self.calc_leaf_weight(g, h)}

        # 遍历所有特征、所有分割阈值，寻找最优分裂
        n_features = X.shape[1]
        best_gain = -np.inf    # 最优分裂增益
        best_split = None      # 存储最优分裂信息：特征、阈值、左右索引

        for feat_idx in range(n_features):
            # 当前特征所有唯一取值，作为候选分割阈值
            thresholds = np.unique(X[:, feat_idx])
            for thresh in thresholds:
                # 按阈值切分左右子集
                left_mask = X[:, feat_idx] <= thresh
                right_mask = X[:, feat_idx] > thresh

                # 避免分裂后某一侧无样本，无效分裂直接跳过
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # 计算左右子集分数
                score_left = self.calc_score(g[left_mask], h[left_mask])
                score_right = self.calc_score(g[right_mask], h[right_mask])
                score_parent = self.calc_score(g, h)

                # XGB分裂增益 = 左分数 + 右分数 - 父节点分数
                gain = 0.5 * (score_left + score_right - score_parent)

                # 更新最优分裂
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feat_idx, thresh, left_mask, right_mask)

        # 终止条件3：最优增益小于gamma，没必要分裂（剪枝正则）
        if best_gain < self.gamma or best_split is None:
            return {"weight": self.calc_leaf_weight(g, h)}

        # 解包最优分裂信息
        feat_idx, thresh, left_mask, right_mask = best_split

        # 递归构建左右子树，深度+1
        left_node = self._build_tree(X[left_mask], g[left_mask], h[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], g[right_mask], h[right_mask], depth + 1)

        # 返回当前分裂节点信息
        return {
            "feature": feat_idx,
            "threshold": thresh,
            "left": left_node,
            "right": right_node
        }

    # 训练单棵树：传入特征、一阶导g、二阶导h
    def fit(self, X, g, h):
        self.tree = self._build_tree(X, g, h, depth=0)

    # 单样本递归预测
    def predict_one(self, x, node):
        # 如果是叶子节点，直接返回权重
        if "weight" in node:
            return node["weight"]
        # 非叶子节点，按特征阈值判断走左/右子树
        feat_val = x[node["feature"]]
        if feat_val <= node["threshold"]:
            return self.predict_one(x, node["left"])
        else:
            return self.predict_one(x, node["right"])

    # 批量预测
    def predict(self, X):
        res = [self.predict_one(x, self.tree) for x in X]
        return np.array(res)

# ==============================================
# 手写 完整XGBoost 框架
# 核心：梯度提升迭代 + 累加每棵树预测结果
# ==============================================
class MyXGBoost:
    def __init__(self, n_estimators=10, learning_rate=0.1,
                 max_depth=2, min_samples_split=2, reg_lambda=1.0, gamma=0.1):
        self.n_estimators = n_estimators   # 迭代轮数 = 树的总数量
        self.learning_rate = learning_rate # 学习率，收缩每棵树权重，防过拟合
        self.max_depth = max_depth         # 每棵树最大深度
        self.min_samples_split = min_samples_split # 单树最小分裂样本
        self.reg_lambda = reg_lambda       # L2正则
        self.gamma = gamma                 # 分裂增益阈值
        self.trees = []                    # 存放所有训练好的树
        self.base_score = None             # 全局初始基准预测值

    # 计算损失的 一阶导数g、二阶导数h
    # 回归任务MSE损失：g = y_pred - y，h = 1
    def _get_grad_hess(self, y_true, y_pred):
        g = y_pred - y_true
        h = np.ones_like(y_true)
        return g, h

    # ==============================================
    # 梯度提升 核心迭代流程
    # ==============================================
    def fit(self, X, y):
        # 1. 初始化全局基准预测（全局均值）
        self.base_score = np.mean(y)
        # 强制浮点类型，避免int/float相加报错
        y_pred = np.full_like(y, self.base_score, dtype=np.float64)

        # 2. 梯度提升迭代：循环训练每一棵树
        for epoch in range(self.n_estimators):
            # 2.1 计算当前预测的一阶、二阶导数
            g, h = self._get_grad_hess(y, y_pred)

            # 2.2 初始化并训练单棵树，拟合梯度g、h
            tree = XGBTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma
            )
            tree.fit(X, g, h)

            # 2.3 累加当前树的预测，学习率缩放
            # 梯度提升核心：不断叠加弱学习器修正预测
            y_pred += self.learning_rate * tree.predict(X)

            # 2.4 保存当前树，用于后续预测
            self.trees.append(tree)

    # 整体模型预测：初始值 + 所有树累加结果
    def predict(self, X):
        # 初始基准值
        pred = np.full(X.shape[0], self.base_score, dtype=np.float64)
        # 累加每一棵树的输出
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred

# ==============================================
# 测试运行
# ==============================================
if __name__ == "__main__":
    # 简单线性数据 y = 2*x
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

    # 初始化手写XGB
    model = MyXGBoost(
        n_estimators=30,        # 迭代30棵树
        learning_rate=0.3,      # 学习率
        max_depth=2,            # 单树深度
        min_samples_split=3,    # 至少3个样本才允许分裂
        reg_lambda=1.0,         # L2正则
        gamma=0.05              # 分裂最小增益
    )

    # 训练（梯度提升迭代全过程）
    model.fit(X, y)
    # 预测
    y_pred = model.predict(X)

    print("真实值：", y)
    print("预测值：", np.round(y_pred, 2))