import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Hàm chia node thành 2 node con dựa trên ngưỡng
def split_node(column, threshold_split):  
    left_node = column[column <= threshold_split].index  
    right_node = column[column > threshold_split].index  
    return left_node, right_node  

# Hàm tính entropy
def entropy(y_target):  
    values, counts = np.unique(y_target, return_counts=True)  
    result = -np.sum([(count / len(y_target)) * np.log2(count / len(y_target)) for count in counts])  
    return result  

# Hàm tính information gain
def info_gain(column, target, threshold_split):  
    entropy_start = entropy(target)  

    left_node, right_node = split_node(column, threshold_split) 

    n_target = len(target) 
    n_left = len(left_node) 
    n_right = len(right_node)  

    # Tính entropy cho các node con
    entropy_left = entropy(target[left_node])  
    entropy_right = entropy(target[right_node])  

    # Tính tổng entropy của các node con
    weight_entropy = (n_left / n_target) * entropy_left + (n_right / n_target) * entropy_right

    # Tính Information Gain
    ig = entropy_start - weight_entropy  
    return ig

# Hàm tìm feature và threshold tốt nhất để chia thông tin thu được từ hàm info_gain
def best_split(dataX, target, feature_id):  
    best_ig = -1  
    best_feature = None 
    best_threshold = None 
    for _id in feature_id:
        column = dataX.iloc[:, _id]  
        thresholds = set(column) 
        for threshold in thresholds:  
            ig = info_gain(column, target, threshold) 
            if ig > best_ig:  
                best_ig = ig  
                best_feature = dataX.columns[_id]  
                best_threshold = threshold  
    return best_feature, best_threshold  

# F trong node lá
def most_value(y_target):  
    if y_target.empty:  
        return None  
    value = y_target.value_counts().idxmax()  
    return value  
def grow_tree(self, X, y, depth=0):  
    n_samples, n_feats = X.shape  
    n_classes = len(np.unique(y))  

    # Điều kiện dừng
    if (depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples_split):
        leaf_value = most_value(y)  
        if leaf_value is None:  
            return Node(value=0)  
        return Node(value=leaf_value)

# Lớp Node đại diện cho từng node trong cây
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature  
        self.threshold = threshold  
        self.left = left  
        self.right = right 
        self.value = value  

    def is_leaf_node(self):  # Hàm kiểm tra có phải là node lá hay không
        return self.value is not None  
    
# Lớp Decision Tree Classification
class DecisionTreeClass:
    def __init__(self, min_samples_split=2, max_depth=10, n_features=None):
        self.min_samples_split = min_samples_split 
        self.max_depth = max_depth  
        self.root = None  
        self.n_features = n_features 

    def grow_tree(self, X, y, depth=0):  
        n_samples, n_feats = X.shape  
        n_classes = len(np.unique(y))  

        # Điều kiện dừng
        if (depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples_split):
            leaf_value = most_value(y)  
            return Node(value=leaf_value)  

        # Lấy số cột ngẫu nhiên khi tham số n_features khác None
        feature_id = np.random.choice(n_feats, self.n_features, replace=False)

        # Tìm feature và threshold tốt nhất để chia
        best_feature, best_threshold = best_split(X, y, feature_id)

        # Tách node thành node trái và phải
        left_node, right_node = split_node(X[best_feature], best_threshold)

        # Dùng đệ quy để xây dựng cây con
        left = self.grow_tree(X.loc[left_node], y.loc[left_node], depth + 1)
        right = self.grow_tree(X.loc[right_node], y.loc[right_node], depth + 1)

        # Trả về node hiện tại với thông tin chia và 2 node con
        return Node(best_feature, best_threshold, left, right)

    def fit(self, X, y):  # X là DataFrame, y là series
        self.n_features = X.shape[1] if self.n_features is None else min(X.shape[1], self.n_features)
        self.root = self.grow_tree(X, y)  

    def traverse_tree(self, x, node):  # Hàm duyệt cây để dự đoán, x là series

        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)

    def predict(self, X):  
        return np.array([self.traverse_tree(x, self.root) for index, x in X.iterrows()])  # Duyệt qua từng dòng trong X

# Hàm tính độ chính xác
def accuracy(y_actual, y_pred):  
    acc = np.mean(y_actual == y_pred)  
    return acc * 100  

# Nhập dữ liệu
data = pd.read_csv('drug200.csv')
print(data)

# Tạo tập X và y
X = data[['Sex', 'BP', 'Cholesterol']]  
y = data['Drug']  

X = data[['Sex', 'BP', 'Cholesterol']].copy()

# Biến đổi dữ liệu định tính sang định lượng
X.loc[:, 'Sex'] = X['Sex'].map({'M': 0, 'F': 1})  
X.loc[:, 'BP'] = X['BP'].map({'HIGH': 2, 'NORMAL': 1, 'LOW': 0})  
X.loc[:, 'Cholesterol'] = X['Cholesterol'].map({'NORMAL': 0, 'HIGH': 1})  
y = y.map({'drugA': 0, 'drugB': 1, 'drugC': 2, 'drugX': 3, 'DrugY': 4}) 

print(X)
print(y)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train)
print(y_train)

# Sử dụng mô hình Decision Tree
decisionTree = DecisionTreeClass(min_samples_split=2, max_depth=10)  
decisionTree.fit(X_train, y_train)  

# Dự đoán trên tập kiểm tra
y_pred_dt = decisionTree.predict(X_test)
print("Dự đoán của Decision Tree:", y_pred_dt)
print("Giá trị thực tế:", y_test.values)

# Tính độ chính xác
accuracy_dt = accuracy(y_test.values, y_pred_dt)  # Tính độ chính xác
print(f'Độ chính xác của Decision Tree: {accuracy_dt:.2f}%')