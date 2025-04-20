import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Hàm để thực hiện lấy mẫu bootstrap
def bootstrap(X, y):
    n_sample = X.shape[0]
    _id = np.random.choice(n_sample, n_sample, replace=True)
    return X.iloc[_id], y.iloc[_id]

# Lớp Random Forest tùy chỉnh
class RandomForest:
    def __init__(self, n_trees=5, max_depth=10, min_samples_split=2):
        self.n_trees = n_trees  
        self.max_depth = max_depth  
        self.min_samples_split = min_samples_split 
        self.trees = []  

    def fit(self, X, y):
        # Khởi tạo danh sách các cây
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            X_sample, y_sample = bootstrap(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Thu thập dự đoán từ mỗi cây
        arr_pred = np.array([tree.predict(X) for tree in self.trees])
        final_pred = []
        for i in range(arr_pred.shape[1]):
            sample_pred = arr_pred[:, i]
            final_pred.append(pd.Series(sample_pred).mode()[0])  
        return np.array(final_pred) 

# Tải và tiền xử lý dữ liệu
data = pd.read_csv('drug200.csv')

# Định nghĩa ma trận đặc trưng X và vector mục tiêu y
X = data[['Sex', 'BP', 'Cholesterol']]
y = data['Drug']

# Ánh xạ các biến phân loại sang giá trị số để phục vụ huấn luyện mô hình
X.loc[:, 'Sex'] = X['Sex'].map({'M': 0, 'F': 1})  
X.loc[:, 'BP'] = X['BP'].map({'HIGH': 2, 'NORMAL': 1, 'LOW': 0})  
X.loc[:, 'Cholesterol'] = X['Cholesterol'].map({'NORMAL': 0, 'HIGH': 1})  
y = y.map({'drugA': 0, 'drugB': 1, 'drugC': 2, 'drugX': 3, 'DrugY': 4})  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Huấn luyện và đánh giá mô hình Random Forest
random_forest = RandomForest(n_trees=10, max_depth=10, min_samples_split=2)  # Tăng số lượng cây
random_forest.fit(X_train, y_train)  # Huấn luyện mô hình trên dữ liệu huấn luyện

# Dự đoán trên tập kiểm tra
y_pred_rf = random_forest.predict(X_test)  
# Tính độ chính xác của các dự đoán
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Xuất kết quả dự đoán và độ chính xác
print("Dự đoán của Random Forest:", y_pred_rf)
print("Giá trị thực tế:", y_test.values)
print(f'Độ chính xác của Random Forest: {accuracy_rf:.2f}')