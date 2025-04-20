import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Hàm để thực hiện lấy dữ liệu
def loadCsv(filename) -> pd.DataFrame:
    try:
        data = pd.read_csv(filename)
        return data
    except Exception as e:
        return None

# Hàm biến đổi cột định tính, dùng phương pháp one hot
def transform(data, columns_trans): 
    for i in columns_trans:
        unique = data[i].unique() + '-' + i 
        matrix_0 = np.zeros((len(data), len(unique)), dtype=int)
        frame_0 = pd.DataFrame(matrix_0, columns=unique)
        for index, value in enumerate(data[i]):
            frame_0.at[index, value + '-' + i] = 1
        data[unique] = frame_0
    return data

# Hàm scale dữ liệu về [0,1] (min-max scaler)
def scale_data(data, columns_scale): 
    for i in columns_scale:  
        _max = data[i].max()
        _min = data[i].min()
        if _max - _min == 0:
            data[i] = 0  # Hoặc xử lý theo cách khác nếu cần
        else:
            min_max_scaler = lambda x: round((x - _min) / (_max - _min), 3)
            data[i] = data[i].apply(min_max_scaler)
    return data

# Hàm tính khoảng cách Cosine
def cosine_distance(train_X, test_X):
    dict_distance = {}
    for index, value in enumerate(test_X, start=1):
        for j in train_X:
            result = np.sqrt(np.sum((j - value)**2))
            if index not in dict_distance:
                dict_distance[index] = [result]
            else:
                dict_distance[index].append(result)
    return dict_distance 

# Hàm gán kết quả theo k
def pred_test(k, train_X, test_X, train_y):
    lst_predict = []
    dict_distance = cosine_distance(train_X, test_X)
    train_y = train_y.to_frame(name='target').reset_index(drop=True)
    frame_concat = pd.concat([pd.DataFrame(dict_distance), train_y], axis=1)
    
    for i in range(1, len(dict_distance) + 1):
        sort_distance = frame_concat[[i, 'target']].sort_values(by=i, ascending=True)[:k]
        target_predict = sort_distance['target'].value_counts(ascending=False).index[0]
        lst_predict.append([i, target_predict])
    
    return lst_predict

# Tải dữ liệu từ drug200.csv
data = loadCsv('drug200.csv')

# xử lý dữ liệu
df = transform(data, ['Sex', 'BP', 'Cholesterol']).drop(['Sex', 'BP', 'Cholesterol'], axis=1)
df = scale_data(df, ['Age', 'Na_to_K'])

# Tạo data_X và target
data_X = df.drop(['Drug'], axis=1).values
data_y = df['Drug']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=0)

# Dự đoán và đánh giá mô hình
test_pred = pred_test(6, X_train, X_test, y_train)

# Tạo DataFrame cho dự đoán và giá trị thực tế
df_test_pred = pd.DataFrame(test_pred).drop([0], axis=1)
df_test_pred.index = range(1, len(test_pred) + 1)
df_test_pred.columns = ['Predict']

df_actual = pd.DataFrame(y_test)
df_actual.index = range(1, len(y_test) + 1)
df_actual.columns = ['Actual']

# Kết hợp dự đoán và giá trị thực tế
results = pd.concat([df_test_pred, df_actual], axis=1)

# Tính độ chính xác của mô hình
def calculate_accuracy(predictions, actuals):
    predictions = predictions.reset_index(drop=True)
    actuals = actuals.reset_index(drop=True)
    correct = sum(predictions == actuals)
    accuracy = correct / len(actuals) * 100
    return accuracy

# Chuyển đổi danh sách dự đoán thành Series để so sánh
predicted_series = pd.Series([pred[1] for pred in test_pred])

# Tính độ chính xác
accuracy = calculate_accuracy(predicted_series, y_test)
print(f"Độ chính xác của mô hình là: {accuracy:.2f}%")