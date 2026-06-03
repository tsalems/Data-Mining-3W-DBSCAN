import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    # Bước 1: Dò tìm dấu phân cách (separator)
    # Đọc nháp với dấu phẩy
    df_temp = pd.read_csv(file_path)
    sep_used = ','
    
    if df_temp.shape[1] == 1:
        # Nếu chỉ có 1 cột, chắc chắn dùng khoảng trắng/tab
        sep_used = r'\s+'
        df_temp = pd.read_csv(file_path, sep=sep_used)
        
    # Bước 2: Kiểm tra xem file có Header hay không
    # Nếu tên cột toàn là số (ví dụ: '15.26', '1', '0.89'), nghĩa là dòng dữ liệu đầu tiên đã bị nuốt làm header
    is_no_header = all(str(c).replace('.', '', 1).replace('-', '', 1).isdigit() for c in df_temp.columns)
    
    # Bước 3: Đọc file chính thức với đầy đủ thông số chuẩn (separator và header)
    if is_no_header:
        df = pd.read_csv(file_path, header=None, sep=sep_used)
    else:
        df = pd.read_csv(file_path, sep=sep_used)

    df = df.dropna()

    # Bước 4: Tách X (features) và y (labels)
    label_col = None
    for col in df.columns:
        if str(col).strip().lower() in ['class', 'label', 'target', 'y']:
            label_col = col
            break
    
    if label_col is not None:
        raw_y = df[label_col].values
        X = df.drop(columns=[label_col]).values 
    else:
        X = df.iloc[:, :-1].values
        raw_y = df.iloc[:, -1].values

    # Bước 5: Chuẩn hóa kiểu dữ liệu an toàn
    le = LabelEncoder()
    y = le.fit_transform(raw_y)
    X = X.astype(float)
    
    return X, y