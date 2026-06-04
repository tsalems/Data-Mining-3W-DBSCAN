import numpy as np
from sklearn.metrics import f1_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

def match_labels(true_labels, pred_labels):
    """
    Thuật toán Hungarian matching (Khớp nhãn).
    Có bẫy lỗi cho trường hợp mảng rỗng hoặc thuật toán phân loại toàn bộ là nhiễu.
    """
    # 1. Bẫy lỗi: Trả về nguyên gốc nếu mảng rỗng
    if len(true_labels) == 0 or len(pred_labels) == 0:
        return pred_labels
    
    unique_true = np.unique(true_labels)
    unique_pred = np.unique(pred_labels)
    
    # 2. Lọc bỏ nhãn nhiễu (-1) ra khỏi thuật toán đếm
    unique_pred = unique_pred[unique_pred != -1]
    
    if len(unique_pred) == 0:
        return pred_labels
        
    contingency_matrix = np.zeros((len(unique_true), len(unique_pred)), dtype=np.int64)
    
    # 3. Tạo ánh xạ liên tục (0, 1, 2...) để chống lỗi index out-of-bounds
    true_idx_map = {val: idx for idx, val in enumerate(unique_true)}
    pred_idx_map = {val: idx for idx, val in enumerate(unique_pred)}
    
    for t, p in zip(true_labels, pred_labels):
        if p != -1:  
            contingency_matrix[true_idx_map[t], pred_idx_map[p]] += 1
            
    # 4. Kiểm tra safety một lần nữa trước khi chạy Hungarian
    if contingency_matrix.size == 0:
        return pred_labels
        
    row_ind, col_ind = linear_sum_assignment(contingency_matrix.max() - contingency_matrix)
    
    # 5. Tạo dictionary map ngược từ nhãn dự đoán sang nhãn thực
    label_mapping = {unique_pred[col]: unique_true[row] for row, col in zip(row_ind, col_ind)}
    mapped_preds = np.array([label_mapping.get(p, -1) for p in pred_labels])
    
    return mapped_preds

def calculate_accuracy(true_labels, pred_labels):
    """Tính toán Accuracy (Acc) - Eq. (11)"""
    if len(true_labels) == 0: return 0.0
    mapped_preds = match_labels(true_labels, pred_labels)
    correct = np.sum(true_labels == mapped_preds)
    return correct / len(true_labels)

def calculate_f1(true_labels, pred_labels):
    """Tính F-measure (F1) - Eq. (12)"""
    if len(true_labels) == 0: return 0.0
    mapped_preds = match_labels(true_labels, pred_labels)
    return f1_score(true_labels, mapped_preds, average='macro')

def calculate_nmi(true_labels, pred_labels):
    """Tính NMI - Eq. (13)"""
    if len(true_labels) == 0: return 0.0
    return normalized_mutual_info_score(true_labels, pred_labels)

def calculate_soft_metrics(model, n_samples):
    """
    Tính toán các chỉ số Soft Clustering: gamma, alpha, alpha_star 
    theo định nghĩa của bài báo (Maji et al. và Zhang).
    """
    pos_mass = 0
    total_mass = 0
    alpha_sum = 0
    c = len(model.POS.keys())
    
    if c == 0:
        return 0.0, 0.0, 0.0
        
    for k in model.POS.keys():
        len_pos = len(model.POS[k])
        len_bnd = len(model.BND[k])
        
        pos_mass += len_pos
        total_mass += (len_pos + len_bnd)
        
        if (len_pos + len_bnd) > 0:
            alpha_sum += (len_pos / (len_pos + len_bnd))
            
    gamma = pos_mass / n_samples
    alpha = alpha_sum / c
    alpha_star = pos_mass / total_mass if total_mass > 0 else 0
    
    return gamma, alpha, alpha_star