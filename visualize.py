import os
import matplotlib.pyplot as plt
import numpy as np

# Tự động tạo thư mục chứa ảnh nếu chưa có
os.makedirs("figures/3W", exist_ok=True)

# Bảng màu và ký hiệu chuẩn
COLORS = ['blue', 'magenta', 'green', 'orange', 'cyan', 'purple', 'brown', 'pink', 'olive', 'teal', 'navy', 'crimson']
MARKERS = ['v', '*', 's', 'D', '^', 'p', 'H', 'X', 'd', '>', '<', '8']

def plot_2way_clusters(ax, X, labels, title):
    unique_labels = set(labels)
    for k in unique_labels:
        if k == -1:
            mask = (labels == k)
            ax.scatter(X[mask, 0], X[mask, 1], c='black', marker='x', s=15, label='Noise')
        else:
            mask = (labels == k)
            color = COLORS[k % len(COLORS)]
            ax.scatter(X[mask, 0], X[mask, 1], c=color, s=25, label=f'C{k+1}')
            
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize=7, ncol=4, framealpha=0.8)

def plot_3way_clusters(ax, X, model, title):
    for k in model.POS.keys():
        color = COLORS[k % len(COLORS)]
        marker = MARKERS[k % len(MARKERS)]
        
        pos_idx = list(model.POS[k])
        if len(pos_idx) > 0:
            ax.scatter(X[pos_idx, 0], X[pos_idx, 1], c=color, marker='o', s=25, label=f'POS(C{k+1})')
            
        bnd_idx = list(model.BND[k])
        if len(bnd_idx) > 0:
            ax.scatter(X[bnd_idx, 0], X[bnd_idx, 1], facecolors='none', edgecolors=color, 
                       marker=marker, s=40, linewidths=1.2, label=f'BND(C{k+1})')
            
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize=7, ncol=4, framealpha=0.8)

def plot_4_panels(X, y, ds_labels, ce3_model, tw_model, dataset_name):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    # Thêm tiêu đề lớn chứa tên dataset ở trên cùng
    fig.suptitle(f"Clustering Results on {dataset_name.upper()} Dataset", fontsize=16, fontweight='bold', y=0.98)
    
    plot_2way_clusters(axs[0, 0], X, y, "(a) The original distribution")
    plot_2way_clusters(axs[0, 1], X, ds_labels, "(b) DScale-DBSCAN")
    plot_3way_clusters(axs[1, 0], X, ce3_model, "(c) CE3-kmeans")
    plot_3way_clusters(axs[1, 1], X, tw_model, "(d) 3W-DBSCAN")
    
    plt.tight_layout(pad=3.0)
    fig.subplots_adjust(top=0.92) # Dịch các hình nhỏ xuống một chút để nhường chỗ cho tiêu đề lớn
    
    # Lưu ảnh vào folder figures
    plt.savefig(f"figures/Figure_{dataset_name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_figure_10(eta_values, dataset_f1_dict):
    """Vẽ Figure 10: F1 on different datasets with different eta values"""
    plt.figure(figsize=(10, 7))
    
    for i, (ds_name, f1_list) in enumerate(dataset_f1_dict.items()):
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        plt.plot(eta_values, f1_list, marker=marker, linestyle='-', color=color, linewidth=2, label=ds_name)
    
    plt.xlabel('Parameter $\eta$', fontsize=12)
    plt.ylabel('F1 Measure', fontsize=12)
    plt.title('Figure 10: F1 on different datasets with different $\eta$ values', fontsize=14, fontweight='bold')
    
    # Đưa bảng chú giải ra một góc rõ ràng
    plt.legend(loc='best', fontsize=10, ncol=2)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig("figures/3W/Figure_10_F1_vs_eta.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()