import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("figures/LE3W", exist_ok=True)

COLORS = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'brown', 'pink', 'olive', 'teal']
MARKERS = ['v', '*', 's', 'D', '^', 'p', 'H', 'X', 'd', '>']

def plot_original(ax, X, y, title):
    unique_labels = set(y)
    for k in unique_labels:
        if k == -1:
            mask = (y == k)
            ax.scatter(X[mask, 0], X[mask, 1], c='black', marker='x', s=15, label='Noise')
        else:
            mask = (y == k)
            color = COLORS[k % len(COLORS)]
            ax.scatter(X[mask, 0], X[mask, 1], c=color, s=20, label=f'C{k+1}')
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='lower right', fontsize=8, framealpha=0.8)

def plot_threeway(ax, X, model, title):
    for k in model.POS.keys():
        color = COLORS[(k-1) % len(COLORS)]
        marker = MARKERS[(k-1) % len(MARKERS)]
        
        pos_idx = list(model.POS[k])
        if len(pos_idx) > 0:
            ax.scatter(X[pos_idx, 0], X[pos_idx, 1], c=color, marker='o', s=20, label=f'POS(C{k})')
            
        bnd_idx = list(model.BND[k])
        if len(bnd_idx) > 0:
            ax.scatter(X[bnd_idx, 0], X[bnd_idx, 1], facecolors='none', edgecolors=color, 
                       marker=marker, s=35, linewidths=1.0, label=f'BND(C{k})')
            
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, framealpha=0.8)

def plot_paper_comparison(X, y, tw_model, le3w_model, dataset_name):
    """Vẽ format 3 hình (a, b, c) ngang nhau cho các Hình"""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    plot_original(axs[0], X, y, "(a) Original")
    plot_threeway(axs[1], X, tw_model, "(b) 3W-DBSCAN")
    plot_threeway(axs[2], X, le3w_model, "(c) LE3W-DBSCAN")
    
    plt.tight_layout()
    plt.savefig(f"figures/LE3W/Figure_{dataset_name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()