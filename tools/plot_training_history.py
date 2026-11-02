"""
把 cgan/train.py 輸出的 training_history.json 畫成驗證集 IoU 曲線。

用法（在 repo 根目錄）：
    python tools/plot_training_history.py runs/cgan/training_history.json --out runs/cgan/iou_curve.png
"""
import argparse
import json

import matplotlib.pyplot as plt

AFTER = '#2a78d6'       # 校正後
BEFORE = '#8a8984'      # 校正前（參考線）
SURFACE = '#fcfcfb'
TEXT = '#0b0b0b'
TEXT_2 = '#52514e'
GRID = '#e4e3df'


def main():
    parser = argparse.ArgumentParser(description='繪製 cGAN 訓練過程的驗證集 IoU')
    parser.add_argument('history', help='training_history.json')
    parser.add_argument('--out', default='iou_curve.png', help='輸出圖檔')
    parser.add_argument('--title', default='Validation IoU of pseudo-label boxes during cGAN training')
    args = parser.parse_args()

    with open(args.history, encoding='utf-8') as f:
        history = json.load(f)
    epochs = [h['epoch'] for h in history]
    before = [h['mean_iou_before'] for h in history]
    after = [h['mean_iou_after'] for h in history]
    best = max(history, key=lambda h: h['delta_iou'])

    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': GRID, 'axes.labelcolor': TEXT_2, 'xtick.color': TEXT_2, 'ytick.color': TEXT_2,
        'axes.grid': True, 'grid.color': GRID, 'axes.spines.top': False, 'axes.spines.right': False,
        'font.size': 11,
    })
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, after, color=AFTER, linewidth=2, label='After cGAN calibration')
    ax.plot(epochs, before, color=BEFORE, linewidth=2, linestyle='--', label='Before (K-fold pseudo-labels)')
    ax.annotate(f'{after[-1]:.3f}', (epochs[-1], after[-1]), xytext=(6, 0),
                textcoords='offset points', va='center', color=TEXT_2)
    ax.annotate(f'{before[-1]:.3f}', (epochs[-1], before[-1]), xytext=(6, 0),
                textcoords='offset points', va='center', color=TEXT_2)
    ax.set_title(args.title, loc='left', fontsize=13, fontweight='bold', color=TEXT, pad=24)
    ax.text(0, 1.02, f'Best ΔIoU +{best["delta_iou"]:.3f} at epoch {best["epoch"]}',
            transform=ax.transAxes, color=TEXT_2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean IoU with ground truth')
    ax.set_xlim(0, epochs[-1] * 1.07)
    ax.legend(frameon=False, loc='center right')
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f'已儲存：{args.out}')


if __name__ == '__main__':
    main()
