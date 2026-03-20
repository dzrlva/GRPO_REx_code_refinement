import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

COLORS = {
    'Pretrained': '#E91E8C',
    'PPO':        '#C084FC',
    'GRPO':       '#60A5FA',
    'TGPR':       '#1E3A8A',
}
MODEL_LABELS = {
    'Pretrained': 'Qwen2.5-7B-Pretrained',
    'PPO':        'Qwen2.5-7B-PPO',
    'GRPO':       'Qwen2.5-7B-GRPO',
    'TGPR':       'Qwen2.5-7B-TGPR (Ours)',
}

# Stagger labels: Pretrained/PPO go inward, GRPO/TGPR go outward
MODEL_OFFSETS = {
    'Pretrained': -0.16,
    'PPO':        -0.07,
    'GRPO':        0.07,
    'TGPR':        0.16,
}

categories = ['LiveCodeBench', 'HumanEval', 'MBPP', 'APPS', 'Codeforces']
N = len(categories)

data_p1 = {
    'Pretrained': [32.5, 68.0, 59.0, 41.0, 48.2],
    'PPO':        [48.2, 81.2, 74.5, 56.5, 62.5],
    'GRPO':       [51.2, 83.3, 77.9, 58.7, 65.3],
    'TGPR':       [55.8, 87.1, 82.1, 62.4, 65.8],
}

data_p10 = {
    'Pretrained': [44.2, 84.7, 75.1, 47.8, 58.3],
    'PPO':        [63.5, 94.5, 82.2, 60.2, 72.0],
    'GRPO':       [65.1, 95.2, 89.4, 61.5, 74.8],
    'TGPR':       [74.3, 98.5, 94.2, 72.7, 75.2],
}


def make_radar(ax, data_dict, r_min=0, r_max=100):
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ring_vals = np.linspace(r_min, r_max, 5)

    # Grid rings
    for rv in ring_vals:
        ring = [(rv - r_min) / (r_max - r_min)] * (N + 1)
        ax.plot(angles, ring, color='#CCCCCC', linewidth=0.9,
                linestyle='--', zorder=1)

    # Spokes
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 1.0], color='#CCCCCC',
                linewidth=0.9, zorder=1)

    # Model polygons
    for model, vals in data_dict.items():
        norm_vals = [(v - r_min) / (r_max - r_min) for v in vals]
        norm_vals += norm_vals[:1]
        ax.plot(angles, norm_vals,
                color=COLORS[model], linewidth=2.8,
                marker='o', markersize=8, zorder=3,
                label=MODEL_LABELS[model])
        ax.fill(angles, norm_vals, color=COLORS[model], alpha=0.08, zorder=2)

    # Category axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=21, fontweight='bold',
                       color='#222222')
    ax.tick_params(axis='x', pad=30)
    ax.set_yticks([])
    ax.set_yticklabels([])



    # Data value labels — staggered per model so they don't overlap
    for model, vals in data_dict.items():
        norm_vals = [(v - r_min) / (r_max - r_min) for v in vals]
        offset = MODEL_OFFSETS[model]
        for angle, norm, raw in zip(angles[:-1], norm_vals, vals):
            ax.text(angle, norm + offset, f'{raw}',
                    ha='center', va='center',
                    fontsize=16, color=COLORS[model],
                    fontweight='bold',
                    transform=ax.transData,
                    clip_on=False,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white',
                              ec='none', alpha=0.7))

    ax.set_ylim(0, 1.2)
    ax.set_xlim(0, 2 * np.pi)
    ax.spines['polar'].set_visible(False)
    ax.set_facecolor('#FAFAFA')


for metric, data_dict in [('p@1', data_p1), ('p@10', data_p10)]:
    safe_name = metric.replace('@', '')
    pdf_path = f'/mnt/user-data/outputs/qwen_capability_{safe_name}.pdf'

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(13, 12))
        fig.patch.set_facecolor('white')

        ax = fig.add_subplot(111, polar=True)
        ax.set_position([0.1, 0.12, 0.8, 0.8])
        make_radar(ax, data_dict)

        handles = [
            plt.Line2D([0], [0], color=COLORS[m], linewidth=2.5,
                       marker='o', markersize=11, label=MODEL_LABELS[m])
            for m in ['Pretrained', 'PPO', 'GRPO', 'TGPR']
        ]
        ax.legend(handles=handles,
                  loc='lower center',
                  bbox_to_anchor=(0.5, -0.22),
                  ncol=2,
                  fontsize=20,
                  frameon=True,
                  fancybox=True,
                  edgecolor='#DDDDDD',
                  labelspacing=0.8,
                  handlelength=2.5)

        fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.14)
        pdf.savefig(fig, bbox_inches='tight', dpi=150)
        plt.close(fig)

    print("Saved:", pdf_path)
