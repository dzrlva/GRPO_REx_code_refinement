import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

COLORS = {
    'GPT35':     '#E91E8C',
    'DeepSeek':  '#C084FC',
    'GPT4o':     '#60A5FA',
    'TGPR':      '#1E3A8A',
}
MODEL_LABELS = {
    'GPT35':     'GPT-3.5-Turbo',
    'DeepSeek':  'DeepSeek-Coder-V2',
    'GPT4o':     'GPT-4o',
    'TGPR':      'Qwen2.5-7B-TGPR (Ours)',
}

categories = ['Syntax', 'Semantic', 'Algorithmic', 'Runtime/TLE', 'Interpretation', 'Hallucination']
N = len(categories)

data = {
    'GPT35':    [14.5, 20.2, 24.8, 13.5, 18.0, 19.5],
    'DeepSeek': [11.5, 16.5, 19.1, 12.1, 14.2, 15.5],
    'GPT4o':    [10.1, 14.1, 18.2, 11.2,  9.2,  7.8],
    'TGPR':     [ 8.2, 12.8, 10.5,  8.4,  8.5,  5.4],
}

# Radial offset per model — push label outward from actual point
# Innermost model (TGPR) pushed inward, outermost (GPT35) pushed outward
RADIAL_OFFSETS = {
    'TGPR':     -0.10,
    'GPT4o':    -0.04,
    'DeepSeek':  0.04,
    'GPT35':     0.10,
}

def make_radar(ax, data_dict, r_min=0, r_max=27):
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
    for model in ['GPT35', 'DeepSeek', 'GPT4o', 'TGPR']:
        vals = data_dict[model]
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
    ax.tick_params(axis='x', pad=36)
    ax.set_yticks([])
    ax.set_yticklabels([])

    # Data value labels — offset in Cartesian space to avoid overlap
    # Convert polar label position to cartesian, then nudge tangentially
    for model, vals in data_dict.items():
        norm_vals = [(v - r_min) / (r_max - r_min) for v in vals]
        r_off = RADIAL_OFFSETS[model]
        # small tangential nudge per model so labels at same radius don't stack
        ang_nudge = {'TGPR': -0.07, 'GPT4o': -0.02, 'DeepSeek': 0.02, 'GPT35': 0.07}[model]
        for angle, norm, raw in zip(angles[:-1], norm_vals, vals):
            label_r = max(0, norm + r_off)
            label_a = angle + ang_nudge
            ax.text(label_a, label_r, f'{raw}',
                    ha='center', va='center',
                    fontsize=16, color=COLORS[model],
                    fontweight='bold',
                    clip_on=False,
                    zorder=5,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white',
                              ec='none', alpha=0.8))

    ax.set_ylim(0, 1.18)
    ax.set_xlim(0, 2 * np.pi)
    ax.spines['polar'].set_visible(False)
    ax.set_facecolor('#FAFAFA')


pdf_path = '/mnt/user-data/outputs/error_taxonomy.pdf'
with PdfPages(pdf_path) as pdf:
    fig = plt.figure(figsize=(15, 14))
    fig.patch.set_facecolor('white')

    ax = fig.add_subplot(111, polar=True)
    ax.set_position([0.05, 0.16, 0.90, 0.84])
    make_radar(ax, data)

    handles = [
        plt.Line2D([0], [0], color=COLORS[m], linewidth=2.5,
                   marker='o', markersize=11, label=MODEL_LABELS[m])
        for m in ['GPT35', 'DeepSeek', 'GPT4o', 'TGPR']
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

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.18)
    pdf.savefig(fig, bbox_inches='tight', dpi=150)
    plt.close(fig)

print("Saved:", pdf_path)
