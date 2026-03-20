#!/usr/bin/env python3
"""
Generate ablation study bar chart (Figure for Appendix).

"""

import matplotlib.pyplot as plt
import numpy as np

methods = ['GRPO\n(Base)', 'GRPO+DR', 'TGPR\n(Full)']

x = np.arange(len(methods))
width = 0.32

fig, ax = plt.subplots(figsize=(8, 5.5))

bars1 = ax.bar(x - width/2, mbpp, width, color='#7B68EE',
               label='MBPP', zorder=3, edgecolor='white', linewidth=0.8)
bars2 = ax.bar(x + width/2, apps, width, color='#2ECC71',
               label='APPS', zorder=3, edgecolor='white', linewidth=0.8)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}', ha='center', va='bottom',
            fontsize=10, color='#7B68EE', fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}', ha='center', va='bottom',
            fontsize=10, color='#2ECC71', fontweight='bold')

ax.text(0.5, 76, '+1.6', ha='center', fontsize=9, color='#7B68EE', style='italic')
ax.text(1.5, 76, '+2.6', ha='center', fontsize=9, color='#7B68EE', style='italic')
ax.text(0.5, 56, '+1.4', ha='center', fontsize=9, color='#2ECC71', style='italic')
ax.text(1.5, 56, '+2.3', ha='center', fontsize=9, color='#2ECC71', style='italic')

ax.set_ylabel('Pass@1 (%)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.set_ylim(54, 85)
ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
ax.grid(True, axis='y', alpha=0.2, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig('ablation_study.png', dpi=300, bbox_inches='tight')
plt.show()

print("Saved: ablation_study.png")
