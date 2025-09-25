import matplotlib.pyplot as plt
import numpy as np

# 各语言模型的 AUC 数据（含 all）
data = {
    "ch": {
        "MPU": [0.6934, 0.6508, 0.6249, 0.6671, 0.5098, 0.7283, 0.8142, 0.6051],
        "OpenAI": [0.5546, 0.6046, 0.6088, 0.6216, 0.5431, 0.6044, 0.7317, 0.5913],
        "RADAR": [0.7832, 0.6291, 0.6257, 0.4836, 0.5792, 0.5763, 0.8058, 0.5720],
        "ArguGPT": [0.5064, 0.6945, 0.6394, 0.7156, 0.7714, 0.6745, 0.8044, 0.6410]
    },
    "de": {
        "MPU": [0.7354, 0.5980, 0.6109, 0.6490, 0.4868, 0.6785, 0.7558, 0.5805],
        "OpenAI": [0.5771, 0.6206, 0.5930, 0.6582, 0.5721, 0.6209, 0.7616, 0.6075],
        "RADAR": [0.8438, 0.6372, 0.5975, 0.4562, 0.5986, 0.5507, 0.7924, 0.5711],
        "ArguGPT": [0.5124, 0.6349, 0.5795, 0.6553, 0.7260, 0.6400, 0.7604, 0.6023]
    },
    "fr": {
        "MPU": [0.7146, 0.5823, 0.5925, 0.6285, 0.4523, 0.6946, 0.7246, 0.5653],
        "OpenAI": [0.5246, 0.5907, 0.5719, 0.6311, 0.5685, 0.6081, 0.7400, 0.5863],
        "RADAR": [0.8370, 0.6278, 0.5950, 0.4685, 0.5959, 0.5545, 0.7280, 0.5699],
        "ArguGPT": [0.4945, 0.6264, 0.5748, 0.6564, 0.7051, 0.6368, 0.7179, 0.5925]
    }
}

labels = ['xsum', 'pubmedqa', 'squad', 'writingprompts', 'openreview', 'blog', 'tweets', 'all']
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# 设置画布
fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
languages = ['ch', 'de', 'fr']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

for i, lang in enumerate(languages):
    ax = axs[i]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0.4, 0.9)
    ax.set_title(f'{lang.upper()} - AUC Radar', fontsize=14)

    for j, (model_name, values) in enumerate(data[lang].items()):
        values = values + values[:1]
        ax.plot(angles, values, label=model_name, color=colors[j])
        ax.fill(angles, values, alpha=0.1, color=colors[j])

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=10)

plt.suptitle("AUC Radar Charts by Language (Translate, Training Scale: 500)", fontsize=16)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig("./detectors/opensource/auc_radar_translate_languages.png", dpi=300)
plt.show()
