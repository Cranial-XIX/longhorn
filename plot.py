import matplotlib.pyplot as plt
import torch
import numpy as np


plt.figure(figsize=(5, 3))
min_iter = np.inf
dataset = "slimpajama"
#dataset = "openwebtext"


for i, model in enumerate([
    #"openwebtext_2.5_block1024_online_implicit2_seed1337.stats",
    #"openwebtext_2.5_block1024_online_implicit3_seed1337.stats",
    #"openwebtext_2.5_block1024_llama_seed1337.stats",
    "openwebtext_2.5_block1024_mamba_seed1337.stats",  # mamba 1
    "openwebtext_2.5_block1024_gla_seed1337.stats",  # mamba 1
    "openwebtext_2.5_block1024_gla_online_seed1337.stats",  # mamba 1
    "openwebtext_2.5_block1024_gla_online2_seed1337.stats",  # mamba 1
    ##"openwebtext_2.5_block1024_mamba2_seed1337.stats",  # mamba 2

    #"openwebtext_2.5_block1024_online_k2_seed1337.stats",
    #"openwebtext_2.5_block1024_online_k2_norm_seed1337.stats",
    "openwebtext_2.5_block1024_online_k2_seed1337.stats",
    #"openwebtext_2.5_block1024_online_k2_actv_seed1337.stats",
    "openwebtext_2.5_block1024_online_k2_fast_seed1337.stats",
    "openwebtext_2.5_block1024_online_k2_fast_nonorm_seed1337.stats",
    "openwebtext_2.5_block1024_online_k2_no_norm_fast_seed1337.stats",
    "openwebtext_2.5_block1024_online_k2_no_norm_fast_B_seed1337.stats",
    #"openwebtext_2.5_block1024_online_a2k_seed1337.stats",
    #"openwebtext_2.5_block1024_online_a2k_norm_seed1337.stats",
    #"openwebtext_2.5_block1024_online_a2k_norma_seed1337.stats",

    #"openwebtext_2.5_block1024_mamba_64_seed1337.stats",
    #"openwebtext_2.5_block1024_online_a2k_64_seed1337.stats",
    #"openwebtext_2.5_block1024_online_a2k_norm_64_seed1337.stats",
    #"openwebtext_2.5_block1024_online_k2_norm_64_seed1337.stats",
    #"openwebtext_2.5_block1024_online_k2_64_seed1337.stats",
    #"openwebtext_2.5_block1024_online_a2k_norma_64_seed1337.stats",

    #"openwebtext_2.5_block1024_mamba2_naturalgd_seed1337.stats",  # mamba 2 + natural GD
    #"openwebtext_2.5_block1024_mamba_naturalgd_seed1337.stats",  # mamba + natural GD

    #"openwebtext_2.5_block1024_online_natural_gd_seed1337.stats",
    #"openwebtext_2.5_block1024_online_natural_gd_baseline_seed1337.stats",

    #"openwebtext_2.5_block1024_online_final_seed1337.stats",  # Widrow-Hoff + K
    #"openwebtext_2.5_block1024_online_final2_seed1337.stats",  # Widrow-Hoff

    #"openwebtext_2.5_block1024_gla_natural_gd_seed1337.stats",  # Widrow-Hoff
    #"openwebtext_2.5_block1024_gla_natural_gd_baseline_seed1337.stats",  # Widrow-Hoff

    #"openwebtext_2.5_block1024_gla_unified_seed1337.stats",  # Widrow-Hoff
    #"openwebtext_2.5_block1024_online_exact_seed1337.stats",  # Widrow-Hoff
    #"openwebtext_2.5_block1024_online_exact_lowrank_seed1337.stats",  # Widrow-Hoff
    #"openwebtext_2.5_block1024_online_exact_lowrank2_seed1337.stats",  # Widrow-Hoff
    #"openwebtext_2.5_block1024_online_exact_lowrank_baseline_seed1337.stats",  # Widrow-Hoff
    #"openwebtext_2.5_block1024_online_exact_lowrank_baseline2_seed1337.stats",  # Widrow-Hoff

    #"openwebtext_2.5_block1024_online_exact_mul_seed1337.stats",  # Widrow-Hoff
    #"openwebtext_2.5_block1024_online_exact_mul_noq_seed1337.stats",  # Widrow-Hoff
    #"openwebtext_2.5_block1024_gla_natural_gd2_seed1337.stats",  # Widrow-Hoff
    #"openwebtext_2.5_block1024_online_exact_mul_noq_accurate_seed1337.stats",  # Widrow-Hoff
    #"openwebtext_2.5_block1024_online_exact_mul_noq2_seed1337.stats",  # Widrow-Hoff
    #"openwebtext_2.5_block1024_online_exact_mul_noq3_seed1337.stats",  # Widrow-Hoff
    #"openwebtext_2.5_block1024_online_exact_mul_noqk_seed1337.stats",  # Widrow-Hoff

]):
    filepath = f"./results/{model}"
    filepath = filepath.replace("openwebtext", dataset)
    #filepath = filepath.replace("2.5", "6.9")
    try:
        stats = torch.load(filepath)
        params = stats["model/params"]
        #iters = np.array(stats["iter"])
        iters = np.array(stats["iter"]) * 480 * 1024 / 1e9
        min_iter = min(iters[-1], min_iter)
        val_loss = np.array(stats["val/loss"])
        best_loss = np.min(val_loss)
        model_name = "_".join(model.split("_")[3:])
        plt.plot(
            iters,
            val_loss,
            label=f"{model_name} ({params/1e6:.1f}M - best {best_loss:.4f})",
            color=f"C{i}",
            linewidth=0.5,
            linestyle='-' if 'fw' not in model else '-',
        )
    except:
        print(model)
        continue

axs = plt.gca()
for side in ["right", "top"]:
    axs.spines[side].set_visible(False)

plt.title(f"Val Loss")
plt.grid(linestyle='--', color='gray')
plt.xlabel("# Iterations")
plt.ylabel("Val Loss")
plt.legend(loc='upper right', fontsize=6)
plt.tight_layout()
plt.savefig(f"./new_ssm.pdf")
plt.close()

