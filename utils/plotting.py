import matplotlib.pyplot as plt

def plot_rewards(steps, rewards, out_path):
    plt.figure(figsize=(6,4))
    plt.plot(steps, rewards, linewidth=1.5)
    plt.xlabel('Environment Steps')
    plt.ylabel('Episode Return')
    plt.title('Training Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
