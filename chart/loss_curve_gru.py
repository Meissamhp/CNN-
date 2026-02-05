import matplotlib.pyplot as plt

epochs = list(range(0, 200, 20))
loss = [4.9603, 1.3833, 1.0543, 1.0263, 1.0016, 0.9904, 0.9864, 0.9840, 0.9816, 0.9793]

plt.figure(figsize=(7, 5))
plt.plot(epochs, loss, marker='o', color='green')
plt.title("Training Loss Curve - GRU Model", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("loss_curve_gru.png", dpi=300)
plt.show()
