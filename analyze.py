import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    classification_report,
    confusion_matrix,
)

pred = np.load("data/predictions.npy")
data = np.load("data/train_test.npz")
y_test = data["y_test"]

direction_actual = np.sign(y_test[:, 0])
direction_pred = np.sign(pred[:, 0])

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax1 = axes[0, 0]
ax1.plot(y_test[:50, 0], label="Actual", alpha=0.8, linewidth=1.5)
ax1.plot(pred[:50, 0], label="Predicted", alpha=0.8, linewidth=1.5)
ax1.set_title("First 50 samples - Next tick delta prediction")
ax1.set_xlabel("Sample")
ax1.set_ylabel("Price delta")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.scatter(y_test[:, 0], pred[:, 0], alpha=0.5, s=20)
ax2.plot(
    [y_test[:, 0].min(), y_test[:, 0].max()],
    [y_test[:, 0].min(), y_test[:, 0].max()],
    "r--",
    label="Perfect prediction",
)
ax2.set_title("Actual vs Predicted")
ax2.set_xlabel("Actual delta")
ax2.set_ylabel("Predicted delta")
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
errors = (y_test - pred).flatten()
ax3.hist(errors, bins=50, alpha=0.7, edgecolor="black")
ax3.axvline(x=0, color="r", linestyle="--", label="Zero")
ax3.axvline(
    x=np.mean(errors), color="g", linestyle="--", label=f"Mean: {np.mean(errors):.6f}"
)
ax3.set_title("Error distribution")
ax3.set_xlabel("Error (Actual - Predicted)")
ax3.set_ylabel("Frequency")
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
cm = confusion_matrix(direction_actual, direction_pred, labels=[-1, 0, 1])
im = ax4.imshow(cm, cmap="Blues")
ax4.set_xticks([0, 1, 2])
ax4.set_yticks([0, 1, 2])
ax4.set_xticklabels(["Down", "Neutral", "Up"])
ax4.set_yticklabels(["Down", "Neutral", "Up"])
ax4.set_xlabel("Predicted")
ax4.set_ylabel("Actual")
ax4.set_title("Direction Confusion Matrix")
for i in range(3):
    for j in range(3):
        ax4.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)
plt.colorbar(im, ax=ax4)

plt.tight_layout()
plt.savefig("data/prediction_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved prediction_analysis.png")

mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

mape = np.mean(np.abs(y_test) > 1e-6) * np.mean(np.abs((y_test - pred) / y_test) * 100)

print("\n" + "=" * 50)
print("OVERALL QUALITY METRICS")
print("=" * 50)
print(f"MSE:                {mse:.8f}")
print(f"RMSE:               {rmse:.8f}")
print(f"MAE:                {mae:.8f}")
print(f"R² Score:           {r2:.4f}")
print(f"Direction Accuracy: {(direction_actual == direction_pred).mean() * 100:.1f}%")

print("\n" + "=" * 50)
print("PER-TIMESTEP METRICS")
print("=" * 50)
n_steps = y_test.shape[1]
print(f"{'Step':<6} {'MSE':<12} {'MAE':<12} {'RMSE':<12}")
for t in range(n_steps):
    mse_t = mean_squared_error(y_test[:, t], pred[:, t])
    mae_t = mean_absolute_error(y_test[:, t], pred[:, t])
    rmse_t = np.sqrt(mse_t)
    print(f"{t + 1:<6} {mse_t:<12.8f} {mae_t:<12.6f} {rmse_t:<12.6f}")

print("\n" + "=" * 50)
print("DIRECTION ANALYSIS")
print("=" * 50)
unique_actual = sorted(set(direction_actual.tolist()))
unique_pred = sorted(set(direction_pred.tolist()))
labels = list(set(unique_actual + unique_pred))
label_names = {-1: "Down", 0: "Neutral", 1: "Up"}
target_names = [label_names.get(l, str(l)) for l in labels]
print(f"Actual classes: {unique_actual}, Predicted classes: {unique_pred}")
if len(labels) > 1:
    print(
        classification_report(
            direction_actual, direction_pred, labels=labels, target_names=target_names
        )
    )
else:
    print(f"All predictions are {target_names[0]}")

results = {
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
    "direction_accuracy": float((direction_actual == direction_pred).mean() * 100),
}
pd.DataFrame([results]).to_csv("data/metrics_summary.csv", index=False)
print("\nMetrics saved to data/metrics_summary.csv")
