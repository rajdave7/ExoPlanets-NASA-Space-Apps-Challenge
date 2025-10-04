"""
Kepler KOI Binary Classification Pipeline (Planet vs Not Planet)
Complete ML pipeline with SMOTE, preprocessing, and model comparison
"""

import pandas as pd
import numpy as np
import os
import time
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ==== EDIT THESE PATHS ====
KOI_CSV = "data/kepler_koi.csv"

# Output directories
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("KOI_CSV =", KOI_CSV)
print("Output folders: models/, plots/, results/")
print("=" * 80)

# ============================================================================
# 1. LOAD AND INITIAL EXPLORATION
# ============================================================================
print("\n[1] LOADING DATA")
print("-" * 80)

if not os.path.exists(KOI_CSV):
    raise FileNotFoundError(f"KOI CSV not found at {KOI_CSV}")

df = pd.read_csv(KOI_CSV)
print(f"Loaded KOI dataset: {df.shape}")
print(f"Columns (first 50): {df.columns.tolist()[:50]}")
print(f"\nFirst 3 rows:\n{df.head(3)}")

# ============================================================================
# 2. DATA PREPROCESSING - BINARY CLASSIFICATION
# ============================================================================
print("\n[2] DATA PREPROCESSING - BINARY CLASSIFICATION")
print("-" * 80)

# Target variable - BINARY CLASSIFICATION
target_col = "koi_disposition"
print(f"Target column: {target_col}")
print(f"Original class distribution:\n{df[target_col].value_counts()}")

# *** BINARY CLASSIFICATION: Combine CANDIDATE and CONFIRMED as PLANET (1), FALSE POSITIVE as NOT PLANET (0) ***
print(
    "\n*** CREATING BINARY LABELS: Planet (CANDIDATE + CONFIRMED) vs Not Planet (FALSE POSITIVE) ***"
)
df["binary_label"] = df[target_col].apply(
    lambda x: 1 if x in ["CANDIDATE", "CONFIRMED"] else 0
)
print(f"\nBinary class distribution:")
print(f"Planet (1): {(df['binary_label'] == 1).sum()}")
print(f"Not Planet (0): {(df['binary_label'] == 0).sum()}")
print(
    f"Class ratio: {(df['binary_label'] == 0).sum() / (df['binary_label'] == 1).sum():.2f}:1 (Not Planet:Planet)"
)

# Separate features and target
X = df.drop(
    columns=[target_col, "binary_label", "kepoi_name", "kepler_name"], errors="ignore"
)
y = df["binary_label"].values

print(f"\nFinal class distribution:")
print(f"Planet: {np.sum(y == 1)}")
print(f"Not Planet: {np.sum(y == 0)}")

# Select only numeric columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X_numeric = X[numeric_cols]
print(f"\nSelected {len(numeric_cols)} numeric features")

# Handle missing values
print(f"Missing values before imputation: {X_numeric.isnull().sum().sum()}")
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X_numeric)
X_imputed = pd.DataFrame(X_imputed, columns=numeric_cols)
print(f"Missing values after imputation: {X_imputed.isnull().sum().sum()}")

# Remove features with zero variance
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.0)
X_variance = selector.fit_transform(X_imputed)
selected_features = X_imputed.columns[selector.get_support()].tolist()
print(f"Features after variance threshold: {len(selected_features)}")

# Train-test split BEFORE SMOTE (important!)
X_train, X_test, y_train, y_test = train_test_split(
    X_variance, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")
print(
    f"Train class distribution - Planet: {np.sum(y_train == 1)}, Not Planet: {np.sum(y_train == 0)}"
)
print(
    f"Test class distribution - Planet: {np.sum(y_test == 1)}, Not Planet: {np.sum(y_test == 0)}"
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling completed")

# ============================================================================
# 3. APPLY SMOTE - CLASS BALANCING (IMBALANCE FIXING)
# ============================================================================
print("\n" + "=" * 80)
print("[3] *** APPLYING SMOTE FOR CLASS BALANCING (IMBALANCE FIXING) ***")
print("=" * 80)
print("BEFORE SMOTE:")
print(f"  Planet samples: {np.sum(y_train == 1)}")
print(f"  Not Planet samples: {np.sum(y_train == 0)}")
print(f"  Imbalance ratio: {np.sum(y_train == 0) / np.sum(y_train == 1):.2f}:1")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"\nAFTER SMOTE:")
print(f"  Planet samples: {np.sum(y_train_smote == 1)}")
print(f"  Not Planet samples: {np.sum(y_train_smote == 0)}")
print(
    f"  Imbalance ratio: {np.sum(y_train_smote == 0) / np.sum(y_train_smote == 1):.2f}:1"
)
print(f"  Total training samples increased from {len(y_train)} to {len(y_train_smote)}")
print("=" * 80)

# ============================================================================
# 4. MODEL DEFINITION (9 MODELS)
# ============================================================================
print("\n[4] DEFINING MODELS")
print("-" * 80)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, eval_metric="logloss"
    ),
    "CatBoost": CatBoostClassifier(iterations=100, random_state=42, verbose=0),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=100, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(
        kernel="rbf", probability=True, random_state=42, class_weight="balanced"
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Naive Bayes": GaussianNB(),
}

print(f"Total models to compare: {len(models)}")
print(f"Models: {list(models.keys())}")

# ============================================================================
# 5. CROSS-VALIDATION AND TRAINING
# ============================================================================
print("\n[5] CROSS-VALIDATION AND TRAINING")
print("-" * 80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, model in models.items():
    print(f'\n{"="*80}')
    print(f"TRAINING: {name}")
    print(f'{"="*80}')

    # Create model-specific subfolder
    model_folder = f'plots/{name.replace(" ", "_").lower()}'
    os.makedirs(model_folder, exist_ok=True)

    # Cross-validation
    print("Running 5-fold cross-validation...")
    cv_start = time.time()
    cv_results = cross_validate(
        model,
        X_train_smote,
        y_train_smote,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
        n_jobs=-1,
        return_train_score=True,
    )
    cv_time = time.time() - cv_start

    print(
        f'CV Accuracy: {cv_results["test_accuracy"].mean():.4f} (+/- {cv_results["test_accuracy"].std():.4f})'
    )
    print(f'CV Precision: {cv_results["test_precision"].mean():.4f}')
    print(f'CV Recall: {cv_results["test_recall"].mean():.4f}')
    print(f'CV F1-Score: {cv_results["test_f1"].mean():.4f}')
    print(f'CV ROC-AUC: {cv_results["test_roc_auc"].mean():.4f}')
    print(f"CV Time: {cv_time:.2f}s")

    # Train on full training set
    print("Training on full training set...")
    train_start = time.time()
    model.fit(X_train_smote, y_train_smote)
    train_time = time.time() - train_start

    # Test predictions
    print("Making predictions on test set...")
    test_start = time.time()
    y_pred = model.predict(X_test_scaled)
    test_time = time.time() - test_start

    # Predict probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        y_pred_proba = None
        roc_auc = None

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Inference time per sample
    inference_time_per_sample = test_time / len(X_test_scaled) * 1000  # in ms

    # Approximate FLOPS calculation
    if name == "Logistic Regression":
        flops = X_train_smote.shape[1] * 2 * len(X_test_scaled)
    elif name in [
        "Random Forest",
        "XGBoost",
        "HistGradientBoosting",
        "Gradient Boosting",
    ]:
        avg_depth = 10
        flops = avg_depth * X_train_smote.shape[1] * 100 * len(X_test_scaled)
    elif name == "CatBoost":
        avg_depth = 6
        flops = avg_depth * X_train_smote.shape[1] * 100 * len(X_test_scaled)
    elif name == "Support Vector Machine":
        flops = len(X_train_smote) * X_train_smote.shape[1] * len(X_test_scaled)
    elif name == "K-Nearest Neighbors":
        flops = len(X_train_smote) * X_train_smote.shape[1] * len(X_test_scaled) * 5
    elif name == "Decision Tree":
        avg_depth = 15
        flops = avg_depth * X_train_smote.shape[1] * len(X_test_scaled)
    elif name == "AdaBoost":
        avg_depth = 1
        flops = avg_depth * X_train_smote.shape[1] * 100 * len(X_test_scaled)
    elif name == "Naive Bayes":
        flops = X_train_smote.shape[1] * len(X_test_scaled)
    else:
        flops = 0

    print(f"\n--- Test Set Performance ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if roc_auc:
        print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"\n--- Timing ---")
    print(f"Training Time: {train_time:.2f}s")
    print(f"Test Time: {test_time:.4f}s")
    print(f"Inference Time per Sample: {inference_time_per_sample:.4f}ms")
    print(f"Approximate FLOPS: {flops:,.0f}")

    # Store results
    results.append(
        {
            "Model": name,
            "CV_Accuracy": cv_results["test_accuracy"].mean(),
            "CV_Precision": cv_results["test_precision"].mean(),
            "CV_Recall": cv_results["test_recall"].mean(),
            "CV_F1": cv_results["test_f1"].mean(),
            "CV_ROC_AUC": cv_results["test_roc_auc"].mean(),
            "Test_Accuracy": accuracy,
            "Test_Precision": precision,
            "Test_Recall": recall,
            "Test_F1": f1,
            "Test_ROC_AUC": roc_auc if roc_auc else 0,
            "Train_Time_s": train_time,
            "Test_Time_s": test_time,
            "Inference_ms_per_sample": inference_time_per_sample,
            "Approx_FLOPS": flops,
        }
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Planet", "Planet"],
        yticklabels=["Not Planet", "Planet"],
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"{model_folder}/confusion_matrix.png", dpi=150)
    plt.close()

    # ROC Curve
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})", linewidth=2)
        plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{model_folder}/roc_curve.png", dpi=150)
        plt.close()

    # Classification report
    report = classification_report(
        y_test, y_pred, target_names=["Not Planet", "Planet"]
    )
    print(f"\n--- Classification Report ---\n{report}")

    # Save classification report to file
    with open(f"{model_folder}/classification_report.txt", "w") as f:
        f.write(f"Classification Report - {name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)
        f.write(f"\n\nAccuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        if roc_auc:
            f.write(f"ROC-AUC: {roc_auc:.4f}\n")

# ============================================================================
# 6. RESULTS COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("[6] RESULTS COMPARISON")
print("=" * 80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Test_Accuracy", ascending=False)
print("\nComplete Results (sorted by Test Accuracy):")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv("results/binary_classification_results.csv", index=False)
print("\n✓ Results saved to results/binary_classification_results.csv")

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================
print("\n[7] CREATING COMPREHENSIVE VISUALIZATIONS")
print("-" * 80)

# Create comprehensive comparison plot
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

# 1. Test Accuracy
axes[0, 0].barh(results_df["Model"], results_df["Test_Accuracy"], color="skyblue")
axes[0, 0].set_xlabel("Accuracy", fontsize=11)
axes[0, 0].set_title("Test Accuracy Comparison", fontsize=13, fontweight="bold")
axes[0, 0].set_xlim([0, 1])
for i, v in enumerate(results_df["Test_Accuracy"]):
    axes[0, 0].text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=9)
axes[0, 0].grid(axis="x", alpha=0.3)

# 2. Precision, Recall, F1
x = np.arange(len(results_df))
width = 0.25
axes[0, 1].bar(
    x - width, results_df["Test_Precision"], width, label="Precision", alpha=0.8
)
axes[0, 1].bar(x, results_df["Test_Recall"], width, label="Recall", alpha=0.8)
axes[0, 1].bar(x + width, results_df["Test_F1"], width, label="F1-Score", alpha=0.8)
axes[0, 1].set_xlabel("Model", fontsize=11)
axes[0, 1].set_ylabel("Score", fontsize=11)
axes[0, 1].set_title(
    "Precision, Recall, F1-Score Comparison", fontsize=13, fontweight="bold"
)
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(results_df["Model"], rotation=45, ha="right", fontsize=8)
axes[0, 1].legend()
axes[0, 1].set_ylim([0, 1])
axes[0, 1].grid(axis="y", alpha=0.3)

# 3. ROC-AUC Score
axes[1, 0].barh(results_df["Model"], results_df["Test_ROC_AUC"], color="lightcoral")
axes[1, 0].set_xlabel("ROC-AUC Score", fontsize=11)
axes[1, 0].set_title("ROC-AUC Score Comparison", fontsize=13, fontweight="bold")
axes[1, 0].set_xlim([0, 1])
for i, v in enumerate(results_df["Test_ROC_AUC"]):
    axes[1, 0].text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=9)
axes[1, 0].grid(axis="x", alpha=0.3)

# 4. Training and Test Time
x_pos = np.arange(len(results_df))
axes[1, 1].bar(
    x_pos - width / 2, results_df["Train_Time_s"], width, label="Train Time", alpha=0.8
)
axes[1, 1].bar(
    x_pos + width / 2, results_df["Test_Time_s"], width, label="Test Time", alpha=0.8
)
axes[1, 1].set_xlabel("Model", fontsize=11)
axes[1, 1].set_ylabel("Time (seconds)", fontsize=11)
axes[1, 1].set_title(
    "Training and Test Time Comparison", fontsize=13, fontweight="bold"
)
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(results_df["Model"], rotation=45, ha="right", fontsize=8)
axes[1, 1].legend()
axes[1, 1].grid(axis="y", alpha=0.3)

# 5. Inference Time per Sample
axes[2, 0].barh(
    results_df["Model"], results_df["Inference_ms_per_sample"], color="coral"
)
axes[2, 0].set_xlabel("Time (milliseconds)", fontsize=11)
axes[2, 0].set_title("Inference Time per Sample", fontsize=13, fontweight="bold")
for i, v in enumerate(results_df["Inference_ms_per_sample"]):
    axes[2, 0].text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)
axes[2, 0].grid(axis="x", alpha=0.3)

# 6. FLOPS Comparison
axes[2, 1].barh(results_df["Model"], results_df["Approx_FLOPS"], color="lightgreen")
axes[2, 1].set_xlabel("Approximate FLOPS", fontsize=11)
axes[2, 1].set_title(
    "Computational Complexity (Approximate FLOPS)", fontsize=13, fontweight="bold"
)
for i, v in enumerate(results_df["Approx_FLOPS"]):
    axes[2, 1].text(v + v * 0.02, i, f"{v:,.0f}", va="center", fontsize=8)
axes[2, 1].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("plots/comprehensive_model_comparison.png", dpi=150, bbox_inches="tight")
print("✓ Saved plots/comprehensive_model_comparison.png")
plt.close()

# Additional: Top 5 models by accuracy
top5 = results_df.head(5)
fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(top5))
bars = ax.bar(
    x_pos,
    top5["Test_Accuracy"],
    color=["gold", "silver", "#CD7F32", "lightblue", "lightgreen"],
)
ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Test Accuracy", fontsize=12)
ax.set_title("Top 5 Models by Test Accuracy", fontsize=14, fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels(top5["Model"], rotation=45, ha="right")
ax.set_ylim([0, 1])
ax.grid(axis="y", alpha=0.3)

for i, (idx, row) in enumerate(top5.iterrows()):
    ax.text(
        i,
        row["Test_Accuracy"] + 0.02,
        f"{row['Test_Accuracy']:.4f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig("plots/top5_models.png", dpi=150, bbox_inches="tight")
print("✓ Saved plots/top5_models.png")
plt.close()

# ============================================================================
# 8. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("[8] SUMMARY - BINARY CLASSIFICATION (Planet vs Not Planet)")
print("=" * 80)
print(f'\n🏆 Best Model (by Test Accuracy): {results_df.iloc[0]["Model"]}')
print(f'   Test Accuracy: {results_df.iloc[0]["Test_Accuracy"]:.4f}')
print(f'   Test Precision: {results_df.iloc[0]["Test_Precision"]:.4f}')
print(f'   Test Recall: {results_df.iloc[0]["Test_Recall"]:.4f}')
print(f'   Test F1-Score: {results_df.iloc[0]["Test_F1"]:.4f}')
print(f'   Test ROC-AUC: {results_df.iloc[0]["Test_ROC_AUC"]:.4f}')
print(f"\n📊 Class Distribution:")
print(
    f"   Training (after SMOTE): Planet={np.sum(y_train_smote == 1)}, Not Planet={np.sum(y_train_smote == 0)}"
)
print(f"   Test: Planet={np.sum(y_test == 1)}, Not Planet={np.sum(y_test == 0)}")
print(f"\n📁 All outputs saved to:")
print(f"   - results/binary_classification_results.csv")
print(f"   - plots/comprehensive_model_comparison.png")
print(f"   - plots/top5_models.png")
print(
    f"   - plots/<model_name>/ (individual model folders with confusion matrix, ROC curve, and report)"
)
print("\n" + "=" * 80)
print("✅ BINARY CLASSIFICATION PIPELINE COMPLETE!")
print("=" * 80)
