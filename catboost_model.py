"""
CatBoost Exoplanet Detection Model - Binary Classification
Combines ALL available datasets for maximum performance
Classes: EXOPLANET (Candidate + Confirmed) vs FALSE POSITIVE
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import warnings
import os
warnings.filterwarnings('ignore')

class BinaryExoplanetDetector:
    """
    Binary CatBoost model combining multiple datasets.
    EXOPLANET (CONFIRMED + CANDIDATE) vs FALSE POSITIVE
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = RobustScaler()
        self.feature_names = None
        self.class_names = ['FALSE POSITIVE', 'EXOPLANET']
        
    def load_and_combine_datasets(self, data_dir='data'):
        """Load and combine all available Kepler datasets."""
        print("="*70)
        print("LOADING AND COMBINING ALL DATASETS")
        print("="*70)
        
        datasets = []
        
        # Try to load all datasets
        files_to_try = [
            'kepler_koi_cumulative.csv',
            'kepler_koi.csv',
            'k2pandc.csv'
        ]
        
        for filename in files_to_try:
            filepath = os.path.join(data_dir, filename)
            if not os.path.exists(filepath):
                print(f"\n⚠️  Skipping {filename} (not found)")
                continue
                
            try:
                print(f"\n📂 Loading {filename}...")
                df = pd.read_csv(filepath)
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {len(df.columns)}")
                
                # Identify target column
                target_col = None
                for col in ['koi_disposition', 'koi_pdisposition', 'exo_disposition']:
                    if col in df.columns:
                        target_col = col
                        break
                
                if target_col is None:
                    print(f"   ❌ No disposition column found, skipping")
                    continue
                
                print(f"   Target column: {target_col}")
                
                # Create binary labels: 1 = EXOPLANET, 0 = FALSE POSITIVE
                def classify_disposition(x):
                    x_upper = str(x).upper()
                    if 'CONFIRMED' in x_upper or 'CANDIDATE' in x_upper:
                        return 1  # EXOPLANET
                    else:
                        return 0  # FALSE POSITIVE
                
                df['target'] = df[target_col].apply(classify_disposition)
                df['source_dataset'] = filename
                
                # Show distribution
                dist = df['target'].value_counts().sort_index()
                print(f"   Distribution:")
                for class_val, count in dist.items():
                    print(f"     {self.class_names[class_val]}: {count}")
                
                datasets.append(df)
                print(f"   ✓ Successfully loaded!")
                
            except Exception as e:
                print(f"   ❌ Error loading {filename}: {e}")
                continue
        
        if not datasets:
            raise ValueError("No datasets could be loaded!")
        
        print(f"\n{'='*70}")
        print(f"SUCCESSFULLY LOADED {len(datasets)} DATASET(S)")
        print(f"{'='*70}")
        
        # Combine datasets
        combined_df = pd.concat(datasets, ignore_index=True, sort=False)
        
        print(f"\nCombined dataset shape: {combined_df.shape}")
        print(f"\nOverall distribution:")
        total_dist = combined_df['target'].value_counts().sort_index()
        for class_val, count in total_dist.items():
            pct = count / len(combined_df) * 100
            print(f"  {self.class_names[class_val]}: {count} ({pct:.2f}%)")
        
        # Identify common features across all datasets
        print(f"\n{'='*70}")
        print("FEATURE SELECTION")
        print(f"{'='*70}")
        
        # Priority features based on transit method
        priority_features = [
            'koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 
            'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_steff', 
            'koi_slogg', 'koi_srad', 'koi_impact', 'koi_time0bk',
            'koi_time0', 'koi_eccen', 'koi_longp', 'koi_trans_depth',
            'koi_ror', 'koi_dor', 'koi_incl', 'koi_limbdark_mod',
        ]
        
        # Find available features
        available_features = []
        for feat in priority_features:
            if feat in combined_df.columns:
                missing_pct = combined_df[feat].isnull().sum() / len(combined_df) * 100
                if missing_pct < 80:  # Use features with less than 80% missing
                    available_features.append(feat)
                    print(f"  ✓ {feat:<20} ({missing_pct:.1f}% missing)")
                else:
                    print(f"  ✗ {feat:<20} ({missing_pct:.1f}% missing - excluded)")
        
        if len(available_features) < 3:
            raise ValueError("Not enough features available!")
        
        print(f"\n{len(available_features)} features selected for training")
        
        # Extract features and targets
        X = combined_df[available_features].copy()
        y = combined_df['target'].copy()
        
        self.feature_names = available_features
        
        # Handle missing values
        print(f"\nHandling missing values...")
        X = X.fillna(X.median())
        
        # Remove rows with remaining NaN
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        print(f"Final dataset shape: {X.shape}")
        print(f"Final samples: {len(X)}")
        print(f"✓ Data preparation complete!\n")
        
        return X, y
    
    def train(self, X, y, test_size=0.3, use_smote=True, iterations=500):
        """Train binary exoplanet detector."""
        print("\n" + "="*70)
        print("TRAINING BINARY EXOPLANET DETECTOR")
        print("="*70)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"\nTrain set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Features: {X_train.shape[1]}")
        
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if use_smote:
            print("\nApplying SMOTE...")
            print(f"Before: FP={np.sum(y_train==0)}, Exo={np.sum(y_train==1)}")
            smote = SMOTE(random_state=self.random_state, k_neighbors=5)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            print(f"After: FP={np.sum(y_train==0)}, Exo={np.sum(y_train==1)}")
        
        class_counts = np.bincount(y_train)
        class_weights = {0: len(y_train)/(2*class_counts[0]), 
                        1: len(y_train)/(2*class_counts[1])}
        
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=self.random_state,
            verbose=100,
            early_stopping_rounds=50,
            class_weights=list(class_weights.values()),
            use_best_model=True,
            task_type='CPU',
            border_count=128,
        )
        
        print("\nTraining model...")
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=(X_test_scaled, y_test),
            plot=False
        )
        
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        print("\n✓ Training complete!")
        return self.model
    
    def predict(self, X):
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled).flatten()
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self):
        """Evaluate model on test set."""
        y_pred = self.model.predict(self.X_test).flatten()
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred)
        rec = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_proba)
        
        print("\n" + "="*70)
        print("MODEL EVALUATION RESULTS")
        print("="*70)
        print(f"\nAccuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc:.4f}")
        
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"{'':>20} {'Pred FP':<12} {'Pred Exo':<12}")
        print("-" * 46)
        print(f"{'True FALSE POSITIVE':>20} {cm[0,0]:<12} {cm[0,1]:<12}")
        print(f"{'True EXOPLANET':>20} {cm[1,0]:<12} {cm[1,1]:<12}")
        
        # Additional metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        print(f"\nAdditional Metrics:")
        print(f"Specificity: {specificity:.4f}")
        print(f"True Positives (Exoplanets found): {tp}")
        print(f"False Negatives (Exoplanets missed): {fn}")
        print(f"False Positives (Incorrect detections): {fp}")
        print(f"True Negatives (Correct rejections): {tn}")
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=self.class_names, digits=4))
        
        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': auc,
            'confusion_matrix': cm,
            'specificity': specificity
        }
    
    def plot_feature_importance(self, save_dir='results', top_n=15):
        """Plot feature importance."""
        os.makedirs(save_dir, exist_ok=True)
        
        importance = self.model.get_feature_importance()
        feat_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 6))
        plt.barh(feat_df['Feature'], feat_df['Importance'], color='steelblue')
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{save_dir}/feature_importance_binary.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_dir}/feature_importance_binary.png")
        plt.close()
    
    def plot_roc_curve(self, save_dir='results'):
        """Plot ROC curve."""
        os.makedirs(save_dir, exist_ok=True)
        
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        auc = roc_auc_score(self.y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Exoplanet Detection', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/roc_curve_binary.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_dir}/roc_curve_binary.png")
        plt.close()
    
    def save_model(self, filename='exoplanet_binary_detector.cbm'):
        """Save model."""
        self.model.save_model(filename)
        print(f"✓ Model saved: {filename}")
    
    def load_model(self, filename='exoplanet_binary_detector.cbm'):
        """Load model."""
        self.model = CatBoostClassifier()
        self.model.load_model(filename)
        print(f"✓ Model loaded: {filename}")


def main():
    print("\n" + "="*70)
    print(" BINARY EXOPLANET DETECTION MODEL")
    print(" NASA Space Apps Challenge 2025")
    print(" EXOPLANET (Candidate + Confirmed) vs FALSE POSITIVE")
    print("="*70)
    
    detector = BinaryExoplanetDetector(random_state=42)
    
    # Load and combine all datasets
    X, y = detector.load_and_combine_datasets(data_dir='data')
    
    # Train model
    print("\n" + "-"*70)
    print("TRAINING MODEL")
    print("-"*70)
    detector.train(X, y, test_size=0.3, use_smote=True, iterations=500)
    
    # Evaluate
    print("\n" + "-"*70)
    print("EVALUATION")
    print("-"*70)
    results = detector.evaluate()
    
    # Visualizations
    print("\n" + "-"*70)
    print("GENERATING VISUALIZATIONS")
    print("-"*70)
    detector.plot_feature_importance(save_dir='results')
    detector.plot_roc_curve(save_dir='results')
    
    # Save model
    detector.save_model()
    
    print("\n" + "="*70)
    print(" TRAINING COMPLETE")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  Accuracy:  {results['accuracy']*100:.2f}%")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1']:.4f}")
    print(f"  ROC-AUC:   {results['roc_auc']:.4f}")
    
    return detector, results


if __name__ == "__main__":
    detector, results = main()