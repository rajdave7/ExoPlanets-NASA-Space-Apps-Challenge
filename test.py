# === SECTION 1: DATA LOADING AND STANDARDIZATION ===
DATA_PATHS = {
    'KOI': 'data/kepler_koi.csv',
    'TOI': 'data/tess_toi.csv'
}

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             roc_curve, accuracy_score, precision_score, recall_score, 
                             f1_score)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import time
import pickle
from pathlib import Path

# Deep Learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization, LSTM, 
                                     Conv1D, MaxPooling1D, Flatten, MultiHeadAttention,
                                     LayerNormalization, GlobalAveragePooling1D, Reshape)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Create directories
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

def load_csv(path):
    if path is None or not os.path.exists(path):
        print(f'WARNING: path not found: {path}')
        return pd.DataFrame()
    return pd.read_csv(path)

def standardize_koi(df):
    out = pd.DataFrame()
    out['id'] = df.get('kepoi_name')
    disp = df.get('koi_disposition').astype(str).str.upper() if 'koi_disposition' in df.columns else pd.Series([np.nan]*len(df))
    out['label'] = disp.replace({'CANDIDATE':1,'CONFIRMED':1,'FALSE POSITIVE':0,'FALSE_POSITIVE':0})
    out['period'] = pd.to_numeric(df.get('koi_period'), errors='coerce')
    out['duration'] = pd.to_numeric(df.get('koi_duration'), errors='coerce')
    out['depth'] = pd.to_numeric(df.get('koi_depth'), errors='coerce')
    out['prad'] = pd.to_numeric(df.get('koi_prad'), errors='coerce')
    out['snr'] = pd.to_numeric(df.get('koi_model_snr'), errors='coerce') if 'koi_model_snr' in df.columns else np.nan
    out['st_teff'] = np.nan
    out['st_rad'] = np.nan
    return out

def standardize_toi(df):
    out = pd.DataFrame()
    out['id'] = df.get('toi')
    disp = df.get('tfopwg_disp').astype(str).str.upper() if 'tfopwg_disp' in df.columns else pd.Series([np.nan]*len(df))
    out['label'] = disp.replace({'PC':1,'KP':1,'CONFIRMED':1,'CANDIDATE':1,'FP':0,'FALSE POSITIVE':0})
    out['period'] = pd.to_numeric(df.get('pl_orbper'), errors='coerce')
    out['duration'] = pd.to_numeric(df.get('pl_trandurh'), errors='coerce')
    out['depth'] = pd.to_numeric(df.get('pl_trandep'), errors='coerce')
    out['prad'] = pd.to_numeric(df.get('pl_rade'), errors='coerce')
    out['snr'] = np.nan
    out['st_teff'] = pd.to_numeric(df.get('st_teff'), errors='coerce') if 'st_teff' in df.columns else np.nan
    out['st_rad'] = pd.to_numeric(df.get('st_rad'), errors='coerce') if 'st_rad' in df.columns else np.nan
    return out

# Load data
df_koi = load_csv(DATA_PATHS.get('KOI'))
df_toi = load_csv(DATA_PATHS.get('TOI'))

std_koi = standardize_koi(df_koi) if not df_koi.empty else pd.DataFrame()
std_toi = standardize_toi(df_toi) if not df_toi.empty else pd.DataFrame()

print('Standardized shapes:')
print('KOI:', std_koi.shape, 'TOI:', std_toi.shape)

# === SECTION 2: DATA BALANCING FOR TOI ===
print('\n=== BALANCING TOI DATASET ===')
print('Original TOI label distribution:')
print(std_toi['label'].value_counts(dropna=False))

toi_binary = std_toi[std_toi['label'].isin([0, 1])].copy()
print(f'\nAfter filtering to binary (0,1): {toi_binary.shape[0]} samples')
print(toi_binary['label'].value_counts())

def undersample_balanced(df, target_col='label', random_state=42):
    df = df.dropna(subset=[target_col])
    counts = df[target_col].value_counts()
    if len(counts) < 2:
        return df
    min_count = counts.min()
    parts = []
    for cls, grp in df.groupby(target_col):
        parts.append(grp.sample(min_count, random_state=random_state, replace=False))
    return pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)

toi_balanced = undersample_balanced(toi_binary, random_state=42)
print(f'\nAfter balancing: {toi_balanced.shape[0]} samples')
print(toi_balanced['label'].value_counts())

# === SECTION 3: FEATURE ENGINEERING ===
print('\n=== FEATURE ENGINEERING ===')

def engineer_features(df):
    df = df.copy()
    if 'period' in df.columns:
        df['log_period'] = np.log1p(df['period'])
    if 'duration' in df.columns:
        df['log_duration'] = np.log1p(df['duration'])
    if 'depth' in df.columns:
        df['log_depth'] = np.log1p(df['depth'])
    if 'period' in df.columns and 'duration' in df.columns:
        df['period_duration_ratio'] = df['period'] / (df['duration'] + 1e-6)
    if 'depth' in df.columns and 'prad' in df.columns:
        df['depth_prad_product'] = df['depth'] * df['prad']
    if 'st_teff' in df.columns and 'st_rad' in df.columns:
        df['stellar_luminosity'] = (df['st_rad'] ** 2) * (df['st_teff'] / 5778) ** 4
    return df

toi_engineered = engineer_features(toi_balanced)
koi_engineered = engineer_features(std_koi)

BASE_FEATURES = ['period', 'duration', 'prad', 'depth', 'st_teff', 'st_rad', 'snr']
ENGINEERED_FEATURES = ['log_period', 'log_duration', 'log_depth', 
                       'period_duration_ratio', 'depth_prad_product', 'stellar_luminosity']
ALL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

# === SECTION 4: TRAIN/VAL/TEST SPLIT ===
print('\n=== CREATING TRAIN/VAL/TEST SPLITS ===')

def prepare_data_splits(df, feature_cols, test_size=0.15, val_size=0.15, random_state=42):
    df = df.dropna(subset=['label'])
    df = df[df['label'].isin([0, 1])]
    
    available_features = [f for f in feature_cols if f in df.columns]
    X = df[available_features].copy()
    y = df['label'].astype(int)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state
    )
    
    print(f'Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}')
    print(f'Train labels: {y_train.value_counts().to_dict()}')
    print(f'Val labels: {y_val.value_counts().to_dict()}')
    print(f'Test labels: {y_test.value_counts().to_dict()}')
    
    return X_train, X_val, X_test, y_train, y_val, y_test, available_features

X_train_toi, X_val_toi, X_test_toi, y_train_toi, y_val_toi, y_test_toi, toi_features = \
    prepare_data_splits(toi_engineered, ALL_FEATURES, random_state=42)

# === SECTION 5: DATA PREPROCESSING PIPELINE ===
print('\n=== PREPROCESSING DATA ===')

imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_toi_imputed = imputer.fit_transform(X_train_toi)
X_val_toi_imputed = imputer.transform(X_val_toi)
X_test_toi_imputed = imputer.transform(X_test_toi)

X_train_toi_scaled = scaler.fit_transform(X_train_toi_imputed)
X_val_toi_scaled = scaler.transform(X_val_toi_imputed)
X_test_toi_scaled = scaler.transform(X_test_toi_imputed)

print(f'Preprocessed feature shape: {X_train_toi_scaled.shape[1]} features')

# Reshape for sequence models (LSTM, CNN-LSTM)
X_train_seq = X_train_toi_scaled.reshape(X_train_toi_scaled.shape[0], X_train_toi_scaled.shape[1], 1)
X_val_seq = X_val_toi_scaled.reshape(X_val_toi_scaled.shape[0], X_val_toi_scaled.shape[1], 1)
X_test_seq = X_test_toi_scaled.reshape(X_test_toi_scaled.shape[0], X_test_toi_scaled.shape[1], 1)

print(f'Sequence shape: {X_train_seq.shape}')

# === SECTION 6: MODEL DEFINITIONS ===

def create_dnn_model(input_dim, learning_rate=0.001):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc'), 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    return model

def create_residual_block(x, units, dropout_rate=0.3):
    fx = Dense(units, activation='relu')(x)
    fx = BatchNormalization()(fx)
    fx = Dropout(dropout_rate)(fx)
    fx = Dense(units, activation='linear')(fx)
    fx = BatchNormalization()(fx)
    
    if x.shape[-1] != units:
        x = Dense(units, activation='linear')(x)
    
    out = layers.Add()([x, fx])
    out = layers.Activation('relu')(out)
    return out

def create_residual_dnn(input_dim, learning_rate=0.001):
    inputs = keras.Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = create_residual_block(x, 128, dropout_rate=0.3)
    x = create_residual_block(x, 64, dropout_rate=0.3)
    x = create_residual_block(x, 32, dropout_rate=0.2)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc'),
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    return model

def create_lstm_model(input_shape, learning_rate=0.001):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc'),
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    return model

def create_cnn_lstm_model(input_shape, learning_rate=0.001):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(32, kernel_size=3, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc'),
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    return model

def create_transformer_model(input_dim, learning_rate=0.001):
    inputs = keras.Input(shape=(input_dim,))
    x = Reshape((input_dim, 1))(inputs)
    
    # Transformer block
    attn_output = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    attn_output = Dropout(0.3)(attn_output)
    x = LayerNormalization(epsilon=1e-6)(x + attn_output)
    
    # Feed-forward network
    ffn = Dense(64, activation='relu')(x)
    ffn = Dropout(0.3)(ffn)
    ffn = Dense(input_dim)(ffn)
    x = LayerNormalization(epsilon=1e-6)(x + ffn)
    
    # Global pooling and output
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc'),
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    return model

# === SECTION 7: TRAINING UTILITIES ===

def plot_training_history(history, title, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss Over Epochs')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 1].set_title('Accuracy Over Epochs')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(history.history['auc'], label='Train AUC')
    axes[1, 0].plot(history.history['val_auc'], label='Val AUC')
    axes[1, 0].set_title('AUC Over Epochs')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(history.history['precision'], label='Train Precision', alpha=0.7)
    axes[1, 1].plot(history.history['val_precision'], label='Val Precision', alpha=0.7)
    axes[1, 1].plot(history.history['recall'], label='Train Recall', alpha=0.7)
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall', alpha=0.7)
    axes[1, 1].set_title('Precision & Recall Over Epochs')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved training plot: {save_path}')

def calculate_comprehensive_metrics(model, X_test, y_test, model_name, is_dl=True):
    """Calculate comprehensive metrics including FLOPs, timing, etc."""
    
    # Prediction timing
    start_time = time.perf_counter()
    if is_dl:
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    test_time = time.perf_counter() - start_time
    
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_pred_proba),
        'test_time': test_time,
        'test_time_per_sample': test_time / len(y_test)
    }
    
    # Calculate FLOPs for deep learning models
    if is_dl:
        try:
            # Approximate FLOPs calculation
            total_params = model.count_params()
            # Rough estimate: 2 FLOPs per parameter per sample
            metrics['flops_estimate'] = total_params * 2
            metrics['total_params'] = total_params
        except:
            metrics['flops_estimate'] = None
            metrics['total_params'] = None
    
    return metrics, y_pred, y_pred_proba

def save_evaluation_plot(y_test, y_pred, y_pred_proba, model_name, save_path):
    """Save confusion matrix and ROC curve"""
    cm = confusion_matrix(y_test, y_pred)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Not Planet', 'Planet'],
                yticklabels=['Not Planet', 'Planet'])
    ax1.set_title(f'{model_name} - Confusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    ax2.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)
    ax2.plot([0, 1], [0, 1], 'k--', label='Random')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title(f'{model_name} - ROC Curve')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved evaluation plot: {save_path}')

# === SECTION 8: TRAIN ALL MODELS ===

print('\n' + '='*80)
print('TRAINING ALL MODELS')
print('='*80)

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)

all_models = {}
all_metrics = {}
all_train_times = {}

# 1. Standard DNN
print('\n=== TRAINING STANDARD DNN ===')
dnn_model = create_dnn_model(X_train_toi_scaled.shape[1])
t0 = time.perf_counter()
dnn_history = dnn_model.fit(
    X_train_toi_scaled, y_train_toi,
    validation_data=(X_val_toi_scaled, y_val_toi),
    epochs=100, batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
dnn_train_time = time.perf_counter() - t0
all_models['DNN'] = dnn_model
all_train_times['DNN'] = dnn_train_time
plot_training_history(dnn_history, 'Standard DNN Training', 'plots/dnn_training.png')

# 2. Residual DNN
print('\n=== TRAINING RESIDUAL DNN ===')
res_dnn_model = create_residual_dnn(X_train_toi_scaled.shape[1])
t0 = time.perf_counter()
res_dnn_history = res_dnn_model.fit(
    X_train_toi_scaled, y_train_toi,
    validation_data=(X_val_toi_scaled, y_val_toi),
    epochs=100, batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
res_dnn_train_time = time.perf_counter() - t0
all_models['Residual_DNN'] = res_dnn_model
all_train_times['Residual_DNN'] = res_dnn_train_time
plot_training_history(res_dnn_history, 'Residual DNN Training', 'plots/res_dnn_training.png')

# 3. LSTM
print('\n=== TRAINING LSTM ===')
lstm_model = create_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]))
t0 = time.perf_counter()
lstm_history = lstm_model.fit(
    X_train_seq, y_train_toi,
    validation_data=(X_val_seq, y_val_toi),
    epochs=100, batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
lstm_train_time = time.perf_counter() - t0
all_models['LSTM'] = lstm_model
all_train_times['LSTM'] = lstm_train_time
plot_training_history(lstm_history, 'LSTM Training', 'plots/lstm_training.png')

# 4. CNN-LSTM
print('\n=== TRAINING CNN-LSTM ===')
cnn_lstm_model = create_cnn_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]))
t0 = time.perf_counter()
cnn_lstm_history = cnn_lstm_model.fit(
    X_train_seq, y_train_toi,
    validation_data=(X_val_seq, y_val_toi),
    epochs=100, batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
cnn_lstm_train_time = time.perf_counter() - t0
all_models['CNN_LSTM'] = cnn_lstm_model
all_train_times['CNN_LSTM'] = cnn_lstm_train_time
plot_training_history(cnn_lstm_history, 'CNN-LSTM Training', 'plots/cnn_lstm_training.png')

# 5. Transformer
print('\n=== TRAINING TRANSFORMER ===')
transformer_model = create_transformer_model(X_train_toi_scaled.shape[1])
t0 = time.perf_counter()
transformer_history = transformer_model.fit(
    X_train_toi_scaled, y_train_toi,
    validation_data=(X_val_toi_scaled, y_val_toi),
    epochs=100, batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
transformer_train_time = time.perf_counter() - t0
all_models['Transformer'] = transformer_model
all_train_times['Transformer'] = transformer_train_time
plot_training_history(transformer_history, 'Transformer Training', 'plots/transformer_training.png')

# 6. Random Forest
print('\n=== TRAINING RANDOM FOREST ===')
rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', 
                                   random_state=42, n_jobs=-1)
t0 = time.perf_counter()
rf_model.fit(X_train_toi_scaled, y_train_toi)
rf_train_time = time.perf_counter() - t0
all_models['Random_Forest'] = rf_model
all_train_times['Random_Forest'] = rf_train_time

# 7. XGBoost
print('\n=== TRAINING XGBOOST ===')
xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6,
                          random_state=42, n_jobs=-1, eval_metric='logloss')
t0 = time.perf_counter()
xgb_model.fit(X_train_toi_scaled, y_train_toi)
xgb_train_time = time.perf_counter() - t0
all_models['XGBoost'] = xgb_model
all_train_times['XGBoost'] = xgb_train_time

# === SECTION 9: EVALUATE ALL MODELS ===

print('\n' + '='*80)
print('EVALUATING ALL MODELS')
print('='*80)

for model_name, model in all_models.items():
    print(f'\n=== EVALUATING {model_name} ===')
    
    # Determine if sequence model
    if model_name in ['LSTM', 'CNN_LSTM']:
        X_test_use = X_test_seq
    else:
        X_test_use = X_test_toi_scaled
    
    # Determine if DL model
    is_dl = model_name not in ['Random_Forest', 'XGBoost']
    
    # Calculate metrics
    metrics, y_pred, y_pred_proba = calculate_comprehensive_metrics(
        model, X_test_use, y_test_toi, model_name, is_dl=is_dl
    )
    
    # Add training time
    metrics['train_time'] = all_train_times[model_name]
    
    # Store metrics
    all_metrics[model_name] = metrics
    
    # Print classification report
    print(classification_report(y_test_toi, y_pred, target_names=['Not Planet', 'Planet']))
    
    # Print detailed metrics
    print(f'\nDetailed Metrics for {model_name}:')
    print(f'Accuracy: {metrics["accuracy"]:.4f}')
    print(f'Precision: {metrics["precision"]:.4f}')
    print(f'Recall: {metrics["recall"]:.4f}')
    print(f'F1-Score: {metrics["f1_score"]:.4f}')
    print(f'ROC AUC: {metrics["auc"]:.4f}')
    print(f'Train Time: {metrics["train_time"]:.3f} sec')
    print(f'Test Time: {metrics["test_time"]:.3f} sec')
    print(f'Test Time per Sample: {metrics["test_time_per_sample"]*1000:.3f} ms')
    
    if 'total_params' in metrics and metrics['total_params'] is not None:
        print(f'Total Parameters: {metrics["total_params"]:,}')
        print(f'Estimated FLOPs: {metrics["flops_estimate"]:,}')
    
    # Save evaluation plots
    save_evaluation_plot(y_test_toi, y_pred, y_pred_proba, model_name, 
                        f'plots/{model_name.lower()}_evaluation.png')

# === SECTION 10: COMPREHENSIVE METRICS COMPARISON ===

print('\n' + '='*80)
print('COMPREHENSIVE MODEL COMPARISON')
print('='*80)

# Create comprehensive comparison DataFrame
comparison_data = []
for model_name, metrics in all_metrics.items():
    row = {
        'Model': model_name,
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1_score'],
        'ROC AUC': metrics['auc'],
        'Train Time (s)': metrics['train_time'],
        'Test Time (s)': metrics['test_time'],
        'Test Time/Sample (ms)': metrics['test_time_per_sample'] * 1000
    }
    
    if 'total_params' in metrics and metrics['total_params'] is not None:
        row['Parameters'] = metrics['total_params']
        row['FLOPs (estimate)'] = metrics['flops_estimate']
    
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)

# Sort by ROC AUC
comparison_df = comparison_df.sort_values('ROC AUC', ascending=False)

print('\n=== METRICS COMPARISON TABLE ===')
print(comparison_df.to_string(index=False))

# Save to CSV
comparison_df.to_csv('plots/model_comparison_metrics.csv', index=False)
print('\nSaved metrics comparison to: plots/model_comparison_metrics.csv')

# === SECTION 11: VISUALIZATION OF COMPARISONS ===

print('\n=== CREATING COMPARISON VISUALIZATIONS ===')

# 1. Performance Metrics Comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'ROC AUC']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]
    bars = ax.bar(comparison_df['Model'], comparison_df[metric], color=colors[:len(comparison_df)])
    ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric, fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('plots/performance_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: plots/performance_metrics_comparison.png')

# 2. Training and Testing Time Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Training time
bars1 = ax1.bar(comparison_df['Model'], comparison_df['Train Time (s)'], color=colors[:len(comparison_df)])
ax1.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('Time (seconds)', fontsize=12)
ax1.set_xlabel('Model', fontsize=12)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)

for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}s', ha='center', va='bottom', fontsize=9)

# Testing time per sample
bars2 = ax2.bar(comparison_df['Model'], comparison_df['Test Time/Sample (ms)'], color=colors[:len(comparison_df)])
ax2.set_title('Inference Time per Sample', fontsize=14, fontweight='bold')
ax2.set_ylabel('Time (milliseconds)', fontsize=12)
ax2.set_xlabel('Model', fontsize=12)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}ms', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('plots/time_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: plots/time_comparison.png')

# 3. ROC Curves Comparison
plt.figure(figsize=(10, 8))

for model_name, model in all_models.items():
    # Determine if sequence model
    if model_name in ['LSTM', 'CNN_LSTM']:
        X_test_use = X_test_seq
    else:
        X_test_use = X_test_toi_scaled
    
    # Get predictions
    is_dl = model_name not in ['Random_Forest', 'XGBoost']
    if is_dl:
        y_pred_proba = model.predict(X_test_use, verbose=0).flatten()
    else:
        y_pred_proba = model.predict_proba(X_test_use)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test_toi, y_pred_proba)
    auc = all_metrics[model_name]['auc']
    
    plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC={auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - All Models Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: plots/roc_curves_comparison.png')

# 4. Model Complexity vs Performance
if 'Parameters' in comparison_df.columns:
    dl_models_df = comparison_df[comparison_df['Parameters'].notna()].copy()
    
    if not dl_models_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(dl_models_df['Parameters'], 
                           dl_models_df['ROC AUC'],
                           s=200, alpha=0.6, c=range(len(dl_models_df)), 
                           cmap='viridis')
        
        for idx, row in dl_models_df.iterrows():
            ax.annotate(row['Model'], 
                       (row['Parameters'], row['ROC AUC']),
                       fontsize=10, ha='center', va='bottom')
        
        ax.set_xlabel('Number of Parameters', fontsize=12)
        ax.set_ylabel('ROC AUC Score', fontsize=12)
        ax.set_title('Model Complexity vs Performance', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/complexity_vs_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('Saved: plots/complexity_vs_performance.png')

# 5. Feature Importance (for tree-based models)
print('\n=== FEATURE IMPORTANCE ANALYSIS ===')

rf_importances = rf_model.feature_importances_
xgb_importances = xgb_model.feature_importances_

# Ensure feature names match importances length
feature_names_used = toi_features[:len(rf_importances)]

feature_importance_df = pd.DataFrame({
    'Feature': feature_names_used,
    'Random Forest': rf_importances,
    'XGBoost': xgb_importances
})

feature_importance_df = feature_importance_df.sort_values('Random Forest', ascending=False)
print(feature_importance_df.to_string(index=False))

# Save feature importance
feature_importance_df.to_csv('plots/feature_importance.csv', index=False)

# Plot feature importance
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(feature_names_used))
width = 0.35

bars1 = ax.bar(x - width/2, feature_importance_df['Random Forest'], width, 
               label='Random Forest', alpha=0.8, color='#2ca02c')
bars2 = ax.bar(x + width/2, feature_importance_df['XGBoost'], width, 
               label='XGBoost', alpha=0.8, color='#d62728')

ax.set_xlabel('Features', fontsize=12)
ax.set_ylabel('Importance', fontsize=12)
ax.set_title('Feature Importance Comparison (Tree-based Models)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(feature_importance_df['Feature'], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: plots/feature_importance.png')

# === SECTION 12: SAVE MODELS AND PREPROCESSORS ===

print('\n' + '='*80)
print('SAVING MODELS AND PREPROCESSORS')
print('='*80)

# Save deep learning models
dnn_model.save('models/dnn_model.h5')
print('Saved: models/dnn_model.h5')

res_dnn_model.save('models/res_dnn_model.h5')
print('Saved: models/res_dnn_model.h5')

lstm_model.save('models/lstm_model.h5')
print('Saved: models/lstm_model.h5')

cnn_lstm_model.save('models/cnn_lstm_model.h5')
print('Saved: models/cnn_lstm_model.h5')

transformer_model.save('models/transformer_model.h5')
print('Saved: models/transformer_model.h5')

# Save traditional ML models
with open('models/rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print('Saved: models/rf_model.pkl')

with open('models/xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print('Saved: models/xgb_model.pkl')

# Save preprocessors
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print('Saved: models/scaler.pkl')

with open('models/imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)
print('Saved: models/imputer.pkl')

# Save feature names
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(toi_features, f)
print('Saved: models/feature_names.pkl')

# Save metrics
with open('models/model_metrics.pkl', 'wb') as f:
    pickle.dump(all_metrics, f)
print('Saved: models/model_metrics.pkl')

# === SECTION 13: SUMMARY REPORT ===

print('\n' + '='*80)
print('FINAL SUMMARY REPORT')
print('='*80)

# Best model by different criteria
best_auc_model = comparison_df.loc[comparison_df['ROC AUC'].idxmax()]
best_accuracy_model = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
fastest_train_model = comparison_df.loc[comparison_df['Train Time (s)'].idxmin()]
fastest_inference_model = comparison_df.loc[comparison_df['Test Time/Sample (ms)'].idxmin()]

print('\n=== BEST MODELS BY CRITERIA ===')
print(f'\n🏆 Best ROC AUC: {best_auc_model["Model"]} ({best_auc_model["ROC AUC"]:.4f})')
print(f'🏆 Best Accuracy: {best_accuracy_model["Model"]} ({best_accuracy_model["Accuracy"]:.4f})')
print(f'⚡ Fastest Training: {fastest_train_model["Model"]} ({fastest_train_model["Train Time (s)"]:.2f}s)')
print(f'⚡ Fastest Inference: {fastest_inference_model["Model"]} ({fastest_inference_model["Test Time/Sample (ms)"]:.2f}ms)')

print('\n=== KEY FINDINGS ===')
print(f'• Total models trained: {len(all_models)}')
print(f'• Dataset size: {len(toi_balanced)} samples (balanced)')
print(f'• Train/Val/Test split: {len(y_train_toi)}/{len(y_val_toi)}/{len(y_test_toi)}')
print(f'• Number of features: {len(toi_features)}')
print(f'• Best overall AUC: {comparison_df["ROC AUC"].max():.4f}')
print(f'• Average AUC across all models: {comparison_df["ROC AUC"].mean():.4f}')

print('\n=== RECOMMENDATIONS ===')
if best_auc_model['Model'] == fastest_inference_model['Model']:
    print(f'✅ RECOMMENDED: {best_auc_model["Model"]} - Best performance AND fastest inference!')
else:
    print(f'✅ For ACCURACY: Use {best_auc_model["Model"]} (AUC: {best_auc_model["ROC AUC"]:.4f})')
    print(f'✅ For SPEED: Use {fastest_inference_model["Model"]} (Inference: {fastest_inference_model["Test Time/Sample (ms)"]:.2f}ms)')
    print(f'✅ For BALANCE: Consider ensemble of top 3 models')

print('\n=== DATA IMPROVEMENT SUGGESTIONS ===')
print('1. Feature Engineering:')
print('   - Add more derived features (period harmonics, transit shape metrics)')
print('   - Include stellar characteristics (metallicity, age, spectral type)')
print('   - Add temporal features if light curve data available')
print('')
print('2. Data Augmentation:')
print('   - Use SMOTE or ADASYN for synthetic oversampling')
print('   - Add noise augmentation for robust training')
print('   - Cross-mission data fusion (combine Kepler + TESS)')
print('')
print('3. Advanced Preprocessing:')
print('   - Outlier detection and handling')
print('   - Feature selection using mutual information')
print('   - PCA or autoencoders for dimensionality reduction')
print('')
print('4. Model Improvements:')
print('   - Hyperparameter tuning with Optuna or GridSearch')
print('   - Ensemble methods (stacking, voting)')
print('   - Attention mechanisms for feature importance')
print('   - Transfer learning from pre-trained models')
print('')
print('5. Data Collection:')
print('   - Include K2 mission data')
print('   - Add ground-based follow-up observations')
print('   - Incorporate spectroscopic data')
print('   - Use archival data from other surveys')

print('\n' + '='*80)
print('✅ TRAINING COMPLETE! All models saved to models/ directory')
print('✅ All plots saved to plots/ directory')
print('✅ Ready for web application deployment!')
print('='*80)

print('\n=== NEXT STEPS ===')
print('1. Run the web application: streamlit run app.py')
print('2. Test with new data using the web interface')
print('3. Monitor model performance and retrain as needed')
print('4. Consider deploying to cloud (AWS, Azure, GCP)')
print('5. Implement CI/CD pipeline for automated retraining')

print('\n📊 Check the plots/ folder for all visualizations')
print('📁 Check the models/ folder for saved models')
print('🌐 Launch web app with: streamlit run app.py')
print('\nHappy exoplanet hunting! 🪐✨')