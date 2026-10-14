"""
Model training, evaluation, and persistence pipeline.
Supports Linear Ridge Baseline, Random Forest, and XGBoost with TimeSeries validation.
"""
from typing import Dict, Any, Tuple, List
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from config import MODELS_DIR, ProjectConfig, DEFAULT_CONFIG

def split_time_series_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'Target_Return',
    config: ProjectConfig = DEFAULT_CONFIG
) -> Dict[str, Any]:
    """
    Performs chronological train/val/test splits without look-ahead leakage.
    Scales features using only training set statistics.
    """
    n = len(df)
    train_end = int(n * config.train_ratio)
    val_end = int(n * (config.train_ratio + config.val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    X_train_raw = train_df[feature_cols].values
    X_val_raw = val_df[feature_cols].values
    X_test_raw = test_df[feature_cols].values

    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values

    # Standardize features based ONLY on training statistics
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    return {
        'X_train': X_train, 'y_train': y_train, 'train_df': train_df,
        'X_val': X_val, 'y_val': y_val, 'val_df': val_df,
        'X_test': X_test, 'y_test': y_test, 'test_df': test_df,
        'scaler': scaler,
        'feature_cols': feature_cols
    }

def get_models(random_state: int = 42) -> Dict[str, Any]:
    """Initializes competitive prediction models."""
    return {
        "Ridge_Baseline": Ridge(alpha=10.0, random_state=random_state),
        "Random_Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=6,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1
        )
    }

def train_and_evaluate_models(
    split_data: Dict[str, Any],
    save_models: bool = True,
    models_dir: Path = MODELS_DIR
) -> Tuple[Dict[str, Any], str]:
    """
    Trains all candidate models, evaluates on validation set, and selects the champion model.
    """
    X_train, y_train = split_data['X_train'], split_data['y_train']
    X_val, y_val = split_data['X_val'], split_data['y_val']

    models = get_models()
    trained_models = {}
    val_rmse_scores = {}

    print("\n" + "="*50)
    print(" TRAINING MODELS (Sequential Time-Series Split) ")
    print("="*50)

    for name, model in models.items():
        print(f"--> Training {name}...")
        if name == "XGBoost":
            # Early stopping with validation set to avoid overfitting
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            model.fit(X_train, y_train)

        # Validation score
        val_preds = model.predict(X_val)
        val_rmse = np.sqrt(np.mean((val_preds - y_val) ** 2))
        val_rmse_scores[name] = val_rmse
        trained_models[name] = model
        print(f"    Validation RMSE: {val_rmse:.6f}")

    best_model_name = min(val_rmse_scores, key=val_rmse_scores.get)
    print(f"\n[Champion Model Selected]: {best_model_name} (Val RMSE: {val_rmse_scores[best_model_name]:.6f})")

    # Save models and scaler
    if save_models:
        joblib.dump(split_data['scaler'], models_dir / "scaler.joblib")
        for name, model in trained_models.items():
            joblib.dump(model, models_dir / f"{name}.joblib")
        print(f"[ModelPipeline] Models and Scaler saved to {models_dir}")

    return trained_models, best_model_name

def generate_test_predictions(
    model: Any,
    scaler: StandardScaler,
    test_df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Generates out-of-sample predictions and converts return predictions to price forecasts.
    """
    X_test_scaled = scaler.transform(test_df[feature_cols].values)
    pred_return = model.predict(X_test_scaled)

    results_df = test_df[['Close', 'Target_Return', 'Target_Direction', 'Target_Next_Close']].copy()
    results_df['Pred_Return'] = pred_return
    results_df['Pred_Direction'] = (pred_return > 0).astype(int)
    
    # Forecasted Next Close: P_{t+1} = Close_t * exp(Pred_Return)
    results_df['Pred_Next_Close'] = results_df['Close'] * np.exp(pred_return)
    
    return results_df
