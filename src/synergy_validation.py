import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
from scipy.stats import spearmanr
import pandas as pd
# cross modality

def train_and_evaluate(X_train, y_train, X_test, y_test, model_name):
    lgb_params = {
        'objective': 'regression', 'metric': 'rmse', 'n_estimators': 300, 
        'learning_rate': 0.05, 'verbose': -1, 'n_jobs': -1, 'seed': 42
    }
    model = lgb.LGBMRegressor(**lgb_params)
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

    # 1. train
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse', callbacks=callbacks)

    # 2. predict
    y_pred = model.predict(X_test)
    
    # 3. eval
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rho, _ = spearmanr(y_test, y_pred)

    print(f"\n--- {model_name} result ---")
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE (Log): {rmse:.4f}")
    print(f"Spearman Rho: {rho:.4f}")
    
    return {'R2': r2, 'RMSE': rmse, 'Spearman Rho': rho}


# --- Shuffle ---

def run_cross_modal_ablation(X_img, X_txt, y_data):
    """
    CLIP Multimodal Real with Shuffled test.
    
    Args:
        X_img (np.ndarray): CLIP Image (N x 1024)
        X_txt (np.ndarray): CLIP text (N x 1024)
        y_data (np.ndarray): Log Count (N x 1)
    """
    
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # 2. divide
    N = len(y_data)
    X_indices = np.arange(N)
    
    # train/test
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        X_indices, y_data, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # --------------------------------------------------
    # 3. CLIP Multimodal (REAL) 
    # --------------------------------------------------
    print("--------------------------------------------------")
    print("3. CLIP Multimodal (REAL) ")
    print("--------------------------------------------------")
    
    X_train_clip_real = np.hstack([X_img[X_train_idx], X_txt[X_train_idx]])
    X_test_clip_real = np.hstack([X_img[X_test_idx], X_txt[X_test_idx]])
    
    results_real = train_and_evaluate(
        X_train_clip_real, y_train, 
        X_test_clip_real, y_test, 
        'CLIP Multimodal (REAL)'
    )


    # --------------------------------------------------
    # 4. CLIP Multimodal (SHUFFLED)
    # --------------------------------------------------
    print("\n--------------------------------------------------")
    print("4. CLIP Multimodal (SHUFFLED)")
    print("--------------------------------------------------")
    
    X_test_img_shuffled = X_img[X_test_idx].copy()
    np.random.shuffle(X_test_img_shuffled) 
    
    X_test_txt_original = X_txt[X_test_idx]
    X_test_clip_shuffled = np.hstack([X_test_img_shuffled, X_test_txt_original])
    
    # 3. model REAL train real, SHUFFLED test eval.
    #    -> check the differences
    
    results_shuffled = train_and_evaluate(
        X_train_clip_real, y_train, 
        X_test_clip_shuffled, y_test, 
        'CLIP Multimodal (SHUFFLED - Alignment Destroyed)'
    )
    
    print("\n[shuffled Test Compare]")
    print(f"R2 Score: REAL ({results_real['R2']:.4f}) vs SHUFFLED ({results_shuffled['R2']:.4f})")
    print(f"Spearman Rho: REAL ({results_real['Spearman Rho']:.4f}) vs SHUFFLED ({results_shuffled['Spearman Rho']:.4f})")


# --- Execute ---

if __name__ == '__main__':
    
    try:
       # Sample
        N_SAMPLES = 10000 
        D_FEAT = 1024
        
        # NOTE: real feature file by npy.
        # 예시: X_img = np.load('clip_image_features.npy')
        # 예시: X_txt = np.load('clip_text_features.npy')
        # 예시: y_data = np.load('log_view_counts.npy')
        
        # simulation data (if no data)
        np.random.seed(42)
        X_img = np.random.randn(N_SAMPLES, D_FEAT) * 0.1
        X_txt = np.random.randn(N_SAMPLES, D_FEAT) * 0.1
        
        simulated_weights_img = np.random.rand(D_FEAT) * 0.5
        simulated_weights_txt = np.random.rand(D_FEAT) * 0.3
        y_data = (X_img @ simulated_weights_img) + (X_txt @ simulated_weights_txt)
        y_data = np.log1p(y_data - y_data.min() + 1) 
        y_data = (y_data - y_data.mean()) / y_data.std() # Normalization
        
        print(f"Simulation data load complete: N={N_SAMPLES}")
        
    except Exception as e:
        print(f"[ERROR] Check the path: {e}")
        exit()

    # 2. 실험 실행
    run_cross_modal_ablation(X_img, X_txt, y_data)