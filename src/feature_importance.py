import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import os

# ----------------------------------------------------------------------
# 1. Feature anlysis visualization
# ----------------------------------------------------------------------

def plot_multimodal_feature_importance(X_features: np.ndarray, y_data: np.ndarray, model_name: str):
    
    # CLIP
    D_IMG = 1024 # CLIP Image Embedding 
    D_TXT = 1024 # CLIP Text Embedding 
    TOTAL_D = X_features.shape[1]
    
    if TOTAL_D != D_IMG + D_TXT:
        print(f"[WARN] Input Feature dim inconsistency: Now {TOTAL_D}dim. fail.")
        return

    print(f"LightGBM Train...")

    # 1. train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_data, test_size=0.2, random_state=42
    )

    # 2. LightGBM Train
    lgb_params = {
        'objective': 'regression', 'metric': 'rmse', 'n_estimators': 300, 
        'learning_rate': 0.05, 'verbose': -1, 'n_jobs': -1, 'seed': 42
    }
    model = lgb.LGBMRegressor(**lgb_params)
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse', callbacks=callbacks)

    # 3. Feature Importanity
    importance = model.feature_importances_
    
    # 4. Image and Text Feature Importance
    img_importance_sum = importance[:D_IMG].sum()
    txt_importance_sum = importance[D_IMG:].sum()
    
    # 5. Vizualization (Pie Chart)
    labels = ['CLIP Image Feature Contribution', 'CLIP Text Feature Contribution']
    sizes = [img_importance_sum, txt_importance_sum]
    
    # 폰트 설정 (Mac: AppleGothic, Windows: Malgun Gothic)
    if os.name == 'posix': # Mac, Linux
        plt.rcParams['font.family'] = 'AppleGothic' 
    else: # Windows
        plt.rcParams['font.family'] = 'Malgun Gothic'
        
    plt.figure(figsize=(8, 8))
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, 
            colors=['#3F51B5', '#1E88E5'], 
            wedgeprops={'edgecolor': 'black', 'linewidth': 0.5},
            textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    plt.title(f"feature Contribution Analysis: {model_name}", fontsize=16)
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
# 2. Main execute
# ----------------------------------------------------------------------

if __name__ == '__main__':
    try:
        # Sample data
        N_SAMPLES = 1000
        D_FEATURES = 2048 
        
        # 1. Multimodal feature vectors (X_features)
        X_clip_multi = np.random.randn(N_SAMPLES, D_FEATURES) * 0.1
        X_clip_multi[:, :1024] += np.random.rand(N_SAMPLES, 1024) * 0.5         
        # 2. Log view Count 
        simulated_weights = np.zeros(D_FEATURES)
        simulated_weights[:100] = np.random.rand(100) * 5 
        y_data = X_clip_multi @ simulated_weights + np.random.randn(N_SAMPLES) * 0.5
        y_data = (y_data - y_data.mean()) / y_data.std() # Normalization
        
        print("Complete.")
        
        # 3. vizulization
        plot_multimodal_feature_importance(X_clip_multi, y_data, 'CLIP Multimodal Fusion')
        
    except Exception as e:
        print(f"[Error] Code error. Check the library")
        print(f"Detail Error: {e}")