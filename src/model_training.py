import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BertTokenizer, BertModel, CLIPImageProcessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, root_mean_squared_error, mean_squared_log_error
from scipy.stats import spearmanr
import lightgbm as lgb
import cv2
from tqdm import tqdm

# --- 1. Variable ---
# DEVICE 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# File Path
CSV_FILE = "youtube_shorts_data_CLEAN.csv" # csv file
THUMBNAIL_DIR = "shorts_thumbnails" #  directory

# Model
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
BERT_MODEL_NAME = "bert-base-multilingual-cased" 
RESNET_MODEL_NAME = "resnet18" # ResNet model 

# Hyper Parameter
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Thumnail padding remove info (480x360)
PADDING_WIDTH = 139 

# --- 2. Preprocessing ---

def image_crop_and_preprocess(video_id: str, clip_image_processor: CLIPImageProcessor):
    """
    1. JPG/PNG file exist or not?.
    2. In 480x360 frames, crop balck padding
    3. CLIP standard preprocessing.
    """
    # 1. exist?
    image_path_jpg = os.path.join(THUMBNAIL_DIR, f"{video_id}.jpg")
    image_path_png = os.path.join(THUMBNAIL_DIR, f"{video_id}.png")
    
    image_path = None
    if os.path.exists(image_path_jpg):
        image_path = image_path_jpg
    elif os.path.exists(image_path_png):
        image_path = image_path_png
    else:
        return None

    # 2. image load
    try:
        img_raw = cv2.imread(image_path)
        if img_raw is None: 
            return None 
            
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

        # padding remove
        start_x = PADDING_WIDTH
        end_x = 480 - PADDING_WIDTH
        cropped_img = img_rgb[:, start_x:end_x] # [height, width]

        # CLIP 
        img_pil = Image.fromarray(cropped_img)
        
        processed_input = clip_image_processor(
            images=img_pil, return_tensors="pt"
        )
        return processed_input['pixel_values'].squeeze(0) # unsqueeze(0) for batch dimension

    except Exception as e:
        return None

# --- 3. feature extract (Frozen Weights) ---

@torch.no_grad()
def extract_features(df, clip_model, bert_model, resnet_model, device): 
    """4 Modality."""
    
    features = {
        'clip_img': [], 'clip_txt': [], 'bert_txt': [], 'resnet_img': [], 'log_y': []
    }
    
    # Model Proecessor & Tokenizer 
    clip_processor_all = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_image_processor = clip_processor_all.image_processor
    clip_tokenizer = clip_processor_all.tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    # ResNet Preprocessing- ResNet standard.
    preprocess_resnet = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    clip_model.eval()
    bert_model.eval()
    resnet_model.eval()

    # tqdm
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Features"):
        log_y = row['log_viewCount']
        vid = row['videoId']
        title = row['title']
        
        current_features = {}

        # --- 1. Image Feature ---
        
        # 1-1. CLIP Image 
        image_tensor_clip_raw = image_crop_and_preprocess(vid, clip_image_processor)
        if image_tensor_clip_raw is None:
            continue 

        image_tensor_clip = image_tensor_clip_raw.unsqueeze(0).to(device)
        
        try:
            clip_img_feat = clip_model.get_image_features(image_tensor_clip)
            current_features['clip_img'] = clip_img_feat.squeeze(0).cpu().numpy()
        
        except Exception:
            continue 

        # 1-2. ResNet Image 
        try:
            image_path_jpg = os.path.join(THUMBNAIL_DIR, f"{vid}.jpg")
            image_path_png = os.path.join(THUMBNAIL_DIR, f"{vid}.png")
            
            img_raw = cv2.imread(image_path_jpg) if os.path.exists(image_path_jpg) else cv2.imread(image_path_png)
            img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            cropped_img = img_rgb[:, PADDING_WIDTH:480 - PADDING_WIDTH]
            img_pil_cropped = Image.fromarray(cropped_img)
            
            image_tensor_resnet = preprocess_resnet(img_pil_cropped).unsqueeze(0).to(device)
            resnet_output = resnet_model(image_tensor_resnet)
            current_features['resnet_img'] = resnet_output.squeeze().cpu().numpy()
            
        except Exception:
            current_features['resnet_img'] = np.zeros(512) 
        
        
        # --- 2. Text Feature ---
        try:
            # 2-1. CLIP Text
            text_inputs_clip = clip_tokenizer(title, return_tensors="pt", padding='max_length', truncation=True, max_length=77).to(device)
            clip_txt_feat = clip_model.get_text_features(**text_inputs_clip)
            current_features['clip_txt'] = clip_txt_feat.squeeze(0).cpu().numpy()
            
            # 2-2. BERT Text
            text_inputs_bert = bert_tokenizer(title, return_tensors="pt", padding='max_length', truncation=True, max_length=77).to(device)
            bert_output = bert_model(**text_inputs_bert)
            bert_cls_feat = bert_output.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            current_features['bert_txt'] = bert_cls_feat

            # --- 3. Feature save ---
            features['clip_img'].append(current_features['clip_img'])
            features['clip_txt'].append(current_features['clip_txt'])
            features['resnet_img'].append(current_features['resnet_img'])
            features['bert_txt'].append(current_features['bert_txt'])
            features['log_y'].append(log_y)

        except Exception as e:
            continue

    # NumPy Convert
    for key in features:
        features[key] = np.array(features[key])
        
    return features
# --- 4. visualization ---

def plot_parity_and_residual(y_test_log, y_pred_log, model_name):
    """
    Log Scale Predict and Real (Parity Plot) and Residual Distribution.
    """
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    # Residuals
    residuals = y_test_log - y_pred_log

    fig, axes = plt.subplots(2, 1, figsize=(10, 10)) 
    fig.suptitle(f"Model Name: {model_name} (Log Scale based)", fontsize=16, fontweight='bold')
    
    ax1 = axes[0]
    ax1.scatter(y_test_log, y_pred_log, alpha=0.3, color='#1E88E5', s=5)
    
    min_val = min(y_test_log.min(), y_pred_log.min())
    max_val = max(y_test_log.max(), y_pred_log.max())
    ax1.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1.5)
    
    ax1.set_title('Predict vs Real (Log Parity Plot)')
    ax1.set_xlabel('Real Log ViewCount')
    ax1.set_ylabel('Predict Log ViewCount')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = axes[1]
    ax2.hist(residuals, bins=50, color='#FFC107', edgecolor='black', alpha=0.8)
    ax2.axvline(0, color='red', linestyle='--', linewidth=1)
    
    ax2.set_title('Log Residuals')
    ax2.set_xlabel('Residual (Real - Predict) [Log Scale]')
    ax2.set_ylabel('ViewCount')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plt.show()

# --- 5. Train ---

def train_and_evaluate(X_train, y_train, X_test, y_test, model_name):
    # LightGBM Regression
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': RANDOM_STATE
    }
    
    model = lgb.LGBMRegressor(**lgb_params)
    
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

    model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)], 
              eval_metric='rmse', 
              callbacks=callbacks)
    
    # Y predict
    y_pred = model.predict(X_test)
    
    # Eval (log)
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    spearman, _ = spearmanr(y_test, y_pred) 
     
    y_test_original = np.expm1(y_test)
    y_pred_original = np.expm1(y_pred)
    
    # RMSLE (Root Mean Squared Logarithmic Error) 
    rmsle = np.sqrt(mean_squared_log_error(y_test_original + 1, y_pred_original + 1)) 
    
    # MAPE 
    mape = mean_absolute_percentage_error(y_test_original, y_pred_original) * 100
    
    return {
        'R2': r2, 'RMSE (Log)': rmse, 'RMSLE': rmsle,
        'Spearman Rho': spearman, 'MAPE (%)': mape,
        'y_test_log': y_test,
        'y_pred_log': y_pred
    }


# --- 5. main ---

def main_experiment():
    
    if not os.path.exists(CSV_FILE):
        print(f"[CRITICAL] CSV File '{CSV_FILE}' Not Found. check path.")
        return

    # 1. Data load
    print("1. data load and preprocessing...")
    df_raw = pd.read_csv(CSV_FILE, encoding='utf-8-sig')
    df_raw.drop_duplicates(subset=['videoId'], inplace=True)
    df_raw.dropna(subset=['title', 'viewCount'], inplace=True)
    
    # log(ViewCount + 1)
    df_raw['log_viewCount'] = np.log1p(df_raw['viewCount'])
    
    # Model (GPU on)
    print(f"2. Model Load Begin (Device: {DEVICE})...")
    # (SSL/Model loading code omitted for brevity)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    bert_model = BertModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)
    # ResNet  
    resnet_model = torch.hub.load('pytorch/vision:v0.10.0', RESNET_MODEL_NAME, pretrained=True)
    # Fully Connected layer remove
    resnet_model = nn.Sequential(*(list(resnet_model.children())[:-1])).to(DEVICE)
    
    # 3. feature extract
    print("3. feature extract start...")
    features = extract_features(df_raw, clip_model, bert_model, resnet_model, DEVICE)
    
    # valid?
    N = len(features['log_y'])
    print(f"4. valid sample: {N}개")

    if N < 100:
        print("[CRITICAL] valid sampel is a few. check Path")
        return

    # 4. train/test
    y_data = features['log_y']
    X_indices = np.arange(N)
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        X_indices, y_data, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    results = {}
    
    # --- 5. experiment ---

    # 5-1. CLIP Image only
    X_img = features['clip_img']
    X_train_img = X_img[X_train_idx]
    X_test_img = X_img[X_test_idx]
    results['CLIP_IMAGE_ONLY'] = train_and_evaluate(X_train_img, y_train, X_test_img, y_test, 'CLIP Image')

    # 5-2. CLIP Text only
    X_txt = features['clip_txt']
    X_train_txt = X_txt[X_train_idx]
    X_test_txt = X_txt[X_test_idx]
    results['CLIP_TEXT_ONLY'] = train_and_evaluate(X_train_txt, y_train, X_test_txt, y_test, 'CLIP Text')

    # 5-3. CLIP Multimodal 
    X_clip_multi = np.hstack([X_img, X_txt])
    X_train_clip_multi = X_clip_multi[X_train_idx]
    X_test_clip_multi = X_clip_multi[X_test_idx]
    results['CLIP_MULTIMODAL'] = train_and_evaluate(X_train_clip_multi, y_train, X_test_clip_multi, y_test, 'CLIP Multimodal')

    # 5-4. ResNet + BERT 
    X_resnet = features['resnet_img']
    X_bert = features['bert_txt']
    X_resnet_bert = np.hstack([X_resnet, X_bert])
    X_train_resnet_bert = X_resnet_bert[X_train_idx]
    X_test_resnet_bert = X_resnet_bert[X_test_idx]
    results['RESNET_BERT_MULTIMODAL'] = train_and_evaluate(X_train_resnet_bert, y_train, X_test_resnet_bert, y_test, 'ResNet+BERT Multimodal')

    
    # --- 6. Result ---
    print("\n\n=============== Experiment Results (Log Scale 기준) ===============")
    results_df = pd.DataFrame.from_dict(results, orient='index')
    print(results_df.to_markdown())
    print("=================================================================")
    print("MAPE(%)는 log transformed original viewcount based.")
    
    print("\n--- predict result (Log Scale) ---")
    
    # 1. CLIP
    res_clip = results['CLIP_MULTIMODAL']
    plot_parity_and_residual(res_clip['y_test_log'], res_clip['y_pred_log'], 'CLIP_MULTIMODAL')
    
    # 2. ResNet+BERT 
    res_resnet_bert = results['RESNET_BERT_MULTIMODAL']
    plot_parity_and_residual(res_resnet_bert['y_test_log'], res_resnet_bert['y_pred_log'], 'RESNET_BERT_MULTIMODAL')


if __name__ == "__main__":
    main_experiment()