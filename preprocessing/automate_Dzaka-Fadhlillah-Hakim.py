import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_data(filepath):
    """Load dataset dari file CSV."""
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset berhasil dimuat. Shape: {df.shape}")
    return df

def handle_zero_values(df):
    """Ganti nilai 0 yang tidak valid dengan median kolom."""
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for col in cols_with_zeros:
        df[col] = df[col].replace(0, np.nan)
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"[INFO] Kolom '{col}' — nilai 0 diganti dengan median: {median_val:.2f}")
    
    return df

def split_features_target(df, target_col='Outcome'):
    """Pisahkan fitur (X) dan target (y)."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    print(f"[INFO] Fitur: {list(X.columns)}")
    print(f"[INFO] Target: {target_col}")
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data menjadi train dan test set."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Train size: {X_train.shape}, Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """Scaling fitur menggunakan StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    print("[INFO] Scaling selesai.")
    return X_train_scaled, X_test_scaled

def save_data(X_train, X_test, y_train, y_test, output_dir='diabetes_preprocessing'):
    """Simpan hasil preprocessing ke folder output."""
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)
    
    print(f"[INFO] Data berhasil disimpan di folder '{output_dir}'")

def main():
    # Path ke dataset raw
    filepath = 'diabetes_raw.csv'
    
    # Jalankan pipeline preprocessing
    df = load_data(filepath)
    df = handle_zero_values(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    save_data(X_train_scaled, X_test_scaled, y_train, y_test)
    
    print("\n✅ Preprocessing selesai!")

if __name__ == "__main__":
    main()