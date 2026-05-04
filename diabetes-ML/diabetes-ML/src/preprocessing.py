import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler

def remove_outliers_iqr(df):
    # Özellikle Insulin ve SkinThickness gibi çok sapan sütunları temizleyelim çünkü bu sütunlardaki aşırı değerler modelin performansını olumsuz etkileyebilir.
    cols = ['Glucose', 'BloodPressure', 'Insulin', 'BMI']
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Uç değerleri sınır değerlere baskılıyoruz (Capping)
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    return df

def handle_missing_and_clean(df):
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
    
    it_imputer = IterativeImputer(random_state=42)
    df[cols_to_fix] = it_imputer.fit_transform(df[cols_to_fix])
    
    # Outlier temizliğini imputer'dan sonra yapıyoruz
    df = remove_outliers_iqr(df)
    return df

def engineer_features(df):
    # 1. Glikoz Eşiği (Kritik değer 140)
    df['High_Glucose'] = (df['Glucose'] > 140).astype(int)
    
    # 2. BMI Kategorileri (Binning)
    # 0-18.5: Zayıf, 18.5-25: Normal, 25-30: Kilolu, 30+: Obez
    df['BMI_Cat'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(float)
    
    # 3. Mevcut skorlar
    df['Risk_Score'] = df['BMI'] * df['Age']
    df['Metabolic_Ratio'] = df['Glucose'] / (df['Insulin'] + 1)
    return df

def get_preprocessor():
    return RobustScaler() # Veriyi daha esnek ölçeklendirir ## StandardScaler yerine RobustScaler kullanıyoruz
                          # Bu sayede aykırı değerler (insülin 800 vb.) modelin dengesini bozamaz