from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def balance_and_train(X_train, y_train):
    """
    SMOTE ile veriyi dengeler ve Hibrit Stacking (RF + XGB + ANN) mimarisini eğitir.
    """
    # 1. Veri Dengeleme (Diyabetli vakaları sentetik olarak artırıyoruz)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    # 2. Baz Modeller (Uzman Ekibi)
    # Random Forest: Ağaç sayısını 500'e çıkardık, daha derin ve detaylı analiz yapar.
    rf = RandomForestClassifier(
        n_estimators=500, 
        max_depth=15, 
        min_samples_leaf=2, 
        class_weight='balanced_subsample', 
        random_state=42
    )
    
    # XGBoost: Öğrenme hızını düşürdük (0.05 -> 0.03) ve ağaç sayısını artırdık, daha hassas öğrenir.
    xgb = XGBClassifier(
        n_estimators=400, 
        max_depth=7, 
        learning_rate=0.03, 
        subsample=0.8, 
        eval_metric='logloss', 
        random_state=42
    )
    
    # Yapay Sinir Ağı (ANN): Doğrusal olmayan karmaşık ilişkileri yakalaması için eklendi.
    ann = MLPClassifier(
        hidden_layer_sizes=(64, 32), 
        max_iter=1000, 
        activation='relu', 
        solver='adam', 
        random_state=42
    )
    
    # 3. Meta-Model (Başhekim - Stacking)
    # Tüm modellerin tahminlerini alıp son kararı Lojistik Regresyon verir.
    base_models = [
        ('rf', rf),
        ('xgb', xgb),
        ('ann', ann)
    ]
    
    stack_model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        cv=5,
        passthrough=False # Sadece modellerin tahminlerine odaklanması için
    )
    
    print("🧠 Hibrit Stacking Modeli eğitiliyor (RF + XGB + ANN)...")
    stack_model.fit(X_res, y_res)
    
    # Feature Importance çizimi için RF'i ayrıca eğitip döndürüyoruz
    rf_standalone = rf.fit(X_res, y_res)
    
    return {
        "Stacking Model (Final)": stack_model,
        "Random Forest": rf_standalone
    }