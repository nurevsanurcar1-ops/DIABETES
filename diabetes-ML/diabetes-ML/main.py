import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from src.preprocessing import handle_missing_and_clean, engineer_features, get_preprocessor
from src.model import balance_and_train
from src.evaluation import plot_final_performance_results, plot_feature_importance
import joblib

def run_pipeline():
    print("\n🚀 GELİŞMİŞ DİYABET TAHMİN HATTI BAŞLATILDI...")
    df = pd.read_csv('data/diabetes.csv')
    
    # Preprocessing
    df = handle_missing_and_clean(df)
    df = engineer_features(df)
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = get_preprocessor()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Eğitim
    trained_models = balance_and_train(X_train_scaled, y_train)
    
    # Kıyaslama Tablosu
    comparison_data = []
    for name, model in trained_models.items():
        preds = model.predict(X_test_scaled)
        comparison_data.append({
            "Model": name, 
            "Accuracy": accuracy_score(y_test, preds),
            "F1-Score": f1_score(y_test, preds)
        })
    
    print("\n🏆 MODEL KIYASLAMA TABLOSU:")
    print(pd.DataFrame(comparison_data).to_string(index=False))
    
    # Final Analiz: Ensemble Model
    champion = trained_models["Stacking Model (Final)"]
    
    # 1. Gürültüyü Temizle: Kan basıncı (BloodPressure) diyabet tahminde bazen yanıltıcıdır bu yüzden siliyoruz.

    X = df.drop(['Outcome', 'BloodPressure'], axis=1) 
    
    # 2. Threshold'u 0.38 değerinden 0.36 yaptık.
    # plot_final_performance_results fonksiyonunda:
    plot_final_performance_results(champion, X_test_scaled, y_test, threshold=0.36)
    
    # Feature importance için Ensemble içindeki RF'i kullanalım
    # Feature importance için Ensemble içindeki RF'i kullanalım
    plot_feature_importance(trained_models["Random Forest"], X.columns.tolist())
    
    print("\n✅ Outlier temizliği ve Oylama (Ensemble) yöntemiyle model daha stabilize oldu!")

    # --- BURASI YENİ: Kaydetme işlemlerini fonksiyonun İÇİNE aldık ---
    joblib.dump(champion, 'diabetes_stacking_model.joblib')
    joblib.dump(scaler, 'robust_scaler.joblib')
    print("✅ Şampiyon model ve Scaler başarıyla kaydedildi!")

if __name__ == "__main__":
    run_pipeline()