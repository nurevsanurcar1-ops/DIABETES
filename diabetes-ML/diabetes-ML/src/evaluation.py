import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, matthews_corrcoef, brier_score_loss

def plot_final_performance_results(model, X_test, y_test, threshold=0.38):
    """
    Modelin hem temel hem de ileri seviye metriklerini analiz eder.
    """
    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = (y_probs >= threshold).astype(int)
    
    # 1. Temel Metrikler
    acc = accuracy_score(y_test, y_preds)
    report_dict = classification_report(y_test, y_preds, output_dict=True)
    report_text = classification_report(y_test, y_preds, target_names=['Sağlıklı', 'Diyabet'])
    
    # 2. İleri Seviye Analiz Metrikleri (İstediğin kısımlar)
    loss = log_loss(y_test, y_probs)
    mcc = matthews_corrcoef(y_test, y_preds)
    brier = brier_score_loss(y_test, y_probs)
    
    print("\n" + "="*50)
    print("🎯 ŞAMPİYON MODEL FİNAL PERFORMANS ANALİZİ")
    print("="*50)
    print(f"📈 Genel Doğruluk (Accuracy): %{acc*100:.2f}")
    print(f"🩺 Diyabeti Yakalama (Recall): %{report_dict['1']['recall']*100:.2f}")
    print(f"🚧 Eşik Değeri (Threshold): {threshold}")
    
    print("\n" + "-"*50)
    print("🧪 İLERİ SEVİYE ANALİZ METRİKLERİ")
    print("-"*50)
    print(f"📉 Log Loss (Hata Payı): {loss:.4f}")
    print(f"🧬 MCC (Korelasyon Katsayısı): {mcc:.4f}")
    print(f"🎯 Brier Score (Tahmin Kalitesi): {brier:.4f}")
    print("-" * 50)
    
    print("\n📄 DETAYLI SINIFLANDIRMA RAPORU:")
    print(report_text)
    print("="*50)
    
    # Görselleştirme: Confusion Matrix
    cm = confusion_matrix(y_test, y_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=['Tahmin: Sağlıklı', 'Tahmin: Diyabet'],
                yticklabels=['Gerçek: Sağlıklı', 'Gerçek: Diyabet'])
    plt.title(f'Final Karmaşıklık Matrisi\nAccuracy: %{acc*100:.2f}')
    plt.savefig('final_results_matrix.png')
    print("✅ 'final_results_matrix.png' oluşturuldu.")

def plot_feature_importance(model, feature_names):
    """
    En önemli faktörleri analiz eder ve görselleştirir.
    """
    importances = model.feature_importances_
    
    if len(importances) != len(feature_names):
        min_len = min(len(importances), len(feature_names))
        importances = importances[:min_len]
        feature_names = feature_names[:min_len]

    indices = np.argsort(importances)[::-1]

    print("\n🧐 DİYABETİ TETİKLEYEN EN ÖNEMLİ 3 FAKTÖR:")
    for i in range(min(3, len(indices))):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

    plt.figure(figsize=(10, 7))
    sns.barplot(
        x=importances[indices], 
        y=[feature_names[i] for i in indices], 
        hue=[feature_names[i] for i in indices],
        palette='magma',
        legend=False
    )
    
    plt.title('Diyabet Teşhisinde Belirleyici Özellikler', fontsize=15, fontweight='bold')
    plt.xlabel('Önem Skoru', fontsize=12)
    plt.ylabel('Vücut Değerleri', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    print("✅ 'feature_importance.png' başarıyla oluşturuldu.")