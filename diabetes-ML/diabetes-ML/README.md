🩺 Predictive Intelligence in Diabetes Diagnosis 

🔗 Veri Seti Linki (Kaggle) Kullanılan veri setine ve detaylarına Kaggle üzerinden erişebilirsiniz:

Pima Indians Diabetes Database : https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?select=diabetes.csv 

🚀 Kurulum ve Ortam Hazırlığı
Projenin taşınabilirliği ve farklı bilgisayarlarda aynı sonuçları vermesi için version pinning (sürüm sabitleme) uygulanmıştır.

Terminalinizi açın ve proje ana dizinine gidin.

Gerekli tüm kütüphaneleri (XGBoost, Scikit-learn, Imbalanced-learn vb.) kurmak için şu komutu çalıştırın:

Bash
pip install -r requirements.txt
Projeyi çalıştırmak için:

Bash
python main.py
🛠️ Teknik Mimari ve Mühendislik Kararları
Proje, tıbbi teşhis doğruluğunu maksimize etmek amacıyla şu 4 kritik sütun üzerine inşa edilmiştir:

1. Modüler Klasör Yapısı
Proje, profesyonel standartlarda modüler bir mimariye sahiptir:

src/preprocessing.py: Sahte sıfır değerlerinin temizlenmesi (IterativeImputer), outlier yönetimi (IQR Capping) ve RobustScaler dönüşümleri.

src/model.py: SMOTE ile veri dengeleme ve Hybrid Stacking (RF + XGB + ANN) mimarisi.

src/evaluation.py: Tıbbi metriklerin (Recall, MCC, Log-Loss) analizi ve görsel çıktı yönetimi.

main.py: Tüm pipeline'ı yöneten orkestra şefi.

2. Akıllı Veri Onarımı & Temizliği
Veri setinde biyolojik olarak imkansız olan "0" değerleri (Glikoz, Kan Basıncı vb.) tespit edilmiştir.

IterativeImputer: Eksik veriler basit ortalamalar yerine, diğer özelliklerden yola çıkılarak istatistiksel tahminlerle doldurulmuştur.

IQR Capping: Uç değerler silinmemiş, modelin dengesini bozmaması için en mantıklı sınırlara baskılanmıştır.

3. Özellik Mühendisliği (Feature Engineering)
Modelin tıbbi sezgilerini güçlendirmek için ham veriden yeni anlamlar türetilmiştir:

Metabolic_Ratio: Glikoz ve İnsülin dengesini temsil eden oran.

Risk_Score: Yaş ve BMI etkileşimini ölçen özel puanlama.

High_Glucose: Kritik tıbbi eşiklerin (140 mg/dL) dijitalleştirilmesi.

4. Hybrid Stacking & SMOTE
SMOTE: Diyabetli vakaların azlığından kaynaklanan yanlılığı gidermek için sentetik veri üretilerek eğitim seti dengelenmiştir.

Stacking: Random Forest, XGBoost ve Yapay Sinir Ağları (ANN) uzmanlar grubu olarak kullanılmış; son kararı bu modellerin tahminlerini yorumlayan bir Meta-Model (Logistic Regression) vermiştir.

🧠 Design Choices & FAQ (Teknik Kararlar)
Neden Threshold (Eşik Değeri) 0.36 Olarak Belirlendi?

Problem: Tıbbi teşhislerde bir hastayı "sağlıklı" sanıp eve göndermek (False Negative), sağlıklı birine "kontrol olmalısın" demekten (False Positive) çok daha risklidir.

Çözüm: Eşik değerini 0.50'den 0.36'ya çekerek modelin "şüphecilik" düzeyini artırdık. Bu sayede kaçan hasta sayısını 27'den 12'ye düşürerek Recall oranını zirveye taşıdık.

Neden RobustScaler Tercih Edildi?

İnsülin gibi değişkenlerde uç değerler (outliers) yoğun olduğu için, bu değerlere karşı dirençli olan ve medyan bazlı çalışan RobustScaler kullanılarak modelin matematiksel stabilitesi korunmuştur.

Neden Stacking Mimarisi?

Tek bir algoritmanın kör noktalarını, farklı çalışma mantıklarına sahip (Ağaç tabanlı ve Nöral tabanlı) modelleri birleştirerek kapattık. Bu sayede model daha stabilize ve genelleme yeteneği yüksek bir hale geldi
