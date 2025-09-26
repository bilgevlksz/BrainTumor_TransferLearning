📂 Veri Seti Klasörlerinin İncelenmesi
Bu hücrede, beyin tümörü MRI veri setinin dizin yapısı kontrol edilmektedir.

base_dir değişkeni ile veri setinin temel yolu tanımlanır.
os.listdir() fonksiyonu kullanılarak ana klasördeki alt klasörler listelenir.
Daha sonra Training (eğitim) ve Testing (test) klasörlerinin içerikleri yazdırılır.
Bu sayede, veri setindeki sınıflar (örneğin: glioma, meningioma, pituitary, no_tumor) görüntülenebilir.

🧠 Eğitim Verilerinin Sınıf Dağılımı
Bu hücrede, Training klasöründeki her bir sınıfa ait görüntü sayısı hesaplanmaktadır.

os.listdir(train_dir) ifadesi ile her bir sınıf (örneğin: glioma, meningioma, pituitary, no_tumor) listelenir.
glob.glob() fonksiyonu, her sınıf klasöründeki tüm görüntü dosyalarını bulur.
Her sınıfın sahip olduğu görüntü sayısı ekrana yazdırılır.
Bu sayede, veri setinin dengesiz (imbalanced) olup olmadığını hızlıca görebiliriz.



🖼️ Eğitim Verilerinden Örnek Görsellerin Görselleştirilmesi
Bu hücrede, Training klasöründeki her bir sınıftan rastgele birer görüntü seçilerek görselleştirme yapılmaktadır.

os.listdir(train_dir) ile sınıf isimleri alınır.
random.choice() fonksiyonu ile her sınıftan rastgele bir görüntü seçilir.
matplotlib kütüphanesi kullanılarak her bir sınıfın örnek görseli 2x2 subplot halinde gösterilir.
Görsellerin başlıklarında sınıf isimleri yer alır.
Bu görselleştirme, veri setinin yapısını ve sınıflar arasındaki görsel farklılıkları anlamak için faydalıdır.


📂 Veri Seti Özeti: Dosya Yolları ve Sınıf Dağılımı
Bu hücrede, veri setinin genel yapısı incelenmektedir.
Amaç: Tüm klasörleri gezerek her bir görselin dosya yolunu ve ait olduğu sınıf etiketini bir tabloya (DataFrame) kaydetmek.

os.listdir(base_dir) ile temel dizindeki alt klasörler (sınıflar) bulunur.
glob.glob() ile her sınıftaki tüm görüntü dosyalarının yolları alınır.
Her görüntünün dosya yolu (path) ve sınıf etiketi (label) ayrı listelere eklenir.
Bu bilgiler bir Pandas DataFrame içine aktarılır.
Son olarak, value_counts() ile her sınıfın kaç adet görüntüye sahip olduğu ekrana yazdırılır.
Bu sayede, veri setinin dengeli (balanced) olup olmadığı analiz edilir.

⚙️ Görsel Verilerin Ön İşlenmesi ve Generator’ların Oluşturulması
Bu hücrede, model eğitimi için kullanılacak olan veri artırma (data augmentation) ve normalize etme işlemleri uygulanmaktadır.
Amaç: Görselleri modele uygun formata dönüştürmek ve eğitim sürecinde çeşitlilik sağlamak.

🔹 1. ImageDataGenerator Kullanımı
rescale=1./255: Piksel değerlerini [0,1] aralığına dönüştürür.
rotation_range=15: Görselleri rastgele 15 dereceye kadar döndürür.
width_shift_range & height_shift_range: Görselleri hafifçe kaydırır.
zoom_range=0.1: Görselleri %10 oranında yakınlaştırır.
horizontal_flip=True: Görselleri yatay olarak çevirir.
validation_split=0.15: Eğitim verisinin %15’ini doğrulama (validation) için ayırır.
🔹 2. Train / Validation Generator’ları
train_gen: Eğitim verilerini üretir.
val_gen: Doğrulama verilerini üretir.
Her iki generator da aynı train_dir dizininden, farklı subset parametresiyle veri çeker.
🔹 3. Test Generator
test_gen: Test verilerini üretir.
Sadece normalize işlemi yapılır, augmentation uygulanmaz.
shuffle=False sayesinde test verilerinin sırası korunur.
Bu şekilde, veri kümesi eğitim, doğrulama ve test olmak üzere üç parçaya ayrılmış olur.



🧠 Transfer Learning: EfficientNetB0 Temel Modelinin Yüklenmesi
Bu hücrede, ImageNet üzerinde önceden eğitilmiş (pretrained) bir EfficientNetB0 modeli yüklenmektedir.

🔹 Amaç
Mevcut beyin tümörü MRI veri setinde, güçlü bir ön öğrenme (feature extraction) yeteneğine sahip EfficientNetB0 modelini kullanmak.
Transfer Learning sayesinde, sınırlı veriyle yüksek doğruluk elde etmek.
🔹 Açıklamalar
weights='imagenet': Model, ImageNet veri setinde önceden öğrenilmiş ağırlıkları kullanır.
include_top=False: Modelin sonundaki tam bağlı (fully connected) katman çıkarılır, böylece kendi sınıflarımıza uygun yeni katmanlar eklenebilir.
input_shape=(IMG_SIZE, IMG_SIZE, 3): Giriş boyutu 224x224 RGB görüntülerdir.
Bu aşamada modelin özellik çıkarıcı (feature extractor) kısmı alınır; daha sonra üzerine yeni katmanlar eklenerek kendi sınıflarımız için özelleştirme yapılacaktır.


🧠 EfficientNetB0 ile Transfer Learning ve Fine-Tuning
Bu hücrede, EfficientNetB0 tabanlı bir derin öğrenme modeli kurulmuş ve iki aşamalı eğitim stratejisi uygulanmıştır.

🔹 1. Modelin Kurulumu
EfficientNetB0 modeli, ImageNet ağırlıkları ile yüklendi.
include_top=False parametresiyle son sınıflandırma katmanı çıkarıldı.
Üzerine yeni katmanlar eklendi:
GlobalAveragePooling2D(): Özellik haritalarını sıkıştırır.
Dropout(0.3): Aşırı öğrenmeyi (overfitting) önler.
Dense(..., activation='softmax'): Sınıf sayısına göre çıktı katmanı.
🔹 2. Feature Extraction (Özellik Çıkarımı) Aşaması
base_model katmanları donuk (trainable=False) hale getirildi.
Sadece eklenen yeni katmanlar eğitildi.
Optimizasyon: Adam(learning_rate=1e-4)
Kayıp fonksiyonu: categorical_crossentropy
Metrik: accuracy
Bu aşama, modelin genel özellik çıkarma yeteneğini koruyarak, yeni veri setine uyum sağlamasını sağlar.

🔹 3. Callbacks
EarlyStopping: Doğrulama kaybı artarsa eğitimi durdurur, en iyi ağırlıkları geri yükler.
ReduceLROnPlateau: Modelin gelişimi durduğunda öğrenme oranını düşürür.
🔹 4. Fine-Tuning (İnce Ayar) Aşaması
base_model’in son 20 katmanı tekrar açıldı (trainable=True).
Küçük öğrenme oranı (1e-5) ile ikinci bir eğitim turu yapıldı.
Bu aşama, modelin yüksek seviyeli özelliklerini veri setine göre optimize eder.

🔹 5. Değerlendirme
Test verisi ile model.evaluate() kullanılarak genel performans ölçüldü.
Sonuç: Test Accuracy (%) ekrana yazdırılır.
Bu iki aşamalı yaklaşım (önce özellik çıkarımı, sonra ince ayar), sınırlı veride daha dengeli ve yüksek doğrulukta sonuçlar verir 🚀

🧠 EfficientNetB0 ile Görüntü Sınıflandırma Modeli
Bu notebook’ta, önceden eğitilmiş EfficientNetB0 mimarisi kullanılarak bir görüntü sınıflandırma modeli oluşturulmaktadır.
Aşamalar:

Hyperparametrelerin tanımlanması
Görsel veri jeneratörlerinin oluşturulması
Sınıf ağırlıklarının hesaplanması
Modelin tanımlanması ve derlenmesi
Callbacks tanımlanması
İlk eğitim (base model donuk)
Fine-tuning (bazı katmanlar açılarak)
Değerlendirme ve metriklerin hesaplanması
add Codeadd Markdown
1️⃣ Hiperparametreler
Modelde kullanılacak temel ayarlar burada belirlenir:

IMG_SIZE: Görsellerin yeniden boyutlandırılacağı hedef boyut
BATCH_SIZE: Eğitimde kullanılacak mini-batch boyutu
EPOCHS_INITIAL ve EPOCHS_FINETUNE: Eğitim döngü sayıları
add Codeadd Markdown
2️⃣ Görsel Jeneratörlerinin Hazırlanması
ImageDataGenerator ile veriler normalize edilir.
Eğitim verileri için veri artırma (augmentation) teknikleri uygulanır.
Validation ve test verileri sadece normalize edilir.
add Codeadd Markdown
3️⃣ Sınıf Ağırlıklarının Hesaplanması
Veri dengesizliği varsa, class_weight parametresi ile azınlık sınıflarına daha fazla ağırlık verilir.

4️⃣ Modelin Oluşturulması
EfficientNetB0, ImageNet ağırlıklarıyla yüklenir.
Son katmanlar yerine, kendi problemimize uygun Dense katman eklenir.
İlk etapta temel model (base_model) dondurulur.
5️⃣ Callbacks Tanımlama
EarlyStopping: Aşırı öğrenmeyi engeller.
ReduceLROnPlateau: Validasyon kaybı iyileşmiyorsa öğrenme oranını düşürür.
7️⃣ Fine-Tuning (Son Katmanların Açılması)
Base modelin son 20 katmanı eğitime dahil edilir.
Daha düşük öğrenme oranı ile hassas ayar yapılır.
8️⃣ Değerlendirme
Modelin test seti üzerindeki performansı ölçülür:

Test doğruluğu
Sınıflandırma raporu
ROC-AUC skoru
Karışıklık matrisi (Confusion Matrix)



🎉🎉🎉🎉🎉KAGGLE LİNKİ AŞAĞIDADIR

https://www.kaggle.com/code/bilgeevleksiz/braintumor-transferlearning





