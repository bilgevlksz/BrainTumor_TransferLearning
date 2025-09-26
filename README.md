📂 Veri Seti Klasörlerinin İncelenmesi Bu hücrede, beyin tümörü MRI veri setinin dizin yapısı kontrol edilmektedir.

base_dir değişkeni ile veri setinin temel akışını sağlar. os.listdir() fonksiyonu kullanılarak ana klasördeki alt klasörler listelenir. Daha sonra Training (eğitim) ve Testing (test) klasörlerinin içerikleri yazdırılır. Bu sayede, veri setindeki sınıflar (örneğin: glioma, meningioma, hipofiz, no_tumor) görüntülenebilir.

🧠 Eğitim Verilerinin Sınıf Dağılımı Bu hücrede, Eğitim birimlerindeki her bir sınıfa ait görüntü sayıları hesaplanmaktadır.

os.listdir(train_dir) ifadesi ile her bir sınıf (örneğin: glioma, menenjiyom, hipofiz, no_tümör) listelenir. glob.glob() fonksiyonu, her sınıfın içindeki tüm görüntüleri bulur. Her sınıfın sahip olduğu görüntü sayısı yazdırılır. Bu sayede veri setinin dengesiz (dengesiz) olup olmadığını kısaca göremiyoruz.

🖼️ Eğitim Verilerinden Örnek görsellerin görselleştirilmesi Bu hücrede, Eğitim bölümündeki her sınıftan rastgele bir görüntü seçilerek görselleştirme yapılmaktadır.

os.listdir(train_dir) ile sınıf isimleri alınır. random.choice() fonksiyonu ile her sınıftan rastgele bir görüntü seçilir. matplotlib yapılandırması kullanılarak her sınıfta görseli 2x2 alt grafik halinde gösterilir. hikayelerin başlıklarında sınıf adlarını yer alır. Bu görselleştirme, veri setinin taşınması ve sınıflar arasındaki görsel farklılıkların işlenmesi için faydalıdır.

📂 Veri Seti Özeti: Dosya Yolları ve Sınıf Dağılımı Bu hücrede, veri setinin genel yapısı incelenmektedir. Amaç: Tüm klasörleri bir görsel dosya yolunu gezerek ve ait olduğu sınıf etiketini bir tabloya (DataFrame) yazdırın.

os.listdir(base_dir) ile temel dizindeki alt klasörler (sınıflar) bulunur. glob.glob() ile her sınıftaki tüm görüntü kenarlarının yolları alınır. Her fotoğraf dosyası yolu (yol) ve sınıf etiketi (etiket) ayrı listelere eklenir. Bu bilgiler bir Pandas DataFrame içine aktarılır. Son olarak, value_counts() ile her sınıfın kaç adet görüntüye sahip olduğu ekranı yazdırılır. Bu sayede veri setinin dengeli (dengeli) olup olmadığı analiz edilir.

⚙️ Görsel Verilerin Ön İşlenmesi ve Jeneratörlerin Oluşturulması Bu hücrede, model eğitimi için kullanılacak olan doğrulama (veri artırma) ve normalleştirme işlemleri yapılmaktadır. Amaç: Görselleri modele uygun formata dönüştürmek ve eğitim sürecinde çeşitlilik sağlamak.

🔹 1. ImageDataGenerator kullanımı rescale=1./255: Piksel değerleri [0,1] aralığına eklenir. rotasyon_range=15: stratejiler rastgele 15 döngüye kadar. width_shift_range & height_shift_range: görselleri hafifçe kaydırır. zoom_range=0.1: görselleri %10 oranında yakınlaştırır. Horizontal_flip=True: Görselleri Extreme olarak çevirir. validation_split=0.15: Eğitim verisinin %15'ini çalıştırmak (validation) için ayırmak. 🔹 2. Train / Validation Generator'ları train_gen: Eğitimler gerçekleşir. val_gen: Doğrulama üretir. Her iki jeneratör de aynı train_dir dizininden, farklı alt küme parametreleriyle veri çeker. 🔹 3. Test Generator test_gen: Test parçaları üretir. Sadece normalleştirme işlemi yapılır, büyütme uygulanmaz. shuffle=False sayesinde test verilerinin sırası korunur. Bu şekilde veri ayarı eğitim, sürekli ve test olmak üzere üç parçaya ayrılır.

🧠 Transfer Öğrenme: EfficientNetB0 Temel Modelinin Yüklenmesi Bu hücrede, ImageNet üzerinde önceden eğitilmiş (önceden eğitilmiş) bir EfficientNetB0 modeli yüklenmektedir.

🔹 Amaç Mevcut beyin tümörü MRI veri setinde, güçlü bir ön öğrenme (özellik çıkarımı) yeteneğine sahip EfficientNetB0 adını kullanmak. Transfer Öğrenme sayesinde, sınırlanmış veriyle yüksek doğruluk elde etmek. 🔹 Açıklamalarweights='imagenet': Model, ImageNet veri setinde önceden öğrenilmiş ağırlıkları kullanır. include_top=False: Modelin sonundaki tam bağlı (tamamen bağlı) katman çıkarılabilir, böylece kendi sınıflarımıza uygun yeni katmanlar eklenebilir. input_shape=(IMG_SIZE, IMG_SIZE, 3): Giriş boyutu 224x224 RGB görüntülerdir. Bu aşamada modelin özellik çıkarıcı (özellik çıkarıcı) kısmı alınır; daha sonra üzerine yeni katmanlar eklenerek kendi sınıflarımız için kişiselleştirme yapılabilir.

🧠 EfficientNetB0 ile Transfer Öğrenme ve İnce Ayar Bu hücrede, EfficientNetB0 tabanlı bir derin öğrenme modeli kurulmuş ve iki öğrenim eğitim sistemi hazırlanmıştır.

🔹 1. Modelin Kurulumu EfficientNetB0 modeli, ImageNet ağırlıkları ile yüklendi. include_top=Yanlış parametresiyle sonuncusu taramaları kaldırıldı. Üzerine yeni katmanlar eklenir: GlobalAveragePooling2D(): Özel haritalarını sıkıştırır. Dropout(0.3): Aşırı öğrenmeyi (aşırı uyum) önler. Yoğun(..., aktivasyon='softmax'): Sınıftaki parçalara göre genişletilmiş katman. 🔹 2. Özellik Çıkarımı (Özellik Çıkarımı) Aşaması base_model katmanları donuk (trainable=False) hale getirilir. Sadece miktarı yeni katmanlar içerirdi. Optimizasyon: Adam(learning_rate=1e-4) Kayıp fonksiyonu: categorical_crossentropy Metrik: doğruluk Bu aşama, modelin genel özelliklerini çıkarma kabiliyetini koruyarak, yeni veri setine uyumunu sağlamayı sağlar.

🔹 3. Erken Durdurma: Doğrulama kaybı artarsa ​​antrenman durur, en iyi ağırlıklar geri yüklenir. ReduceLROnPlateau: Modelin gelişimindeki gelişim bozukluklarında eksiklikler. 🔹 4. İnce Ayar (İnce Ayar) Aşaması base_model'in son 20 katmanı tekrar açıldı (trainable=True). Küçük öğrenme oranı (1e-5) ile ikinci bir eğitim turu yapıldı. Bu aşamada, modelin yüksek seviyeli özellikleri veriye göre optimize edilir.

🔹 5. Değerlendirme Test verisi ile model.evaluate() kullanılarak genel performans ölçüldü. Sonuç: Test Doğruluğu (%) yazdırılır. Bu iki tedavi yaklaşımı (önce özellik çıkarımı, sonra ince ayar), sınırlı veride daha gelişmiş ve yüksek doğrulukta sonuçlar verir 🚀

🧠 EfficientNetB0 ile Görüntü Sınıflandırma Modeli Bu notebook'ta, sınıflandırılmış EfficientNetB0 mimari kullanılarak bir görüntü görünümü modeli oluşturulmaktadır. Aşamalar:

Hiperparametrelerin görsel verileri jeneratörlerinin geniş sınıf ağırlıklarının değişimi Modelin değişimleri ve derlenmesi Geri aramaların İlk eğitim (temel model donuk) İnce ayar (bazı katmanlar açılarak) değerlendirilmesi ve metriklerin parçacıklarının eklenmesi Codeadd Markdown 1️⃣ Hiperparametreler Modelde yapılacak temel ayarlar burada belirlenir:

IMG_SIZE: görsellerin yeniden boyutlandırılacağı hedef boyut BATCH_SIZE: Eğitimde kullanılacak mini-batch boyutu EPOCHS_INITIAL ve EPOCHS_FINETUNE: Eğitim döngüsü listesi add Codeadd Markdown 2️⃣ Görsel Jeneratörlerin Hazırlanması ImageDataGenerator ile veriler normalleştirilir. Eğitim verilerini doğrulamak için artırma (arttırma) teknikleri uygulanır. Doğrulama ve test verileri sadece normalleştirilir. add Codeadd Markdown 3️⃣ Sınıf Ağırlıklarının Hesaplanması Veri dengesizliği varsa, class_weight parametreleri ile azınlık sınıflarına daha fazla ağırlık verilir.

4️⃣ Modelin Oluşturulması EfficientNetB0, ImageNet ağırlıklarıyla yüklenir. Son katmanlar yerine, kendi problemimize uygun Yoğun katman ekler. İlk etapta temel model (base_model) dondurulur. 5️⃣ Geri Arama Tanımlama Erken Durdurma: Aşırı öğrenmeyi engeller. ReduceLROnPlateau: Validasyon kaybı iyileşmiyorsa öğrenme bozuklukları azalır. 7️⃣ İnce Ayar (Son Katmanların Açılması) Temel modelin son 20 katman eğitime dahil edilir. Daha düşük öğrenme oranı ile hassas ayar yapılır. 8️⃣ Değerlendirme Modelin test seti üzerindeki performansı:

Test değişimi Sınıflandırma performansı ROC-AUC skoru Karışıklık matrisi (Confusion Matrix)

🎉🎉🎉🎉🎉KAGGLE LİNKİ AŞAĞIDADIR

https://www.kaggle.com/code/bilgeevleksiz/braintumor-transferlearning
