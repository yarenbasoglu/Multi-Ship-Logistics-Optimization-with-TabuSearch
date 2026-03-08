# Gemi Rota Planlama (Rolling Horizon)

Bu proje, çok gemili dağıtım planlamasını `data_ship.xlsx` verisi ile yapar.

Model, günlük kayan pencere ile çalışır:
- Gün 1: `[1..10]`
- Gün 2: `[2..11]`
- Gün 3: `[3..12]`
- ...

Amaç:
- Müşteri taleplerini zaman penceresi içinde karşılamak
- Depo stoklarını kritik eşik altına düşürmemek
- Uygun yerlerde iş birleştirme yapmak

## Veri Kaynağı

Ana veri dosyası:
- `data_ship.xlsx`

Kullanılan ana sayfalar:
- `Musteri_Talepleri`
- `Gun_Maliyet`
- `Kapasite`
- `Mesguliyetler`
- `Max Stock`
- `Initial Stock`
- `Safety Stock`
- `Tuketim`
- `Rafineri Talebi`

## Algoritma Akışı

`adimlar.py` içindeki aktif akış:

1. Event-driven simulation
- Her gün stok tüketimi düşülür.
- Kritik depo eşikleri kontrol edilir.
- Acil müşteri işleri öne alınır.

2. Time-window scheduling
- Müşteri işleri `±2` pencere ile değerlendirilir.
- İş birleştirme adayları pencere, seyir farkı ve rota kurallarına göre seçilir.

3. Backward scheduling
- İşler mümkün olan en geç uygun güne yerleştirilir.
- Erken günler boş bırakılarak sonraki işler için esneklik korunur.

4. Greedy/selection mantığı (aktif kullanımda)
- Feasible rota-gemi seçenekleri çıkarılır.
- Seçim modu işe göre değişir (müşteri/depo).

5. Tabu search
- Bu dosyada aktif çözüm akışında kullanılmıyor.
- Rolling horizon çıktısı bu aşama olmadan üretiliyor.

## Rota Kullanımı

Mevcut kodda rota seti iki parçadan oluşur:
- Excel `Gun_Maliyet` rotaları (`X...`)
- Kodda üretilen ek rotalar:
  - sentetik müşteri birleştirme rotaları (`SYN_...`)
  - depo rotaları (`DIRECT_...`, `..._M6/_M7/_M8`)

Not: Sadece Excel rotalarıyla çalışmak istenirse bu ek rota üretimi kapatılmalıdır.

## Çalıştırma

Ortam:
- Python 3.10+ önerilir


## Çıktı

Kompakt modda (`COMPACT_CONSOLE_OUTPUT = True`) özet olarak:
- Günlük satır: pencere, kritik iş sayısı, müşteri/depo atama adedi, kuyruk
- Servis edilen işler
- Gemi kullanım özeti (kullanılan/boş gemiler, iş bazlı aralıklar)
- Depo kısa özeti

## Önemli Parametreler

`adimlar.py` içinde:
- `DEPOT_MIN_SAILING_DAYS`: depo kritik eşik hesaplarında seyir süresi katsayısı
- `MERGE_WAIT_LOOKAHEAD_DAYS`: birleştirme için ileri bakış günü
- `COMPACT_CONSOLE_OUTPUT`: kısa/uzun çıktı modu

## Not

Bu proje operasyonel planlama mantığına odaklıdır. Model davranışı, veri kalitesi ve rota setindeki kısıtlara doğrudan bağlıdır.
