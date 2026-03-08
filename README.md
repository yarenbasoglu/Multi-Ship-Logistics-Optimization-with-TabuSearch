# Gemi Rota Planlama (Rolling Horizon)

Bu proje günlük kayan pencere ile gemi ataması yapar ve aktif giriş dosyası `route_ship_final.py`'dir.

## Veri Kaynağı

Sadece aşağıdaki klasör kullanılır:
- `/Gemi-Rota/data/from_data_ship`

Kullanılan dosyalar:
- `Musteri_Talepleri.csv`
- `Gun_Maliyet.csv`
- `Kapasite.csv`
- `Mesguliyetler.csv`
- `Max_Stock.csv`
- `Initial_Stock.csv`
- `Safety_Stock.csv`
- `Tuketim.csv`
- `Rafineri_Talebi.csv`

Not:
- Eski `data_ship.xlsx` artık kullanılmaz.
- Planlama rotaları sadece `Gun_Maliyet.csv` içindeki rotalardır (`se` kolonu -> `X{se}`).
- Sentetik rota (`SYN_`) ve ek depo rota (`DIRECT_`, `*_M6/*_M7/*_M8`) aktif akışta kullanılmaz.

## Algoritma Akışı

1. Event-driven simulation
- Günlük stok/talep kontrolü yapılır, acil işler öne alınır.

2. Time-window scheduling
- Müşteri işleri `±2` gün penceresinde değerlendirilir, uygun paketleme denenir.

3. Backward scheduling
- İşler uygun olan en geç güne yerleştirilir.

## Çalıştırma

```bash
python3 /Gemi-Rota/route_ship_final.py
```

## Çıktı

Kompakt modda:
- Günlük özet satırı (`[DAY xx]`)
- Servis edilen işler
- Gemi kullanım özeti
- Depo kısa özeti

## Önemli Parametreler

`route_ship_final.py` içinde:
- `DEPOT_MIN_SAILING_DAYS`
- `MERGE_WAIT_LOOKAHEAD_DAYS`
- `COMPACT_CONSOLE_OUTPUT`
