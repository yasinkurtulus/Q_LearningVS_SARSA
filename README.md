# Q-Learning, SARSA ve Deep Q-Network

Pekiştirmeli öğrenmeyi iki ayrı örnekle inceleyen Python projesi:

| Ortam | Algoritma | Durum temsili | Değerlerin tutulması |
|---|---|---|---|
| 6 × 6 GridWorld | Q-learning ve SARSA | Izgara konumu | Q tablosu |
| MountainCar-v0 | Standart DQN (Vanilla DQN) | Konum ve hız | Sinir ağı |

## MountainCar: kendi DQN uygulamamız

`mountaincar_dqn.py`, Gymnasium'un `MountainCar-v0` ortamında DQN eğitimini
hazır bir RL kütüphanesi kullanmadan uygular. PyTorch yalnızca sinir ağı,
otomatik türev ve optimizer için kullanılır. Mevcut GridWorld ayrı çalışır.

Proje klasöründe önce sanal ortamı oluşturun (mevcut `.venv` varsa ilk
komutu atlayın), ardından bağımlılıkları tek bir terminalde kurun:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dqn.txt
```

Önce modeli eğitin; değerlendirme komutları kaydedilmiş modeli gerektirir:

```powershell
.venv\Scripts\python.exe mountaincar_dqn.py --episodes 1000
.venv\Scripts\python.exe mountaincar_dqn.py --evaluate --episodes 20
.venv\Scripts\python.exe mountaincar_dqn.py --evaluate --render --episodes 5
```

### Ortam ve varsayılan ayarlar

Gözlem `[konum, hız]` biçimindedir; konum aralığı `[-1.2, 0.6]`, hız aralığı
`[-0.07, 0.07]` olur. Eylemler `0: sola kuvvet`, `1: motor kuvveti yok`,
`2: sağa kuvvet` şeklindedir. Amaç `konum >= 0.5` noktasındaki hedefe ulaşmaktır.

| Ayar | Varsayılan değer |
|---|---|
| Ağ | 2 → 128 → 128 → 3, gizli katmanlarda ReLU |
| Optimizer / öğrenme oranı | Adam / 0.0005 |
| Kayıp | Huber (Smooth L1) |
| Gamma | 0.99 |
| Replay buffer / batch | 50.000 / 64 deneyim |
| Eğitim başlangıcı | 1.000 deneyim |
| Eğitim sıklığı | Her 4 ortam adımı |
| Target ağı kopyalama | Her 1.000 ortam adımı |
| Epsilon | İlk 80.000 adımda 1.0'dan 0.05'e doğrusal azalma |
| Değerlendirme epsilon'u | 0 (greedy) |
| Tohum / bölüm sayısı | 42 / 1.000 |

`return`, bölümdeki ödüllerin toplamıdır. `-200`, 200 adım atıldığını gösterir;
başarı durumunu CSV'deki `success` sütunu belirtir. Daha az negatif return,
hedefe daha az adımda ulaşıldığında elde edilir. `mean100` son en fazla 100
bölümün ortalamasıdır. Loss'un azalması tek başına oyunun öğrenildiğini göstermez.

### Öğrenme akışı

Kodun okuma sırası:

1. `QNetwork`: iki sayı (konum, hız) -> 128 -> 128 -> üç Q değeri.
2. `choose_action`: epsilon-greedy ile sol, hızlanma yok veya sağ seçimi.
3. `ReplayBuffer`: `(s, a, r, s_next, terminated)` kaydı ve rastgele örnekleme.
4. `bellman_targets`: `y = r + gamma * (1 - terminated) * max Q_target(s_next)`.
5. `train_step`: seçilen eylemin `Q(s,a)` tahmini ile hedef arasında Huber
   loss; geri yayılım yalnızca ana ağa yapılır.
6. `train`: deneyim toplama, her dört adımda eğitim, her 1000 ortam adımında
   target ağına kopyalama. İlk 1000 deneyim eğitimden önce toplanır.
7. `evaluate`: kaydedilmiş modeli keşif olmadan, ayrı başlangıç tohumlarıyla ölçme.

Ödül değiştirilmez: her adım -1. Hedefe ulaşmak `terminated`, 200 adımlık
süre sınırı `truncated` olur. İkisinde de bölüm biter; yalnızca gerçek terminal
durumda gelecekteki Q katkısı sıfırlanır. Hedef ağ optimal cevabı bilmez;
ana ağın daha seyrek güncellenen kopyasıdır.

`runs/mountaincar/training.csv` bölüm ödüllerini, kaybı ve başarıyı;
`q_network.pt` son modelin ağırlıklarını saklar. Aynı output klasöründeki yeni
eğitim bu çıktıları yeniler; ayrı deneyler için `--output runs/deney2` kullanın.
Bu kayıt eğitim devam ettirme checkpoint'i değildir (optimizer/bellek saklanmaz).
Kısa çalışma testi öğrenme başarısını göstermez. MountainCar keşif açısından
zordur; 1000 bölüm başarı garantisi değildir. Greedy değerlendirmedeki ortalama
ödül ve hedefe ulaşma sayısını kontrol edin. Eğitim CPU üzerinde çalışır.

Kaynaklar: [MountainCar ortamı](https://gymnasium.farama.org/environments/classic_control/mountain_car/),
[DQN mekanizması](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).

### Doğrulama

```powershell
.venv\Scripts\python.exe -m unittest test_mountaincar_dqn.py
.venv\Scripts\python.exe mountaincar_dqn.py --episodes 2 --warmup 64 --output runs/smoke_test
.venv\Scripts\python.exe mountaincar_dqn.py --evaluate --episodes 1 --output runs/smoke_test
```

Testler, her deneyim için ayrı maksimum Q seçildiğini, terminal maskesini ve
geri yayılımın target ağı değiştirmediğini kontrol eder. Kısa çalışma denemesi
eğitim/kaydetme/yükleme akışını doğrular; performans ölçümü değildir.

## GridWorld: Q-learning ve SARSA karşılaştırması

`gridworld.py`, aynı 6 × 6 GridWorld ortamında **Q-learning** ve **SARSA** algoritmalarını eğitir ve sonuçlarını karşılaştırır. Eğitim sonunda her iki algoritmanın Q tablosu ve bu tablolardan elde edilen greedy politika terminale yazdırılır; ayrıca eğitim boyunca Q tablolarının değişimini gösteren bir animasyon açılır.

### Son eğitim Q tabloları

Aşağıdaki görsel, 42 rastgelelik tohumu (`seed=42`) ile 5000 bölüm eğitimden sonra elde edilen Q tablolarını gösterir:

![SARSA ve Q-learning algoritmalarının son Q tabloları](images/q-tables.png)

Tablolarda her satır bir durumu, sütunlar ise sırasıyla **yukarı, aşağı, sol ve sağ** eylemlerini temsil eder. Durum indeksi `satır × 6 + sütun` biçiminde hesaplanır. Hücrelerdeki değerler, ilgili durum-eylem çiftinin öğrenilmiş Q değerleridir.

### Ortam ve ödüller

- Izgara boyutu: **6 × 6**
- Başlangıç: **(0, 0)**
- Hedef: **(5, 5)**
- Engeller: `(1, 1)`, `(1, 2)`, `(2, 2)`, `(3, 3)`, `(4, 4)`, `(2, 5)`, `(5, 2)`, `(1, 4)`, `(3, 0)`
- Normal bir adım: **−1**
- Hedefe ulaşma: **+10**
- Izgara dışına çıkmayı deneme: **−10** (ajan bulunduğu karede kalır)
- Engelli kareye girme: **−10** (ajan bu kareye girer)

### Algoritmalar

Her iki algoritma da eylem seçimi için **ε-greedy** politika kullanır. Kodda `ε = 0.1` olduğundan ajan çoğunlukla mevcut en iyi eylemi seçer, ancak keşif amacıyla zaman zaman rastgele bir eylem dener.

- **SARSA (on-policy):** Bir sonraki durumdaki, politikanın gerçekten seçtiği eylemin Q değeriyle güncellenir:  
  `Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') − Q(s,a)]`
- **Q-learning (off-policy):** Bir sonraki durum için mümkün olan en yüksek Q değerini kullanır:  
  `Q(s,a) ← Q(s,a) + α [r + γ maxₐ' Q(s',a') − Q(s,a)]`

Hedef durum için güncelleme hedefi doğrudan alınan ödüldür. Öğrenme oranı `α = 0.1`, indirim katsayısı `γ = 0.9` ve eğitim uzunluğu algoritma başına **5000 bölüm** olarak ayarlanmıştır.

### Gereksinimler ve çalıştırma

Python 3 ve `numpy`, `matplotlib` paketleri gerekir:

```bash
python -m pip install numpy matplotlib
python gridworld.py
```

Program önce her algoritma için son Q tablosunu ve greedy politikayı terminale yazdırır, ardından eğitim sırasında kaydedilen Q tablolarını karşılaştıran bir animasyon penceresi açar. Pencereyi kapatmak için grafik penceresini kapatın.

## Proje dosyaları

- [`gridworld.py`](gridworld.py): GridWorld ortamı, SARSA ve Q-learning eğitimi, politika çıktısı ve animasyon.
- [`mountaincar_dqn.py`](mountaincar_dqn.py): DQN ağı, replay buffer, eğitim ve değerlendirme.
- [`requirements-dqn.txt`](requirements-dqn.txt): MountainCar DQN bağımlılıkları.
- [`test_mountaincar_dqn.py`](test_mountaincar_dqn.py): DQN hedefi ve gradyan doğruluk testleri.
- [`images/q-tables.png`](images/q-tables.png): Son eğitim Q tablolarının karşılaştırmalı görseli.

Sanal ortam, Python önbelleği ve `runs/` altındaki eğitim çıktıları Git'e eklenmez.
