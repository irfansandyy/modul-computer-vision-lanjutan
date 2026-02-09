<!-- markdownlint-disable MD060 -->

# Modul Computer Vision Lanjutan

## Daftar Isi

### Teori

- [Dasar](#dasar)
- [Data dan Sensor](#data-dan-sensor)
- [Preprocessing](#preprocessing)
- [Inference](#inference)
- [Postprocessing](#postprocessing)
- [Transformasi Koordinat 2D ke 3D](#transformasi-koordinat-2d-ke-3d)

### Implementasi

- [OpenCV](#opencv)
- [Roboflow](#roboflow)
- [Yolo](#yolo)
- [Alat Inference](#alat-inference)
- [Website Penting](#website-penting)
- [Belajar Lebih Lanjut](#belajar-lebih-lanjut)

## [Dasar](#dasar)

Section ini membahas penerapan computer vision pada Unmanned Aerial Vehicles (UAV) atau UAV. Fokus utama adalah pada teknik pengolahan citra dan visi yang digunakan untuk berbagai aplikasi UAV, seperti pemetaan, pengawasan, dan inspeksi.

### Garis Besar

```mermaid
---
title: Garis Besar Computer Vision untuk UAV
---
flowchart LR
    A["Sensor (Kamera)"]
    B["Preprocessing (OpenCV)"]
    C["Inference Model (Yolo dalam bentuk ONNX)"]
    D["Postprocessing (OpenCV / Yolo)"]
    E["Coordinate Transformation"]
    I["Mavlink Protocol"]
    J["Flight Controller (PX4/ArduPilot)"]
    F["Training Model (Yolo)"]
    G["Dataset (Gambar Berlabel)"]
    H["Labelling (Roboflow)"]
    A --> B --> C --> D --> E --> I --> J
    H --> G --> F --> C
```

### UAV Computer Vision

Computer vision pada UAV itu real-time dan terbatas sumber daya

#### Tabel Perbandingan Computer Vision Biasa dan UAV Computer Vision

| Aspek | CV Biasa | CV UAV |
|-------|----------|--------|
| Sumber Daya | Tak Terbatas | Fixed dan Terbatas Sumber Daya |
| Daya Listrik | Tak Terbatas | Terbatas Baterai |
| Latensi | Tidak Terbatas | Real-time / Low Latency |
| Kegagalan | Bisa Restart | Harus Tahan Gagal atau Crash |
| Lingkungan | Terkontrol | Dinamis dan Tidak Terduga |
| Kamera | Statis | Bergerak / Dinamis |

Contoh: Model yolo yang memiliki akurasi 95% tetapi sering miss frame lebih buruk daripada model dengan akurasi 80% tetapi stabil di fps yg sama. Biasanya ada tradeoff antara akurasi dan kecepatan.

### Fungsi dan Peran Computer Vision pada UAV

#### Fungsi Umum untuk UAV

- Object Detection (Orang, Kendaraan, Bangunan, Api)
- Visual Tracking
- Obstacle Avoidance
- Mapping

#### Peran Computer Vision pada UAV

- Manual: Kesadaran Operator
- Assisted: Peringatan dan pemberian informasi
- Autonomous (Otonom): Kontrol penuh UAV dan pengambil keputusan

Visi tidak menerbangkan UAV, tetapi visi memberikan informasi kepada pengendali UAV atau sistem otonom untuk membuat keputusan.

### End to End Pipeline Computer Vision pada UAV

```mermaid
---
title: UAV CV End to End Pipeline
---
flowchart LR
    Z["Photon / Cahaya"]
    A["Sensor (Kamera)"]
    X["Image Signal Processor (Onboard ISP Processor)"]
    B["Preprocessing (OpenCV)"]
    C["Inference Model (Yolo dalam bentuk ONNX)"]
    D["Postprocessing (OpenCV / Yolo)"]
    F["Decision Making / Control (ROS)"]
    G["Actuator (Motor / Servo)"]
    Z --> A --> X --> B --> C --> D --> F --> G
```

### Batasan Hardware UAV

#### Compute Platforms

- Embedded Systems: NVIDIA Jetson, Google Coral, Raspberry Pi
- Custom Hardware: FPGA, ASIC

#### Batasan yang Perlu Diperhatikan

- Komputasi Terbatas: Pilih model ringan
- Daya Terbatas: Optimalkan konsumsi daya
- Penyimpanan Memory Terbatas: Gunakan model kompresi / quantized
- Thermal Throttling

Pertanyaan penting: Bisakah ini dijalankan 20 menit dibawah sinar matahari tanpa overheat dan menguras baterai?

### Latency vs Accuracy Tradeoff

Kompromi inti UAV

Kita tidak bisa memaksimalkan semuanya

```text
Accuracy naik -> Latency naik -> Control Delay naik -> Safety turun
Latency turun -> Resolusi turun -> Accuracy turun -> Control Delay turun -> Safety naik
```

#### Contoh nyata

- 640x640 @ 30 FPS -> Flyable
- 1280x1280 @ 8 FPS -> Bahaya

Belajar budgeting, bukan mengejar angka dan metrik

### Karakteristik sensor yang berpengaruh terhadap Visi

#### Faktor Kamera

- Resolusi: Lebih tinggi lebih baik, tapi butuh komputasi lebih
- Frame Rate: Lebih tinggi lebih baik untuk tracking dan obstacle avoidance
- Shutter Type: Global shutter sangat direkomendasikan untuk UAV agresif atau presisi tinggi. Rolling shutter masih dapat digunakan dengan mitigasi yang tepat.
- Motion Blur: Gunakan exposure time pendek untuk mengurangi blur
- Exposure Time vs FPS: Untuk mendapatkan FPS tinggi, exposure time biasanya perlu lebih pendek. Tetapi semakin pendek exposure time, semakin gelap gambarnya (perlu kompensasi dari lighting/ISO/gain).
- FOV vs Detection scale: Semakin lebar FOV, semakin kecil objek di gambar, tetapi cakupan area lebih besar

#### Faktor Lingkungan

- Pencahayaan: Kondisi cahaya berubah-ubah
- Sudut Matahari: Lens Flare
- Bayangan
- Debu, kabut, hujan
- Background Clutter

### Detection vs Tracking vs Segmentation

#### Object Detection (Deteksi Objek)

Deteksi adalah tahap paling dasar di mana CV mencoba menjawab pertanyaan: "Apa objek itu dan di mana lokasinya?"

- Cara Kerja: CV akan mengidentifikasi kategori objek (misal: orang, mobil) dan menggambar kotak pembatas yang disebut Bounding Box di sekitar objek tersebut.
- Output: Koordinat kotak (x, y, w, h) dan label kelas beserta skor keyakinannya (confidence score).
- Kegunaan: Sangat cepat dan efisien untuk menghitung jumlah objek dalam satu gambar statis.

#### Object Tracking (Pelacakan Objek)

Pelacakan melangkah lebih jauh dari deteksi dengan menjawab pertanyaan: "Ke mana objek itu pergi?"

- Cara Kerja: Setelah objek dideteksi pada satu frame video, algoritma pelacakan akan memberikan ID unik pada objek tersebut. Algoritma ini kemudian mencoba mencocokkan objek yang sama pada frame-frame berikutnya.
- Pentingnya: Tanpa tracking, CV hanya melihat sekumpulan deteksi yang terputus-putus. Dengan tracking, CV tahu bahwa "Mobil A" di detik ke-1 adalah "Mobil A" yang sama di detik ke-5, meskipun posisinya berpindah.Kegunaan: Menghitung arus lalu lintas, mengikuti pergerakan pemain bola, atau memantau jalur terbang UAV.

#### Image Segmentation (Segmentasi Gambar)

Segmentasi adalah teknik yang paling mendetail karena bekerja pada level pixel. Ia menjawab: "Manakah bagian (pixel) yang merupakan bagian dari objek ini?"

Terdapat dua jenis utama:

- Semantic Segmentation: Mengelompokkan semua pixel dengan kategori yang sama. Misalnya, semua pixel "jalan" diberi warna merah, dan semua "trotoar" diberi warna biru.
- Instance Segmentation: Tidak hanya membedakan kategori, tapi juga membedakan tiap individu objek. Misalnya, jika ada tiga orang, masing-masing orang akan memiliki "masker" warna yang berbeda (misal: merah, hijau, kuning).
- Kegunaan: Mobil otonom (untuk mengetahui batas jalan yang presisi), diagnosis medis (mengukur luas tumor), dan penentuan area pendaratan UAV.

#### Perbandingan Ringkas

| Fitur | Detection | Tracking | Segmentation |
|-------|-----------|----------|--------------|
| Output Utama | Kotak (Bounding Box) | ID Unik + Jalur Gerak | Masker (pixel demi pixel) |
| Fokus Utama | Keberadaan Objek | Kontinuitas Gerakan | Bentuk & Batas Presisi |
| Beban Komputasi | Sedang | Ringan (setelah deteksi) | Berat |

### Kegagalan Umum pada UAV CV

- Manuver: Saat UAV belok tajam, CV sering "buta" karena blur.
- Ketinggian: Makin tinggi UAV, makin banyak deteksi palsu (false positive).
- Cahaya: Fajar/senja membuat CV gagal mengenali target karena minim kontras.
- Panas: CV bisa kepanasan (overheat), membuat kecepatan deteksi (FPS) anjlok drastis.
- Cuaca: Hujan, kabut, debu mengurangi kualitas gambar.
- Kamera: Lensa kotor, goyang, atau salah kalibrasi.

### Peran Sistem dan Batasan

- Vision System: Sees, interprets, reports confidence
- Flight Controller: Decides, acts, stabilizes

## [Data dan Sensor](#data-dan-sensor)

Tujuan: Mengontrol, memahami, dan mempersiapkan data visi di sumbernya sebelum pra-pemrosesan, sebelum model.

### Fungsi Layer Data dan Sensor

Ide Utama: Setiap kegagalan CV pada UAV biasanya berawal dari data/sensor yang buruk.

Section ini membahas:

- Apa yang dilihat kamera?
- Apa pengaruh motion terhadap gambar?
- Bagaimana bisa raw frames menjadi gambar yang bisa diproses CV?

### Fundamental Kamera UAV

#### Cara Kerja Kamera

Kamera mengubah cahaya menjadi sinyal listrik melalui sensor gambar (image sensor) yang terdiri dari jutaan fotodioda kecil (pixel). Setiap pixel mengukur intensitas cahaya yang jatuh padanya dan mengubahnya menjadi nilai digital. Sensor gambar utama adalah CCD (Charge-Coupled Device) dan CMOS (Complementary Metal-Oxide-Semiconductor).

#### Tipe Kamera yang digunakan di UAV

- RGB (Paling umum)
- Thermal
- Multispectral (Kamera yang bisa menangkap beberapa spektrum cahaya tidak cuma visible light)
- Depth Camera (Untuk mengukur jarak objek)
- Night Vision Camera (Biasanya kegunaan militer)

Fokus section ini adalah kamera RGB biasa.

#### Mekanisme Shutter Kamera

| Tipe | Cara Kerja | Dampak pada UAV |
|------|------------|-----------------|
| Global Shutter | Menangkap seluruh pixel secara serentak | Sangat baik untuk gerakan cepat: objek cenderung lebih tajam meski UAV bergerak cepat/bergetar |
| Rolling Shutter | Menangkap gambar baris demi baris (atas ke bawah) | Kurang ideal untuk manuver cepat: dapat muncul efek miring (skew), wobble, atau efek jello |

##### Mengapa Data Training Harus Sesuai dengan Tipe Shutter?

Jika kalian menggunakan model CV di lapangan dengan kamera Rolling Shutter, namun melatihnya (training) hanya dengan gambar sempurna dari internet (yang biasanya Global Shutter atau foto statis), maka:

- Model Tidak Siap: CV tidak pernah "belajar" mengenali objek yang miring atau terdistorsi.
- Akurasi Rendah: Di lingkungan nyata, akurasi akan terjun bebas karena ada perbedaan visual yang signifikan antara data latihan dan input kamera UAV.
- Solusi: Jika terpaksa menggunakan kamera murah, kalian harus menambahkan augmentasi data berupa geometric distortion atau motion blur pada saat training agar CV lebih "kebal" terhadap efek rolling shutter.

#### Lensa dan Field of View (FOV)

| Tipe FOV | Kelebihan | Kekurangan |
| -------- | --------- | ---------- |
| Narrow (Sempit) | Detail sangat tinggi, objek terlihat besar | Area cakupan kecil, sulit mencari target yang hilang |
| Wide (Lebar) | Area cakupan luas, navigasi lebih mudah | Objek terlihat sangat kecil, banyak distorsi di pinggir lensa |

##### Hubungan Pixel-to-Meter (Resolusi Spasial)

Dalam Computer Vision, kita harus memahami berapa banyak area di dunia nyata yang diwakili oleh satu pixel di layar (GSD - Ground Sample Distance).

- Logikanya: Jika kamera Anda memiliki resolusi 1080p dan menggunakan lensa Wide, satu pixel mungkin mewakili 10 cm di tanah. Jika menggunakan lensa Narrow, satu pixel mungkin hanya mewakili 2 cm.
- Dampak pada CV: Algoritma deteksi membutuhkan jumlah pixel minimal (misal: 20x20 pixel) untuk mengenali sebuah objek. Jika nilai pixel-to-meter terlalu besar (satu pixel mencakup area yang luas), objek kecil seperti manusia hanya akan terlihat seperti satu titik kecil yang mustahil dideteksi.

##### Perubahan Ukuran Objek terhadap Ketinggian

Ukuran objek pada gambar tidaklah statis, ia berubah secara berbanding terbalik dengan ketinggian UAV.

- Fenomena: Jika UAV naik dua kali lebih tinggi, ukuran objek di layar akan mengecil menjadi seperempatnya (secara area).
- Masalah Deteksi: Model CV yang dilatih pada ketinggian 10 meter mungkin akan gagal total saat UAV terbang di ketinggian 50 meter karena fitur-fitur visual objek tersebut menjadi terlalu kecil untuk dikenali oleh convolutional layers pada model.

### Image Signal Path (Sensor to Image)

```mermaid
---
title: Image Signal Path
---
flowchart LR
    A["Photon / Cahaya"]
    B["Lensa Kamera"]
    C["Image Sensor (CMOS/CCD)"]
    D["Analog Signal"]
    E["Image Signal Processor (ISP)"]
    F["Digital Image"]
    A --> B --> C --> D --> E --> F
```

Tuning ISP dapat berdampak besar pada kualitas input CV, dan pada beberapa kasus dampaknya bisa lebih besar daripada perubahan kecil pada hyperparameter model.

### Format Umum Gambar

- RAW: Data mentah dari sensor, belum diproses
- YUV: Format kamera umum, memisahkan luminance (Y) dan chrominance (U,V)
- HSV: Warna berdasarkan Hue, Saturation, Value
- RGB: Format umum untuk CV, tiga channel warna (Red, Green, Blue)
- BGR: Format default OpenCV, tiga channel warna (Blue, Green, Red)
- Grayscale: Satu channel intensitas cahaya, mengurangi kompleksitas komputasi
- JPEG/PNG: Tidak ideal untuk CV low-level atau geometric vision, tapi sangat umum dan cukup untuk detection berbasis CNN

Penting: Salah satu sumber bug yang sangat umum pada CV adalah ketidaksesuaian BGR vs RGB.

Berikut adalah ringkasan untuk poin 1.5 hingga 1.7 dalam konteks Computer Vision untuk UAV (UAV):

### Frame Timing & Synchronization

Sinkronisasi waktu adalah kunci agar UAV tidak "bingung" dengan datanya sendiri.

- FPS vs Exposure: FPS tinggi butuh cahaya banyak. Jika gelap, exposure lama akan menurunkan FPS asli.
- Dropped Frames: Frame yang hilang karena kabel longgar atau CPU sibuk. CV harus bisa menangani jeda waktu ini.
- UAV Issues:
  - Camera FPS =! Inference FPS: Kamera kirim 60 gambar/detik, tapi CV mungkin hanya sanggup proses 15 gambar/detik.
  - Sync IMU & Image: Data kemiringan UAV (IMU) harus pas dengan gambar. Jika telat 0.1 detik saja, estimasi posisi bisa meleset jauh.

Penting: Data gambar tanpa catatan waktu yang akurat akan sulit dipakai untuk navigasi otonom (sinkronisasi dengan IMU/estimasi state jadi mudah meleset).

### Motion Artifacts & Environmental Effects

#### Masalah Gerak

- Motion Blur: Kabur karena gerak cepat.
- Rolling Shutter Skew: Objek terlihat miring saat UAV bermanuver.
- Jitter: Getaran halus dari motor UAV yang merusak detail pixel.

#### Masalah Lingkungan

- Sun Glare: Pantulan cahaya matahari langsung ke lensa.
- Shadows: Bayangan tajam yang sering dianggap objek oleh AI.
- Haze/Dust: Debu atau kabut yang menurunkan kontras.

Penting: Expect imperfect data. Jangan berharap gambar sejelas foto studio. Data di lapangan akan selalu berantakan.

### Dataset Design for UAV Vision

Jangan hanya mengumpulkan gambar, tapi rancanglah sebuah sistem data.

#### Dataset Sources

- Real Flight Footage: Data terbaik karena asli, tapi mahal dan berisiko jatuh.
- Ground-based Simulation: Mengambil gambar dari darat dengan sudut pandang UAV (misal dari balkon) untuk menghemat biaya.
- Synthetic Data: Menggunakan game engine (Unity/Unreal) untuk membuat ribuan gambar secara otomatis.

#### Dataset Diversity Axes (Sumbu Keberagaman)

Dataset yang bagus harus memiliki variasi pada:

- Altitude: Objek harus difoto dari berbagai ketinggian.
- Camera Angle: Sudut tegak lurus vs sudut miring .
- Time of Day: Pagi, siang, dan sore (perubahan bayangan).
- Weather: Cerah, berawan, hingga berkabut.

Penting: Dataset != folder of images. Dataset adalah representasi terukur dari semua kondisi yang mungkin ditemui UAV di dunia nyata.

### Sumber Dataset UAV Terbuka

- Roboflow Universe: [universe.roboflow.com](https://universe.roboflow.com)
- Kaggle Datasets: [www.kaggle.com/datasets](https://www.kaggle.com/datasets)
- Mendeley Data: [data.mendeley.com](https://data.mendeley.com)
- Google Dataset Search: [datasetsearch.research.google.com](https://datasetsearch.research.google.com)
- Hugging Face: [huggingface.co/datasets](https://huggingface.co/datasets)
- VisUAV: [github.com/VisUAV/VisUAV-Dataset](https://github.com/VisUAV/VisUAV-Dataset)

### Tools untuk Membuat Dataset (Labelling)

- Roboflow Annotate: [roboflow.com/annotate](https://roboflow.com/annotate)
- Label Studio: [labelstud.io](https://labelstud.io/)
- CVAT: [github.com/openvinotoolkit/cvat](https://github.com/openvinotoolkit/cvat)
- LabelMe: [github.com/wkentaro/labelme](https://github.com/wkentaro/labelme)

### Struktur Dataset Yolo

#### Directory Layout

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

#### Label Format

Setiap baris dalam file label mewakili satu objek dengan format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

#### Normalisasi

Disini kita membahas tentang normalisasi koordinat bounding box pada format label Yolo.

##### Apa itu Normalisasi?

Normalisasi adalah proses mengubah koordinat pixel (misal: 0-1920) menjadi rentang 0.0 sampai 1.0.

##### Mengapa ini dilakukan?

Agar label tetap akurat meskipun Anda mengubah ukuran gambar (resizing). Jika Anda melatih model dengan gambar 1080p lalu melakukan inferensi pada gambar 720p, koordinat absolut akan berantakan, tetapi koordinat normalisasi tetap tepat di tengah objek.

##### Cara Menghitung Normalisasi

Misalkan Anda punya gambar dengan resolusi 1000 x 1000 pixel dan sebuah kotak deteksi (bounding box).

| Komponen | Rumus | Contoh Perhitungan | Hasil (YOLO) |
|----------|-------|--------------------|--------------|
| x_center | x_pusat_pixel / lebar_gambar | 500 / 1000 | 0.5 |
| y_center | y_pusat_pixel / tinggi_gambar | 500 / 1000 | 0.5 |
| width | lebar_kotak_pixel / lebar_gambar | 200 / 1000 | 0.2 |
| height | tinggi_kotak_pixel / tinggi_gambar | 300 / 1000 | 0.3 |

### Labeling for UAV Use Cases

Melabeli data UAV berbeda dengan foto biasa karena sudut pandang dari atas.

#### Aturan Spesifik

- Ukuran Minimum: Jangan labeli objek yang terlalu kecil (misal < 10 pixel) karena CV hanya akan belajar dari noise.
- Occlusion (Terhalang): Tetap gambar kotak jika Anda tahu objek ada di sana (misal orang di bawah pohon), asalkan masih terlihat sebagian.

#### Kesalahan Umum

- Over-tight Boxes: Kotak terlalu mepet sehingga memotong fitur penting objek.
- Labeling Motion Blur: Jangan melabeli bayangan kabur sebagai objek jika bentuk aslinya sudah tidak dikenali.
- Inconsistent Class: Sekarang disebut "daun_sakit", besok "daun_bercak". Ini merusak logika AI.

Penting: Consistency > Precision. Lebih baik semua label sedikit meleset tapi seragam, daripada sangat akurat tapi tidak konsisten.

## [Preprocessing](#preprocessing)

Tujuan: Mengonversi frame kamera mentah menjadi tensor yang siap digunakan untuk model dengan cepat, akurat, dan deterministik.

### Peran preprocessing pada CV UAV

Ide Utama: Preprocessing adalah bagian dari pipeline real-time, bukan sekedar tahapan bantuan.

Preprocessing yang buruk menyebabkan:

- Latency Tinggi dan Spikes: Jika preprocessing lambat, keseluruhan sistem CV akan melambat.
- Distorsi Data: Transformasi yang salah bisa merusak fitur penting pada gambar.
- Ketidakstabilan Inferensi: Frame rate yang tidak konsisten membuat kontrol UAV goyah.
- False Detections: Data yang tidak tepat menyebabkan model salah mengenali objek.

### Ringkasan Tahapan Preprocessing

```mermaid
---
title: UAV CV Preprocessing Pipeline
---
flowchart LR
    A["Camera Frame Capture"]
    B["Format Conversion (BGR to RGB)"]
    C["Resize / Letterbox"]
    D["Normalize (0-1)"]
    E["Layout Transform (HWC to CHW)"]
    F["Tensor Upload (GPU/NPU)"]
    A --> B --> C --> D --> E --> F
```

- Camera Frame Capture: Dapatkan frame mentah dari kamera.
- Format Conversion: Ubah warna dari BGR ke RGB agar CV tidak salah mengenali warna objek.
- Resize / Letterbox: Sesuaikan ukuran gambar ke input model tanpa merusak rasio aslinya.
- Normalize: Ubah nilai pixel 0-255 menjadi 0.0-1.0 agar perhitungan matematika CV lebih stabil.
- Layout Transform: Susun ulang urutan data dari (H, W, C) menjadi (C, H, W) sesuai standar GPU / CPU.
- Tensor Upload: Kirim data akhir ke memori GPU atau NPU untuk diproses oleh model.

### Strategi Input Resolusi

#### Fixed Resolution

- Fixed Resolution: Semua frame diubah ke ukuran tetap (misal: 640x480). Lebih mudah dioptimalkan, tapi bisa kehilangan detail pada objek kecil.

Penting: Pada UAV biasanya dipilih fixed resolution untuk kestabilan dan prediktabilitas (timing, budgeting, dan integrasi downstream lebih mudah).

#### Typical UAV Choices

- Common Size: 640x640
- Trade-offs: Resolusi lebih tinggi = akurasi lebih baik, tapi butuh komputasi lebih banyak dan latency lebih tinggi.

### Resize vs Letterbox

#### Resize

- Keunggulan: simpel, cepat
- Kekurangan: distorsi objek

#### Letterbox

- Keunggulan: mempertahankan rasio aspek
- Kekurangan: padding, kompleksitas scaling

#### Realitas UAV

- Lebih disukai menggunakan letterbox untuk deteksi
- Harus melacak offset padding untuk pemrosesan selanjutnya

### Color Space Handling

- Common Conversions: Proses mengubah cara warna direpresentasikan, seperti dari format transmisi video ke format tampilan atau pemrosesan AI.
- Contoh: YUV (sinyal video) -> RGB (warna standar) -> BGR (format default OpenCV).
- Model Expects RGB: Hampir semua model deteksi (YOLO) dilatih dengan data RGB, sehingga input harus disesuaikan.
- GPU vs CPU Conversion Cost: Melakukan konversi warna di CPU bisa menyebabkan bottleneck, sehingga lebih baik dilakukan langsung di GPU/NPU menggunakan hardware accelerator.

### Teknik Normalisasi dan Quantization

- Scale Pixel Values [0,1]: Mengubah angka pixel 0-255 menjadi 0.0-1.0 agar input model lebih seragam.
- Mean/Std Scaling: Menyelaraskan distribusi warna gambar dengan data yang digunakan saat training model.
- FP32 vs FP16 vs INT8: FP32 paling akurat, tapi INT8 jauh lebih cepat dan hemat memori pada perangkat edge.

### Batch Size dan Stream Handling

- Batch Size = 1: Pada UAV, kita memproses gambar satu per satu segera setelah ditangkap untuk meminimalkan jeda waktu.
- Mengapa batching merusak latency: Batching meningkatkan throughput (jumlah total gambar per detik), tapi dapat memperburuk latency (waktu respon per gambar).

### Determinism dan Timing Guarantees

- Variable Preprocessing Time: Waktu proses yang berubah-ubah (misal karena CPU panas) bisa mengacaukan kontrol UAV.
- Frame Queue Buildup: Antrean gambar yang menumpuk karena CV lebih lambat dari kamera, menyebabkan deteksi "terlambat" dari posisi UAV aslinya.
- Frame Drop Policy: Kebijakan untuk membuang frame lama jika CV masih sibuk, agar CV selalu memproses gambar terbaru (fresh).

Penting: Deterministic > fast. Sistem yang sedikit lebih lambat tapi stabil waktunya jauh lebih aman daripada sistem yang sangat cepat tapi sering tersendat.

### Calibration-Aware Preprocessing

- Undistortion Before Resize: Melakukan koreksi distorsi lensa (efek fisheye) pada resolusi asli sebelum gambar diperkecil.
  - Contoh: Menghilangkan kelengkungan garis cakrawala agar objek di pinggir foto tidak terlihat melengkung.
- Fixed Camera Matrix Usage: Menggunakan parameter intrinsik kamera (focal length, optical center) yang sudah dihitung secara permanen.
  - Contoh: Memasukkan file .yaml hasil kalibrasi kamera spesifik UAV Anda ke dalam kode OpenCV.

Penting: Koreksi di awal. Jika distorsi tidak dibuang, CV akan kesulitan mendeteksi objek di area pinggir lensa karena bentuknya yang melengkung tidak wajar.

### Geometry-Safe Resizing

- Aspect Ratio Preservation: Menjaga perbandingan lebar dan tinggi gambar agar objek tidak terlihat "gepeng".
  - Contoh: Menggunakan teknik Letterboxing (menambah bar hitam) alih-alih menarik gambar secara paksa ke ukuran 640x640.
- Pixel Mapping Correctness: Memastikan setiap pixel pada gambar hasil resize tetap memetakan koordinat yang benar di dunia nyata.
  - Contoh: Menggunakan interpolasi INTER_LINEAR atau INTER_AREA di OpenCV yang sesuai dengan tipe objek yang dideteksi.

### Preprocessing untuk Training vs Inference

- Training-only (Augmentation): Menggunakan teknik acak agar model belajar dari berbagai kondisi sulit.
- Inference-only (Deterministic): Proses harus baku dan tidak boleh ada elemen acak agar hasil deteksi konsisten.

### Contoh kesalahan dalam Preprocessing

- Wrong Aspect Ratio: Gambar yang "ketarik" membuat objek bulat jadi lonjong, sehingga CV gagal mengenali bentuknya.
- Wrong Normalization: Lupa membagi 255 atau salah nilai mean/std membuat skor keyakinan (confidence) CV anjlok drastis.
- Incorrect Padding Handling: Kesalahan saat menambah border hitam membuat kotak deteksi (bounding box) bergeser dari objek aslinya.

## [Inference](#inference)

Tujuan: Lakukan inferensi neural secara terprediksi, aman, dan cukup cepat untuk penerbangan, menggunakan output preprocessing.

Section ini membahas:

- Kapan inference jalan
- Seberapa sering inference jalan
- Bagaimana inference terintegrasi dalam sistem UAV

### Peran Inference pada UAV CV

#### Garis Besar Peletakan Inference dalam Sistem UAV

```mermaid
---
title: UAV CV Inference Integration
---
flowchart LR
    A["Preprocessed Tensor"]
    B["Inference Engine (Yolo Model)"]
    C["Raw Predictions"]
    A --> B --> C
```

Inference adalah:

- Tugas real-time: Harus selesai dalam batas waktu tertentu agar UAV bisa bereaksi cepat.
- Pesaing sumber daya (CPU/GPU/NPU)
- Penyumbang latensi

### Model Pengaturan Waktu Inferensi

#### Model umum

- Frame-synchronous (every frame)
- Fixed rate (contoh 10 Hz)
- Event-driven (triggered)

#### Realitas UAV untuk Inferensi

- Deteksi seringkali berjalan lebih lambat daripada FPS kamera
- Frame dilewati dengan sengaja

Penting: Melewatkan frame adalah fitur, bukan bug.

### Inference Frequency & Control Loop Impact

Di dunia UAV, CV tidak berdiri sendiri. Ia adalah "mata" yang membimbing "tangan" (sistem kendali/flight controller).

- Control-Loop Coupling: Sistem kontrol UAV bekerja sangat cepat. Jika CV memberikan data terlalu lambat, perintah gerak UAV akan menjadi tidak sinkron.
  - Vision Latency memengaruhi stabilitas kontrol: Semakin lama CV memproses gambar (latency), semakin tidak stabil gerakan UAV. UAV akan mulai goyang (oscillate) karena ia bereaksi terhadap posisi objek yang sudah lewat.
  - Inference lambat -> Deteksi basi

- Konsep Penting yang Harus Dipahami:
  - Time-to-useful-detection: Waktu total yang dibutuhkan mulai dari sensor menangkap cahaya sampai sistem kendali menerima instruksi. Nilai ini tergantung dinamika UAV dan jenis misi.
  - Detection Freshness: Seberapa "segar" data deteksi tersebut. Data yang segar adalah data yang diambil kurang dari beberapa milidetik yang lalu.

Penting: Tingkat inferensi harus sesuai dengan kebutuhan misi.

### Standar I/O

#### Input

Input ke model harus bersifat "Immutable" (tidak boleh berubah strukturnya):

- Fixed-shape tensor: Untuk sistem real-time (UAV), biasanya lebih aman memakai input fixed-shape (mis. 640x640) supaya latency lebih stabil. Dynamic shape bisa dipakai, tapi perlu pengujian dan budgeting yang lebih ketat.
- Known layout: Pastikan formatnya sudah pasti.
- Timestamped: Setiap data yang masuk harus membawa catatan waktu kapan gambar tersebut diambil oleh kamera, bukan kapan ia sampai di CV.

#### Output

Output dari model hanyalah angka mentah (Raw Prediction). Jangan langsung membuat keputusan gerak di sini. Biarkan modul lain yang menerjemahkan angka ini menjadi koordinat atau perintah terbang.

### Latency Budgeting

Dalam UAV, kita bekerja dengan "Hard Deadlines". Jika UAV terbang 5 m/s, keterlambatan 100ms berarti UAV sudah berpindah 50cm sebelum ia sadar ada rintangan.

| Tahapan | Anggaran Maksimal (Contoh) | Deskripsi |
| ------- | -------------------------- | --------- |
| Preprocessing | 8 ms | Resize, Normalize, Upload ke GPU |
| Inference | 20 ms | Waktu CV "berpikir" (Running model) |
| Postprocessing | 5 ms | Non Maximum Suppression, koordinat transform |
| TOTAL | 33 ms | Setara dengan 30 FPS |

### Model Optimization Techniques

- Pruning: Menghapus neuron yang tidak penting untuk mempercepat inferensi.
- Quantization: Mengurangi presisi angka (misal dari FP32 ke INT8) untuk mempercepat komputasi.
- Knowledge Distillation: Melatih model kecil (student) untuk meniru model besar (teacher) agar tetap akurat namun lebih ringan.
- Hardware Acceleration: Memanfaatkan GPU, NPU, atau TPU untuk mempercepat inferensi.

### Integrasi dengan Postprocessing

#### Peran Inference vs. Postprocessing

| Karakteristik | Inference Engine | Postprocessing |
| ------------- | ---------------- | -------------- |
| Output | Raw Tensor (skor numerik) | Objek Terstruktur (Koordinat, Kelas, ID) |
| Konteks | Hanya tahu pixel gambar | Tahu koordinat GPS, tinggi UAV, & waktu |
| Tujuan | Klasifikasi & Lokalisasi mentah | Pengambilan keputusan & Filter data |

#### Mengapa Postprocessing Diperlukan?

- Interpretasi Data: Model hanya mengeluarkan angka (misal: 0.85). Postprocessing mengubahnya menjadi informasi bermakna
- Penyaringan (Filtering): Model sering mengeluarkan ribuan kotak prediksi yang tumpang tindih. Postprocessing menggunakan algoritma seperti Non-Maximum Suppression (NMS) untuk memilih satu kotak terbaik.
- Transformasi Koordinat: Mengubah posisi objek dari sistem koordinat gambar (pixel) ke sistem koordinat dunia nyata (Latitude/Longitude) berdasarkan sensor IMU UAV.
- Temporal Consistency: Memastikan bahwa objek yang terdeteksi di frame ke-1 adalah objek yang sama di frame ke-2 agar tidak terjadi lonjakan data (flickering).

#### Struktur Data Inference

- Bounding Box Array: $[x, y, w, h]$ dalam format normalisasi.
- Confidence Scores: Nilai probabilitas $[0.0 - 1.0]$.
- Class Indices: ID kategori objek $[0, 1, 2, ...]$.
- Metadata: Timestamp sinkronisasi dan ID kamera.

## [Postprocessing](#postprocessing)

Tujuan: Mengubah output mentah dari model menjadi informasi bermakna yang dapat digunakan untuk pengambilan keputusan UAV secara real-time.

Modul ini menjawab pertanyaan-pertanyaan berikut:

- Apa arti tensor-tensor ini?
- Deteksi mana yang harus kita percayai?
- Bagaimana deteksi bertahan seiring waktu?
- Bagaimana CV memengaruhi keputusan UAV?

### Peran Postprocessing pada UAV CV

#### Garis Besar Peletakan Postprocessing dalam Sistem UAV

```mermaid
---
title: UAV CV Postprocessing Integration
---
flowchart LR
    A["Raw Predictions from Inference"]
    B["Postprocessing Module"]
    C["Structured Detections"]
    A --> B --> C
```

Penting: Output mentah YOLO sangat berantakan. seringkali terdapat ratusan kotak untuk satu objek yang sama.

### Interpretasi Output Mentah Model

Sebelum menyaring data, kita harus mengerti apa yang dikeluarkan oleh model:

- Bounding Boxes: Koordinat lokasi objek (biasanya $[x, y, w, h]$).
- Confidence Scores: Seberapa yakin CV bahwa ada "sesuatu" di kotak tersebut.
- Class Probabilities: Seberapa yakin CV bahwa "sesuatu" itu adalah objek spesifik (misal: Orang).

Website untuk visualisasi input output mentah (file ONNX):

- Netron: [netron.app](https://netron.app/)

### Confidence Thresholding

Tujuan: Buang semua "sampah" deteksi berkualitas rendah.

- Global vs Class-specific: Kadang kita butuh standar berbeda untuk tiap objek.
- Contoh UAV:
  - Search & Rescue (Manusia): Gunakan threshold rendah (misal 0.3) agar tidak melewatkan korban, meski banyak deteksi palsu.
  - Pendaratan Otonom (Heliport): Gunakan threshold tinggi (0.8) karena kita butuh kepastian mutlak sebelum mendarat.

### Non-Maximum Suppression (NMS)

Tujuan: Hapus deteksi yang tumpang tindih untuk satu objek yang sama.

- Duplicate Detections: Model sering mendeteksi bagian-bagian berbeda dari objek yang sama.
- Solusi: NMS membandingkan semua kotak yang tumpang tindih dan hanya menyisakan satu kotak dengan skor keyakinan tertinggi.

#### Tipe

- Standard NMS: Hapus kotak yang tumpang tindih di atas threshold IoU tertentu.
- Soft-NMS: Alih-alih menghapus, kurangi skor keyakinan kotak yang tumpang tindih.
- Weighted NMS: Gabungkan kotak yang tumpang tindih menjadi satu kotak baru berdasarkan skor keyakinan mereka.

### Koreksi Geometri

Tahap ini adalah membatalkan (undo) semua perubahan yang dilakukan saat preprocessing di Module 2.

Langkah: Buang padding letterbox, kembalikan skala kotak ke ukuran gambar asli, dan pastikan koordinat tidak keluar dari batas gambar.

Lihat bagian [Transformasi Koordinat 2D ke 3D](#transformasi-koordinat-2d-ke-3d) untuk detail lebih lanjut.

### Filter Semantic dan Kontekstual

- Size Threshold: Abaikan deteksi yang terlalu kecil (mungkin hanya noise).
- Edge Filtering: Abaikan objek yang terpotong di pinggir lensa karena bentuknya tidak lengkap.

### Motion and Kinematic Reasoning

Menggunakan logika fisika sederhana untuk membuang deteksi yang mustahil.

Contoh: Jika sebuah objek tiba-tiba "melompat" dari ujung kiri ke ujung kanan layar dalam 1 milidetik, itu adalah deteksi palsu. Objek fisik memiliki batasan kecepatan dan arah gerak yang masuk akal.

Penting: Logika gerak membantu membuang gangguan visual sesaat.

### Confidence Over Time

- Accumulation: Jika objek terdeteksi terus menerus dalam 10 frame, tingkat kepercayaan sistem meningkat.
- Decay: Jika objek hilang sesaat, jangan langsung dihapus; turunkan kepercayaannya perlahan.

Penting: Drone harus tahu kapan ia boleh percaya pada matanya dan kapan ia harus ragu.

### Interface to Flight Systems

Menentukan batas tanggung jawab yang sangat jelas.

- Output Vision: Berupa Metadata (koordinat, ID) atau Estimasi Posisi.
- Larangan Mutlak: Modul vision tidak boleh mengirim perintah langsung ke motor/ESC/servo.

Penting: Vision mengirim data ke Flight Controller, dan Flight Controller-lah yang mengatur kestabilan motor.

## [Transformasi Koordinat 2D ke 3D](#transformasi-koordinat-2d-ke-3d)

Tujuan: Ubah koordinat objek dari sistem 2D gambar ke sistem 3D dunia nyata (X, Y, Z) menggunakan data sensor UAV.

### Matrix Transformasi

![Intrinsic Matrix](img/matrix.png)

### Transformasi Koordinat: Image to World Projection via Ray Casting

Masalah Utama: Output dari YOLO adalah (x: 320, y: 320) dalam satuan pixel. Flight Controller (FC) tidak mengerti pixel. FC butuh koordinat dalam meter (misal: "Target ada 5 meter di Utara, 2 meter di Timur").

Bagian ini membahas cara mengubah data 2D (gambar) menjadi data 3D (dunia nyata).

#### Konsep Pinhole Camera Model

Kamera hanyalah sebuah lubang jarum (pinhole) yang memproyeksikan dunia 3D ke bidang 2D. Untuk membalikkan prosesnya (2D ke 3D), kita perlu memahami parameter intrinsik kamera.

#### Matriks Intrinsik ($K$)

Setiap kamera memiliki "sidik jari" optik yang disebut Matriks Intrinsik:

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

$f_x, f_y$: Panjang fokus (focal length) dalam satuan pixel.

$c_x, c_y$: Titik tengah optik (biasanya pusat gambar, misal 320, 320 pada gambar 640x640).

**Cara Mendapatkan $K$:** Lakukan Kalibrasi Kamera (menggunakan papan catur/checkerboard di OpenCV).

#### Pipeline Transformasi (Ray Casting)

Karena gambar 2D kehilangan informasi kedalaman (depth), kita tidak bisa langsung tahu posisi $X,Y,Z$ objek. Yang kita tahu adalah arah objek tersebut dari kamera. Kita membayangkan sebuah garis lurus (Ray) yang ditembakkan dari pusat kamera menembus pixel deteksi menuju tanah.

```mermaid
---
title: Coordinate Transformation Pipeline
---
flowchart LR
    A["Pixel Coordinate (u,v)"] --> B["Normalized Plane (x,y)"]
    B --> C["Camera Frame (3D Ray)"]
    C --> D["Body Frame (Relative to Drone)"]
    D --> E["Inertial Frame (NED - North East Down)"]
```

#### Langkah 1: Pixel ke Normalized Coordinates

Mengubah pixel menjadi koordinat tanpa satuan (di bidang fokus).

$$x_{norm} = \frac{(u - c_x)}{f_x}, \quad y_{norm} = \frac{(v - c_y)}{f_y}$$

#### Langkah 2: Flat Earth Assumption (Asumsi Tanah Datar)

Jika kita tidak punya sensor LiDAR atau Depth Camera, kita gunakan data Altitude (tinggi terbang) dari barometer/GPS.

Jika drone terbang rata (level) pada ketinggian $H$ meter:

- Jarak Depan ($X_{body}$) = $H \times y_{norm}$ (jika kamera menghadap bawah)
- Jarak Samping ($Y_{body}$) = $H \times x_{norm}$

Catatan: Rumus di atas berubah drastis jika drone miring (roll/pitch) atau ada Gimbal. Lihat bagian [Gimbal Awareness](#gimbal-awareness).

Catatan lagi: Ray casting tanpa kompensasi attitude, terrain model, atau range sensor hanya cocok untuk estimasi kasar. Jangan digunakan untuk kontrol presisi.

#### Hirarki Frame Koordinat

- Image Frame: 2D (u, v). Origin: Pojok kiri atas.
- Camera Frame: 3D. Origin: Sensor kamera.
  - OpenCV default: Z forward, X right, Y down
- Body Frame (FRD): 3D. Origin: Pusat Flight Controller. X=Forward (Depan), Y=Right (Kanan), Z=Down (Bawah).
- Inertial Frame (NED): 3D. Origin: Titik Takeoff. North, East, Down.

Tugas CV Engineer: Mengirim data dalam Body Frame atau Inertial Frame ke Flight Controller.

### Gimbal Awareness

Masalah Utama: Kamera pada drone canggih biasanya dipasang pada Gimbal 3-Axis.Jika gimbal menunduk (pitch) 45 derajat, maka "tengah gambar" bukan lagi "depan drone", melainkan "serong bawah drone". Jika kamu mengabaikan sudut gimbal, drone akan salah menghitung lokasi target hingga puluhan meter.

#### Dampak Rotasi Gimbal

Bayangkan sebuah objek berada tepat di tengah gambar $(u=320, v=320)$.

- Kasus A (Gimbal Lurus 0°): Objek berada sejajar dengan ketinggian drone (di cakrawala). Jarak = Tak Terhingga.
- Kasus B (Gimbal Bawah -90°): Objek berada tepat di bawah perut drone. Jarak Horizontal = 0 meter.

Posisi pixel sama, tapi lokasi dunia nyata berbeda total. Inilah kenapa kita butuh Rotation Matrix.

#### Integrasi Data Telemetri

Untuk menghitung lokasi akurat, CV Module harus "berbicara" dengan Gimbal/FC secara real-time.

**Input yang dibutuhkan:**

- Gimbal Pitch ($\theta$): Menunduk/Menengadah.
- Gimbal Yaw ($\psi$): Menoleh kiri/kanan.
- Gimbal Roll ($\phi$): Miring (biasanya 0 pada gimbal stabil).

Rumus Transformasi dengan Rotasi

Alih-alih rumus sederhana, kita gunakan Matriks Rotasi ($R_{gimbal}$) untuk memutar vektor pandangan.

Jika vector sinar di Camera Frame adalah $V_{cam} = [x_{norm}, y_{norm}, 1]$, maka vector di Body Frame $V_{body}$ adalah:

$$V_{body} = R_{gimbal} \times V_{cam}$$

Setelah mendapatkan arah vektor $V_{body}$ yang sudah dikoreksi rotasi gimbal, barulah kita kalikan dengan ketinggian ($Altitude$) untuk mencari titik temu di tanah.

### Pose Estimation dengan PnP (Perspective-n-Point)

Masalah: Ray Casting (bab sebelumnya) hanya memberitahu kita koordinat $X, Y$ di tanah dengan asumsi kita tahu ketinggian drone. Namun, bagaimana jika kita ingin mendarat presisi di atas *charging station* atau terbang melewati jendela? Kita butuh lebih dari sekadar posisi. Kita butuh Orientasi 3D (6-DOF Pose).

Solusi: Algoritma PnP (Perspective-n-Point).

#### Apa itu PnP?

PnP adalah metode matematika untuk menghitung posisi dan orientasi kamera relatif terhadap sebuah objek, asalkan kita mengetahui ukuran asli objek tersebut di dunia nyata.

Jika Ray Casting menjawab: "Benda itu ada di koordinat mana?"
PnP menjawab: "Di mana posisi dan arah hadap drone saya terhadap benda itu?"

#### Input yang Dibutuhkan

Untuk menjalankan PnP (biasanya `cv2.solvePnP` di OpenCV), kita membutuhkan 4 hal:

- 2D Image Points: Koordinat pixel dari titik-titik sudut objek di gambar (hasil deteksi YOLO/OpenCV).
- 3D World Points: Koordinat asli objek tersebut di dunia nyata (kita definisikan sendiri).
  - Contoh: Jika kita mendeteksi ArUco Marker ukuran 20cm x 20cm, kita definisikan titik sudutnya sebagai $(0,0,0), (0.2,0,0), (0.2,0.2,0), (0,0.2,0)$ dalam meter.
- Camera Matrix ($K$): Matriks intrinsik hasil kalibrasi (sama seperti bab sebelumnya).
- Distortion Coefficients: Parameter distorsi lensa.

#### Output PnP: $rvec$ dan $tvec$

Algoritma akan menghasilkan dua vektor vital:

- Translation Vector ($tvec$): Posisi objek relatif terhadap kamera $(x, y, z)$.
  - $z$ di sini adalah depth. PnP memberikan estimasi $Z$ geometrik yang independen dari barometer, namun tetap perlu difusi sensor untuk kestabilan UAV.
- Rotation Vector ($rvec$): Orientasi objek relatif terhadap kamera.
  - Ini memberitahu kita apakah drone sedang miring, tegak lurus, atau mendekati objek dari samping.

#### Pipeline Implementasi

```mermaid
---
title: PnP Flowchart
---
flowchart LR
    A["Deteksi Sudut Objek (YOLO/CornerSubPix)"]
    B["Definisi Model 3D (Ukuran Fisik Objek)"]
    C["Algoritma PnP (cv2.solvePnP)"]
    D["Output: Rotasi & Translasi (rvec, tvec)"]
    E["Konversi ke Euler Angles (Pitch, Roll, Yaw)"]
    F["Kirim ke Flight Controller"]
    A --> C
    B --> C
    C --> D --> E --> F
```

### Kapan Menggunakan Ray Casting vs PnP?

| Fitur | Ray Casting | PnP |
|-------|-------------|-----|
| Target | Objek sembarang (mobil, orang, pohon) | Objek yang diketahui ukurannya (QR Code, Pad H, Gawang) |
| Syarat | Butuh sensor ketinggian (Barometer/LiDAR) | PnP dapat memberikan estimasi Z tanpa barometer, namun dalam sistem UAV biasanya tetap difuse dengan barometer / rangefinder |
| Output | Lokasi 2D di tanah (X,Y) | Lokasi & Orientasi 3D (X,Y,Z,Roll,Pitch,Yaw) |
| Kegunaan | Surveillance, Mapping, Tracking | Precision Landing, Indoor Flight, Docking |

## [OpenCV](#opencv)

Lihat: [Modul OpenCV](https://github.com/magang-bayucaraka-2026/modul-opencv)

### OpenCV untuk pipeline deteksi YOLO (praktis)

OpenCV sering dipakai untuk bagian **preprocessing** dan **postprocessing** (di luar model) karena cepat, stabil, dan mudah diintegrasikan dengan pipeline kamera UAV.

#### Capture frame + timestamp

Timestamp sebaiknya diambil saat frame diterima (bukan setelah inference) agar sinkronisasi dengan IMU lebih masuk akal.

```python
import time
import cv2

cap = cv2.VideoCapture(0)

ok, frame_bgr = cap.read()
ts_ns = time.monotonic_ns()  # contoh timestamp monotonic
```

#### Letterbox (resize tanpa merusak aspect ratio)

Letterbox menyimpan rasio aspek dan menambah padding. Nilai `r` dan `(dw, dh)` perlu disimpan untuk mengembalikan koordinat box ke ukuran frame asli.

```python
import cv2
import numpy as np

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
  h0, w0 = im.shape[:2]
  h, w = new_shape

  r = min(h / h0, w / w0)
  new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
  dw, dh = (w - new_unpad[0]) / 2, (h - new_unpad[1]) / 2

  if (w0, h0) != new_unpad:
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

  top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
  left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
  im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
  return im, r, (dw, dh)
```

#### BGR -> RGB, normalisasi, dan layout tensor

Mayoritas model deteksi dilatih di RGB. Pastikan urutan channel dan normalisasi sesuai saat training/export.

```python
img_lb, r, (dw, dh) = letterbox(frame_bgr, (640, 640))
img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)

x = img_rgb.astype(np.float32) / 255.0
x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
x = np.expand_dims(x, 0)        # CHW -> NCHW
```

#### De-letterbox (kembalikan koordinat box ke frame asli)

Jika hasil postprocessing menghasilkan box dalam koordinat image letterbox (misal format `xyxy`), konversi balik ke ukuran asli:

```python
def scale_boxes_xyxy(boxes_xyxy, r, dw, dh, w0, h0):
  boxes = boxes_xyxy.copy()
  boxes[:, [0, 2]] -= dw
  boxes[:, [1, 3]] -= dh
  boxes /= r
  boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0 - 1)
  boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0 - 1)
  return boxes
```

Catatan: detail format output tergantung model/engine (ada yang output `xywh`, ada yang `xyxy`, ada yang masih normalisasi). Yang penting: simpan parameter resize/padding dan konsisten di postprocessing.

### Kalibrasi Kamera

Beberapa pinhole camera menimbulkan distorsi yang signifikan pada gambar. Dua jenis distorsi utama adalah distorsi radial dan distorsi tangensial.

Distorsi radial menyebabkan garis lurus tampak melengkung. Distorsi radial menjadi lebih besar semakin jauh titik-titik tersebut dari pusat gambar. Misalnya, sebuah gambar ditunjukkan di bawah ini di mana dua tepi papan catur ditandai dengan garis merah. Namun, Anda dapat melihat bahwa batas papan catur bukanlah garis lurus dan tidak sesuai dengan garis merah. Semua garis lurus yang diharapkan tampak melengkung.

![Chessboard](img/chess.png)

#### Distorsi

##### Distorsi Radial

Distorsi radial menyebabkan garis lurus tampak melengkung. Distorsi meningkat seiring dengan semakin jauhnya titik dari pusat gambar.

Hal ini dapat dimodelkan sebagai:

$$x_{\text{distorted}} = x (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$
$$y_{\text{distorted}} = y (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$

##### Distorsi Tangensial

Distorsi tangensial terjadi karena lensa pengambil gambar tidak sejajar sempurna dengan bidang pencitraan. Akibatnya, beberapa area gambar mungkin tampak lebih dekat atau lebih jauh dari yang diharapkan.

Besarnya distorsi tangensial dapat dinyatakan sebagai:

$$x_{\text{distorted}} = x + \left[ 2 p_1 x y + p_2 \left( r^2 + 2 x^2 \right) \right]$$
$$y_{\text{distorted}} = y + \left[ p_1 \left( r^2 + 2 y^2 \right) + 2 p_2 x y \right]$$

##### Koefisien Distorsi

Singkatnya, kita perlu menemukan lima parameter, yang dikenal sebagai koefisien distorsi:

$$\text{Koefisien distorsi} = ( k_1, k_2, p_1, p_2, k_3 )$$
di mana:

- $k_1, k_2, k_3$ mewakili distorsi radial
- $p_1, p_2$ mewakili distorsi tangensial

##### Parameter Intrinsik Kamera

Selain koefisien distorsi, kita membutuhkan parameter intrinsik kamera. Parameter ini spesifik untuk kamera dan meliputi:

- Panjang fokus: $f_x, f_y$
- Pusat optik: $c_x, c_y$

Parameter-parameter ini membentuk matriks intrinsik kamera:

$\text{Matriks Kamera} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$
Matriks kamera unik untuk kamera tertentu dan dapat digunakan kembali untuk semua gambar yang diambil oleh kamera yang sama.

##### Parameter Ekstrinsik Kamera

Parameter ekstrinsik sesuai dengan vektor rotasi dan translasi yang mengubah koordinat titik 3D dari sistem koordinat dunia ke sistem koordinat kamera.

##### Gambaran Umum Kalibrasi

Untuk aplikasi penglihatan stereo dan rekonstruksi 3D, distorsi lensa harus dikoreksi terlebih dahulu.

Untuk memperkirakan koefisien distorsi dan parameter kamera, kami menyediakan beberapa gambar dari pola kalibrasi yang terdefinisi dengan baik (misalnya, papan catur).

Kami mendeteksi titik fitur spesifik yang posisi relatifnya diketahui (misalnya, titik sudut papan catur). Karena kami mengetahui:

- koordinat 3D dari titik-titik ini di ruang dunia nyata, dan
- koordinat 2D yang sesuai dalam gambar,

kami dapat menyelesaikan parameter intrinsik kamera, parameter ekstrinsik, dan koefisien distorsi.

Untuk akurasi yang lebih baik, setidaknya 10 gambar kalibrasi yang diambil dari sudut pandang yang berbeda direkomendasikan.

#### Kode Kalibrasi Kamera dengan OpenCV

##### Persiapan Data

Untuk melakukan kalibrasi, kita memerlukan minimal 10 pola pengujian. Umumnya digunakan papan catur (chessboard):

- **Object Points (3D):** Titik koordinat dunia nyata. Agar mudah, kita asumsikan papan catur berada pada bidang $Z = 0$. Koordinat ditentukan berdasarkan indeks kotak, misalnya $(0,0,0), (1,0,0), (2,0,0), \dots$
- **Image Points (2D):** Lokasi pixel di mana sudut-sudut kotak hitam bertemu pada gambar.

#### Setup

Jadi untuk menemukan pola pada papan catur, kita dapat menggunakan fungsi `cv.findChessboardCorners()`. Kita juga perlu menentukan jenis pola yang kita cari, seperti grid 8x8, grid 5x5, dll. Dalam contoh ini, kita menggunakan grid 7x6. (Biasanya papan catur memiliki kotak 8x8 dan sudut dalam 7x7). Fungsi ini mengembalikan titik-titik sudut dan nilai kembalian yang akan bernilai `True` jika pola ditemukan. Sudut-sudut ini akan ditempatkan dalam urutan tertentu (dari kiri ke kanan, atas ke bawah). Setelah menemukan sudut-sudutnya, kita dapat meningkatkan akurasinya menggunakan `cv.cornerSubPix()`. Kita juga dapat menggambar polanya menggunakan `cv.drawChessboardCorners()`.

```python
import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()
```

##### Hasil Titik

![alt text](img/chess-points.png)

#### Kalibrasi

Sekarang setelah kita memiliki titik objek dan titik gambar, kita siap untuk melakukan kalibrasi. Kita dapat menggunakan fungsi `cv.calibrateCamera()` yang mengembalikan matriks kamera, koefisien distorsi, vektor rotasi dan translasi, dll.

```python
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
```

#### Undistort Gambar

Sekarang, kita dapat mengambil gambar dan menghilangkan distorsinya. OpenCV menyediakan dua metode untuk melakukan ini. Namun pertama, kita dapat menyempurnakan matriks kamera berdasarkan parameter penskalaan bebas menggunakan `cv.getOptimalNewCameraMatrix()`. Jika parameter penskalaan alpha=0, fungsi ini mengembalikan gambar yang tidak terdistorsi dengan pixel yang tidak diinginkan seminimal mungkin. Jadi, fungsi ini bahkan dapat menghilangkan beberapa pixel di sudut gambar. Jika alpha=1, semua pixel dipertahankan dengan beberapa gambar hitam tambahan. Fungsi ini juga mengembalikan ROI gambar yang dapat digunakan untuk memotong hasilnya.

```python
# get one of the image from the set
img = cv.imread('left12.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)
```

##### Hasil Undistort

Hasilnya merupakan papan catur yang memiliki garis lurus tanpa distorsi.

![alt text](img/fixed-chessboard.png)

### SolvePNP (Perspective-n-Point)

Masalah estimasi pose terdiri dari mencari rotasi (rotation) dan translasi (translation) yang meminimalkan "reprojection error" dari korespondensi titik 3D-2D. Pose ini menentukan bagaimana objek diposisikan relatif terhadap kamera.

![alt text](img/pose-estimation.png)

#### Konsep Utama

- Persamaan Proyeksi: Titik 3D di dunia nyata diproyeksikan ke bidang gambar 2D menggunakan matriks intrinsik kamera (K) dan koefisien distorsi.
- Output:
  - `rvec` (Rotation Vector): Vektor rotasi.
  - `tvec` (Translation Vector): Vektor translasi.
- Koordinat Kamera: Sumbu X ke kanan, Y ke bawah, dan Z ke depan.

##### Model Matematika Proyeksi

Titik dalam world frame $\mathbf{X}_w$ diproyeksikan ke bidang gambar $[u, v]$ menggunakan model proyeksi perspektif $\Pi$ dan matriks parameter intrinsik kamera $\mathbf{A}$:

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{A} \Pi ^c\mathbf{T}_w \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$

Secara mendalam, persamaannya adalah sebagai berikut:

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix} \begin{bmatrix} r_{11} & r_{12} & r_{13} & t_x \\ r_{21} & r_{22} & r_{23} & t_y \\ r_{31} & r_{32} & r_{33} & t_z \\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$

##### Transformasi Koordinat Dunia ke Kamera

Pose yang diestimasi (rvec dan tvec) memungkinkan transformasi titik 3D dari world frame ke camera frame:

$$\begin{bmatrix} X_c \\ Y_c \\ Z_c \\ 1 \end{bmatrix} = ^c\mathbf{T}_w \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix} = \begin{bmatrix} r_{11} & r_{12} & r_{13} & t_x \\ r_{21} & r_{22} & r_{23} & t_y \\ r_{31} & r_{32} & r_{33} & t_z \\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$

#### Metode SolvePnP yang Sering Digunakan

- `SOLVEPNP_ITERATIVE`: Berbasis optimasi Levenberg-Marquardt. Membutuhkan minimal 4 titik (planar) atau 6 titik (non-planar).
- `SOLVEPNP_P3P / AP3P`: Berbasis solusi aljabar untuk tepat 4 titik.
- `SOLVEPNP_EPNP`: Metode efisien untuk n-buah titik.
- `SOLVEPNP_IPPE`: Khusus untuk titik objek yang berada dalam satu bidang (koplanar), sering digunakan untuk estimasi marker.
- `SOLVEPNP_SQPNP`: Solusi optimal global untuk 3 titik atau lebih.

#### Fungsi Penting

- `cv.solvePnP()`: Mengembalikan satu solusi pose terbaik.
- `cv.solvePnPRansac()`: Menggunakan skema RANSAC untuk menangani pencilan (outliers/data kotor).
- `cv.solvePnPGeneric()`: Memungkinkan pengambilan semua solusi yang mungkin (beberapa metode menghasilkan lebih dari satu pose).
- `cv.solvePnPRefineLM()` / VVS: Digunakan untuk memperbaiki (refine) akurasi pose yang sudah ada menggunakan minimisasi non-linear.

#### Kode `SolvePnP`

```python
import cv2
import numpy as np

# 1. Siapkan titik 3D (Object Points) dan titik 2D (Image Points)
# Contoh: 4 titik sudut persegi di dunia nyata (satuan mm)
obj_points = np.array([[0, 0, 0],
                       [50, 0, 0],
                       [50, 50, 0],
                       [0, 50, 0]], dtype=np.float32)

# Koordinat piksel yang terdeteksi di gambar
img_points = np.array([[245, 120],
                       [510, 125],
                       [515, 380],
                       [240, 375]], dtype=np.float32)

# 2. Masukkan parameter kamera (Intrinsic Matrix & Distortion Coefficients)
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4,1)) # Asumsi tanpa distorsi

# 3. Hitung Pose
success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

if success:
    print("Rotation Vector:\n", rvec)
    print("Translation Vector:\n", tvec)
```

## [Roboflow](#roboflow)

Roboflow adalah platform manajemen dataset model computer vision yang menyediakan alat untuk membuat dataset, melabeli data, dan export dataset ke YOLO.

Lihat link berikut ini untuk mulai melabel dataset dengan Roboflow: [roboflow.com/annotate](https://roboflow.com/annotate)
Link dokumentasi Roboflow: [docs.roboflow.com](https://docs.roboflow.com/)
Link Quickstart Roboflow: [blog.roboflow.com/getting-started-with-roboflow](https://blog.roboflow.com/getting-started-with-roboflow/)

### Alur kerja Roboflow (praktis)

Tujuan utama Roboflow di pipeline ini adalah memastikan dataset rapi, konsisten, dan mudah diekspor ke format training (misalnya YOLO).

#### Buat project dan definisikan kelas

- Tentukan *class list* dari awal dan jangan berubah-ubah nama kelas di tengah jalan.
- Pilih tipe anotasi sesuai kebutuhan: **Detection** (bounding box) atau **Segmentation** (mask).

#### Upload data (raw flight footage lebih baik)

- Usahakan data merepresentasikan kondisi UAV: variasi ketinggian, angle kamera, waktu (pagi/siang/sore), cuaca, dan blur.
- Hindari hanya mengambil data “gambar bagus dari internet” jika deployment-nya pakai rolling shutter / banyak motion blur.

#### Labeling + quality control

Checklist cepat yang biasanya paling sering bikin training gagal:

- **Bounding box konsisten** (jangan hari ini super ketat, besok longgar).
- **Objek terlalu kecil**: tentukan aturan minimum (misal area/pixel threshold) supaya model tidak belajar dari noise.
- **Occlusion**: sepakati aturan tim (tetap label sebagian vs skip) agar konsisten.

#### Generate dataset version (split + augmentasi)

- Pastikan split **train/val/test** benar. Default yang aman: train dominan, test kecil tapi representatif.
- Augmentasi pakai seperlunya dan masuk akal untuk UAV: motion blur ringan, brightness/contrast, perspective/rotate kecil.

#### Export ke format training

- Untuk YOLO: export ke **YOLOv5/YOLOv8** (format folder `images/` + `labels/` dan file `.yaml`).
- Untuk ekosistem lain: export ke **COCO** (umum untuk training/benchmarking).

Catatan penting: export format bukan cuma “beda folder”. Pastikan juga **konvensi koordinat** dan **nama kelas** sesuai tool training yang dipakai.

## [Yolo](#yolo)

Ultralytics YOLO adalah framework populer untuk deteksi/segmentasi/pose yang menyeimbangkan kecepatan dan akurasi. Framework ini menyediakan antarmuka Python dan Command-Line Interface (CLI).

### Instalasi YOLO

Metode tercepat adalah menggunakan `pip`. Pastikan sistem sudah terinstal Python. Untuk akselerasi GPU, instal PyTorch yang sesuai dengan versi CUDA/driver yang tersedia.

#### Instalasi Standar

```bash
pip install -U ultralytics

```

### Alur Kerja Dasar dengan Python

Antarmuka Python YOLO memungkinkan integrasi yang mulus ke dalam proyek aplikasi. Berikut adalah alur kerja utamanya:

```python
from ultralytics import YOLO

# 1. Load Model
# Contoh: gunakan model pretrained yang tersedia di Ultralytics.
# (Nama file bisa berbeda tergantung versi Ultralytics yang kamu pakai.)
model = YOLO("yolov8n.pt")

# 2. Training (Opsional)
# Melatih model pada dataset kustom
results = model.train(data="coco8.yaml", epochs=3, imgsz=640)

# 3. Validasi
# Mengevaluasi performa model pada set validasi
metrics = model.val()

# 4. Prediksi / Inferensi
# Menjalankan deteksi pada gambar, video, atau URL
results = model("https://ultralytics.com/images/bus.jpg")

# 5. Export
# Mengonversi model ke format lain (ONNX, TensorRT, OpenVINO)
model.export(format="onnx")

```

### Alur kerja via CLI (ringkas)

CLI berguna untuk workflow cepat dan reproducible:

```bash
# Training
yolo detect train model=yolov8n.pt data=path/to/dataset.yaml epochs=100 imgsz=640

# Validasi
yolo detect val model=runs/detect/train/weights/best.pt data=path/to/dataset.yaml

# Predict
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/images

# Export (contoh ONNX)
yolo export model=runs/detect/train/weights/best.pt format=onnx
```

Catatan UAV: fokuskan pada kestabilan runtime (FPS stabil, spike rendah) dan *freshness* deteksi, bukan hanya mAP.

### Pipeline YOLO

#### Dataset YAML (minimal)

Ultralytics menggunakan file `.yaml` untuk mendeskripsikan dataset.

```yaml
# dataset.yaml
path: /abs/path/to/dataset
train: images/train
val: images/val

names:
  0: person
  1: car
```

Checklist dataset:

- Struktur folder konsisten: `images/{train,val,test}` dan `labels/{train,val,test}`.
- Label YOLO ter-normalisasi (0..1) dan sesuai `names`.
- Pastikan variasi data sesuai kondisi UAV (altitude/angle/lighting/motion blur).

#### Preprocessing & postprocessing harus “match”

Untuk deployment, *yang paling sering bikin hasil meleset bukan modelnya, tapi mismatch preprocessing/postprocessing*.

- Preprocess umum YOLO export: **letterbox** -> **BGR to RGB** -> **/255** -> **NCHW float32**.
- Postprocess umum: confidence threshold -> per-class filtering (opsional) -> NMS -> koreksi letterbox -> clamp ke ukuran frame asli.

#### Export checklist (agar ONNX/engine tidak bikin kejutan)

- Tetapkan input size (mis. `imgsz=640`) dan usahakan fixed-shape untuk latency stabil.
- Validasi hasil export minimal dengan 1–2 gambar yang sama: PyTorch (`.pt`) vs ONNX (cek apakah box kira-kira sama).
- Simpan metadata yang dibutuhkan deployment: `imgsz`, jenis letterbox yang dipakai, class list, threshold.

#### Benchmarking cepat sebelum terbang

- Ukur latency end-to-end (capture→preprocess→inference→postprocess), bukan hanya waktu inference.
- Lakukan warmup beberapa kali; batch size 1 biasanya paling relevan untuk UAV.

### Mode Operasi YOLO

YOLO memiliki beberapa mode utama yang mencakup seluruh siklus pengembangan model:

#### Train Mode

Digunakan untuk melatih model pada dataset kustom.

```python
model.train(data="path/to/dataset.yaml", epochs=100, imgsz=640, device=0)

```

#### Predict Mode

Digunakan untuk melakukan inferensi pada berbagai sumber data. YOLO mendukung banyak format input:

```python
# Dari webcam
results = model.predict(source="0", show=True)

# Dari folder berisi banyak gambar
results = model.predict(source="folder_path/", save=True)

# Dari ndarray (OpenCV)
import cv2
img = cv2.imread("image.jpg")
results = model.predict(source=img)

```

#### Track Mode

Digunakan untuk pelacakan objek secara *real-time* dalam aliran video.

```python
# Menggunakan tracker default (BoT-SORT) atau ByteTrack
results = model.track(source="video.mp4", show=True, tracker="bytetrack.yaml")

```

#### Export Mode

Penting untuk deployment di hardware khusus. Mode ini mengubah model `.pt` menjadi format yang lebih optimal.

```python
model.export(format="engine")    # Export ke TensorRT
model.export(format="openvino")  # Export ke OpenVINO

```

Catatan: export `engine` (TensorRT) biasanya butuh environment NVIDIA yang sesuai (driver/CUDA/TensorRT). Jika gagal, mulai dari export `onnx` dulu lalu optimasi bertahap.

### Konfigurasi Global (Settings)

Ultralytics memiliki `SettingsManager` untuk mengatur direktori penyimpanan dataset dan hasil eksperimen secara permanen.

```python
from ultralytics import settings

# Melihat semua konfigurasi saat ini
print(settings)

# Mengubah direktori penyimpanan dataset
settings.update({"datasets_dir": "/path/to/custom/datasets"})

# Reset ke pengaturan awal
settings.reset()

```

### Ringkasan Fitur Lanjutan

| Fitur | Deskripsi |
| --- | --- |
| **Multi-Task** | Mendukung Detection, Segmentation, OBB (Oriented Bounding Boxes), Pose, dan Classify. |
| **Benchmark** | Tool otomatis untuk membandingkan kecepatan model di berbagai format export (`onnx`, `openvino`, `tensorrt`). |
| **Pythonic API** | Semua hasil inferensi dikembalikan dalam bentuk objek yang mudah diakses (misal: `result.boxes`, `result.masks`). |
| **Trainers** | Arsitektur modular yang memungkinkan modifikasi komponen internal model bagi peneliti. |

### Contoh Visualisasi Hasil (Python)

```python
results = model("bus.jpg")

for r in results:
    # Menampilkan gambar dengan bounding box terplot
    r.show()
    
    # Menyimpan hasil ke file
    r.save(filename="result.jpg")
    
    # Mengambil koordinat box (format xyxy)
    print(r.boxes.xyxy)

```

## [Alat Inference](#alat-inference)

Inference adalah proses menjalankan model machine learning yang sudah dilatih untuk membuat prediksi pada data baru. Dalam konteks computer vision, inference sering digunakan untuk mendeteksi objek, mengklasifikasikan gambar, atau melakukan segmentasi gambar secara real-time. Pada section ini kita belajar bagaimana menjalankan model yang dihasilkan dari yolo menggunakan beberapa inference engine populer seperti ONNX Runtime, NVIDIA TensorRT, dan OpenVINO.

### Ringkasan pemilihan inference engine

| Engine | Cocok untuk | Kelebihan | Catatan / Risiko |
| --- | --- | --- | --- |
| ONNX Runtime | Validasi & deployment umum (CPU/GPU) | Setup relatif mudah, banyak provider | Hasil sangat bergantung preprocessing/postprocess; GPU butuh kompatibilitas driver/CUDA |
| TensorRT | NVIDIA GPU/Jetson (latency rendah) | Performa tinggi, optimasi FP16/INT8 | Setup paling sensitif versi CUDA/TensorRT; debugging lebih sulit |
| OpenVINO | CPU/iGPU/NPU Intel | Optimasi bagus di hardware Intel, opsi AUTO/HETERO | Konversi/kompatibilitas operator bisa jadi pembatas |

### ONNX Runtime

ONNX Runtime adalah inference engine general-purpose untuk menjalankan model format ONNX di berbagai backend (CPU, CUDA, TensorRT, OpenVINO).

#### Instalasi ONNX Runtime

Terdapat dua paket Python untuk ONNX Runtime. Hanya satu dari paket ini yang boleh diinstal pada satu environment. Paket GPU mencakup sebagian besar fungsi CPU.

##### Install ONNX Runtime CPU

```bash
pip install onnxruntime
```

##### Install ONNX Runtime GPU (CUDA 12.x)

```bash
pip install onnxruntime-gpu
```

##### Install ONNX Runtime GPU (CUDA 11.8)

```bash
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
```

#### ONNX Inference

##### Contoh inferensi minimal (Python)

Contoh ini menunjukkan cara menjalankan model ONNX, mengambil nama input, dan menjalankan `session.run()`.

```python
import numpy as np
import onnxruntime as ort

# 1) Buat session (default CPU)
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

# 2) Ambil metadata input
input0 = session.get_inputs()[0]
input_name = input0.name
print("Input name:", input_name)
print("Input shape:", input0.shape)
print("Input type:", input0.type)

# 3) Siapkan tensor input (contoh random). Pastikan dtype/layout sesuai modelmu.
# Banyak model deteksi export dari YOLO memakai layout NCHW dan float32.
x = np.random.randn(1, 3, 640, 640).astype(np.float32)

# 4) Run inferensi. `None` = ambil semua output.
outputs = session.run(None, {input_name: x})
print("Num outputs:", len(outputs))
```

##### Inference Dengan Pilihan GPU atau CPU

List Inference Provider: [onnxruntime.ai/docs/execution-providers](https://onnxruntime.ai/docs/execution-providers)

```python
import onnxruntime as ort

print("Available providers:", ort.get_available_providers())

session = ort.InferenceSession(
  "model.onnx",
  providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
```

##### Menambahkan Session Options

```python
import onnxruntime as ort

options = ort.SessionOptions()
options.enable_profiling = True

session = ort.InferenceSession(
  "model.onnx",
  sess_options=options,
  providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
```

#### Checklist umum agar hasil tidak “aneh”

- Pastikan **preprocessing** sama dengan saat training/export (resize/letterbox, urutan channel, normalisasi, mean/std).
- Cocokkan **layout** input (umumnya NCHW untuk ONNX dari YOLO) dan **dtype** (umumnya `float32`).
- Selalu cek **nama input** lewat `session.get_inputs()` (jangan hardcode kalau tidak yakin).
- Output model deteksi biasanya masih mentah: perlu **confidence threshold + NMS + koreksi letterbox** di postprocessing.

#### Pipeline inference

Urutan yang aman untuk deployment:

1. Jalankan `.pt` (Ultralytics) untuk dapatkan baseline hasil.
2. Export ke ONNX fixed-shape (mis. 640x640) dan jalankan di ONNX Runtime CPU.
3. Pastikan hasil “masuk akal” (box/label) sebelum optimasi performa.
4. Baru pindah ke GPU provider (CUDAExecutionProvider) atau engine spesifik (TensorRT/OpenVINO).
5. Optimasi: FP16/INT8 (jika hardware mendukung) + profiling latency end-to-end.

##### Profiling latency yang realistis

Contoh sederhana

```python
import time
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
x = np.random.randn(1, 3, 640, 640).astype(np.float32)

# warmup
for _ in range(10):
  session.run(None, {input_name: x})

t0 = time.perf_counter()
iters = 50
for _ in range(iters):
  session.run(None, {input_name: x})
t1 = time.perf_counter()

print("avg ms:", (t1 - t0) * 1000 / iters)
```

### NVIDIA TensorRT

NVIDIA TensorRT adalah SDK untuk optimasi model deep learning guna menghasilkan inferensi berperforma tinggi. TensorRT mencakup optimizer inferensi dan runtime untuk eksekusi, yang memungkinkan model berjalan dengan throughput lebih tinggi dan latensi lebih rendah.

![tensorrt](img/tensorrt.png)

#### Instalasi TensorRT

Terdapat beberapa metode utama untuk menginstal TensorRT tergantung pada kebutuhan lingkungan pengembanganmu.

##### Menggunakan Pip (Python Wheel)

Catatan: instalasi via `pip` tidak selalu tersedia/berhasil untuk semua kombinasi OS, versi Python, driver GPU, dan versi CUDA. Untuk workflow yang paling konsisten, biasanya gunakan **NGC container** atau install dari repo NVIDIA.

```bash
pip install tensorrt
```

##### Menggunakan File Debian (Ubuntu/Linux)

```bash
# Contoh untuk Debian-based
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2x04-cudaX.X-trtX.X.X.X_1.0-1_amd64.deb
sudo apt-get update
sudo apt-get install tensorrt
```

##### Menggunakan Container (NVIDIA NGC)

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/tensorrt:24.01-py3
```

#### Alur Kerja Dasar (Workflow)

Untuk men-deploy model, kamu perlu mengikuti lima langkah dasar berikut:

1. Export Model: Ubah model dari framework asli (PyTorch/TF) ke format perantara (seperti ONNX).
2. Select Precision: Pilih presisi numerik (FP32, FP16, INT8).
3. Convert Model: Ubah model menjadi TensorRT Engine.
4. Deploy Model: Jalankan engine menggunakan runtime API.

#### Konversi dan Deployment

![konversi](img/convertion.png)

##### Opsi Konversi

- Torch-TensorRT: Integrasi langsung di dalam PyTorch.
- ONNX Conversion (Otomatis): Menggunakan tool trtexec untuk mengubah file .onnx menjadi engine.
- Nsight Deep Learning Designer: Tool berbasis GUI untuk visualisasi dan konversi.
- API Network Definition: Membangun jaringan secara manual lapis demi lapis (C++ atau Python).

##### Opsi Deployment

- Standalone Runtime API: Performa tertinggi dengan overhead paling rendah.
- Triton Inference Server: Untuk deployment skala produksi/cloud yang mendukung HTTP/gRPC.

#### Contoh Implementasi: ONNX ke TensorRT

##### Konversi dengan trtexec

Gunakan tool baris perintah ini untuk mengubah model ONNX menjadi engine TensorRT dengan presisi FP16:

```bash
trtexec --onnx=model.onnx --saveEngine=model_engine.engine --fp16
```

##### Inferensi dengan Python Runtime

Implementasi Python TensorRT lengkap biasanya melibatkan:

- Membaca engine lalu membuat execution context
- Menyiapkan bindings untuk *semua* input & output
- Mengelola CUDA stream + memcpy HtoD/DtoH

Karena contoh lengkapnya cukup panjang dan bergantung versi TensorRT (API bisa berubah), untuk tahap awal biasanya lebih aman:

- Gunakan `trtexec` untuk konversi + benchmark.
- Untuk sistem produksi, gunakan **Triton Inference Server** (HTTP/gRPC) atau contoh resmi TensorRT.

#### Referensi API (Runtime)

##### nvinfer1::ICudaEngine

Class ini merepresentasikan model yang sudah dioptimasi (Engine).

- create_execution_context(): Membuat konteks eksekusi untuk menjalankan inferensi.
- get_tensor_shape(tensor_name): Mengambil dimensi input/output dari model.
- get_tensor_dtype(tensor_name): Mengambil tipe data tensor.

##### nvinfer1::IExecutionContext

Class utama untuk mengatur state eksekusi dan menjalankan inferensi.

- execute_v2(bindings): Menjalankan inferensi secara sinkron pada batch data (deprecated di versi terbaru, gunakan V3).
- enqueue_v3(stream): Menjalankan inferensi secara asinkron pada CUDA stream tertentu.
- set_input_shape(name, shape): Menentukan dimensi input jika menggunakan dynamic shapes.

### OpenVINO

OpenVINO (Open Visual Inference and Neural Network Optimization) adalah toolkit open-source dari Intel untuk mengoptimalkan dan men-deploy model AI dari berbagai framework (PyTorch, TensorFlow, ONNX) ke perangkat keras Intel (CPU, GPU terintegrasi, NPU).

![OpenVINO](img/openvino.png)

#### Instalasi OpenVINO

##### Install OpenVINO Runtime (Python)

```bash
pip install openvino
```

##### Install OpenVINO Development Tools (Untuk Konversi Model)

```bash
pip install openvino-dev
```

#### Alur Kerja OpenVINO

- Convert: Mengonversi model dari format asli (PyTorch/TF) menjadi format OpenVINO IR (.xml dan .bin) menggunakan API ov.convert_model.
- Optimize: Menerapkan teknik kompresi seperti kuantisasi (INT8) menggunakan NNCF (Neural Network Compression Framework).
- Deploy: Menjalankan inferensi menggunakan OpenVINO Runtime API yang secara otomatis mengoptimalkan beban kerja di hardware Intel.

#### Implementasi Inferensi (Python)

##### Load Model dan Eksekusi

Berikut adalah cara dasar untuk memuat model dan menjalankan inferensi:

```python
import openvino as ov
import numpy as np

# 1. Inisialisasi Core
core = ov.Core()

# 2. Baca model (format IR, ONNX, atau PaddlePaddle)
model = core.read_model("model.xml")

# 3. Compile model untuk perangkat tertentu (misal: CPU atau GPU)
compiled_model = core.compile_model(model, "CPU")

# 4. Jalankan inferensi
infer_request = compiled_model.create_infer_request()
input_data = np.random.randn(1, 3, 224, 224)
results = compiled_model([input_data])[compiled_model.output(0)]
```

##### Konversi Model Langsung dari PyTorch

Pada beberapa kasus, OpenVINO bisa mengonversi dari model PyTorch tanpa harus mengekspor ke ONNX terlebih dahulu (kompatibilitas tergantung operator/model):

```python
import openvino as ov
import torch

model = MyPyTorchModel() # Model pytorch
ov_model = ov.convert_model(model)
ov.save_model(ov_model, "model_ir.xml")
```

#### Advanced Features

##### Automated Device Configuration

- AUTO (Automatic Device Selection): Secara otomatis memilih hardware terbaik yang tersedia (misal: jika ada iGPU, ia akan memakainya, jika tidak, balik ke CPU).
- HETERO (Heterogeneous Execution): Membagi beban kerja model ke beberapa hardware sekaligus (misal: sebagian di CPU, sebagian di NPU).

##### Model Optimization (NNCF)

- Quantization-Aware Training (QAT): Melatih model agar tetap akurat meski menggunakan presisi INT8.
- Post-Training Quantization (PTQ): Mengompres model yang sudah dilatih tanpa perlu retraining.

##### Performance Hints

- LATENCY: Cocok untuk aplikasi real-time (seperti drone yang harus mendeteksi penyakit tanaman saat terbang).

- THROUGHPUT: Cocok jika kamu memproses ribuan foto sekaligus di server setelah drone mendarat.

```python
# Contoh setting prioritas Latency
compiled_model = core.compile_model(model, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
```

## [Website Penting](#website-penting)

### Website Sumber Dataset UAV Terbuka

- Roboflow Universe: [universe.roboflow.com](https://universe.roboflow.com)
- Kaggle Datasets: [www.kaggle.com/datasets](https://www.kaggle.com/datasets)
- Mendeley Data: [data.mendeley.com](https://data.mendeley.com)
- Google Dataset Search: [datasetsearch.research.google.com](https://datasetsearch.research.google.com)
- Hugging Face: [huggingface.co/datasets](https://huggingface.co/datasets)
- VisUAV: [github.com/VisUAV/VisUAV-Dataset](https://github.com/VisUAV/VisUAV-Dataset)

### Website untuk Membuat Dataset (Labelling)

- Roboflow Annotate: [roboflow.com/annotate](https://roboflow.com/annotate)
- Label Studio: [labelstud.io](https://labelstud.io/)
- CVAT: [github.com/openvinotoolkit/cvat](https://github.com/openvinotoolkit/cvat)
- LabelMe: [github.com/wkentaro/labelme](https://github.com/wkentaro/labelme)

### Website untuk visualisasi model (file ONNX)

- Netron: [netron.app](https://netron.app/)

## [Belajar Lebih Lanjut](#belajar-lebih-lanjut)

- Python Documentation: [docs.python.org](https://docs.python.org/3/)
- Roboflow Documentation: [docs.roboflow.com](https://docs.roboflow.com/)
- OpenCV Documentation: [docs.opencv.org](https://docs.opencv.org/4.x/)
- YOLO Documentation: [docs.ultralytics.com](https://docs.ultralytics.com/)
- ONNX Runtime Documentation: [onnxruntime.ai/docs](https://onnxruntime.ai/docs/)
- TensorRT Documentation: [docs.nvidia.com](https://docs.nvidia.com/deeplearning/tensorrt/latest/)
- OpenVINO Documentation: [docs.openvino.ai](https://docs.openvino.ai/2025/index.html)
