# SRS.md

## Project: G-Transformer

### Version: 1.0

### Title: Energy-Efficient Transformer Architecture Based on Genesis Information Theory (GIT)

### Author: Syamsuddin B. Ideris, S.Pd.MM

### Institution: SMPN 3 Kandangan – Independent Researcher

---

## 1. Introduction

### 1.1 Purpose

Dokumen ini menjelaskan kebutuhan dan rancangan arsitektur **G-Transformer**, model Large Language Model (LLM) hemat energi berdasarkan **Genesis Information Theory (GIT)**. Tujuannya adalah mengurangi konsumsi daya komputasi AI dengan memandang seluruh proses neural network sebagai **transfer dan transformasi informasi (I)** yang tunduk pada hukum energi-informasi (E = k_I \cdot T \cdot I).

### 1.2 Scope

G-Transformer adalah varian arsitektur Transformer yang:

1. Mengoptimalkan efisiensi energi per token (J/token).
2. Meminimalkan operasi berlebih melalui mekanisme **Informational Attention (IA-Attention)**.
3. Mengompresi cache dan mengurangi redundansi data berdasarkan **kontribusi informasi ΔI**.
4. Mengatur konsumsi daya GPU melalui kontrol **DVFS (Dynamic Voltage and Frequency Scaling)** berbasis laju informasi.
5. Menjaga akurasi dan stabilitas seperti Transformer konvensional.

### 1.3 Objectives

* Mengurangi energi inferensi dan pelatihan hingga 80%.
* Meningkatkan rasio **FLOPS/Watt** hingga 3× dibanding FP16 Transformer standar.
* Menghasilkan model yang lebih efisien secara termodinamika, tanpa kehilangan akurasi signifikan.

---

## 2. Theoretical Foundation

### 2.1 Core Principle

GIT memandang energi dan informasi sebagai ekuivalen:
[
E = k_I , T , I
]
di mana (k_I) adalah konstanta informasi fundamental.

### 2.2 Information Flow Equation

Total energi operasi model dinyatakan sebagai:
[
E_{\text{total}} = N_{\text{ops}} E_{\text{op}} + N_{\text{bytes}} E_{\text{bit}} + E_{\text{idle}}
]
Efisiensi informasi dihitung sebagai:
[
\eta_I = \frac{I_{\text{useful}}}{I_{\text{total}}}
]
dan tujuan optimasi adalah memaksimalkan (\eta_I) dengan batasan kehilangan akurasi minimum.

---

## 3. Functional Requirements

### 3.1 Informational Attention (IA-Attention)

* Menghitung **ΔI per token** sebagai ukuran relevansi.
* Mengabaikan koneksi antar token dengan ΔI di bawah ambang ε.
* Kompleksitas efektif: O(n·w + r·n).
* Mengurangi operasi atensi hingga 10×.

### 3.2 Low-Rank Feed Forward Network (LR-FFN)

* Faktorisasi matriks bobot (W = UΣV^T).
* Sparsity 2:4 pada bobot dan aktivasi.
* Presisi FP8 untuk matmul, FP16 akumulasi.
* Penghematan energi 2–3× dibanding FFN standar.

### 3.3 Entropy-Based MoE Router

* Aktivasi expert hanya jika ΔI_expert ≥ ε_expert.
* Fungsi penalti:
  [
  L_{\text{info}} = λ \cdot I_{\text{waste}}
  ]
* Mengurangi FLOPS tanpa degradasi kualitas.

### 3.4 KV-Cache Compression

* Hitung skor informasi (r = ||k|| \cdot ||v||).
* Simpan hanya vektor dengan r > ε_kv.
* Memori KV turun 2–8×.

### 3.5 Delta-Gradient Communication

* Kirim hanya perubahan gradien signifikan:
  [
  Δg = g_t - g_{t-1}
  ]
* Kompresi INT8 dan entropi coding.
* Penghematan energi komunikasi hingga 80%.

### 3.6 DVFS & Information Scheduler

* Sensor energi dan suhu membaca σ lokal GPU.
* Tegangan diturunkan terlebih dahulu sebelum frekuensi.
* Pembagian daya: compute 60%, memory 25%, I/O 10%, control 5%.
* Scheduler menyeimbangkan panas dan laju informasi.

---

## 4. Non-Functional Requirements

| Category        | Description                                                             |
| --------------- | ----------------------------------------------------------------------- |
| Performance     | Latency ≤ 1.2× Transformer FP16, dengan konsumsi daya ≤ 0.2×            |
| Scalability     | Dapat berjalan pada GPU single node hingga cluster multi-node           |
| Reliability     | Self-adaptive terhadap beban data variatif                              |
| Maintainability | Modular (Attention, FFN, Cache, MoE, Scheduler)                         |
| Portability     | Kompatibel dengan PyTorch, TensorFlow, dan framework C++ custom         |
| Energy          | Efisiensi minimum 0.1 J/token (inferensi), 0.5 kWh/epoch (training 13B) |

---

## 5. System Architecture

### 5.1 Main Components

1. **IA-Attention Module** – menyeleksi token informatif.
2. **LR-FFN** – menghemat operasi linear.
3. **KV-Compressor** – kompresi cache.
4. **Entropy-MoE Router** – routing berbasis kontribusi ΔI.
5. **Delta-Gradient Engine** – hemat komunikasi.
6. **DVFS Controller** – optimasi energi runtime.
7. **Information Scheduler** – menjaga keseimbangan beban dan suhu.

### 5.2 Energy Flow

[
E_{\text{total}} = E_{\text{compute}} + E_{\text{memory}} + E_{\text{I/O}} + E_{\text{control}}
]
Target reduksi (E_{\text{total}}) hingga 70–85%.

---

## 6. Hardware Requirements

| Component    | Specification                                       |
| ------------ | --------------------------------------------------- |
| GPU          | NVIDIA H100 / MI300X / RTX 4090                     |
| Memory       | ≥ 96 GB HBM3                                        |
| Storage      | NVMe SSD 4 TB                                       |
| Cooling      | Gitton Cooling System (GCS) hybrid fluid-electronic |
| Power Supply | 2.4 kW efficiency 94%                               |
| Sensors      | Power, temperature, entropy (software metric)       |

---

## 7. Data and Training

### 7.1 Dataset

* Text corpus open-source (Pile, RedPajama, OSCAR).
* Metadata tambahan: entropy score dan ΔI log per batch.

### 7.2 Training Objectives

[
L_{\text{total}} = L_{\text{crossentropy}} + λ L_{\text{info}}
]
dengan:
[
L_{\text{info}} = λ \cdot (I_{\text{total}} - I_{\text{useful}})
]

### 7.3 Monitoring Metrics

* J/token
* FLOPS/Watt
* Memory Access Efficiency
* ΔEntropy per Layer
* Accuracy Delta (Δloss)

---

## 8. Verification and Testing

| Test Type   | Description                       | Metric                    |
| ----------- | --------------------------------- | ------------------------- |
| Unit Test   | Validasi modul IA-Attention & FFN | Accuracy ±1e-5            |
| Integration | Sinkronisasi ΔI antar modul       | Stability over 10k tokens |
| Energy Test | Bandingkan J/token                | ≤ 0.1 J/token             |
| Stress Test | Context 64k token                 | No overflow               |
| Robustness  | Simulasi noise data               | Δloss ≤ 0.5%              |

---

## 9. Future Roadmap

1. Integrasi dengan **GitPU** (GPU berbasis GIT).
2. Implementasi hardware-aware compiler untuk DVFS.
3. Adaptasi ke model multimodal (text, image, speech).
4. Kolaborasi dengan laboratorium energi rendah AI.

---

## 10. References

1. Syamsuddin B. Ideris – *Genesis Information Theory (GIT v2.0)*, 2025.
2. Vaswani et al. – *Attention Is All You Need*, 2017.
3. OpenAI & NVIDIA whitepapers on FP8 training, 2023–2024.
4. MIT CSAIL – *Energy-aware Neural Architecture Design*, 2022.

---
