# build2aiapi — Yapısal Analiz Motoru

Build2AI'nın arka uç servisi. FastAPI + Pydantic + Firebase üzerinde, SAP2000 `.s2k` modellerini uçtan uca çözen bir sonlu elemanlar motoru içerir. Tüm matematik orijinal İTÜ "Sonlu Elemanlar Analizi" ders kitabından (SEA_Book) port edilmiş ve bire bir karakterize edilmiştir.

---

## 1. Büyük Resim

```
┌──────────────────────────────────────────────────────────────────┐
│  build2ai (Nuxt)                                                 │
│    AI Chat · 3D Viewer · Analiz Paneli · Config Modal            │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTPS + Bearer (Firebase ID token)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  build2aiapi (FastAPI)                                           │
│                                                                  │
│    routers/          auth · projects · files · chat · analysis   │
│    services/         firebase · storage · structural_analysis    │
│    repositories/     Firestore CRUD (users/projects/files/       │
│                      analyses)                                   │
│                                                                  │
│    services/structural_analysis/   ← FEM motoru                  │
│      parser/   model/   validation/   elements/                  │
│      assembly/ solver/  recovery/     spectra/                   │
│      results/  pipeline.py                                       │
└──────────────────────────────────────────────────────────────────┘
                             │
                             ▼
                  Firebase Auth · Firestore · Storage
```

Motor iki modda çalışır: komuta satırı (`pytest`, Python scriptleri) ve HTTP API. Aynı `ModelDTO` tüm yolları besler.

---

## 2. FEM Boru Hattı (Pipeline)

Tek bir fonksiyon akışı: `pipeline.run_static_analysis(model, options)`.

```
.s2k metni
   │
   ▼
┌──────────────┐   SAP tablolarını ayıklar
│   PARSE      │   JOINT COORDS, FRAME CONN, AREA CONN, MATERIALS,
│ parser/s2k.. │   SECTIONS, LOADS, COMBINATIONS, DIAPHRAGMS, RELEASES,
└──────┬───────┘   MASS SOURCE → ModelDTO
       ▼
┌──────────────┐   Çözüm öncesi sağlama
│  VALIDATE    │   Eksik atama, sıfır rijitlik, bağlantısız düğüm,
│ validation/  │   mesnet yokluğu — her hata kullanıcıya uyarı
└──────┬───────┘   olarak döner
       ▼
┌──────────────┐   Partitioned numaralama: serbestler [0..N), tutulular [N..M)
│  DOF NUMBER  │   Her düğüm 6 DOF (ux, uy, uz, rx, ry, rz)
│ assembly/    │
└──────┬───────┘
       ▼
┌──────────────┐   Her eleman için lokal K → global dönüşüm → scatter
│  ASSEMBLE K  │   Frame_3d (12×12) + ShellQ4 (24×24) birleşik
│ stiffness_..  │   Sparse COO → CSC
└──────┬───────┘
       ▼
┌──────────────┐   PS (nokta yükleri) + RHS (dağıtılmış/alan/öz ağırlık)
│  ASSEMBLE F  │   + area loads çevre frame'lere
│ load_assembler│  + öz ağırlık ρ·g·A·L
└──────┬───────┘
       ▼
┌──────────────┐   Opsiyonel: CONSTRAINT DEFINITIONS - DIAPHRAGM
│  DIAPHRAGM   │   master-slave dönüşüm matrisi T
│ constraints  │   K_reduced = Tᵀ K T, F_reduced = Tᵀ F
└──────┬───────┘
       ▼
┌──────────────┐   K11 U1 = P1 + RHS1 - K12 U2 (partitioned)
│  STATIC      │   scipy.sparse.linalg.spsolve
│  SOLVE       │   Singular tespit → Tikhonov regularization
└──────┬───────┘
       ▼
┌──────────────┐   U_reduced → expand_U → U_full (tüm DOF'lar)
│  RECOVERY    │   Her düğüme yer değiştirme sözlüğü
│              │   Mesnet reaksiyonları = K U - RHS (tutulu kısım)
└──────┬───────┘
       ▼
       [opsiyonel] KOMBİNASYON — lineer süperpozisyon:
                    U_combo = Σ (factor_i × U_case_i)
       [opsiyonel] MODAL — K φ = ω² M φ  →  scipy.sparse.linalg.eigsh
                           M kütle matrisi MASS SOURCE kuralına göre kurulur
       [opsiyonel] RESPONSE SPECTRUM (TBDY 2018) — modal katılım + SRSS
       ▼
   AnalysisResult → serialize → Firestore (metadata + sonuç)
```

---

## 3. Desteklenen SAP Tabloları

Parser şu `.s2k` tablolarını okur:

| Kategori | Tablo | Motor davranışı |
|---|---|---|
| Temel | PROGRAM CONTROL | birim sistemi (kN, m, C) |
| Malzeme | MATERIAL PROPERTIES 01 + 02 | E, ν, ρ (UnitMass) |
| Kesit | FRAME SECTION PROPERTIES 01 - GENERAL | A, Iy, Iz, J (TorsConst) |
| Kesit | AREA SECTION PROPERTIES | kalınlık |
| Geometri | JOINT COORDINATES | düğüm koordinatları |
| Geometri | CONNECTIVITY - FRAME | frame elemanları |
| Geometri | CONNECTIVITY - AREA | shell elemanları (Q4) |
| Mesnet | JOINT RESTRAINT ASSIGNMENTS | ux/uy/uz/rx/ry/rz kısıtları |
| Kısıt | CONSTRAINT DEFINITIONS - DIAPHRAGM + JOINT CONSTRAINT ASSIGNMENTS | rijit diyafram (axis=Z) |
| Mafsal | FRAME RELEASE ASSIGNMENTS 1 - GENERAL | P/V2/V3/T/M2/M3 — statik kondenzasyon |
| Atama | FRAME SECTION ASSIGNMENTS | frame → kesit + malzeme |
| Atama | AREA SECTION ASSIGNMENTS | shell → kesit + malzeme |
| Yük | LOAD PATTERN DEFINITIONS | G, Q, EQX, EQY vb. (+ SelfWtMult) |
| Yük | JOINT LOADS - FORCE | tekil yükler |
| Yük | FRAME LOADS - DISTRIBUTED | yayılı frame yükleri (gravity/x/y/z/local) |
| Yük | AREA LOADS - UNIFORM TO FRAME | döşeme yükü → çevre frame'lere dağıtım |
| Kombinasyon | COMBINATION DEFINITIONS | lineer kombinasyonlar (CaseType=Linear Static) |
| Kütle | MASS SOURCE | Loads=Yes ise yük pattern'lerinden kütle |

**Henüz ele alınmayanlar:** CASE - STATIC 2/4 (nonlineer), FUNCTION - TIME HISTORY, PARTICIPATION RATIOS (manuel), CASE - RESPONSE SPECTRUM - LOAD ASSIGNMENTS. Spektrum analizinde motor kendi TBDY parametrelerini kullanır.

---

## 4. Eleman Kitaplığı

### 4.1 FrameElement3D (`elements/frame_3d.py`)

- Port kaynağı: SEA_Book `sec5_frame_3d.py`
- 12×12 lokal K; eksen dönüşümü `TOMG @ TALFA @ TBETA`
- Characterization testi: [tests/test_frame_3d.py](src/services/structural_analysis/tests/test_frame_3d.py) — 10 altın karşılaştırma (K_Local, K_Global, TLG, q_Local, B)
- **Frame releases (mafsallar):** `element.hinges = {"start": [...], "end": [...]}` — `p`, `v2`, `v3`, `t`, `m2`, `m3` etiketlerinden biri. Statik kondenzasyon:

  ```
  K_retained = K_rr - K_rc × inv(K_cc) × K_cr
  ```

  Released DOF'lar global birleştirmeye sıfır katkı verir.

### 4.2 PlaneStressQ4 (`elements/plane_stress_q4.py`)

- Port kaynağı: SEA_Book `sec4_plane_stress_rectangle_gauss.py`
- 4 düğümlü plane stress (membran) elemanı
- Gauss 2×2 integrasyon
- Characterization testi: [tests/test_plane_stress_q4.py](src/services/structural_analysis/tests/test_plane_stress_q4.py)

### 4.3 PlateBendingQ4 (`elements/plate_bending_q4.py`)

- Mindlin-Reissner plate bending
- Selective reduced integration:
  - Bending: Gauss 2×2 (tam)
  - Shear: 1×1 (locking'i önlemek için)
- 12×12 lokal K (w, θx, θy × 4 düğüm)
- Kirchhoff limit konvansiyonu: `θ_x = +∂w/∂y`, `θ_y = -∂w/∂x`
- Rigid-body modları (6) null space'te

### 4.4 ShellQ4 (`elements/shell_q4.py`)

Tam shell — 4 düğüm × 6 DOF = 24×24 global K:

```
Lokal 24×24 K'nın iç yapısı:
   membran (in-plane 8 DOF)  → u', v' × 4 düğüm
   plate bending (12 DOF)    → w', θ_x', θ_y' × 4 düğüm
   drilling (4 DOF)          → θ_z' stabilizasyon (membran max diag × 1e-4)
```

Lokal eksenler düğüm konumlarından türetilir:
```
x̂ = (n1 → n2) / |n1 → n2|
n̂ = (n1 → n2) × (n1 → n4)   → eleman normali
ŷ = n̂ × x̂
```

Global dönüşüm: her düğümde 6×6 blok T; 24×24 tam transform `T_big`:
```
K_global = T_bigᵀ × K_local × T_big
```

---

## 5. Assembler Detayları

### 5.1 DOF numaralama (`assembly/dof_numbering.py`)

Partitioned: serbest DOF'lar `[0..N-1]`, tutulular `[N..M-1]`. Bu sıralama partitioned solve (K11 U1 = ...) için doğrudan kullanışlıdır.

### 5.2 Rijit diyafram (`assembly/constraints.py`)

Rijit kat (axis=Z) için master-slave dönüşümü:

```
slave.ux = master.ux - (y_s - y_m) × master.rz
slave.uy = master.uy + (x_s - x_m) × master.rz
slave.rz = master.rz
```

`DiaphragmTransform.T` matrisi: `(M_full × M_reduced)`. Çözüm uzayı küçülür (her diyaframda 3×(n_joints - 1) DOF elenir). Çözümden sonra `expand_U(U_red)` ile tam U geri kurulur; reaksiyonlar full uzayda `K U - RHS` formülüyle bulunur.

### 5.3 Yük birleştirme (`assembly/load_assembler.py`)

Her yük durumu için üç vektör:
- `PS[code]` — düğüm yükleri (gerekliyse node Euler yerel eksenine rotated)
- `RHS[code]` — elemansal katkılar (`global_load_vector` × kod dağıtımı)
- `US[code]` — mesnet çökmeleri (şu an sıfır)

Elemansal yükler:
- **Yayılı frame yükleri**: `local_2` / `local_3` / `gravity` / `x` / `y` / `z` yönleri — element.element_axes_transform() ile global→lokal q
- **Öz ağırlık**: `w = ρ × g × A` kN/m, load pattern'in `SelfWtMult`'ü ile çarpılır
- **Area loads (UnifLoad kN/m²)**: alan × yoğunluk → toplam yük → çevre frame'lere perimetre oranında `q = totalLoad/perimeter` kN/m

### 5.4 Kütle matrisi (`assembly/mass_assembler.py`)

Lumped (yığılı) diagonal matris. SAP **MASS SOURCE** tablosu kurallarına uyar:

```
ms.from_elements = True  → her eleman ρ·V/2 ucuna (frame) veya ρ·V/4 düğümüne (shell)
ms.from_loads    = True  → her load pattern için |F_z| / g × multiplier,
                            düğüme lumped
ms.from_masses   = True  → (henüz yok) manuel nodal kütle atamaları
```

Tipik TBDY kurulumu: `Elements=No, Loads=Yes, G×1 + Q×0.3`. Motor bu satırları olduğu gibi uygular; `ρ` fallback'i (ρ=0 için çelik 7.85, beton 2.5 t/m³) yalnızca element-bazlı modlarda devreye girer.

---

## 6. Çözücüler

### 6.1 Statik (`solver/static_solver.py`)

Partitioned çözüm:
```
K11 U1 = P1 + RHS1 - K12 U2
P2 = K21 U1 + K22 U2 - RHS2
```

- `scipy.sparse.linalg.spsolve` — direct factorization (LU)
- **Tikhonov regularization**: K11 diyagonalinde sıfır-rijitlikli DOF varsa `α = max_diag × 1e-8` ile identity ekler. Sistem hala anlamlıysa pratik olarak sonuçları etkilemez; mekanizma artığı varsa `α`'ya yakın küçük yer değiştirmeler üretir.
- **Diaphragm transform** verilirse K/F önce reduce edilir, solve reduced'da yapılır, U full'a expand edilir.

### 6.2 Modal (`solver/modal_solver.py`)

```
K φ = ω² M φ    (genelleştirilmiş öz-değer)
```

- `scipy.sparse.linalg.eigsh(K11, k, M11, sigma=0, which="LM")` — en küçük `k` özdeğer
- Mod şekli normalize edilir (`max |φ| = 1` — UI görselleştirmesi için)
- `sigma=0` başarısız olursa fallback: `which="SM"`

Çıktı:
```
T_n = 2π / ω_n    f_n = 1 / T_n
```

Modal kütle katılım oranı: her mod için `Γ_n,d = φ_nᵀ M r_d / (φ_nᵀ M φ_n)`, `M_eff = Γ² × m_n_gen`. Bu hesaplama pipeline içinde modal result'lara eklenmiyor (dışarıdan hesaplanır — `tests/` altında örneği var).

### 6.3 Response Spectrum (`solver/response_spectrum.py`)

TBDY 2018 elastik + tasarım spektrumu (`spectra/tbdy_2018.py`):

```
T ≤ TA:          Sae = SDS × (0.4 + 0.6·T/TA)
TA < T ≤ TB:     Sae = SDS
TB < T ≤ TL:     Sae = SD1 / T
T > TL:          Sae = SD1 · TL / T²

SDS = Ss × FS,  SD1 = S1 × F1     (FS, F1 ZA..ZE için Tablo 2.1/2.2)
Ra(T) = R/I    (T ≥ TB)
        D + (R/I - D) × T/TB   (T < TB)
Sar(T) = Sae(T) × I / Ra(T)      tasarım ivmesi
```

Her yön (X, Y, Z) için modal SRSS:

```
per mode:  u_n = Γ_n × Sa(T_n) / ω_n² × φ_n
SRSS:      U_rs = √(Σ_n u_n²)
```

Sonuçlar `EQX_RS` / `EQY_RS` yük durumu olarak case listesine eklenir.

---

## 7. API Yüzeyi

Router: [src/routers/analysis.py](src/routers/analysis.py)

```
GET    /api/projects/{pid}/files/{fid}/preview
   Dosyayı parse et. Yük durumu listesi + kombinasyon listesi + model
   özeti + validator uyarıları. Frontend config modal'ını doldurur.

POST   /api/projects/{pid}/files/{fid}/analyze
   Body: AnalyzeRequestDto
     options:
       selected_load_cases: list | null        (null = hepsi)
       selected_combinations: list | null
       modal: bool
       modal_n_modes: int
       response_spectrum: bool
       spectrum_params: { Ss, S1, soil, R, I, run_x, run_y }
   Yanıt: AnalysisStatusDto (analiz kaydı + summary + warnings)

GET    /api/projects/{pid}/files/{fid}/analyses
GET    /api/projects/{pid}/files/{fid}/analyses/{aid}
GET    /api/projects/{pid}/files/{fid}/analyses/{aid}/summary
GET    /api/projects/{pid}/files/{fid}/analyses/{aid}/displacements[?load_case=]
GET    /api/projects/{pid}/files/{fid}/analyses/{aid}/reactions[?load_case=]
GET    /api/projects/{pid}/files/{fid}/analyses/{aid}/modes
DELETE /api/projects/{pid}/files/{fid}/analyses/{aid}
```

Tüm endpoint'ler Firebase Bearer ile korunur (`dependencies.get_uid`). Analiz sonuçları Firestore'da tek dokümanda saklanır (`users/{uid}/projects/{pid}/files/{fid}/analyses/{aid}`). NaN/inf değerleri JSON'a yazılmadan önce 0.0'a sanitize edilir (yanıt crash'i önlenir).

---

## 8. Doğrulama Felsefesi

Her matematiksel birim iki katman testle korunur:

1. **Altın karakterizasyon (characterization)**: SEA_Book orijinal kodundan üretilmiş referans çıktı (`tests/benchmarks/*.json`), port edilen sınıfın ürettiği matrislerle `numpy.testing.assert_allclose(rtol=1e-12)` ile karşılaştırılır. Bu, matematiği port sırasında değişmediğini garantiler.

2. **Entegrasyon testi**: Gerçek SAP fixture üzerinde uçtan uca smoke test. `sap_dd2_iter3.s2k` (3 katlı RC bina) için DOF sayısı, reaksiyon dengesi (`ΣFz = toplam uygulanan yük` tolerans %0.5), max yer değiştirme, temel periyot kontrol edilir.

66+ test çalışıyor:
- Parser smoke (5) + gerçek fixture (12)
- Frame_3d characterization (10)
- PlaneStressQ4 (5)
- PlateBendingQ4 (6)
- Pipeline E2E (3) + gerçek fixture (3) + advanced (6)
- Response spectrum (6)
- Router HTTP (8)

```bash
cd build2aiapi && source venv/bin/activate && pytest
```

---

## 9. Bilinen Sınırlar ve Yol Haritası

**Çalışıyor ve doğrulandı:**
- Statik lineer + modal + TBDY 2018 response spectrum
- Shell Q4 (membran + plate bending + drilling)
- Rijit diyafram (axis=Z), frame releases (mafsallar), area-to-frame yük dağıtımı
- MASS SOURCE (Loads=Yes — TBDY G+0.3Q formülü)
- Kombinasyonlar — Linear Static + NonLin Static (transitif expand ile iç içe referanslar)
- Otomatik deprem yükleri (AUTO SEISMIC LOADS TO JOINTS) — SAP'ın ±%5 eksantrisite türevleri
- Otomatik rüzgar yükleri (AUTO WIND LOADS TO JOINTS)
- Modal kütle katılım oranları (Mx, My, Mz + kümülatif + TBDY %95 kontrolü)
- Grid Lines (aks/kot) — her düğüme en yakın grid etiketi
- Excel export (tüm sonuç + yük durumu / reaksiyon / modlar ayrı sheet'ler)
- Firebase Storage gzip offload — 1MB Firestore limiti aşan büyük modeller

**Henüz yok (ileride):**
- **Eleman kesit tesirleri (M, V, N boyunca)** — mühendisin açıklık/mesnet momentleri için şart, pipeline'da recovery eksik
- Drift oranı (kat ötelenmesi) hesabı
- Otomatik yük kombinasyonları (kullanıcı SAP'tan almak zorunda; TBDY kombolarını bizden üretmiyoruz)
- P-Δ (geometrik nonlineer)
- CQC modal birleştirme (şu an SRSS; yakın periyotlarda SRSS yanlış)
- Time-history analizi
- Plastik mafsal / pushover (openseespy köprüsü)
- Shell Q9, T3, T6 / Eğrisel kabuk
- Tasarım kontrolleri (TS500, TS498, AISC)
- Trapezoidal yayılı yük (şu an `magnitude_a ≠ magnitude_b` atlanır)
- Mesnet çökmesi (`US`)
- Async jobs + SSE progress (uzun analizlerde HTTP timeout riski)
- PDF rapor şablonu

**Doğruluk notları:**
- Gerçek SAP modeline karşı T₁ eşleşmesi: **±%3-5** aralığında (MASS SOURCE parse edildikten sonra 0.84 s → 1.14 s'ye düzeldi; SAP 1.10 s — fark %3.6).
- Kalan farkın kaynakları:
  - Plate bending formülasyonu (Mindlin + selective reduced integration vs. SAP ShellThin)
  - Drilling stabilizasyon + Tikhonov regularization (küçük rijitlik eklemesi)
  - Q4 shell (SAP'ta opsiyonel DKMQ veya higher-order seçilebiliyor)

---

## 10. Hızlı Başlangıç

```bash
# Yerel geliştirme
cd build2aiapi
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Testler
pytest

# API sunucu (localhost:8000)
python src/app.py

# Interactive swagger
open http://localhost:8000/docs
```

Frontend [build2ai](../build2ai) için `.env` içinde `NUXT_PUBLIC_API_BASE=http://localhost:8000` ile bu sunucuya yönlendirilir.

---

## 11. Kaynak Tanıklığı

Motor matematiği **İTÜ Makina Mühendisliği 'Sonlu Elemanlar Analizi' ders kitabı** kodlarından port edilmiştir:
- Frame_3d: `SEA_Book/sec5_frame_3d.py`
- Plane stress Q4: `SEA_Book/sec4_plane_stress_rectangle_gauss.py`
- Plate bending: Reddy formülasyonundan özel implementasyon
- Static solver: `SEA_Book/solver/__init__.py` partitioned yaklaşımı

TBDY 2018 spektrum formülleri, AFAD deprem yönetmeliği Madde 2.3 ve Tablo 2.1/2.2'ye dayanır.

---

## 12. Son Güncellemeler (Changelog)

Bu bölüm motorun ve UI'nın evrimini kronolojik olarak tutar. En yeni
güncellemeler üstte.

### v0.9 — Real-model tam uyum + UI polish (2026-04-16)

Gerçek SAP modellerinde (340+ düğüm, 216 shell, 90 kombinasyon) uçtan
uca çalışma + mühendislik doğruluğu + büyük UX iyileştirmeleri.

| Değişiklik | Açıklama |
|---|---|
| **Modal kütle katılım oranları** | `ModeDTO.mass_participation` (Mx/My/Mz). UI tablosunda kümülatif satır + TBDY %95 ✓/✗ + mod yönü rozeti (X/Y/Z/lokal) |
| **Aks / Kot etiketleri** | `GRID LINES` tablosu parse, her düğüme 30 cm tolerans ile en yakın grid. UI'da tablo satırlarında **`B-3`** (kolon kesişimi — mavi monospace) veya **`aks B`** (ara düğüm — gri) + **`Z2`** kot rozeti. Hover tooltip ile açıklama |
| **Yük durumu dropdown** | 115 pill'lik liste yerine **arama + gruplandırılmış dropdown**: kapalıyken seçili case trigger'da, açılınca mavi "Yük Patternleri" + mor "Kombinasyonlar" grupları |
| **Kombinasyon detayı (ana ekran)** | Seçili combo için info butonuyla formül açılır: `+1.400·G  +1.600·Q  +1.600·Qr` |
| **Config modal — scroll + arama** | 90+ kombinasyon için max-height scroll alanı + mini search box; her combo'nun yanında detay expander |
| **Excel export** | `openpyxl` ile 4 endpoint: `/export/xlsx` (tüm analiz, her case + mod ayrı sheet), `displacements.xlsx`, `reactions.xlsx`, `modes.xlsx`. UI'da 4 buton + yükleniyor göstergesi |
| **Auth race fix** | `authReady()` promise ile Firebase auth restore bekleme — refresh sonrası "geçmiş analiz boş" bug'ı çözüldü |
| **Firebase Storage gzip offload** | 1MB Firestore limiti aşan payload'larda `cases` ve `modes` otomatik olarak gzip JSON olarak Storage'a yazılır; Firestore'da pointer kalır; okuma tarafı şeffaf |
| **Docker compose + .dockerignore** | Tek-komut deploy (`docker compose up -d --build`). Health check, log rotation, 2GB RAM limit, opsiyonel Traefik profili |

### v0.8 — Shell K + SAP MASS SOURCE (2026-04-15)

"İterasyon B" — gerçek SAP modelinin çözülebilmesi için eksik shell K ve
kütle kaynağı eklemeleri.

| Değişiklik | Etki |
|---|---|
| **PlaneStressQ4** (membran) | 4-düğümlü Q4, SEA_Book `sec4_plane_stress_rectangle_gauss.py` port, altın test ile |
| **PlateBendingQ4** (Mindlin-Reissner) | Bending 2×2 Gauss + shear 1×1 reduced integration (shear locking önlenir) |
| **ShellQ4** (full shell) | 24×24 global K = membran 8 + plate 12 + drilling 4. 3D eksen dönüşümü düğüm konumlarından |
| **Assembler entegrasyonu** | `stiffness_assembler` her Q4 shell'i K'ya ekler; shell kütlesi (ρ × A × t) lumped olarak düğümlere |
| **MASS SOURCE tablosu** | Parser yeni: `Elements=No Loads=Yes` varyasyonu. Mass matrix `load_vectors` kullanarak G×1 + Q×0.3 = 1270 ton çıkardı (önce 683 ton). T₁ = 0.84 → **1.14 s** (SAP 1.10 s ile %3 fark) |
| **Çelik/beton ρ fallback** | Malzeme UnitMass=0 ise `E`'den otomatik: çelik 7.85, beton 2.5 t/m³ |

### v0.7 — Nested kombinasyonlar + AUTO SEISMIC (2026-04-14)

Kullanıcının dosyasında 100 COMBINATION DEFINITIONS satırı ama çoğu
"NonLin Static" → önceden sadece "Linear Static" alınıyordu (7 combo).

| Değişiklik | Etki |
|---|---|
| **`CASE - STATIC 1 - LOAD ASSIGNMENTS` parse** | Türetilmiş case'lerin pattern haritası (örn. `Q+Qr = Q×1 + Qr×1`) çıkarılır |
| **Transitif combo expansion** | Recursive resolve — her combo base pattern'lara indirgenir (max 10 derinlik). 7 combo → **90 combo** |
| **`AUTO SEISMIC LOADS TO JOINTS`** parse | SAP'ın ±%5 eksantrisite türevleri (Exp, Expep, Expen, vs.) — 4560 satır düğüm yükü pattern'lere aktarılır |
| **`AUTO WIND LOADS TO JOINTS`** parse | Rüzgar yükü otomatik pattern'leri |

### v0.6 — Diyafram + frame releases + area loads (2026-04-13)

Real SAP fixture'ı "NORMAL RİJİTLİK temiz.s2k" ilk kez çözülür hale geldi.

| Değişiklik | Etki |
|---|---|
| **Rijit diyafram (master-slave)** | `CONSTRAINT DEFINITIONS - DIAPHRAGM` + `JOINT CONSTRAINT ASSIGNMENTS` parse, transformation matrix T ile reduced solve. 900 slave DOF elenir |
| **Frame releases (mafsallar)** | `FRAME RELEASE ASSIGNMENTS 1` parse, statik kondenzasyon: `K_new = K_rr - K_rc × inv(K_cc) × K_cr` |
| **AREA LOADS - UNIFORM TO FRAME** | Döşeme yoğunluk yükü × alan / perimetre → çevre frame'lere q kN/m |
| **Tikhonov regularization** | Sıfır-rijitlikli DOF varsa küçük α ile stabilize — tam mafsal + shell'siz mekanizma artıkları için |

### v0.5 — TBDY Response Spectrum (2026-04-12)

TBDY 2018 Madde 2.3 elastik + tasarım spektrumu ve SRSS modal birleştirme.
Zemin sınıfı ZA-ZE interpolasyonu dahil. EQX_RS / EQY_RS case'leri üretir.

### v0.4 — Modal + Kombinasyon filtreleme + preview (2026-04-11)

Kullanıcı analiz başlatmadan önce yük durumu + kombinasyonu seçer,
modal analiz opsiyonel. `/preview` endpoint'i ile dosyadaki case'ler
listelenir. AnalysisConfigModal.vue dropdown tabanlı UI.

### v0.3 — FastAPI router + frontend (2026-04-10)

Analiz başlatma (`POST /analyze`), geçmiş listeleme, sonuç endpoint'leri
(displacements/reactions/modes) + Nuxt frontend entegrasyonu.
Firestore'da analiz sonucu saklanır.

### v0.2 — Statik solver + recovery MVP (2026-04-09)

`parse → DOF numaralama → K/F assemble → spsolve → yer değiştirme + reaksiyon`.
Partitioned çözüm (serbestler/tutulular), SEA_Book referans matematik.

### v0.1 — FrameElement3D port (2026-04-08)

SEA_Book `sec5_frame_3d.py` characterization ile port edildi. 10 altın
test ile K_Local, K_Global, TLG, q_Local, B matrisleri birebir korunur.

---

## 13. Yol Haritası

Hâlâ eksik olan özellikler, öncelik ve zorluk sınıfına göre.

### ⚠️ Kritik (üretimde iş yapmak için şart)

| Özellik | Neden kritik | Tahmini efor |
|---|---|---|
| **Eleman kesit tesirleri (M, V, N boyunca)** | Kirişin mesnet ve açıklık momentlerini görmeden tasarım yapılamaz | 1-2 gün |
| **Drift oranı** (kat ötelenmesi) | TBDY deprem şartları | 1 gün |
| **Otomatik yük kombinasyonları** | TBDY kombolarını bizim üretmemiz — kullanıcı SAP'a bağımlı kalmasın | yarım gün |
| **CQC modal birleştirme** | Yakın periyotlu modlarda SRSS yanlış | yarım gün |
| **Trapezoidal yayılı yük** | `magnitude_a ≠ magnitude_b` — şu an atlanıyor | yarım gün |
| **Mesnet çökmesi** (settlement) | Nadir ama gerektiğinde şart | yarım gün |

### 🟡 Bazı senaryolar için (hepsi gerekli değil)

| Özellik | Hangi case'ler için? | Tahmini efor |
|---|---|---|
| **P-Δ** (geometrik nonlineer) | Yüksek katlı bina (>10 kat) | 2-3 gün |
| **Burkulma analizi** (Euler) | Çelik yapılar, narin kolonlar | 2 gün |
| **Yay mesnet** (Winkler, temel) | Radye temelli yapılar | 1 gün |
| **Cable/brace** (sadece çekme) | Asma yapılar, çelik çaprazlar | 1 gün |

### 🔴 Büyük modüller (ayrı sprint)

| Modül | Kapsam |
|---|---|
| **Tasarım kontrolleri** (TS500 RC, TS498 çelik, AISC) | Kapasite/talep oranı, donatı önerisi — 1-2 hafta |
| **Nonlineer analiz** (pushover, plastik mafsal) | `openseespy` köprüsü — 1-2 hafta |
| **Time-history analizi** | İvme kaydıyla dinamik — 1 hafta |
| **3D solid elemanlar** (brick, tetra) | Temel, dolgu duvarı — 1 hafta |
| **Curved shell** (eğrisel kabuk) | Kubbe, silo, kemer |
| **Bridge moving loads** | Köprü tasarımı |

### 🔵 Altyapı / UX

| Eksik | Etki |
|---|---|
| **Async jobs** (RabbitMQ + SSE) | 60 sn+ süren analizler HTTP timeout'a girer |
| **PDF rapor üretimi** | Mühendis kağıt çıktı ister (TBDY format) |
| **Stress kontur haritası** (shell yüzey) | Shell analizinde en değerli görsel |
| **Deformed shape animation** (3D viewer) | Mod şekli + yer değiştirme animasyonu |

---

## 14. "Hangi kullanıcı için hazırız?"

| Profil | Hazırlık durumu |
|---|---|
| **Üniversite öğrencisi / araştırmacı** (temel statik+modal) | ✅ %98 |
| **Yapı mühendisi — ön tasarım** (RC/çelik bina, statik+modal+RS) | ✅ %85 |
| **Yapı mühendisi — detay tasarım** (kesit tesirleri, drift, tasarım kontrolü) | ⚠️ %50 |
| **Yüksek yapı / özel proje** (P-Δ, nonlineer, kapsamlı raporlama) | ⚠️ %30 |
| **Köprü / endüstriyel tesis** (curved shell, moving load, cable) | ❌ %15 |

### Öncelik önerisi

**A) "Mühendise tam analiz aracı" (~3 gün):**
1. Kesit tesirleri (M, V, N diyagramları) — **1-2 gün**
2. Drift oranı hesabı — 1 gün
3. Otomatik TBDY kombinasyonları — yarım gün

Bittikten sonra kullanıcı SAP'tan aldığı modelin çoğunu bu araçta tamamlayabilir.

**B) "Production-grade platform" (~5 gün):**
1. Async jobs + SSE progress — 2 gün
2. PDF rapor şablonu — 2 gün
3. 3D viewer'a deformed shape + mod animasyonu — 1 gün

