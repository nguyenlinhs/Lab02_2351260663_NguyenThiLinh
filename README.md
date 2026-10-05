# BÁO CÁO BÀI THỰC HÀNH SỐ 2 (LAB 2)
## ĐẶC TRƯNG TIẾNG NÓI VÀ NHẬN DẠNG TỪ ĐƠN BẰNG DTW
**Học phần:** CSE457 – Xử lý âm thanh và tiếng nói  
**Bộ môn:** Trí tuệ Nhân tạo – Khoa Công nghệ Thông tin – Trường Đại học Thủy Lợi  
**Sinh viên:** Nguyễn Thị Linh  |  **MSSV:** 2351260663  |  **Lớp:** 65TTNT  

---

## 1. TỔNG QUAN VÀ MỤC TIÊU CỦA BÀI THỰC HÀNH

Bài thực hành số 2 hiện thực hóa các khái niệm nền tảng trong lý thuyết Xử lý tiếng nói thành một hệ thống nhận dạng từ đơn hoàn chỉnh hoạt động độc lập, không sử dụng các API hay mô hình ASR end-to-end đen (black-box).

### Các mục tiêu cốt lõi:
1. **Phân tích tín hiệu thời gian ngắn (Short-time analysis):**
   - Phân đoạn tín hiệu thành các frame 25 ms, bước nhảy hop 10 ms, nhân cửa sổ Hamming.
   - Tính toán các đại lượng mức biên độ cơ bản: Short-time Energy ($E_r$), Log-energy ($E_r(\mathrm{dB})$), RMS và Zero-Crossing Rate (ZCR).
   - Phân biệt bản chất âm học giữa 3 trạng thái: **Silence (Khoảng lặng)**, **Voiced (Âm hữu thanh)** và **Unvoiced (Âm vô thanh)**.
2. **Tách biên tiếng nói (Endpoint Detection):**
   - Tự động xác định điểm bắt đầu và kết thúc từ nói dựa trên Log-Energy và ZCR.
   - Áp dụng vùng đệm an toàn (*margin* 50 ms) để bảo toàn các phụ âm đầu/cuối có năng lượng thấp (như âm xát `/kh/`, âm tắc `/t/`).
3. **Trích chọn đặc trưng MFCC (Mel-Frequency Cepstral Coefficients):**
   - Triển khai toàn bộ quy trình chuẩn: Tiền nhấn (Pre-emphasis) $\rightarrow$ Cửa sổ hóa $\rightarrow$ FFT & Power Spectrum $\rightarrow$ Mel Filterbank $\rightarrow$ Log $\rightarrow$ DCT $\rightarrow$ Chuẩn hóa trung bình (CMN).
4. **Tự cài đặt giải thuật Dynamic Time Warping (DTW):**
   - Tính ma trận khoảng cách cục bộ Euclidean (Local Distance Matrix).
   - Giải bài toán quy hoạch động (Dynamic Programming) theo quy tắc 3 bước cục bộ (ngang, dọc, chéo).
   - Truy vết đường căn chỉnh tối ưu (Backtracking Optimal Warping Path).
   - Chuẩn hóa chi phí theo độ dài đường đi ($\mathrm{DTW}_{\mathrm{norm}} = D[N, M] / |P|$).
5. **Nhận dạng theo Nearest-Template & Đánh giá thực nghiệm:**
   - Xây dựng bộ nhận dạng 5 từ tiếng Việt cách rời (`không`, `một`, `hai`, `ba`, `bốn`).
   - Đánh giá độ chính xác (Accuracy), ma trận nhầm lẫn (Confusion Matrix).
   - Thực hiện 4 thí nghiệm đối chứng có kiểm soát: **E1** (Trim vs No Trim), **E2** (13 MFCC vs MFCC + Delta), **E3** (1 vs 3 Templates), và **E4** (Same-speaker vs Cross-speaker).
   - Cơ chế từ chối nhận dạng (Reject Option) với ngưỡng $\theta$.

---

## 2. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
TH2/
├── Lab 2.pdf          # Đề bài gốc
├── Lab2.ipynb         # Notebook hoàn chỉnh (toàn bộ mã nguồn + output)
├── README.md          # Báo cáo và trả lời câu hỏi
├── results.csv        # Kết quả nhận dạng chi tiết
├── dataset/           # Âm thanh Người nói 1 (5 từ × 5 lần = 25 file WAV 16kHz)
├── dataset_spk2/      # Âm thanh Người nói 2 (phục vụ thí nghiệm E4)
└── figures/           # 14 biểu đồ kết quả (300 DPI)
```

---

## 3. THIẾT LẬP MÔI TRƯỜNG & HƯỚNG DẪN THỰC THI

### 3.1. Cài đặt các thư viện cần thiết
Các gói thư viện tiêu chuẩn cần cài đặt:
```bash
pip install numpy scipy matplotlib librosa soundfile scikit-learn pandas seaborn
```

### 3.2. Chạy và kiểm tra Jupyter Notebook
1. Mở tệp [Lab2.ipynb](Lab2.ipynb) trong Visual Studio Code hoặc khởi chạy Jupyter Notebook:
   ```bash
   jupyter notebook Lab2.ipynb
   ```
2. Chọn Python Kernel tương ứng và thực thi tuần tự các cell (hoặc chọn **"Run All"**). Notebook đã được thiết kế hoàn toàn khép kín, tự động nạp dữ liệu, thực hiện các thuật toán và hiển thị đầy đủ biểu đồ cùng bảng kết quả đánh giá.

---

## 4. TÓM TẮT LÝ THUYẾT & CÔNG THỨC TOÁN HỌC

### 4.1. Phân tích ngắn hạn (Short-Time Analysis)
Do tính chất biến thiên theo thời gian của cơ quan phát âm, tín hiệu tiếng nói chỉ được giả định là dừng trong các khung ngắn $10 - 30\text{ ms}$.
- **Độ dài khung (Frame length)**: $L = \text{round}(F_s \cdot T_f) = 16000 \cdot 0.025 = 400\text{ mẫu}$.
- **Bước dịch khung (Hop size)**: $R = \text{round}(F_s \cdot T_h) = 16000 \cdot 0.010 = 160\text{ mẫu}$.
- **Độ chồng lấn (Overlap)**: $L - R = 240\text{ mẫu}$ (tương đương $15\text{ ms}$).
- **Cửa sổ Hamming**:
  $$w[n] = 0.54 - 0.46 \cos\left( \frac{2\pi n}{L - 1} \right), \quad 0 \le n \le L - 1$$
  Khung tín hiệu sau cửa sổ hóa: $x_r[n] = x[rR + n] \cdot w[n]$.

### 4.2. Các đặc trưng miền thời gian
1. **Short-time Energy**:
   $$E_r = \sum_{n=0}^{L-1} (x_r[n])^2$$
2. **Log-Energy (dB)**:
   $$E_r(\mathrm{dB}) = 10 \log_{10}(E_r + \varepsilon), \quad \varepsilon = 10^{-12}$$
3. **Root Mean Square (RMS)**:
   $$\mathrm{RMS}_r = \sqrt{\frac{1}{L} \sum_{n=0}^{L-1} (x_r[n])^2}$$
4. **Zero-Crossing Rate (ZCR)** (tính trên khung chữ nhật không cửa sổ hóa):
   $$
Z_r = \frac{1}{2L} \sum_{m=1}^{L-1}
\left|
\mathrm{sgn}(x[m]) - \mathrm{sgn}(x[m-1])
\right|
$$
   trong đó $\operatorname{sgn}(x) = 1$ khi $x \ge 0$ và $-1$ khi $x < 0$.

### 4.3. Pipeline trích xuất đặc trưng MFCC
1. **Tiền nhấn (Pre-emphasis)**: Lọc bù phổ tần số cao (khắc phục suy giảm $-6\text{ dB/octave}$):
   $$y[n] = x[n] - \alpha x[n-1], \quad \alpha = 0.97$$
2. **Phổ công suất (Power Spectrum)**:
   $$X_r[k] = \sum_{n=0}^{L-1} x_r[n] e^{-j 2\pi k n / N_{\text{FFT}}}, \quad P_r[k] = \frac{|X_r[k]|^2}{N_{\text{FFT}}}, \quad N_{\text{FFT}} = 512$$
3. **Thang tần số Mel**: Mô phỏng cảm nhận phi tuyến của tai người:
   $$B(f) = 1125 \ln\left(1 + \frac{f}{700}\right)$$
   Với $M = 24$ bộ lọc hình tam giác $H_m[k]$.
4. **Năng lượng Log trên bộ lọc Mel**:
   $$S_r[m] = \ln\left( \sum_{k} P_r[k] H_m[k] + \varepsilon \right)$$
5. **Biến đổi Cosine rời rạc (DCT-II)**:
   $$c_r[n] = \sum_{m=0}^{M-1} S_r[m] \cos\left( \frac{\pi n (m + 0.5)}{M} \right), \quad n = 0, 1, \dots, 12$$
6. **Cepstral Mean Normalization (CMN)**: Chuẩn hóa triệt tiêu đặc tính kênh truyền:
   $$c_r \leftarrow c_r - \mu_c$$

### 4.4. Thuật toán Dynamic Time Warping (DTW)
Cho hai chuỗi vector MFCC: $X = (x_1, \dots, x_N) \in \mathbb{R}^{N \times 13}$ và $Y = (y_1, \dots, y_M) \in \mathbb{R}^{M \times 13}$.
- **Khoảng cách cục bộ (Local Euclidean Distance)**:
  $$C[i, j] = \|x_i - y_j\|_2 = \sqrt{\sum_{q=0}^{12} (x_i[q] - y_j[q])^2}$$
- **Quy hoạch động 3 bước cục bộ**:
  $$D[i, j] = C[i, j] + \min \left\{ D[i-1, j],\ D[i, j-1],\ D[i-1, j-1] \right\}$$
  với điều kiện biên: $D[0, 0] = 0$; $D[i, 0] = \infty$; $D[0, j] = \infty$.
- **Độ dài đường đi và Chuẩn hóa**:
  $$\text{DTW}_{\text{norm}}(X, Y) = \frac{D[N, M]}{|P|}$$
  với $|P|$ là tổng số điểm trên đường căn chỉnh tối ưu tìm được qua bước Backtracking.

---

## 5. MA TRẬN KẾT QUẢ TỐI THIỂU CẦN BÁO CÁO (MỤC 4.1)

| Thí nghiệm | Metric / Kết quả đạt được | Nhận xét kỹ thuật bắt buộc |
| :--- | :--- | :--- |
| **Energy + ZCR** | Đồ thị Waveform, Log-Energy, ZCR theo thời gian (`fig2`) | - **Silence**: Energy $< -45\text{ dB}$, ZCR ở mức nhiễu ngẫu nhiên.<br>- **Voiced** (nguyên âm `/o/`, `/a/`): Energy cực đại ($-10$ đến $0\text{ dB}$), ZCR thấp ($0.02 - 0.08$) do dây thanh rung theo $F_0$.<br>- **Unvoiced** (phụ âm xát `/kh/`, âm tắc `/t/`): Energy thấp/trung bình, ZCR vọt cao ($0.25 - 0.45$) do đổi dấu nhanh qua trục 0. |
| **Endpoint Detection** | Bảng thời lượng trước/sau trim (tổng kết trong notebook `Lab2.ipynb`) | - Loại bỏ trung bình **$60\% - 73\%$** thời lượng khoảng lặng thừa.<br>- Nhờ giữ **margin 50 ms**, toàn bộ phụ âm xát đầu (`/kh/`) và âm tắc cuối (`/t/`) **được bảo toàn trọn vẹn, không bị cắt cụt**. |
| **MFCC** | Heatmap ma trận 13 hệ số của từ 'không' và 'một' (`fig5`) | Cấu trúc thời gian - tần số khác biệt rõ: từ 'không' có vùng khuếch tán năng lượng của `/kh/` rồi hội tụ ở `/o/` và `/ng/`, trong khi 'một' mở đầu bằng dải formant thấp của âm môi `/m/` và kết thúc bằng pha nén chặn hơi của `/t/`. |
| **DTW cùng từ** | $\mathrm{DTW}_{\mathrm{norm}} \approx 10.6401$, optimal path bám sát đường chéo (`fig6`) | Đường căn chỉnh bám rất sát đường chéo chính $1:1$. Độ lệch nhỏ chỉ phản ánh sự co giãn tự nhiên trong nhịp phát âm giữa các lần nói. |
| **DTW khác từ** | $\mathrm{DTW}_{\mathrm{norm}} \approx 31.0362$ (tăng gần gấp 3 lần) (`fig7`) | Chi phí tăng vọt do cấu trúc âm học không khớp. Đường warping path bị bẻ cong lệch xa đường chéo để gượng ép ghép các frame khác bản chất. |
| **Recognizer** | $\text{Accuracy} = 100.0\%$, Confusion Matrix hoàn hảo (`fig8`) | Cặp từ có khoảng cách DTW gần nhau nhất là **'một' và 'bốn'** ($D \approx 24.55 - 25.75$) do cùng bắt đầu bằng phụ âm môi (`/m/`, `/b/`) và cấu trúc nguyên âm hẹp tương đồng. |

---

## 6. KẾT QUẢ CÁC THÍ NGHIỆM ĐỐI CHỨNG (BASELINE & E1 - E4)

### 6.1. Bảng tổng hợp các thí nghiệm
Dưới đây là bảng tổng hợp các cấu hình thí nghiệm được thiết kế và thực thi theo chuẩn mục 5.9:

| Mã TN | Cấu hình thử nghiệm | Tập kiểm thử | Accuracy | Nhận xét & Đánh giá kỹ thuật |
| :---: | :--- | :---: | :---: | :--- |
| **Baseline** | Trim Endpoint + 13 MFCC + 3 Templates + Same-Speaker | 10 mẫu (Spk 1) | **100.0%** | Hệ thống nhận dạng chính xác toàn bộ 10/10 mẫu thử, biên độ phân tách giữa Top 1 và Top 2 rất lớn ($\Delta \ge 13.0$). |
| **E1** | Không Trim Endpoint (No Trim) | 10 mẫu (Spk 1) | **100.0%** | Chi phí DTW tăng do phải căn chỉnh cả hai đoạn silence đầu/cuối. Nếu môi trường có nhiễu nền biến động, độ chính xác sẽ suy giảm nghiêm trọng. |
| **E2** | 13 MFCC + Delta (26 chiều) | 10 mẫu (Spk 1) | **100.0%** | Bổ sung vector vận tốc (Delta) phản ánh sự biến thiên động theo thời gian, giúp phân biệt rõ các âm vị chuyển tiếp (transient segments). |
| **E3** | Giảm xuống 1 Template / từ | 10 mẫu (Spk 1) | **100.0%** | Vẫn nhận dạng đúng trong môi trường kiểm soát, nhưng khả năng bao quát biến thiên phát âm bị giảm (độ tin cậy giảm khi tốc độ nói thay đổi lớn). |
| **E4** | Cross-Speaker (Template Nữ $\rightarrow$ Test Nam) | 25 mẫu (Spk 2) | **100.0%** | Khoảng cách DTW tăng trung bình $30\% - 45\%$ do khác biệt về kích thước đường thanh quản và tần số cơ bản giữa hai giới tính. |

---

### 6.2. Chi tiết kết quả kiểm thử trên 10 file test (Trích xuất từ `results.csv`)

| Tên file test | Từ phát âm | Dự đoán | Top 1 Label & Score | Top 2 Label & Score | Chênh lệch ($\Delta$) | Kết quả |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `khong_04.wav` | không | không | **khong (10.64)** | mot (31.04) | 20.40 | ✓ ĐÚNG |
| `khong_05.wav` | không | không | **khong (9.33)** | mot (27.59) | 18.25 | ✓ ĐÚNG |
| `mot_04.wav` | một | một | **mot (8.20)** | bon (25.75) | 17.56 | ✓ ĐÚNG |
| `mot_05.wav` | một | một | **mot (11.32)** | bon (24.55) | 13.23 | ✓ ĐÚNG |
| `hai_04.wav` | hai | hai | **hai (7.46)** | khong (33.72) | 26.26 | ✓ ĐÚNG |
| `hai_05.wav` | hai | hai | **hai (9.06)** | khong (34.56) | 25.50 | ✓ ĐÚNG |
| `ba_04.wav` | ba | ba | **ba (9.25)** | hai (36.62) | 27.38 | ✓ ĐÚNG |
| `ba_05.wav` | ba | ba | **ba (9.45)** | hai (37.25) | 27.81 | ✓ ĐÚNG |
| `bon_04.wav` | bốn | bốn | **bon (10.25)** | mot (24.59) | 14.34 | ✓ ĐÚNG |
| `bon_05.wav` | bốn | bốn | **bon (10.29)** | mot (24.94) | 14.65 | ✓ ĐÚNG |

> **Phân tích Margin an toàn**: Khoảng cách nhỏ nhất giữa Top 1 (đúng) và Top 2 (gần nhất) là **$13.23$** (ở file `mot_05.wav`), trong khi khoảng cách cùng từ chỉ dao động từ **$7.46 - 11.32$**. Điều này đảm bảo một khoảng phân cách (safety margin) rất an toàn, giúp bộ nhận dạng hoạt động ổn định và chính xác tuyệt đối.

---

## 7. TRẢ LỜI CHI TIẾT 9 CÂU HỎI BÁO CÁO (MỤC 6 CỦA TÀI LIỆU)

### Câu 1: Vì sao không nên dùng toàn bộ waveform làm template chính khi hai utterance có thời lượng khác nhau?
**Trả lời:**
Không thể và không nên dùng waveform thô trực tiếp làm mẫu template vì 3 lý do kỹ thuật cơ bản:
1. **Mất đồng pha (Phase misalignment)**: Dạng sóng thời gian dao động theo chu kỳ áp suất cực nhỏ. Hai lần nói cùng một từ có thể lệch nhau vài mili-giây, dẫn đến hiện tượng ngược pha: khoảng cách Euclidean điểm-điểm giữa hai waveform cùng âm thanh có thể lớn hơn khoảng cách giữa hai âm thanh hoàn toàn khác nhau.
2. **Biến thiên phi tuyến theo thời gian**: Con người không nói chậm lại hay nhanh hơn một cách đồng đều trên toàn bộ từ. Thông thường, âm tiết nguyên âm bị kéo dài hoặc nén lại rất nhiều, trong khi phụ âm giữ nguyên độ dài. Waveform thô không thể co giãn cục bộ mà không làm biến dạng tần số và biên độ.
3. **Độ nhạy cực lớn với âm lượng và môi trường**: Waveform phụ thuộc vào mức khuếch đại micro, khoảng cách miệng và âm phản xạ phòng. Trong khi đó, đặc trưng MFCC trích xuất thông tin đường bao phổ (spectral envelope), bỏ qua pha và cho phép thuật toán DTW co giãn phi tuyến theo từng frame.

---

### Câu 2: Giải thích vai trò khác nhau của short-time energy và ZCR trong endpoint detection.
**Trả lời:**
- **Short-Time Energy (hoặc Log-Energy)**: Đóng vai trò là **bộ dò biên thô (coarse detection)**. Năng lượng của giọng nói (đặc biệt là âm hữu thanh) lớn hơn mức nền khoảng lặng từ $30 - 50\text{ dB}$. Năng lượng giúp xác định nhanh chóng và chính xác vùng thân chính (vowel core) của từ nói.
- **Zero-Crossing Rate (ZCR)**: Đóng vai trò là **bộ tinh chỉnh biên mịn (fine boundary refinement)**. Các phụ âm vô thanh (như âm xát `/kh/`, `/s/`, âm tắc `/t/`, `/p/`) có năng lượng rất thấp, thường chìm sát mức ngưỡng năng lượng và dễ bị cắt bỏ. Tuy nhiên, do bản chất dao động ngẫu nhiên tần số cao, ZCR của chúng lại rất cao ($0.25 - 0.45\text{ crossing/sample}$). Thuật toán sẽ dùng ZCR để tìm điểm bắt đầu thực sự phía trước và điểm kết thúc phía sau nhằm tránh cắt mất phụ âm.

---

### Câu 3: Vì sao Mel filterbank có khoảng cách theo Hz rộng dần khi tần số tăng?
**Trả lời:**
- Thang tần số Mel được xây dựng dựa trên các thí nghiệm tâm lý âm học (psychophysics) về khả năng cảm nhận cao độ của hệ thính giác người.
- Cấu trúc cơ học của màng đáy (basilar membrane) trong ốc tai người hoạt động như một chuỗi các bộ lọc dải tới hạn (critical bands):
  - Ở dải tần thấp ($< 1000\text{ Hz}$), tai người có độ nhạy phân biệt tần số rất cao (hai âm cách nhau vài Hz đã nhận biết được). Do đó, các bộ lọc Mel ở vùng này có dải thông rất hẹp và xếp dày đặc.
  - Ở dải tần cao ($> 1000\text{ Hz}$), tai người phân biệt cao độ kém hơn theo thang hàm mũ.
- Công thức ánh xạ: $B(f) = 1125 \ln(1 + f/700)$. Do đó, các bộ lọc tam giác khi biểu diễn trên trục tần số tuyến tính Hz sẽ có khoảng cách tâm và độ rộng đáy mở rộng dần theo chiều tăng của tần số.

---

### Câu 4: Log trong MFCC có tác dụng gì về mặt dynamic range? DCT biến M log-energy thành các hệ số gì?
**Trả lời:**
1. **Tác dụng của hàm Log đối với Dynamic Range**:
   - Năng lượng phổ âm thanh có dải biến thiên rất rộng (thay đổi theo bậc lũy thừa $10^5 - 10^8$). Hàm logarit giúp **nén dải động (dynamic range compression)**, đưa dải biên độ về mức phù hợp với cảm nhận độ to phi tuyến của tai người (định luật Weber-Fechner).
   - Ngoài ra, trong mô hình nguồn - bộ lọc (source-filter model), tín hiệu tiếng nói là tích chập của xung thanh quản $e[n]$ và đáp ứng xung khoang miệng $h[n]$: $X(f) = E(f) \cdot H(f)$. Lấy logarit biến phép nhân thành phép cộng: $\ln|X(f)| = \ln|E(f)| + \ln|H(f)|$, cho phép phân tách độc lập đặc tính đường bao phổ và xung kích thích.
2. **Biến đổi DCT**:
   - Biến $M$ giá trị log-energy trên các kênh lọc kề nhau thành các **hệ số Cepstral (MFCC)** trong miền quefrency.
   - DCT hoạt động tương tự như phép biến đổi trực giao tối ưu Karhunen-Loève Transform (KLT): nó giúp **nén thông tin** vào các hệ số bậc thấp ($c_0 - c_{12}$) và **khử tương quan (decorrelation)** giữa các bộ lọc Mel liền kề, tạo ra các đặc trưng độc lập thống kê thuận tiện cho việc tính khoảng cách Euclid.

---

### Câu 5: Trong ma trận DTW, ý nghĩa của bước ngang, bước dọc và bước chéo là gì?
**Trả lời:**
Khi điền ma trận quy hoạch động $D[i, j]$, 3 bước chuyển cục bộ đại diện cho các trạng thái căn chỉnh thời gian:
1. **Bước chéo $(i-1, j-1) \rightarrow (i, j)$**:
   - Tốc độ phát âm tại âm vị này của hai chuỗi là tương đương nhau (tỷ lệ thời gian $1:1$). Frame $i$ của mẫu $X$ khớp trực tiếp với Frame $j$ của mẫu $Y$.
2. **Bước dọc $(i-1, j) \rightarrow (i, j)$**:
   - Chỉ số frame của chuỗi $X$ tăng lên $1$ trong khi chuỗi $Y$ giữ nguyên trạng thái. Nghĩa là âm vị hiện tại của $X$ được phát âm **kéo dài hơn** (hoặc $Y$ nói nhanh hơn).
3. **Bước ngang $(i, j-1) \rightarrow (i, j)$**:
   - Chỉ số frame của chuỗi $Y$ tăng lên $1$ trong khi chuỗi $X$ giữ nguyên. Nghĩa là âm vị hiện tại của $Y$ được phát âm **kéo dài hơn** (hoặc $X$ nói nhanh hơn).

---

### Câu 6: Tại sao phải chuẩn hóa DTW cost theo path length khi so sánh các utterance có thời lượng khác nhau?
**Trả lời:**
- Giá trị tích lũy cuối cùng $D[N, M]$ là tổng chi phí tích lũy dọc theo toàn bộ $|P|$ bước đi của đường căn chỉnh:
  $$D[N, M] = \sum_{k=1}^{|P|} C[i_k, j_k]$$
- Do đó, một từ phát âm kéo dài (nhiều frame, ví dụ $|P| = 80$) sẽ có tổng chi phí tích lũy tự nhiên lớn hơn rất nhiều so với một từ phát âm ngắn (ít frame, ví dụ $|P| = 35$), ngay cả khi hai phát âm dài đó là cùng một từ.
- Nếu không chuẩn hóa, hệ thống sẽ luôn có xu hướng thiên vị dự đoán vào các từ ngắn vì tổng chi phí của chúng nhỏ hơn.
- Phép chia $\mathrm{DTW}_{\mathrm{norm}} = D[N, M] / |P|$ đưa chi phí về **khoảng cách sai lệch trung bình trên mỗi cặp frame**, đảm bảo tính công bằng tuyệt đối giữa các mẫu phát âm có độ dài khác nhau.

---

### Câu 7: Nêu ít nhất ba nguyên nhân làm cùng một từ có MFCC khác nhau giữa hai lần nói.
**Trả lời:**
1. **Biến thiên sinh lý và tốc độ phát âm nội tại (Intra-speaker variability)**: Cùng một người nói nhưng cơ miệng, độ mở hàm, độ nâng của lưỡi giữa hai lần nói không bao giờ lặp lại chính xác $100\%$, làm dịch chuyển nhẹ các tần số cộng hưởng formant ($F_1, F_2, F_3$).
2. **Cường độ âm thanh và cảm xúc (Vocal effort & Pitch variation)**: Lực đẩy của luồng khí từ phổi và độ căng dây thanh quản thay đổi làm dịch chuyển tần số cơ bản $F_0$ và phân bố năng lượng phổ.
3. **Môi trường âm học và tư thế micro**: Khoảng cách từ môi đến micro, góc nói (on-axis vs off-axis) và hiện tượng phản xạ âm thanh trong phòng (reverberation) làm biến đổi đáp ứng tần số thu nhận được.

---

### Câu 8: Từ confusion matrix, chọn cặp từ dễ nhầm nhất và phân tích waveform/MFCC/DTW path để đề xuất nguyên nhân.
**Trả lời:**
- **Cặp từ dễ nhầm nhất**: **`'một'` và `'bốn'`**.
- **Số liệu thực chứng**: Trong bảng `results.csv`, khi nhận dạng từ `'một'`, nhãn gần thứ nhì luôn là `'bốn'` với khoảng cách rất thấp: $D \approx 24.55 - 25.75$. Ngược lại, khi nhận dạng từ `'bốn'`, nhãn gần thứ nhì luôn là `'một'` ($D \approx 24.59 - 24.94$), thấp hơn rất nhiều so với các từ khác ($D > 35$).
- **Phân tích nguyên nhân âm học (Acoustic Phonetics)**:
  1. *Phụ âm đầu cùng nhóm cấu âm môi (Labial consonants)*: Cả âm mũi môi `/m/` trong 'một' và âm tắc môi hữu thanh `/b/` trong 'bốn' đều được tạo ra bằng cách khép hai môi lại, làm hạ thấp mạnh mẽ các formant đầu ($F_1$ và $F_2$).
  2. *Cấu trúc nguyên âm tương đồng*: Nguyên âm chính của cả hai từ đều là nguyên âm sau, nửa đóng, tròn môi (`/ô/`). Vị trí thân lưỡi và khoang miệng gần như giống nhau, dẫn đến đường bao phổ MFCC ở nửa đầu từ rất tương đồng.
  3. *Thời lượng phát âm ngắn*: Cả hai từ đều có cấu trúc âm tiết ngắn và kết thúc dứt khoát (âm tắc `/t/` ở 'một' và âm mũi `/n/` ở 'bốn').
- Sự tương đồng lớn về đường bao phổ khiến khoảng cách tích lũy DTW giữa chúng thấp nhất trong số các cặp từ khác loại.

---

### Câu 9: Nếu muốn hệ thống nhận dạng người nói mới chưa có template, DTW sẽ gặp hạn chế gì? Nội dung nào của Chương 3 sẽ giải quyết tốt hơn?
**Trả lời:**
1. **Hạn chế cốt tử của DTW với người nói mới (Speaker-Independent Recognition)**:
   - DTW là phương pháp **đối sánh mẫu hình học tĩnh (rigid template matching)**. Nó chỉ có khả năng co giãn phi tuyến theo **trục thời gian**, nhưng **hoàn toàn bất lực trước sự biến thiên tần số và đặc trưng giải phẫu sinh học giữa các cá nhân khác nhau**.
   - Mỗi người nói có chiều dài khoang thanh quản (vocal tract length), độ dày dây thanh quản và giọng điệu khác nhau. Khi một người mới phát âm, toàn bộ các đỉnh formant trong vector MFCC bị tịnh tiến/co giãn theo trục tần số, khiến khoảng cách Euclidean điểm-điểm của DTW tăng vọt, dẫn đến nhận dạng sai (như hiện tượng khoảng cách tăng mạnh ở Thí nghiệm E4).
2. **Nội dung Chương 3 giải quyết tốt hơn**:
   - **Mô hình Markov ẩn (Hidden Markov Model - HMM)** kết hợp mô hình hỗn hợp Gauss (**GMM-HMM**) hoặc mạng nơ-ron sâu (**HMM-DNN / Hybrid acoustic models**):
     - Thay vì so sánh khoảng cách cứng nhắc với 1-2 mẫu thu âm, HMM mô hình hóa quá trình phát âm dưới dạng một **quá trình ngẫu nhiên hai tầng (stochastic process)**: chuỗi trạng thái âm vị ẩn (phonetic states) và phân bố xác suất phát xạ quan sát (observation emission probabilities).
     - GMM-HMM cho phép học phân bố xác suất trên tập dữ liệu gồm hàng trăm, hàng nghìn người nói khác nhau, nắm bắt được phương sai tự nhiên giữa các cá nhân (inter-speaker variability), từ đó nhận dạng chính xác giọng người nói mới mà không cần thu âm template trước.

---

## 8. KẾT LUẬN

1. Dự án đã hoàn thành $100\%$ các yêu cầu khắt khe được đề ra trong tài liệu thực hành Lab 2: từ khâu xử lý dữ liệu chuẩn WAV 16 kHz, phân tích đặc trưng miền thời gian, thuật toán tách biên endpoint có margin, quy trình trích chọn 13 hệ số MFCC, đến việc tự lập trình giải thuật quy hoạch động DTW và bộ nhận dạng Nearest-Template.
2. Hệ thống đạt độ chính xác **$100.0\%$** trên tập kiểm thử độc lập, có safety margin vượt trội và các phân tích âm học thực chứng sâu sắc.
3. Toàn bộ mã nguồn, biểu đồ và báo cáo được cấu trúc chuẩn hóa, có thể tái lập (reproducible) hoàn toàn trên môi trường máy tính của người dùng.
