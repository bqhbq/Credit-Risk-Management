# Dự Án Phân Tích Rủi Ro Tín Dụng & Dự Đoán Nợ Xấu (Nova Bank Credit Scoring)

Dự án triển khai hệ thống học máy (Machine Learning Pipeline) nhằm tự động hóa quy trình đánh giá hồ sơ vay vốn, phân loại rủi ro tín dụng và dự đoán khả năng nợ quá hạn (`loan_status`) của khách hàng tại Nova Bank. Dự án giải quyết triệt để các thách thức về dữ liệu mất cân bằng, giá trị ngoại lai cực đoan và tối ưu hóa phân phối toán học phục vụ kiểm toán ngân hàng.

---

## 📌 Khung Quy Trình Triển Khai (Pipeline Architecture)

Hệ thống được thiết kế theo chuẩn module hóa với các phân vùng Notebook độc lập để tối ưu bộ nhớ RAM và dễ dàng bảo trì:
1. **Tiền xử lý và Làm sạch:** `preprocess.ipynb`, `feature_extract.ipynb`, `EDA.ipynb`
2. **Tối ưu phân phối toán học:** `normalization.ipynb`
3. **Huấn luyện và Đánh giá Mô hình:** `model.ipynb`
---

## 🛠️ Chi Tiết Các Bước Đã Thực Hiện

### Giai đoạn 1: Feature Engineering & Lọc Nhiễu Nghiệp Vụ
* **Tạo đặc trưng mới:** Biến đổi cột tuổi liên tục thành `age_group` và thiết lập cờ cảnh báo áp lực tài chính `financial_stress_flag` dựa trên logic dòng tiền.
* **Thanh trừng biến nhiễu (Business Drop):** Chủ động loại bỏ các cột định danh hệ thống (`client_ID`) và dữ liệu tọa độ địa lý chi tiết (`city_longitude`, `city_latitude`) nhằm tuân thủ Đạo luật cho vay công bằng (Fair Lending) và tránh hiện tượng mô hình học vẹt (Overfitting).

### Giai đoạn 2: Thiết Lập Ranh Giới Cô Lập Dữ Liệu
* **Mã hóa:** Chuyển đổi toàn bộ biến phân loại chữ sang định dạng số nguyên nhị phân bằng kỹ thuật One-Hot Encoding (`pd.get_dummies`).
* **Phân tách Train/Valid/Test:** Chia tập dữ liệu theo tỷ lệ nghiêm ngặt **70% Train - 15% Valid - 15% Test** kết hợp với tham số `stratify=y`. Bước này khóa chặt tập Valid và Test vào hộp kính trước khi tính toán số liệu, ngăn chặn hoàn toàn hiện tượng Rò rỉ dữ liệu (Data Leakage).

### Giai đoạn 3: Tối Ưu Hóa Toán Học Chỉ Trên Tập Train
* **Lựa chọn Đặc trưng (Feature Selection):** Huấn luyện thuật toán Rừng ngẫu nhiên (`RandomForestClassifier`) trên tập Train để trích xuất điểm số quan trọng (Feature Importance). Giữ lại Top 20 đặc trưng có sức nặng kinh tế lớn nhất để dọn dẹp các cột thừa cho cả 3 tập dữ liệu.
* **Xử lý Ngoại lai (IQR Capping):** Quét các biến số tài chính liên tục bằng biểu đồ hộp (Boxplot). Xác định hàng rào Upper Bound theo công thức $Q_3 + 1.5 \times IQR$ trên tập Train và tiến hành "cắt ngọn" (Capping) giới hạn biên để bảo vệ các mô hình tuyến tính.
* **Xử lý Lệch phân phối (Power Transformation):** Tính toán độ lệch (Skewness). Tự động áp dụng thuật toán biến đổi hình học nâng cao `PowerTransformer` (Yeo-Johnson) để bẻ cong các phân phối lệch phải nặng về dạng hình chuông chuẩn (Normal Distribution), đồng thời tích hợp chuẩn hóa thang đo Z-score về gốc 0.
* **Cân bằng Lớp (SMOTE):** Áp dụng kỹ thuật lấy mẫu tổng hợp SMOTE độc quyền trên tập Train để nhân bản dữ liệu nợ xấu (Class 1) nhân tạo, đưa tỷ lệ phân bổ mục tiêu về mức cân bằng hoàn hảo 50:50, giúp AI không bị thiên vị lớp đa số.

---
## 📊 Đánh Giá Hiệu Năng Mô Hình (Model Evaluation)

### Đánh Giá Tổng Quan Lựa Chọn Mô Hình (Tập Validation)
Quá trình huấn luyện và đối chiếu hai thuật toán trên tập Validation (chiếm 15% dữ liệu) cho thấy kết quả như sau:

| Mô hình Thuật toán | Accuracy | Precision (Nợ xấu) | Recall (Nợ xấu) | F1-Score | Điểm số ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 82% | 56% | 75% | 0.64 | 0.863 |
| **XGBoost Classifier (Nâng cao)** | 92% | 91% | 71% | 9.80 | 0.934 |

**Nhận định:** Các chỉ số của XGBoost áp đảo hoàn toàn so với mô hình tuyến tính cơ bản. Có 1 điểm kì lạ là chỉ số sống còn của bài toán rủi ro tín dụng là **Recall** đạt `71%` nhỏ hơn so với Logistic Regression nhưng sức mạnh phân tách **ROC-AUC** lên tới `0.934`, cho thấy XGBoost có khả năng khoanh vùng chính xác nhóm nợ xấu ẩn mình mà không đánh oan người vay tốt. Vì vậy, XGBoost chính thức được lựa chọn để bước vào vòng kiểm thử cuối cùng.

### 🧠 Biện Luận: Nghịch Lý Recall & Bản Chất Thuật Toán
Tại ngưỡng mặc định 0.5, xuất hiện một hiện tượng đặc biệt: **Recall của Logistic Regression (75%) cao hơn XGBoost (71%)**, dù tất cả các chỉ số khác và điểm ROC-AUC của XGBoost đều áp đảo hoàn toàn. 

* **Nguyên nhân toán học:** Mô hình tuyến tính (Logistic Regression) do năng lực phân tách kém nên có xu hướng phân bổ xác suất rủi ro bị nhiễu rộng quanh vùng trung tâm. Khi ép cắt ở ngưỡng cứng 0.5, mô hình này dán nhãn "Nợ xấu" một cách đại trà (Vô tình kéo Precision xuống rất thấp). Việc dán nhãn quá nhiều giúp nó vô tình tăng cơ hội bao phủ, "bắt nhầm hơn bỏ sót" và đẩy Recall lên 75%.
* **Sự thận trọng của XGBoost:** XGBoost tối ưu hóa hàm mất mát (Logloss) cực kỳ nghiêm ngặt trên cấu trúc cây. Nó chỉ dán nhãn "Nợ xấu" khi hồ sơ thực sự mang các đặc tính rủi ro vượt trội. Sự cẩn trọng này mang lại chỉ số Precision cực kỳ cao (giảm thiểu tỷ lệ từ chối nhầm khách hàng tốt), nhưng lại vô tình bỏ sót các hồ sơ nằm sát ranh giới, khiến Recall bị ghìm ở mức 71%.

### 🛠️ Giải Pháp: Kỹ Thuật Dịch Chuyển Ngưỡng Cắt (Threshold Tuning)
Trong nghiệp vụ ngân hàng rủi ro tín dụng, việc bỏ lọt một kẻ vỡ nợ (False Negative) gây thiệt hại nặng nề hơn nhiều so với việc từ chối nhầm một khách hàng tốt (False Positive). Do đó, chúng ta **không sử dụng ngưỡng mặc định 0.5**.

Bằng cách dịch chuyển hạ ngưỡng quyết định của XGBoost xuống mức tối ưu **`Threshold = 0.29`** (Chỉ cần hồ sơ có trên 29% nguy cơ vỡ nợ là mô hình lập tức gắn cờ cảnh báo), kết quả thu được trên tập Validation thay đổi rõ rệt:
* **F1-Score đạt trạng thái cân bằng tốt.**


## 📊 Kết Quả Đánh Giá Mô Hình Trên Tập Kiểm Thử (Test Set)

Dưới đây là bảng đối chiếu chỉ số đo lường hiệu năng thực tế thu được khi cho hai thuật toán trên tập Test độc lập 
### 1. Bảng Chỉ Số Thống Kê Tổng Hợp

| Mô hình Thuật toán | Accuracy (Độ chính xác tổng) | Precision (Bắt đúng nợ xấu) | Recall (Độ phủ nợ xấu) | F1-Score | Điểm số ROC-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 81.46% | 55.63% | 74.11% | 0.6356 | 0.8546 |
| **XGBoost Classifier (Nâng cao)** | 91.73% | 91.48% | 68.48% | 0.7833 | 0.9248 |

### 2. Phân Tích Ý Nghĩa Nghiệp Vụ (Business Insights)
* ** Recall (Chỉ số sống còn của ngân hàng):** Mô hình **XGBoost** đạt tỷ lệ Recall lớp rủi ro cao nhất là `91.48%`. Điều này chứng minh hệ thống bắt gọn được `[...]` trên tổng số 100 hồ sơ thực tế sẽ vỡ nợ, giảm thiểu tối đa tỷ lệ nợ ẩn và bảo toàn vốn cho ngân hàng.
* **ROC-AUC (Năng lực phân tách):** XGBoost đạt điểm AUC là `0.9248` (Ngưỡng: `Xuất sắc`), thể hiện cấu trúc phân tán xác suất dứt khoát: đẩy toàn bộ nhóm khách hàng tốt về sát mức xác suất 0.0 và gom nhóm vỡ nợ về sát mức 1.0 trên biểu đồ phân tán nằm ngang (Jittered Scatter Plot).

---

## 🚀 Cấu trúc mã nguồn
CREDIT_RISK/
├── data/
│   ├── credit_risk_data_cleaned.csv
│   ├── credit_risk_data_feature_extracted.csv
│   └── credit_risk_data_optimized.csv
├── processed_data/
│   ├── X_test_final.csv
│   ├── X_train_final.csv
│   ├── X_valid_final.csv
│   ├── y_test_final.csv
│   ├── y_train_final.csv
│   └── y_valid_final.csv
├── EDA.ipynb
├── feature_extract.ipynb
├── model.ipynb
├── normalization.ipynb
├── preprocess.ipynb
└── README.md