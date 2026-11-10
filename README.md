# Nguy cơ cháy rừng Việt Nam (63 tỉnh, 2012–2025)

Dữ liệu: [VLUS06/vn-forest-fire-data](https://huggingface.co/datasets/VLUS06/vn-forest-fire-data) trên Hugging Face.

Câu hỏi: chỉ số cháy kiểu Nesterov (P) có đủ tin cậy để cảnh báo cháy rừng ở Việt Nam không? Notebook `05_cau_chuyen` kể câu chuyện đầy đủ, tóm tắt như sau.

- P hơn khí hậu thuần túy nhưng thua một biến đo ngay trong ngày (độ hụt bão hòa) ở cả sáu vùng có rừng, vì khô kéo dài quá một tuần không làm cháy dễ hơn và mưa dưới 5 mm đã đủ dập cháy ở phần lớn các vùng.
- Ngưỡng độ khô để cháy bắt đầu khác nhau gần gấp đôi giữa các vùng và đổi theo vĩ độ, độ cao, nên một thang P chung không có nghĩa như nhau ở mọi nơi.
- Cháy là quá trình có đà, có đợt và có tính vùng: hôm qua có cháy thì hôm nay cháy cao hơn dù cùng độ khô, mùa đã cháy nhiều tiếp tục cháy nhiều, 5% số ngày lớn nhất chứa 31% đến 61% số điểm cháy, và các tỉnh gần nhau đồng bộ.
- Con người để lại dấu vết: cháy giảm khoảng 6 điểm phần trăm quanh Tết, và cháy trên đất nông nghiệp ít phụ thuộc thời tiết hơn cháy rừng.
- Điểm cháy giảm mạnh sau 2015 ở Tây Nguyên và Nam Trung Bộ nhưng chưa rõ do cháy thật hay do dữ liệu.

Mở Jupyter từ thư mục `notebooks/` vì mọi đường dẫn là `../data/...`.

## Nguồn dữ liệu

| Nguồn | Dùng để làm gì | Cách thu thập |
|---|---|---|
| NASA FIRMS, VIIRS S-NPP | điểm cháy vệ tinh (vị trí, ngày giờ, độ tin cậy, công suất cháy), là biến kết quả của đề tài | API theo đoạn 5 ngày, 1.019 đoạn, cần khóa `FIRMS_MAP_KEY` |
| Open-Meteo Archive (ERA5) | nhiệt độ, điểm sương, mưa theo giờ của từng tỉnh; lấy giá trị 13h giờ địa phương, đúng lúc vệ tinh bay qua | API theo tỉnh và năm, 945 tệp |
| ESA WorldCover 2021 | lớp phủ đất 10 m, để phân biệt điểm cháy trên rừng với cháy đồng ruộng và đô thị | đọc ảnh COG trên AWS S3 |
| geoBoundaries ADM1 | ranh giới 63 tỉnh cũ, để gán mỗi điểm cháy vào một tỉnh | tải qua API |
| NOAA CPC | chỉ số ONI, đo El Niño và La Niña | tệp văn bản theo tháng |
| Nhập tay | ngày Tết, Giỗ Tổ (kiểm chứng bằng thư viện lịch âm) | bảng trong `utils_firerisk.py` |

## Mỗi nguồn cho những feature nào và ghép lại thế nào

Bảng `panel.csv` có mỗi dòng là một tỉnh trong một ngày (63 tỉnh x 5.095 ngày = 320.985 dòng, 76 cột). Mỗi nguồn trả lời một câu hỏi khác nhau.

| Nguồn | Câu hỏi nó trả lời | Dữ liệu thô | Feature trong `panel.csv` | Vai trò |
|---|---|---|---|---|
| NASA FIRMS (VIIRS S-NPP) | Ở đâu, ngày nào có cháy? | danh sách điểm cháy: vĩ độ, kinh độ, giờ vệ tinh bay qua, độ tin cậy, công suất cháy | `n_hot_all`, `n_hot_forest`, `fire_any_forest`, `event_start`, `large_event_active` | biến kết quả, thứ cần giải thích |
| Open-Meteo Archive (ERA5) | Hôm đó trời thế nào? | nhiệt độ, điểm sương, mưa theo giờ tại một điểm đại diện của tỉnh | `T13`, `Td13`, `RH13`, `D13_hpa`, `dd13_c`, các cột mưa (`rain_cal`, `rain_3d`, `rain_7d`, `rain_14d`, `rain_30d`), `dsr_*`, `T13_roll7`, `D13_roll7`, `T13_anom`, `D13_anom` và các cột `P_*` | biến giải thích về độ khô; lấy giá trị 13h vì vệ tinh bay qua khoảng 13:30 |
| ESA WorldCover 2021 | Điểm cháy rơi trên loại đất nào? | bản đồ lớp phủ 10 m (cây, cỏ, nông nghiệp, xây dựng) | `n_hot_forest`, `forest_share`, `crop_share`, `grass_share`, `built_share` | tách cháy rừng khỏi đốt đồng ruộng và đô thị |
| geoBoundaries ADM1 | Điểm cháy thuộc tỉnh nào? | đa giác ranh giới 63 tỉnh cũ | khóa `prov`, `area_km2`, `rep_lat`, `rep_lon` | khóa nối giữa điểm cháy và tỉnh |
| NOAA CPC | Năm đó là El Niño hay La Niña? | chỉ số ONI theo tháng | `oni`, `oni_lag3` | kiểm tra ảnh hưởng của khí hậu quy mô lớn |
| Nhập tay | Hôm đó có phải ngày lễ không? | ngày Tết, Giỗ Tổ, Thanh minh, các ngày lễ dương lịch | `days_from_tet`, `tet_win3`, `days_from_giotuo`, `giotuo_win1`, `thanh_minh_win1`, `holiday_solar`, cùng `dow`, `is_weekend`, `month`, `doy`, `year` | kiểm tra dấu vết của con người, ví dụ cháy giảm quanh Tết |

Khóa nối là (tỉnh, ngày địa phương). Việc ghép gồm ba bước:

1. **Điểm cháy thành tỉnh và loại đất.** Mỗi điểm cháy được đối chiếu với đa giác tỉnh để biết nó thuộc tỉnh nào, rồi lấy 9 điểm mẫu quanh nó trên bản đồ WorldCover. Nếu từ một nửa số mẫu trở lên là cây (`tree_frac` >= 0,5) thì điểm đó tính là cháy rừng. Giờ vệ tinh (UTC) được đổi sang giờ địa phương (UTC+7) để có đúng ngày.
2. **Đếm theo tỉnh và ngày.** Số điểm cháy của từng tỉnh trong từng ngày thành `n_hot_all` (mọi loại đất) và `n_hot_forest` (chỉ rừng). `fire_any_forest` bằng 1 nếu `n_hot_forest` lớn hơn 0.
3. **Gắn thời tiết, lịch và đặc trưng tĩnh.** Thời tiết của tỉnh vào ngày đó nối theo khóa (tỉnh, ngày). Lịch và ONI nối theo ngày hoặc tháng. Đặc trưng tĩnh như `forest_share` lặp lại giống nhau ở mọi ngày của cùng một tỉnh.

Các biến trễ như `rain_30d`, `dsr_5` và chỉ số P cộng dồn nhiều ngày được tính từ chuỗi thời tiết riêng của từng tỉnh, bắt đầu từ 01/01/2011, nên những ngày đầu của 2012 các biến này đã có giá trị.

Ví dụ thật, Gia Lai, 10 đến 13/03/2019:

| ngày | T13 | Td13 | D13_hpa | rain_cal | P_vn_5mm_Ddew | n_hot_all | n_hot_forest | fire_any_forest |
|---|---|---|---|---|---|---|---|---|
| 10/03 | 34,5 | 16,9 | 35,4 | 0 | 30.551 | 65 | 22 | 1 |
| 11/03 | 35,8 | 15,5 | 41,1 | 0 | 31.278 | 68 | 15 | 1 |
| 12/03 | 31,0 | 18,9 | 23,1 | 1,4 | 31.653 | 35 | 8 | 1 |
| 13/03 | 31,0 | 19,1 | 22,8 | 3,1 | 32.022 | 26 | 5 | 1 |

Ngày 11/03 nóng và rất khô (T13 35,8 độ C, điểm sương chỉ 15,5 độ C) nên độ hụt bão hòa lên 41,1 hPa, và có 68 điểm cháy trong đó 15 điểm trên rừng. Không có mưa trên 5 mm nên P cứ tích lũy tăng đều; một ngày mưa trên 5 mm sẽ đưa P về 0. Khi mưa 1,4 và 3,1 mm xuất hiện và độ hụt giảm, số điểm cháy giảm dần. Đây cũng là phát hiện chính của bài: biến đo ngay trong ngày (độ hụt bão hòa) bám sát cháy hơn chỉ số P tích lũy.

Những điểm dễ nhầm:

- Điểm cháy thô nằm ở `data/interim/hotspots_enriched.csv`, không nằm trong `panel.csv`. `panel.csv` chỉ giữ số đếm theo ngày.
- `panel.csv` không có các cột mùa (`season`, `in_season`). Các cột này chỉ có ở `panel_clean.csv` vì mùa là giả định của phân tích.
- Mỗi tỉnh chỉ có một điểm thời tiết đại diện, nên thời tiết chỉ là mức thô cho cả tỉnh.
- Điểm cháy do vệ tinh ghi nhận nên không phải toàn bộ cháy thực tế: cảm biến không thấy cháy dưới mây dày và chỉ quan sát khoảng hai lần mỗi ngày.

## Dữ liệu

Toàn bộ dữ liệu (khoảng 1,2 GB, hơn 1.900 tệp CSV) nằm trên Hugging Face, không nằm trong repo này: [VLUS06/vn-forest-fire-data](https://huggingface.co/datasets/VLUS06/vn-forest-fire-data)

Repo giữ sẵn cây thư mục `data/` rỗng. Tải dữ liệu về đúng vị trí bằng một lệnh, chạy từ thư mục gốc của repo:

```bash
pip install -U huggingface_hub
```
```bash
huggingface-cli download VLUS06/vn-forest-fire-data --repo-type dataset --local-dir data
```

Sau khi tải, cây thư mục phải giống như bên dưới. Nếu chỉ muốn xem phân tích thì chỉ cần `data/clean/` và `data/interim/hotspots_enriched.csv`, các notebook 04 và 05 đọc từ đó.

## Cấu trúc thư mục

```
DoAn_CayRung/
├── README.md
├── requirements.txt                    # thư viện cần cài
│
├── notebooks/
│   ├── 01_thu_thap_du_lieu.ipynb       # tải điểm cháy FIRMS, thời tiết Open-Meteo, ONI; gán tỉnh và lớp phủ đất cho điểm cháy
│   ├── 02_gop_du_lieu.ipynb            # ghép các nguồn thành bảng tỉnh x ngày (panel.csv), báo cáo chất lượng
│   ├── 03_tien_xu_ly.ipynb             # kiểm tra chất lượng, xử lý null, chọn mẫu, giảm biến thể P, tạo đặc trưng
│   ├── 04_EDA.ipynb                    # khám phá: cháy ở đâu, khi nào, thời tiết ngày cháy, P và xác suất cháy, ngày lễ
│   ├── 05_cau_chuyen.ipynb             # câu chuyện chính: P có đủ tin cậy để cảnh báo cháy rừng không; bảy chương từ P thua ở đâu đến vì sao và còn thiếu gì
│   ├── utils_firerisk.py               # hàm dùng chung: tên tỉnh, giờ địa phương, chỉ số Nesterov, đặc trưng trễ, lịch, ONI
│   └── test_utils.py                   # kiểm thử cho utils_firerisk.py
│
├── data/                               # toàn bộ là CSV; trong repo chỉ có cây thư mục rỗng, dữ liệu tải từ Hugging Face
│   ├── raw/                            # dữ liệu thô tải về, notebook 01 ghi
│   │   ├── firms/                      # 1.019 tệp, mỗi tệp một đoạn 5 ngày điểm cháy VIIRS S-NPP
│   │   │   └── VIIRS_SNPP_SP_<bbox>_<ngày bắt đầu>_5.csv
│   │   ├── wx/                         # 945 tệp, thời tiết theo giờ của 63 tỉnh x 15 năm (2011–2025)
│   │   │   └── <tỉnh>_<năm>.csv        #   cột: time, temperature_2m, dew_point_2m, precipitation, elevation
│   │   └── README.txt
│   │
│   ├── interim/                        # bảng trung gian, notebook 01 ghi và notebook 02 đọc
│   │   ├── prov_table.csv              # 63 tỉnh: khóa, vùng, toạ độ đại diện, diện tích
│   │   ├── static_prov.csv             # đặc trưng tĩnh mỗi tỉnh: tỉ lệ rừng, nông nghiệp, cỏ, đô thị, độ cao
│   │   ├── hotspots_enriched.csv       # 947.287 điểm cháy trong 63 tỉnh, đã gắn tỉnh, giờ địa phương và lớp phủ WorldCover
│   │   ├── oni.csv                     # chỉ số ONI (ENSO) theo tháng, tự tải khi chưa có
│   │   └── README.txt
│   │
│   ├── processed/                      # bảng gộp, notebook 02 ghi
│   │   ├── panel.csv                   # bảng tỉnh x ngày, 320.985 dòng x 76 cột, có 29 biến thể P (253 MB, notebook 02 tạo ra)
│   │   ├── gate2_province_year.csv     # số ngày và số điểm cháy rừng theo tỉnh-năm
│   │   └── manifest.csv                # tham số và mã băm các tệp của lần chạy notebook 02
│   │
│   └── clean/                          # bảng sạch, notebook 03 ghi
│       ├── panel_clean.csv             # bảng đã làm sạch và thêm đặc trưng, có cờ analysis_sample đánh dấu mẫu phân tích
│       ├── data_dictionary.csv         # từ điển cột: tên, nhóm, kiểu, % null, mô tả
│       └── eligible_provinces.csv      # 49 tỉnh đủ điều kiện (40 tỉnh ở 6 vùng có rừng thuộc mẫu phân tích)
│
└── docs/
    ├── prereg_v1.md                    # tài liệu đăng ký trước giả thuyết (hướng bài báo)
    └── ket_qua_phan_tich_dev.xlsx      # kết quả phân tích ban đầu (bản gốc, không do notebook sinh ra)
```

## Luồng dữ liệu

```
firms/, wx/, ONI, ranh giới, WorldCover
        │  01
        ▼
interim/ ──02──► processed/panel.csv ──03──► clean/panel_clean.csv ──┬──04──► EDA
                                                                     └──05──► câu chuyện
```

## Chạy lại từ đầu

1. `pip install -r requirements.txt`
2. Đặt biến môi trường `FIRMS_MAP_KEY` trước khi mở Jupyter.
3. Chạy lần lượt 01 đến 05. Tệp đã tải trong `data/raw` được dùng lại, không tải lại.

Nếu đã tải dữ liệu từ Hugging Face thì có thể bắt đầu từ notebook 02 (cần `data/raw` và `data/interim`), hoặc chạy thẳng từ notebook 04 với `data/clean/panel_clean.csv`. Chỉ khi muốn thu thập lại từ nguồn gốc mới cần chạy notebook 01 và đặt khóa FIRMS.
