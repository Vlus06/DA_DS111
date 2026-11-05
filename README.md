# Nguy cơ cháy rừng Việt Nam (63 tỉnh, 2012–2025)

Mở Jupyter từ thư mục `notebooks/` vì mọi đường dẫn là `../data/...`.

```
notebooks/
  01_thu_thap_du_lieu.ipynb   tải FIRMS, Open-Meteo, ONI, làm giàu điểm cháy -> data/raw, data/interim
  02_gop_du_lieu.ipynb        ghép thành bảng tỉnh x ngày -> data/processed
  03_tien_xu_ly.ipynb         làm sạch, chọn mẫu, tạo đặc trưng -> data/clean
  04_EDA.ipynb                phân tích khám phá
  05_mo_hinh_du_bao.ipynb     M0/M1/M2, H1-H6, kết quả -> docs/ket_qua/*.csv
  modeling.py                 hàm cho notebook 05
  utils_firerisk.py           hàm dùng chung
  test_utils.py               kiểm thử cho utils_firerisk.py
data/                         toàn bộ là CSV
docs/                         prereg_v1.md, ket_qua_phan_tich_dev.xlsx
```

## Chạy lại từ đầu

1. `pip install -r requirements.txt`
2. Đặt biến môi trường `FIRMS_MAP_KEY` trước khi mở Jupyter.
3. Chạy lần lượt 01 đến 04. Tệp đã tải trong `data/raw` được dùng lại, không tải lại.

Chỉ chạy 03 và 04 thì không cần các bước trên vì `data/processed/panel_dev.csv` đã có sẵn.

Thiếu so với bản đầy đủ: `data/interim/hotspots_enriched.csv` và 767 trong 945 tệp thời tiết `data/raw/wx/<tỉnh>_<năm>.csv`. Notebook 01 sẽ tải bù phần thiếu.

## Quy tắc

`panel_SEALED_DO_NOT_OPEN.csv` (hai mùa cuối của mỗi tỉnh) không được mở hay thống kê cho đến khi chốt mô hình, xem `docs/prereg_v1.md`.
