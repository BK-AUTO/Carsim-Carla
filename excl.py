import pandas as pd
import numpy as np

# Đọc dữ liệu từ file Excel
file_path = "tiremodel.xls"
data = pd.read_excel(file_path)  # không có tiêu đề

Slip = data.iloc[:, 0].values   # cột 1
Roadmu = data.iloc[:, 1].values # cột 2

# Hàm tìm u(s) bằng nội suy tuyến tính
def u_of_s(s):
    return np.interp(s, Slip, Roadmu)

# Ví dụ: tìm giá trị u tại s = -0.93
s_value = -0.95
u_value = u_of_s(s_value)
print(f"u({s_value}) = {u_value}")
