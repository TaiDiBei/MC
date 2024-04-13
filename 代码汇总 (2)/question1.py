import pandas as pd

# 读取CSV文件
df = pd.read_csv('attachment/attachment1.csv', encoding="GBK")

# 打印第一列数据
print(df[df["分拣中心"] == "SC48"]["日期"])
