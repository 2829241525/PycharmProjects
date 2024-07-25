import numpy as np
import pandas as pd

# 设定基准年份和起始年份
base_year = 2000
end_year = 2023
base_salary = 2363  # 假设2000年的起薪为2000元

# 生成年份序列
years = np.arange(base_year, end_year + 1)

# 模拟起薪增长，假设每年增长率介于5%至10%之间，反映经济增长和通货膨胀的影响
growth_rates = np.random.uniform(0.04, 0.06, len(years) - 1)
salaries = [base_salary]
for growth_rate in growth_rates:
    salaries.append(salaries[-1] * (1 + growth_rate))

year_data = []
for data in years:
    year_data.append(data)

years_str = ','.join(map(str, years))
salaries_str = ','.join(map(str, salaries))

print(year_data)
print(salaries)

