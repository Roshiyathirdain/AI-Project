# create_test_data.py
import pandas as pd
import os

data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Create CSV
df_csv = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Department": ["HR", "Engineering", "Marketing", "Sales"],
    "Salary": [60000, 85000, 55000, 70000]
})
df_csv.to_csv(os.path.join(data_dir, "employees.csv"), index=False)
print("Created employees.csv")

# Create Excel
df_excel = pd.DataFrame({
    "Project": ["Alpha", "Beta", "Gamma"],
    "Status": ["Completed", "In Progress", "Planning"],
    "Budget": [100000, 250000, 50000]
})
df_excel.to_excel(os.path.join(data_dir, "projects.xlsx"), index=False, engine='openpyxl')
print("Created projects.xlsx")

# Create Document
with open(os.path.join(data_dir, "roadmap.md"), "w", encoding="utf-8") as f:
    f.write("# Strategic Roadmap 2024\n\n## Quarter 1\n- Launch new backend\n- Security audit\n\n## Quarter 2\n- Market expansion\n- Native mobile app")
print("Created roadmap.md")
