import pandas as pd
import re

# misal dataset
df = pd.read_csv("resumes.csv")

# fungsi bersihkan HTML dan whitespace
def clean_text(text):
    text = re.sub(r"<.*?>", "", str(text))  # hapus tag HTML
    return text.strip().lower()

df["category_clean"] = df["category"].apply(clean_text)

def search_resumes_by_category(category):
    category = category.strip().lower()
    return df[df["category_clean"] == category]

print(search_resumes_by_category("hr"))