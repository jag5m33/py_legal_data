# simple cleaning function
# 2. Write your cleaning function with re. This takes one clause and returns a cleaner version:

    # - lowercase
    # - money becomes moneytoken
    # - numbers become numtoken
    # - remove bracketed list markers like (a) and (ii)
    # - remove remaining punctuation and symbols
    # - collapse extra spaces
 
import re
def clean_text(text):
    text = re.sub(r"\S+@\S+", " emailtoken ", text) # emails
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text) # websites
    text = re.sub(r"[$£€]\s?\d[\d,]*(\.\d+)?", " moneytoken ", text) # money
    text = re.sub(r"\(\s*([a-z]|[ivx]+|\d+)\s*\)", " ", text, flags=re.I) # list markers: (a) (ii) (3)
    text = re.sub(r"\d+([.,]\d+)*", " numtoken ", text) # any other number
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text) # punctuation and symbols
    return re.sub(r"\s+", " ", text).strip()     