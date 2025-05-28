import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import os

try:
    from hazm import stopwords_list, Normalizer
except ImportError:
    print("نصب hazm...")
    os.system("pip install hazm")
    from hazm import stopwords_list, Normalizer

nltk.download('punkt')

df = pd.read_csv('movies_farsi.csv', encoding='utf-8')
df['overview'] = df['overview'].fillna('')

normalizer = Normalizer()
stop_words = set(stopwords_list())

def preprocess(text):
    text = normalizer.normalize(text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

df['overview_clean'] = df['overview'].apply(preprocess)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['overview_clean'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(movie_title, top_n=3):
    if movie_title not in df['title'].values:
        return [f"فیلم «{movie_title}» یافت نشد."]
    idx = df[df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return [f"{df.iloc[i]['title']} (شباهت: {score:.2f})" for i, score in sim_scores]

try:
    import tkinter as tk
    from tkinter import messagebox

    def on_recommend():
        movie = entry.get().strip()
        if not movie:
            messagebox.showwarning("خطا", "لطفاً نام یک فیلم وارد کنید.")
            return
        result = get_recommendations(movie)
        result_box.delete(0, tk.END)
        for item in result:
            result_box.insert(tk.END, item)

    root = tk.Tk()
    root.title("🎬 سیستم توصیه‌گر فیلم‌های فارسی")
    root.geometry("500x400")

    label = tk.Label(root, text="نام فیلم را وارد کنید:", font=("Helvetica", 12))
    label.pack(pady=10)

    entry = tk.Entry(root, width=40, font=("Helvetica", 12))
    entry.pack()

    btn = tk.Button(root, text="پیشنهاد فیلم", command=on_recommend, font=("Helvetica", 12))
    btn.pack(pady=10)

    result_box = tk.Listbox(root, width=60, height=10, font=("Helvetica", 11))
    result_box.pack(pady=10)

    root.mainloop()
except:
    print("محیط گرافیکی Tkinter در این محیط پشتیبانی نمی‌شود. لطفاً در دسکتاپ اجرا کنید.")
   