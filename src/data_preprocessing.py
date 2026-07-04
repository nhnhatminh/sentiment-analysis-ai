"""
preprocessing.py
-----------------
Module Tien xu ly & Lam sach du lieu tho cho bai toan Review Classification.

Nhanh: feature/data-preprocessing

Chuc nang chinh:
    1. Doc du lieu tu file CSV local, loai bo cac dong bi khuyet (NaN/Null)
       o truong noi dung van ban.
    2. Chuan hoa nhan diem so: trich xuat so tu chuoi goc, ap dung bo loc
       nhi phan (4-5 sao -> nhan 1 - Tich cuc; 1-2 sao -> nhan 0 - Tieu cuc;
       3 sao -> drop hoan toan).
    3. Lam sach chuoi van ban (Text Purifying) thong qua class TextCleaner:
       lowercase, bo dau cau/ky tu dac biet/chu so bang Regex, loc Stopwords
       tieng Anh tieu chuan.

San pham ban giao: class TextCleaner (nhan text tho -> tra ve text sach).
"""

import re
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Danh sach Stopwords tieng Anh tieu chuan
# ---------------------------------------------------------------------------
# Uu tien dung bo stopwords chuan cua NLTK neu da duoc tai ve san. Neu khong
# co san (moi truong khong co internet / chua download corpus), fallback ve
# mot danh sach stopwords tieng Anh tieu chuan duoc nhung san trong code de
# dam bao module luon chay duoc, khong phu thuoc buoc tai du lieu ben ngoai.
def _load_stopwords() -> set:
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    except Exception:
        # Danh sach stopwords tieng Anh tieu chuan (tuong duong bo NLTK 'english')
        return {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
            "you're", "you've", "you'll", "you'd", "your", "yours", "yourself",
            "yourselves", "he", "him", "his", "himself", "she", "she's", "her",
            "hers", "herself", "it", "it's", "its", "itself", "they", "them",
            "their", "theirs", "themselves", "what", "which", "who", "whom",
            "this", "that", "that'll", "these", "those", "am", "is", "are",
            "was", "were", "be", "been", "being", "have", "has", "had",
            "having", "do", "does", "did", "doing", "a", "an", "the", "and",
            "but", "if", "or", "because", "as", "until", "while", "of", "at",
            "by", "for", "with", "about", "against", "between", "into",
            "through", "during", "before", "after", "above", "below", "to",
            "from", "up", "down", "in", "out", "on", "off", "over", "under",
            "again", "further", "then", "once", "here", "there", "when",
            "where", "why", "how", "all", "any", "both", "each", "few",
            "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "s", "t",
            "can", "will", "just", "don", "don't", "should", "should've",
            "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren",
            "aren't", "couldn", "couldn't", "didn", "didn't", "doesn",
            "doesn't", "hadn", "hadn't", "hasn", "hasn't", "haven",
            "haven't", "isn", "isn't", "ma", "mightn", "mightn't", "mustn",
            "mustn't", "needn", "needn't", "shan", "shan't", "shouldn",
            "shouldn't", "wasn", "wasn't", "weren", "weren't", "won",
            "won't", "wouldn", "wouldn't",
        }


STOPWORDS = _load_stopwords()


# ---------------------------------------------------------------------------
# 2. TextCleaner - San pham ban giao chinh
# ---------------------------------------------------------------------------
class TextCleaner:
    """
    Class chiu trach nhiem lam sach mot chuoi text tho thanh chuoi text sach,
    san sang cho cac buoc vectorize / feature engineering o cac giai doan sau.

    Pipeline lam sach (Text Purifying):
        1. Ep kieu ve string va chuyen ve lowercase.
        2. Dung Regex loai bo:
            - The HTML (neu co, phong truong hop text crawl tu web).
            - Dau cau, ky tu dac biet (chi giu lai chu cai va khoang trang).
            - Chu so.
        3. Chuan hoa khoang trang thua (nhieu space -> 1 space, strip 2 dau).
        4. Tach tu (tokenize theo khoang trang) va loc bo Stopwords tieng Anh.

    Usage:
        cleaner = TextCleaner()
        clean_text = cleaner.clean("This Product is AMAZING!! 10/10 <3")
        # -> "product amazing"
    """

    # Regex duoc bien dich san 1 lan de tang hieu nang khi goi nhieu lan.
    _HTML_TAG_RE = re.compile(r"<.*?>")
    _NON_ALPHA_RE = re.compile(r"[^a-z\s]")
    _MULTI_SPACE_RE = re.compile(r"\s+")

    def __init__(self, stopwords: set = None):
        """
        Parameters
        ----------
        stopwords : set, optional
            Cho phep truyen vao mot bo stopwords tuy chinh. Neu khong truyen,
            su dung bo stopwords tieng Anh tieu chuan (STOPWORDS o tren).
        """
        self.stopwords = stopwords if stopwords is not None else STOPWORDS

    def clean(self, text) -> str:
        """
        Nhan vao mot chuoi text tho, tra ve chuoi text da duoc lam sach.

        Parameters
        ----------
        text : str
            Chuoi van ban tho (vi du: mot review/comment cua nguoi dung).

        Returns
        -------
        str
            Chuoi van ban da duoc lam sach: lowercase, khong dau cau/so,
            khong stopwords, khong khoang trang thua.
        """
        # Bao ve truong hop dau vao la NaN/None hoac khong phai string.
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return ""
        text = str(text)

        # 1. Lowercase
        text = text.lower()

        # 2. Bo the HTML (neu co)
        text = self._HTML_TAG_RE.sub(" ", text)

        # 3. Bo dau cau, ky tu dac biet va chu so (chi giu chu cai + space)
        text = self._NON_ALPHA_RE.sub(" ", text)

        # 4. Chuan hoa khoang trang
        text = self._MULTI_SPACE_RE.sub(" ", text).strip()

        # 5. Loc Stopwords
        tokens = [word for word in text.split(" ") if word and word not in self.stopwords]

        return " ".join(tokens)

    def clean_batch(self, texts) -> list:
        """Ap dung clean() cho mot iterable (list/Series) cac chuoi text."""
        return [self.clean(t) for t in texts]

    # Cho phep goi truc tiep instance nhu mot ham: cleaner(text)
    def __call__(self, text) -> str:
        return self.clean(text)


# ---------------------------------------------------------------------------
# 3. Chuan hoa nhan diem so (Score -> Label nhi phan)
# ---------------------------------------------------------------------------
def extract_rating(raw_score) -> float:
    """
    Trich xuat so (rating) tu mot chuoi/gia tri tho. Ho tro cac dinh dang
    thuong gap trong du lieu crawl nhu: "5", "5.0", "5 stars", "5/5", "Rating: 4".

    Tra ve None neu khong trich xuat duoc so hop le.
    """
    if raw_score is None or (isinstance(raw_score, float) and pd.isna(raw_score)):
        return None

    if isinstance(raw_score, (int, float)):
        return float(raw_score)

    match = re.search(r"(\d+(\.\d+)?)", str(raw_score))
    if not match:
        return None
    return float(match.group(1))


def score_to_label(score) -> int:
    """
    Ap dung bo loc nhi phan tren diem so (thang 1-5):
        - 4-5 sao -> 1 (Tich cuc)
        - 1-2 sao -> 0 (Tieu cuc)
        - 3 sao hoac gia tri khong hop le -> None (se bi drop)
    """
    rating = extract_rating(score)
    if rating is None:
        return None
    if rating >= 4:
        return 1
    if rating <= 2:
        return 0
    return None  # 3 sao -> trung lap, drop hoan toan


# ---------------------------------------------------------------------------
# 4. Doc & lam sach du lieu tu CSV (Data Loading pipeline)
# ---------------------------------------------------------------------------
def load_and_preprocess(
    csv_path: str,
    text_column: str = "review",
    score_column: str = "rating",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Doc dataset tu file CSV local va thuc hien toan bo buoc tien xu ly:
        1. Doc CSV.
        2. Drop cac dong bi khuyet (NaN/Null) o truong noi dung van ban.
        3. Chuan hoa diem so thanh nhan nhi phan (0/1), drop bai 3 sao.
        4. Lam sach van ban bang TextCleaner.

    Parameters
    ----------
    csv_path : str
        Duong dan toi file CSV local.
    text_column : str
        Ten cot chua noi dung van ban tho.
    score_column : str
        Ten cot chua diem so/rating tho.
    encoding : str
        Encoding cua file CSV (mac dinh utf-8).

    Returns
    -------
    pd.DataFrame
        DataFrame voi 2 cot: 'clean_text' (van ban da lam sach) va
        'label' (0 = Tieu cuc, 1 = Tich cuc).
    """
    df = pd.read_csv(csv_path, encoding=encoding)

    # Buoc 1: Loai bo cac dong bi khuyet (NaN/Null) o truong van ban va diem so
    df = df.dropna(subset=[text_column, score_column])

    # Buoc 2: Chuan hoa nhan diem so, drop bai 3 sao / khong hop le
    df["label"] = df[score_column].apply(score_to_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Buoc 3: Lam sach van ban
    cleaner = TextCleaner()
    df["clean_text"] = df[text_column].apply(cleaner.clean)

    # Loai bo cac dong ma sau khi lam sach text bi rong (khong con noi dung)
    df = df[df["clean_text"].str.len() > 0]

    return df[["clean_text", "label"]].reset_index(drop=True)


if __name__ == "__main__":
    # Vi du chay thu nhanh (smoke test) khi thuc thi truc tiep file nay.
    sample = pd.DataFrame(
        {
            "review": [
                "This product is AMAZING!! 10/10 would buy again <3",
                None,
                "Terrible experience, broke after 2 days...",
                "It's okay, nothing special.",
            ],
            "rating": ["5 stars", "4", "1", "3"],
        }
    )
    sample.to_csv("/tmp/_sample_reviews.csv", index=False)

    result = load_and_preprocess("/tmp/_sample_reviews.csv")
    print(result)