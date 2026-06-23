import os
import pandas as pd

def audit_processed_dataset(file_path):
    if not os.path.exists(file_path):
        print(f"[ERROR] Processed data file not found at: {file_path}")
        return

    print(f"[INFO] Starting comprehensive data quality audit for: {file_path}\n")
    df = pd.read_csv(file_path)

    total_rows = len(df)
    print(f"--- 1. General Dataset Statistics ---")
    print(f"Total processed records: {total_rows}")

    label_counts = df['Target Label'].value_counts()
    pos_count = label_counts.get(1, 0)
    neg_count = label_counts.get(0, 0)
    print(f"Positive labels (1): {pos_count} ({pos_count/total_rows*100:.2f}%)")
    print(f"Negative labels (0): {neg_count} ({neg_count/total_rows*100:.2f}%)")

    empty_reviews = df['Review Text'].isna().sum() + (df['Review Text'].str.strip() == '').sum()
    print(f"\n--- 2. Structural Integrity Check ---")
    print(f"Empty or whitespace-only rows: {empty_reviews}")

    placeholder_count = df['Review Text'].str.contains('review text not found', case=False, na=False).sum()
    print(f"Remaining placeholder text ('review text not found'): {placeholder_count}")

    broken_words = ['don', 've', 'couldn', 'cant', 'st', 'nd', 'rd', 'th']
    print(f"\n--- 3. Broken Token Remnants Check ---")
    for word in broken_words:
        count = df['Review Text'].str.contains(rf'\b{word}\b', case=False, na=False).sum()
        print(f"Occurrences of isolated remnant word '{word}': {count}")

    df['Word Count'] = df['Review Text'].apply(lambda x: len(str(x).split()))
    max_words = df['Word Count'].max()
    min_words = df['Word Count'].min()
    avg_words = df['Word Count'].mean()
    print(f"\n--- 4. Text Density Analysis ---")
    print(f"Maximum word count in a single review: {max_words} words")
    print(f"Minimum word count in a single review: {min_words} words")
    print(f"Average word count across dataset: {avg_words:.2f} words")

if __name__ == "__main__":
    processed_file_path = os.path.join("data", "processed", "clean_data.csv")
    audit_processed_dataset(processed_file_path)