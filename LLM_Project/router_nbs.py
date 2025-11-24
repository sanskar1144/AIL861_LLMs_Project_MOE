import json

with open('/Users/animeshsingh/Desktop/Datasets/AI Experts/data_router.json', 'r') as f:
    data = json.load(f)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import joblib
import numpy as np
from typing import List, Dict, Tuple, Any


def detect_text_label_keys(sample: Dict[str, Any],
                           text_candidates=("text","sentence","body","review","doc","message"),
                           label_candidates=("label","class","target","cat","category")) -> Tuple[str,str]:
    keys = list(sample.keys())
    text_key = None
    label_key = None
    for k in keys:
        kl = k.lower()
        if any(tok in kl for tok in text_candidates) and text_key is None:
            text_key = k
        if any(tok in kl for tok in label_candidates) and label_key is None:
            label_key = k
    if text_key is None or label_key is None:
        if len(keys) >= 2:
            # fallback: first two keys
            return keys[0], keys[1]
        raise ValueError("Couldn't detect text/label keys in sample dict; please provide keys explicitly.")
    return text_key, label_key

def list_of_dicts_to_dict_of_lists(data_list: List[Dict[str, Any]],
                                   text_key: str = None, label_key: str = None) -> Tuple[dict,int]:

    if not isinstance(data_list, list):
        raise TypeError("Input must be a list of dicts.")
    if len(data_list) == 0:
        raise ValueError("Empty list provided.")

    # find first dict to detect keys if not provided
    first_dict = None
    for item in data_list:
        if isinstance(item, dict):
            first_dict = item
            break
    if first_dict is None:
        raise ValueError("List contains no dict items.")

    if text_key is None or label_key is None:
        text_key, label_key = detect_text_label_keys(first_dict)

    texts = []
    labels = []
    skipped = 0
    for item in data_list:
        if not isinstance(item, dict):
            skipped += 1
            continue
        if text_key not in item or label_key not in item:
            skipped += 1
            continue
        t = item[text_key]
        l = item[label_key]
        # skip None or empty text
        if t is None:
            skipped += 1
            continue
        texts.append(str(t))
        labels.append(l)

    if len(texts) == 0:
        raise ValueError("No valid text/label pairs found after conversion.")

    return {"text": texts, "label": labels}, skipped


def train_mnb_tfidf_from_list(data_list: List[Dict[str, Any]],
                              test_size: float = 0.2,
                              random_state: int = 42,
                              do_grid_search: bool = False,
                              grid_alphas=(0.1, 0.5, 1.0),
                              save_pipeline_path: str = "mnb_tfidf_pipeline.joblib",
                              save_labelencoder_path: str = "label_encoder.joblib"):


    data_dict, skipped = list_of_dicts_to_dict_of_lists(data_list)
    texts = np.asarray(data_dict["text"])
    labels_raw = np.asarray(data_dict["label"])

    print(f"Converted {len(texts)} examples (skipped {skipped} malformed entries).")

    # Basic label checks
    label_counts = Counter(labels_raw)
    print("Label distribution:", label_counts)
    if len(label_counts) < 2:
        raise RuntimeError("Need at least 2 distinct labels to train a classifier.")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels_raw)

    # Split (stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=1)),
        ("clf", MultinomialNB())
    ])

    # Optional grid search for alpha
    if do_grid_search:
        param_grid = {"clf__alpha": list(grid_alphas)}
        gs = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        print("Grid search best params:", gs.best_params_)
    else:
        model = pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}\n")
    print("Classification report (encoded labels):")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save
    joblib.dump(model, save_pipeline_path)
    joblib.dump(le, save_labelencoder_path)
    print(f"\nSaved pipeline: {save_pipeline_path}")
    print(f"Saved label encoder: {save_labelencoder_path}")

    return model, le, (X_test, y_test)


if __name__ == "__main__":
 
    model, label_encoder, (X_test, y_test) = train_mnb_tfidf_from_list(
        data,
        test_size=0.2,
        random_state=42,
        do_grid_search=False,
        save_pipeline_path="mnb_tfidf_pipeline.joblib",
        save_labelencoder_path="label_encoder.joblib"
    )

    new_texts = [
        "How long does it typically take for antidepressants to start working?",
        "What are the symptoms of anxiety?"
    ]
    preds_enc = model.predict(new_texts)
    preds = label_encoder.inverse_transform(preds_enc)
    for t, p in zip(new_texts, preds):
        print(f"\"{t}\"  ->  {p}")