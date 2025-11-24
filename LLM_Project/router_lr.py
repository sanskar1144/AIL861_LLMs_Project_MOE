import json

with open('/Users/animeshsingh/Desktop/Datasets/AI Experts/data_router.json', 'r') as f:
    data = json.load(f)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

def convert_list_of_dicts(data, text_key="text", label_key="class"):
    texts = []
    labels = []
    for item in data:
        if isinstance(item, dict):
            if text_key in item and label_key in item:
                texts.append(item[text_key])
                labels.append(item[label_key])
    return {"text": texts, "label": labels}

data = convert_list_of_dicts(data)

def _detect_keys(data_dict):
    """Heuristic to detect which key is text and which is label in the provided dict."""
    keys = list(data_dict.keys())
    if len(keys) != 2:
        raise ValueError("Expecting a dictionary with exactly two keys (text and class).")
    # common text key names
    text_candidates = [k for k in keys if any(tok in k.lower() for tok in ("text","sentence","body","review","doc","message"))]
    label_candidates = [k for k in keys if any(tok in k.lower() for tok in ("label","class","target","cat","category"))]
    if len(text_candidates) == 1 and len(label_candidates) == 1:
        return text_candidates[0], label_candidates[0]
    # fallback: assume first is text, second is label
    return keys[0], keys[1]

def train_logreg_tfidf(data_dict, test_size=0.1, random_state=42, do_grid=False,
                       save_pipeline_path=None, save_labelencoder_path=None):
   
    text_key, label_key = _detect_keys(data_dict)
    texts = np.asarray(data_dict[text_key])
    labels = np.asarray(data_dict[label_key])
    if len(texts) != len(labels):
        raise ValueError("Text and label lists must be the same length.")
    # Encode labels to 0..n-1
    le = LabelEncoder()
    y = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=test_size, stratify=y, random_state=random_state
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=1)),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="multinomial", random_state=random_state))
    ])

    if do_grid:
        param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
        gs = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        print("Grid search best params:", gs.best_params_)
    else:
        model = pipeline.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}\n")
    print("Classification report (labels shown as encoded integers):\n")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    # Optionally save pipeline and label encoder
    if save_pipeline_path:
        joblib.dump(model, save_pipeline_path)
        print(f"Saved pipeline to: {save_pipeline_path}")
    if save_labelencoder_path:
        joblib.dump(le, save_labelencoder_path)
        print(f"Saved label encoder to: {save_labelencoder_path}")

    return model, le, (X_test, y_test)


if __name__ == "__main__":


    model, label_encoder, (X_test, y_test) = train_logreg_tfidf(
        data,
        test_size=0.2,
        random_state=42,
        do_grid=False,
        save_pipeline_path="logreg_tfidf_pipeline.joblib",
        save_labelencoder_path="label_encoder.joblib"
    )

    # How to make predictions on new raw text:
    new_texts = ["best ways to train your puppy", "how to bake a chocolate cake"]
    preds_encoded = model.predict(new_texts)
    preds_labels = label_encoder.inverse_transform(preds_encoded)
    print("Predictions:", list(zip(new_texts, preds_labels)))




