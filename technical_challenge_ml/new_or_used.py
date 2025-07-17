"""
Exercise description
--------------------

Description:
In the context of Mercadolibre's Marketplace an algorithm is needed to predict if an item listed in the markeplace is new or used.

Your tasks involve the data analysis, designing, processing and modeling of a machine learning solution 
to predict if an item is new or used and then evaluate the model over held-out test data.

To assist in that task a dataset is provided in `MLA_100k.jsonlines` and a function to read that dataset in `build_dataset`.

For the evaluation, you will use the accuracy metric in order to get a result of 0.86 as minimum. 
Additionally, you will have to choose an appropiate secondary metric and also elaborate an argument on why that metric was chosen.

The deliverables are:
--The file, including all the code needed to define and evaluate a model.
--A document with an explanation on the criteria applied to choose the features, 
  the proposed secondary metric and the performance achieved on that metrics. 
  Optionally, you can deliver an EDA analysis with other formart like .ipynb


"""


import json
import os
import pandas as pd
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, classification_report
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint, uniform
from tqdm.auto import tqdm
from joblib import parallel_backend, dump, load

# Configure Pandas to not clip output
pd.set_option('display.max_rows', None)


# === Extending RandomizedSearchCV with tqdm ===
class TQDMRandomizedSearchCV(RandomizedSearchCV):
    def fit(self, X, y=None, **fit_params):
        n_candidates = self.n_iter
        n_folds = self.cv if isinstance(self.cv, int) else len(self.cv)
        total = n_candidates * n_folds
        with tqdm(total=total, desc="🔍 Optimización", unit="fit") as pbar:
            self._pbar = pbar
            with parallel_backend("loky"):
                return super().fit(X, y, **fit_params)

    def _run_search(self, evaluate_candidates):
        def wrapped(candidate_params):
            out = evaluate_candidates(candidate_params)
            self._pbar.update(len(candidate_params) * self.cv)
            return out
        return super()._run_search(wrapped)

# === LIST OF COLUMNS TO IGNORE (flattened) ===
_IGNORED_COLUMNS = {
    "id", "url", "secure_url", "thumbnail", "permalink",
    "address_line", "description", "video_id", "catalog_product_id",
    "descriptions", "pictures", "deal_ids", "attributes", "tags",
    "listing_source", "parent_item_id", "coverage_areas",
    "international_delivery_mode", "official_store_id",
    "differential_pricing", "geolocation_latitude", "geolocation_longitude",
    "seller_contact_area_code2", "seller_contact_phone2",
    "seller_contact_webpage", "seller_contact_email",
    "seller_contact_contact", "seller_contact_area_code",
    "seller_contact_other_info", "seller_contact_phone",
    "location_open_hours", "location_neighborhood_name",
    "location_neighborhood_id", "location_longitude", "location_country_name",
    "location_country_id", "location_address_line", "location_latitude",
    "location_zip_code", "location_city_name", "location_city_id",
    "location_state_name", "location_state_id", "shipping_local_pick_up",
    "shipping_methods", "shipping_tags", "shipping_free_shipping",
    "shipping_mode", "shipping_dimensions", "seller_address_comment",
    "seller_address_longitude", "seller_address_id",
    "seller_address_country_name", "seller_address_country_id",
    "seller_address_address_line", "seller_address_latitude",
    "seller_address_search_location_neighborhood_name",
    "seller_address_search_location_neighborhood_id",
    "seller_address_search_location_state_name",
    "seller_address_search_location_state_id",
    "seller_address_search_location_city_name",
    "seller_address_search_location_city_id", "seller_address_zip_code",
    "seller_address_city_name", "seller_address_city_id",
    "seller_address_state_name", "seller_address_state_id", "base_price"
}

# === Utility to flatten and filter ignored columns ===
def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if new_key in _IGNORED_COLUMNS:
            continue
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)

def drop_ignored_columns(data):
    """Remove ignored keys from each dictionary."""
    return [{k: v for k, v in d.items() if k not in _IGNORED_COLUMNS} for d in data]

# === Show class distribution ===
def print_class_distribution(y, title=""):
    counter = Counter(y)
    total = len(y)
    print(f"\nClass distribution {title}:")
    for label, count in sorted(counter.items()):
        pct = (count / total) * 100
        print(f"  Clase '{label}': {count} ({pct:.2f}%)")

# === Build balanced dataset ===
def build_dataset():
    path = os.path.join("technical_challenge_ml", "MLA_100k.jsonlines")
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    data = [x for x in data if "condition" in x and x["condition"] in ("new", "used")]
    y_all = [x["condition"] for x in data]
    print_class_distribution(y_all, "inicial")

    for x in data:
        x.pop("condition", None)

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        data, y_all, test_size=0.1, random_state=42, stratify=y_all
    )

    def balance_data(X_raw, y_raw):
        df = pd.DataFrame(X_raw)
        df["condition"] = y_raw
        min_class_size = df["condition"].value_counts().min()
        df_bal = pd.concat([
            df[df["condition"] == "new"].sample(min_class_size, random_state=42),
            df[df["condition"] == "used"].sample(min_class_size, random_state=42)
        ])
        df_bal = df_bal.sample(frac=1, random_state=42).reset_index(drop=True)
        y = df_bal["condition"].tolist()
        X = df_bal.drop(columns=["condition"]).to_dict(orient="records")
        return X, y

    X_train_bal, y_train_bal = balance_data(X_train_raw, y_train_raw)
    X_test_bal, y_test_bal = balance_data(X_test_raw, y_test_raw)

    return X_train_bal, y_train_bal, X_test_bal, y_test_bal, X_test_raw, y_test_raw

# === Main ===
if __name__ == "__main__":
    print("Loading dataset...")
    X_train_raw, y_train, X_test_raw, y_test, X_test_dicts, y_test_dicts = build_dataset()

    print_class_distribution(y_train, "balanced training")
    print_class_distribution(y_test, "balanced testing")

    print("\nData flattening...")

    # Flatten and clean columns
    X_train = drop_ignored_columns([flatten_dict(x) for x in X_train_raw])
    # ---- Variables remaining after discarding ignored variables ----
    remaining_vars = sorted(set().union(*X_train))
    print("\nVariables that were NOT ruled out:")
    for col in remaining_vars:
        print(" -", col)

    X_test  = drop_ignored_columns([flatten_dict(x) for x in X_test_raw])

    print("\nFirst 10 records of the training set (flattened and cleaned):")
    df_preview = pd.DataFrame(X_train[:100])
    print(df_preview.head(10))

    print("\nPreparing pipeline...")
    pipeline = Pipeline([
        ("vectorizer", DictVectorizer()),
        ("imputer", SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)),
        ("clf", GradientBoostingClassifier(random_state=42))
    ])

    param_distributions = {
        "clf__n_estimators": randint(50, 150),
        "clf__max_depth": randint(3, 10),
        "clf__min_samples_split": randint(2, 10),
        "clf__min_samples_leaf": randint(1, 10),
        "clf__learning_rate": uniform(0.01, 0.2),
        "clf__subsample": uniform(0.7, 0.3)
    }

    search = TQDMRandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=30,
        cv=2,
        scoring="accuracy",
        n_jobs=-1,
        random_state=42
    )

    print("Running random search with progress...")
    search.fit(X_train, y_train)
    
    print("\nBest hyperparameters found:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    print("\nMetrics in TRAINING set:")
    best_model = search.best_estimator_
    y_train_pred = best_model.predict(X_train)
    print("  Accuracy:", round(accuracy_score(y_train, y_train_pred), 4))
    print("  F1 score:", round(f1_score(y_train, y_train_pred, pos_label="new", average="binary"), 4))
    print("  Recall:", round(recall_score(y_train, y_train_pred, pos_label="new", average="binary"), 4))
    print("\nFull Report TRAINING:")
    print(classification_report(y_train, y_train_pred))

    print("\nEvaluating the best model in test...")
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label="new", average="binary")
    recall = recall_score(y_test, y_pred, pos_label="new", average="binary")

    print("\nTest suite metrics:")
    print("  Accuracy:", round(acc, 4))
    print("  F1 score:", round(f1, 4))
    print("  Recall:", round(recall, 4))
    print("\nComplete Test Report:")
    print(classification_report(y_test, y_pred))

    print("\nSaving model (full pipeline)...")
    os.makedirs("Models_Dir", exist_ok=True)
    dump(best_model, "Models_Dir/gb_pipeline.pkl")

    print("\nLoading saved model for prediction...")
    loaded_pipeline = load("Models_Dir/gb_pipeline.pkl")

    print("\nPredictions on 100 records from the test set:")
    sample_dicts = X_test_dicts[:100]
    sample_y = y_test_dicts[:100]
    sample_flat = drop_ignored_columns([flatten_dict(x) for x in sample_dicts])
    preds = loaded_pipeline.predict(sample_flat)

    df_results = pd.DataFrame({
        "Real class": sample_y,
        "Predicted class": preds
    })
    print(df_results)