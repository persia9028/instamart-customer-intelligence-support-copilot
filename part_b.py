"""Part B: deterministic RFM segmentation and pre-cutoff lapse modelling."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "clean"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
NUMERIC = ["recency", "frequency", "monetary", "avg_discount", "cancellation_rate", "avg_delivery", "age", "tenure_days"]
CATEGORICAL = ["city", "gender", "favourite_category"]


def segments(orders, customers):
    completed = orders[orders.status.eq("completed")].copy()
    completed["order_ts"] = pd.to_datetime(completed.order_ts)
    rfm = completed.groupby("customer_id").agg(
        last_order=("order_ts", "max"), frequency=("order_id", "nunique"), monetary=("order_amount", "sum")
    )
    rfm["recency"] = (SNAPSHOT - rfm.pop("last_order")).dt.days
    rfm = rfm[["recency", "frequency", "monetary"]]
    print("RFM summary:", json.dumps({"customers": len(rfm), "recency_median": float(rfm.recency.median()), "recency_min": int(rfm.recency.min()), "recency_max": int(rfm.recency.max()), "frequency_mean": float(rfm.frequency.mean()), "frequency_max": int(rfm.frequency.max()), "monetary_mean": float(rfm.monetary.mean()), "monetary_max": float(rfm.monetary.max())}))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm)
    scores = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(scaled)
        scores.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(silhouette_score(scaled, km.labels_))})
    print("KMEANS:", json.dumps(scores))
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42).fit(scaled)
    rfm["segment_id"] = kmeans.labels_
    profile = rfm.groupby("segment_id").agg(size=("frequency", "size"), recency=("recency", "mean"), frequency=("frequency", "mean"), monetary=("monetary", "mean")).round(2)
    premium_id = int(profile.monetary.idxmax())
    # Business names follow the measured profile; IDs remain arbitrary cluster labels.
    names = {premium_id: "Premium Basket"}
    remaining = [int(cluster) for cluster in profile.sort_values(["frequency", "recency"], ascending=[False, True]).index if cluster != premium_id]
    other_names = ["Frequent Essentials", "Steady Shoppers", "At-Risk Light Buyers"]
    for cluster, name in zip(remaining, other_names):
        names[cluster] = name
    rfm["segment_name"] = rfm.segment_id.map(names)
    rfm["is_premium"] = rfm.segment_id.eq(premium_id)
    print("SEGMENT_PROFILE:", profile.to_json(orient="index"))
    print("SEGMENT_NAMES:", json.dumps(names), "premium_id:", premium_id)
    pca = PCA(n_components=2).fit(scaled)
    print("PCA explained variance:", pca.explained_variance_ratio_.tolist())
    exported = rfm.reset_index()[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]]
    exported.to_csv(OUT / "customer_segments.csv", index=False)
    return exported, rfm


def make_cohort(orders, customers):
    orders = orders.copy()
    orders["order_ts"] = pd.to_datetime(orders.order_ts)
    products = pd.read_csv(DATA / "products.csv", usecols=["product_id", "category"])
    orders = orders.merge(products, on="product_id", how="left", validate="many_to_one")
    pre = orders[orders.order_ts.lt(CUTOFF)].copy()
    pre_complete = pre[pre.status.eq("completed")]
    cohort_ids = pre_complete.customer_id.unique()
    c = customers[customers.customer_id.isin(cohort_ids)].set_index("customer_id")
    feats = pre_complete.groupby("customer_id").agg(
        last_order=("order_ts", "max"), frequency=("order_id", "nunique"), monetary=("order_amount", "sum"),
        avg_discount=("discount_pct", "mean"), avg_delivery=("delivery_minutes", "mean"),
    )
    feats["recency"] = (CUTOFF - feats.pop("last_order")).dt.days
    totals = pre.groupby("customer_id").status.agg(total_final="size", cancelled=lambda s: int(s.eq("cancelled").sum()))
    feats["cancellation_rate"] = totals.cancelled.div(totals.total_final).fillna(0)
    fav = pre_complete.groupby("customer_id").category.agg(lambda s: s.mode().sort_values().iloc[0])
    feats["favourite_category"] = fav
    feats["city"] = c.city
    feats["age"] = c.age
    feats["gender"] = c.gender
    feats["tenure_days"] = (CUTOFF - pd.to_datetime(c.signup_date)).dt.days
    sept = orders.order_ts.ge(CUTOFF) & orders.order_ts.lt(pd.Timestamp("2025-10-01")) & orders.status.eq("completed")
    active_sept = set(orders.loc[sept, "customer_id"])
    y = pd.Series([0 if cid in active_sept else 1 for cid in feats.index], index=feats.index, name="lapsed")
    X = feats[NUMERIC + CATEGORICAL].copy()
    assert not X.isna().any().any(), "Feature matrix contains missing values"
    print("COHORT:", json.dumps({"rows": len(X), "lapsed": int(y.sum()), "lapsed_pct": float(y.mean() * 100)}))
    return X, y, feats


def metric_dict(y, pred):
    return {"confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            "precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
            "f1": float(f1_score(y, pred, pos_label=1, zero_division=0))}


def main():
    orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_ts"])
    customers = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
    segment_file, _ = segments(orders, customers)
    X, y, raw_features = make_cohort(orders, customers)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print("SPLIT:", len(X_train), len(X_test))
    baseline = np.zeros(len(y_test), dtype=int)
    print("BASELINE:", json.dumps({"accuracy": float(accuracy_score(y_test, baseline)), **metric_dict(y_test, baseline)}))
    prep = ColumnTransformer([("num", StandardScaler(), NUMERIC), ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL)], sparse_threshold=0)
    logit = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=1000))])
    grid_log = GridSearchCV(logit, {"model__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1", n_jobs=-1)
    grid_log.fit(X_train, y_train)
    rf = Pipeline([("prep", prep), ("model", RandomForestClassifier(n_estimators=100, random_state=42))])
    grid_rf = GridSearchCV(rf, {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1", n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    pred_log = grid_log.predict(X_test)
    pred_rf = grid_rf.predict(X_test)
    reports = {"logistic": {"best_params": grid_log.best_params_, "cv_f1": float(grid_log.best_score_), **metric_dict(y_test, pred_log)},
               "random_forest": {"best_params": grid_rf.best_params_, "cv_f1": float(grid_rf.best_score_), **metric_dict(y_test, pred_rf)}}
    print("MODEL_REPORT:", json.dumps(reports))
    best_log = grid_log.best_estimator_
    transformed_train = best_log.named_steps["prep"].transform(X_train)
    freq_ix = NUMERIC.index("frequency")
    # StandardScaler is the first transformer and keeps numeric columns in declared order.
    print("FREQUENCY_SCALER_TRAIN_MEAN:", float(best_log.named_steps["prep"].named_transformers_["num"].mean_[freq_ix]), "FULL_COHORT_MEAN:", float(X.frequency.mean()))
    counts = y_train.value_counts()
    rng = np.random.RandomState(42)
    minority_ids = np.flatnonzero(y_train.to_numpy() == 1)
    extra = rng.choice(minority_ids, size=int(counts[0] - counts[1]), replace=True)
    train_balanced = np.concatenate([transformed_train, transformed_train[extra]], axis=0)
    y_balanced = np.concatenate([y_train.to_numpy(), y_train.to_numpy()[extra]])
    balanced_model = LogisticRegression(max_iter=1000, C=best_log.named_steps["model"].C).fit(train_balanced, y_balanced)
    balanced_pred = balanced_model.predict(best_log.named_steps["prep"].transform(X_test))
    reports["logistic_balanced"] = metric_dict(y_test, balanced_pred)
    print("BALANCED_LOGISTIC:", json.dumps(reports["logistic_balanced"]))
    rf_model = grid_rf.best_estimator_.named_steps["model"]
    cat_names = best_log.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(CATEGORICAL).tolist()
    feature_names = NUMERIC + cat_names
    importances = sorted(zip(feature_names, rf_model.feature_importances_), key=lambda x: x[1], reverse=True)[:5]
    print("RF_TOP_FEATURES:", json.dumps([(n, float(v)) for n, v in importances]))
    coef_names = best_log.named_steps["prep"].get_feature_names_out()
    coefs = best_log.named_steps["model"].coef_[0]
    chosen = []
    for wanted in ("frequency", "recency", "monetary"):
        ix = next((i for i, n in enumerate(coef_names) if n.endswith(wanted)), None)
        if ix is not None:
            chosen.append((wanted, float(coefs[ix])))
    print("LOGISTIC_COEFFICIENTS:", json.dumps(chosen))
    # LogisticRegression is selected because its scores are easy to explain and target; it must clear F1 >= .55.
    selected = best_log
    assert reports["logistic"]["f1"] >= 0.55 or reports["random_forest"]["f1"] >= 0.55
    if reports["logistic"]["f1"] < 0.55:
        selected = grid_rf.best_estimator_
    joblib.dump(selected, OUT / "lapse_model.joblib")
    reloaded = joblib.load(OUT / "lapse_model.joblib")
    predictions_match = bool(np.array_equal(selected.predict(X_test), reloaded.predict(X_test)))
    assert predictions_match
    test_ids = X_test.index
    premium_ids = set(segment_file.loc[segment_file.is_premium, "customer_id"])
    mask = [cid in premium_ids for cid in test_ids]
    probs = selected.predict_proba(X_test.loc[mask])[:, 1]
    retention = pd.DataFrame({"customer_id": np.asarray(test_ids)[mask], "lapse_probability": probs, "actually_lapsed": y_test.loc[np.asarray(test_ids)[mask]].to_numpy()})
    retention = retention.sort_values(["lapse_probability", "customer_id"], ascending=[False, True]).head(10)
    retention.to_csv(OUT / "retention_list.csv", index=False)
    print("PERSISTENCE:", json.dumps({"predictions_match": predictions_match, "retention_true_lapsed": int(retention.actually_lapsed.sum()), "retention_rows": len(retention)}))
    reports["retention_true_lapsed"] = int(retention.actually_lapsed.sum())
    reports["retention_rows"] = len(retention)
    (OUT / "part_b_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
