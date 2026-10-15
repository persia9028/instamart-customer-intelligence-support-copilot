"""Part A: audit, clean, store and analyse the synthetic Instamart exports."""
from pathlib import Path
import json
import sqlite3

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
DB = ROOT / "data" / "instamart.db"


def audit():
    customers = pd.read_csv(RAW / "customers.csv")
    products = pd.read_csv(RAW / "products.csv")
    orders = pd.read_csv(RAW / "orders.csv")
    for name, frame in (("customers", customers), ("products", products), ("orders", orders)):
        print(f"\n{name}: shape={frame.shape}")
        print(frame.dtypes.to_string())
        print(pd.DataFrame({"missing_count": frame.isna().sum(), "missing_pct": frame.isna().mean().mul(100).round(2)}).to_string())
    figures = {
        "raw_order_rows": len(orders), "exact_duplicate_rows": int(orders.duplicated().sum()),
        "pending_rows": int(orders.status.eq("pending").sum()), "distinct_order_ids": int(orders.order_id.nunique()),
        "non_numeric_amounts": int(pd.to_numeric(orders.order_amount, errors="coerce").isna().sum()),
        "negative_delivery_minutes": int(pd.to_numeric(orders.delivery_minutes, errors="coerce").lt(0).sum()),
        "blank_delivery_minutes": int(orders.delivery_minutes.eq("").sum()),
        "blank_age": int(customers.age.eq("").sum()),
        "ages_outside_10_100": int((pd.to_numeric(customers.age, errors="coerce").notna() & ~pd.to_numeric(customers.age, errors="coerce").between(10, 100)).sum()),
        "raw_city_spellings": int(customers.city.nunique()),
        "raw_gender_spellings_nonblank": int(customers.loc[customers.gender.ne(""), "gender"].nunique()),
        "blank_gender": int(customers.gender.eq("").sum()),
    }
    print("\nAUDIT CHECKPOINTS")
    for key, value in figures.items():
        print(f"{key}: {value}")
    return figures


def clean():
    customers = pd.read_csv(RAW / "customers.csv", keep_default_na=False)
    products = pd.read_csv(RAW / "products.csv", keep_default_na=False)
    orders = pd.read_csv(RAW / "orders.csv", keep_default_na=False)

    city_changed = int(customers.city.ne(customers.city.astype(str).str.strip().str.title()).sum())
    gender_mapped = customers.gender.astype(str).str.strip().str.lower().map(
        {"male": "male", "m": "male", "female": "female", "f": "female"}
    ).fillna("unknown")
    gender_changed = int(customers.gender.ne(gender_mapped).sum())
    customers["city"] = customers.city.astype(str).str.strip().str.title()
    customers["gender"] = gender_mapped
    customers["age"] = pd.to_numeric(customers.age, errors="coerce")
    ages_invalid = int((customers.age.notna() & ~customers.age.between(10, 100)).sum())
    ages_blank = int(customers.age.isna().sum())
    customers.loc[~customers.age.between(10, 100), "age"] = np.nan
    customers["age"] = customers.age.fillna(customers.groupby("city").age.transform("median"))
    customers["signup_date"] = pd.to_datetime(customers.signup_date)

    products["price_inr"] = pd.to_numeric(products.price_inr)
    orders["order_ts"] = pd.to_datetime(orders.order_ts, format="%Y-%m-%d %H:%M:%S")
    raw_order_rows = len(orders)
    orders = orders.drop_duplicates()
    exact_duplicates_removed = raw_order_rows - len(orders)
    after_exact_duplicates = len(orders)
    orders = orders.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")
    nonfinal_rows_removed = after_exact_duplicates - len(orders)
    orders["quantity"] = pd.to_numeric(orders.quantity)
    orders["discount_pct"] = pd.to_numeric(orders.discount_pct)
    orders["order_amount"] = pd.to_numeric(orders.order_amount, errors="coerce")
    amount_imputed = int(orders.order_amount.isna().sum())
    orders["delivery_minutes"] = pd.to_numeric(orders.delivery_minutes, errors="coerce")
    negative_delivery = int(orders.delivery_minutes.lt(0).sum())
    orders = orders.merge(products[["product_id", "price_inr", "category"]], on="product_id", how="left", validate="many_to_one")
    missing_amount = orders.order_amount.isna()
    orders.loc[missing_amount, "order_amount"] = (
        orders.loc[missing_amount, "price_inr"] * orders.loc[missing_amount, "quantity"] *
        (1 - orders.loc[missing_amount, "discount_pct"] / 100)
    ).round(2)
    orders.loc[orders.delivery_minutes.lt(0), "delivery_minutes"] = np.nan
    completed_missing = orders.status.eq("completed") & orders.delivery_minutes.isna()
    completed_delivery_imputed = int(completed_missing.sum())
    category_median = orders[orders.status.eq("completed")].groupby("category").delivery_minutes.transform("median")
    orders.loc[completed_missing, "delivery_minutes"] = category_median.reindex(orders.index)[completed_missing]
    orders = orders.drop(columns=["price_inr", "category"])
    orders["order_ts"] = orders.order_ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    print("CLEANING CHANGES:", json.dumps({"city_values_standardized": city_changed,
          "gender_values_normalized": gender_changed, "blank_ages": ages_blank, "impossible_ages": ages_invalid,
          "exact_order_duplicates_removed": exact_duplicates_removed, "older_or_pending_order_rows_removed": nonfinal_rows_removed,
          "order_amounts_imputed_after_deduplication": amount_imputed,
          "negative_delivery_values_set_missing": negative_delivery,
          "completed_delivery_values_imputed": completed_delivery_imputed,
          "cancelled_delivery_values_left_missing": int(orders.delivery_minutes.isna().sum())}))

    CLEAN.mkdir(parents=True, exist_ok=True)
    for name, frame in (("customers", customers), ("products", products), ("orders", orders)):
        frame.to_csv(CLEAN / f"{name}.csv", index=False)
    with sqlite3.connect(DB) as con:
        for name, frame in (("customers", customers), ("products", products), ("orders", orders)):
            frame.to_sql(name, con, if_exists="replace", index=False)
    return customers, products, orders


def validate(customers, products, orders):
    completed = orders[orders.status.eq("completed")]
    assert len(customers) == 1200 and customers.city.nunique() == 6
    assert customers.gender.value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}
    assert customers.age.notna().all() and customers.age.between(18, 60).all()
    assert len(products) == 40 and products.product_id.nunique() == 40
    assert len(orders) == 12309 and orders.order_id.is_unique
    assert orders.status.value_counts().to_dict() == {"completed": 11110, "cancelled": 1199}
    assert orders.order_amount.notna().all() and completed.delivery_minutes.notna().all()
    assert orders.loc[orders.status.eq("cancelled"), "delivery_minutes"].isna().all()
    assert round(completed.order_amount.sum(), 2) == 2998668.80
    with sqlite3.connect(DB) as con:
        assert set(pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", con).name) == {"customers", "products", "orders"}
    print("ALL CHECKPOINTS PASSED")


def sql_analysis():
    statements = [s.strip() for s in (ROOT / "sql" / "queries.sql").read_text(encoding="utf-8").split("-- Q") if s.strip()]
    with sqlite3.connect(DB) as con:
        for i, statement in enumerate(statements, 1):
            query = statement[statement.find("\n") + 1:].strip().rstrip(";")
            result = pd.read_sql_query(query, con)
            print(f"\nQ{i}: rows={len(result)}")
            print(result.head(8).to_string(index=False))
            if i == 2:
                print("Q2 revenue total:", round(result.revenue_inr.sum(), 2))
            if i == 5:
                average = pd.read_sql_query(
                    "SELECT AVG(total_spend) AS avg_spend FROM "
                    "(SELECT customer_id, SUM(order_amount) AS total_spend FROM orders "
                    "WHERE status='completed' GROUP BY customer_id)", con
                ).avg_spend.iloc[0]
                print("Q5 average customer spend:", round(average, 2))
            if i == 7:
                print("Q7 last cumulative revenue:", round(result.cumulative_revenue.iloc[-1], 2))


def exploratory_analysis():
    customers = pd.read_csv(CLEAN / "customers.csv", parse_dates=["signup_date"])
    products = pd.read_csv(CLEAN / "products.csv")
    orders = pd.read_csv(CLEAN / "orders.csv", parse_dates=["order_ts"])
    completed = orders.loc[orders.status.eq("completed")].merge(
        customers[["customer_id", "city"]], on="customer_id", validate="many_to_one"
    ).merge(products[["product_id", "category"]], on="product_id", validate="many_to_one")
    pivot = pd.pivot_table(completed, index="city", columns="category", values="order_amount", aggfunc="sum", margins=True, fill_value=0)
    city_stats = completed.groupby("city").order_amount.agg(total="sum", mean="mean", count="count").round(2)
    amounts = completed.order_amount
    q1, median, q3 = amounts.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    correlations = completed[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
    pairs = [(abs(correlations.loc[a, b]), a, b, correlations.loc[a, b])
             for i, a in enumerate(correlations.columns) for b in correlations.columns[i + 1:]]
    pair = max(pairs)
    summary = {"pivot_grand_total": float(pivot.loc["All", "All"]),
               "city_stats": city_stats.reset_index().to_dict(orient="records"),
               "order_amount": {"mean": float(amounts.mean()), "median": float(median), "skew": float(amounts.skew()),
                                "q1": float(q1), "q3": float(q3), "lower_fence": float(lower), "upper_fence": float(upper),
                                "outliers_below": int((amounts < lower).sum()), "outliers_above": int((amounts > upper).sum())},
               "strongest_correlation": {"feature_a": pair[1], "feature_b": pair[2], "value": float(pair[3])},
               "customer_city_counts": customers.city.value_counts().to_dict(),
               "customer_gender_counts": customers.gender.value_counts().to_dict()}
    print("PANDAS_PIVOT_GRAND_TOTAL:", summary["pivot_grand_total"])
    print("CITY_STATS:\n" + city_stats.to_string())
    print("AMOUNT_DISTRIBUTION:", json.dumps(summary["order_amount"]))
    print("CORRELATIONS:\n" + correlations.round(4).to_string())
    print("STRONGEST_CORRELATION:", json.dumps(summary["strongest_correlation"]))
    print("CLEAN_CUSTOMER_COUNTS:", json.dumps({"city": summary["customer_city_counts"], "gender": summary["customer_gender_counts"]}))
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "part_a_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    audit()
    validate(*clean())
    sql_analysis()
    exploratory_analysis()
