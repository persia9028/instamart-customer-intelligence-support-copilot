"""Re-run the Part A acceptance checkpoint assertions."""
import pandas as pd

from part_a import CLEAN, DB, audit, validate

raw_audit = audit()
assert raw_audit["blank_delivery_minutes"] == 2541
assert raw_audit["blank_age"] == 61
assert raw_audit["blank_gender"] == 30

customers = pd.read_csv(CLEAN / "customers.csv", parse_dates=["signup_date"])
products = pd.read_csv(CLEAN / "products.csv")
orders = pd.read_csv(CLEAN / "orders.csv", parse_dates=["order_ts"])
validate(customers, products, orders)
