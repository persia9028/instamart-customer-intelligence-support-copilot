"""Re-run the Part A acceptance checkpoint assertions."""
import pandas as pd

from part_a import CLEAN, DB, validate

customers = pd.read_csv(CLEAN / "customers.csv", parse_dates=["signup_date"])
products = pd.read_csv(CLEAN / "products.csv")
orders = pd.read_csv(CLEAN / "orders.csv", parse_dates=["order_ts"])
validate(customers, products, orders)
