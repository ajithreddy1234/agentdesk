"""Deterministic synthetic data: customers, products and orders relative to "now".

    python -m app.seed             # reset: drop and recreate all demo data
    python -m app.seed --if-empty  # startup-safe: seed only a brand-new database
"""
from __future__ import annotations

import random
from datetime import timedelta

from .config import settings
from .db import Base, Customer, Order, OrderItem, Product, Refund, get_engine, session_factory

PRODUCTS = [
    ("Toned Milk 1L", "Dairy", 58, True), ("Full Cream Milk 1L", "Dairy", 72, True), ("Curd 500g", "Dairy", 45, True),
    ("Paneer 200g", "Dairy", 95, True), ("Butter 100g", "Dairy", 60, True), ("Cow Ghee 500ml", "Dairy", 380, False),
    ("Greek Yogurt 400g", "Dairy", 150, True), ("Bananas 1kg", "Fruits", 60, True), ("Apples 1kg", "Fruits", 210, True),
    ("Alphonso Mangoes 1kg", "Fruits", 520, True), ("Pomegranate 500g", "Fruits", 140, True),
    ("Tomatoes 1kg", "Vegetables", 40, True), ("Onions 1kg", "Vegetables", 38, True), ("Potatoes 1kg", "Vegetables", 35, True),
    ("Spinach 250g", "Vegetables", 30, True), ("Brown Bread", "Bakery", 50, True), ("Multigrain Bread", "Bakery", 65, True),
    ("Basmati Rice 5kg", "Staples", 640, False), ("Whole Wheat Atta 5kg", "Staples", 295, False),
    ("Toor Dal 1kg", "Staples", 165, False), ("Sunflower Oil 1L", "Staples", 155, False), ("Sugar 1kg", "Staples", 48, False),
    ("Orange Juice 1L", "Beverages", 125, False), ("Green Tea 25pc", "Beverages", 180, False),
    ("Almonds 500g", "Dry Fruits", 520, False), ("Cashews 500g", "Dry Fruits", 610, False),
    ("Olive Oil 1L", "Staples", 899, False), ("Organic Honey 500g", "Staples", 345, False),
]
FIRST = ["Aarav", "Diya", "Ishaan", "Ananya", "Kabir", "Meera", "Rohan", "Sneha", "Vikram", "Priya", "Arjun", "Kavya",
         "Nikhil", "Pooja", "Rahul", "Divya", "Siddharth", "Lakshmi", "Varun", "Neha"]
LAST = ["Reddy", "Sharma", "Iyer", "Patel", "Gupta", "Nair", "Rao", "Singh", "Menon", "Das"]
CITIES = ["Hyderabad", "Bengaluru", "Mumbai", "Delhi NCR", "Chennai", "Pune"]


def seed(url: str | None = None, n_customers: int = 60, seed_value: int = 7) -> None:
    rng = random.Random(seed_value)
    now = settings.now()
    from .db import init_db
    from .sql_lookup import drop_views

    eng = get_engine(url)
    drop_views(eng)  # the agent's read views depend on the tables
    Base.metadata.drop_all(eng)
    init_db(url)
    S = session_factory(url)
    with S.begin() as s:
        prods = [Product(id=i, name=n, category=c, price=p, perishable=per) for i, (n, c, p, per) in enumerate(PRODUCTS, 1)]
        s.add_all(prods)
        for cid in range(1, n_customers + 1):
            s.add(Customer(id=cid, name=f"{rng.choice(FIRST)} {rng.choice(LAST)}", phone=f"9{rng.randint(100000000, 999999999)}",
                           city=rng.choice(CITIES), plus_member=rng.random() < 0.3, wallet_balance=round(rng.uniform(0, 800), 2)))
        s.flush()
        oid, iid = 1000, 1
        for cid in range(1, n_customers + 1):
            # each customer gets a spread of orders: just delivered, a few days old, old, in flight, cancelled
            plan = [("delivered", timedelta(hours=rng.randint(2, 20))),          # perishable window open
                    ("delivered", timedelta(days=rng.randint(2, 5))),            # perishable window closed, packaged open
                    ("delivered", timedelta(days=rng.randint(12, 40))),          # everything closed
                    (rng.choice(["placed", "out_for_delivery"]), timedelta(hours=rng.randint(1, 8))),
                    ("cancelled", timedelta(days=rng.randint(3, 20)))]
            if rng.random() < 0.5:
                plan.append(("delivered", timedelta(hours=rng.randint(1, 23))))  # a second fresh order, often large
            for status, age in plan:
                oid += 1
                placed = now - age - timedelta(hours=10)
                items = rng.sample(prods, rng.randint(2, 5))
                lines = []
                for p in items:
                    qty = rng.randint(1, 4)
                    lines.append(OrderItem(id=iid, order_id=oid, product_id=p.id, qty=qty, unit_price=p.price))
                    iid += 1
                total = round(sum(l.qty * l.unit_price for l in lines), 2)
                s.add(Order(id=oid, customer_id=cid, status=status, slot=rng.choice(["morning", "express", "evening"]),
                            payment_method=rng.choice(["upi", "card", "wallet", "cod"]) if total <= 2000 else rng.choice(["upi", "card"]),
                            total=total, placed_at=placed, delivered_at=(now - age) if status == "delivered" else None))
                s.flush()
                s.add_all(lines)
        # a few historical refunds so "complaints in 30 days" logic has data
        for cid in rng.sample(range(1, n_customers + 1), 4):
            old = s.query(Order).filter_by(customer_id=cid, status="delivered").order_by(Order.placed_at).first()
            item = s.query(OrderItem).filter_by(order_id=old.id).first()
            for _ in range(2):
                s.add(Refund(order_id=old.id, product_id=item.product_id, qty=0, amount=0, method="wallet",
                             reason="damaged", status="approved", approved_by="auto", created_at=now - timedelta(days=rng.randint(1, 25))))
    print(f"seeded {n_customers} customers, {len(PRODUCTS)} products, {oid - 1000} orders")


def seed_if_empty(url: str | None = None) -> None:
    """Startup-safe: only seeds a brand-new database; never drops existing data."""
    from sqlalchemy import inspect

    eng = get_engine(url)
    if "customers" in inspect(eng).get_table_names():
        with session_factory(url)() as s:
            if s.query(Customer).first():
                print("database already has data; skipping seed")
                from .db import init_db

                init_db(url)  # add tables/columns introduced by upgrades
                return
    seed(url)


if __name__ == "__main__":
    import sys

    if "--if-empty" in sys.argv:
        seed_if_empty()
    else:
        seed()  # explicit reset: drops and recreates the demo data
