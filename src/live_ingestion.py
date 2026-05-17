import random
from datetime import datetime


PRODUCTS = [
    ("Dept_1", "A"),
    ("Dept_2", "A"),
    ("Dept_3", "B"),
    ("Dept_4", "B"),
    ("Dept_5", "C"),
    ("Dept_6", "A"),
]


def calculate_waste_quantity(current_stock, units_sold, expiry_days):
    overstock = max(0, current_stock - units_sold)

    if expiry_days <= 2:
        waste_quantity = int(overstock * 0.40)
    elif expiry_days <= 5:
        waste_quantity = int(overstock * 0.20)
    else:
        waste_quantity = int(overstock * 0.10)

    return waste_quantity


def generate_food_sales_record():
    product_name, category = random.choice(PRODUCTS)

    today = datetime.now()
    day_of_week = today.weekday()
    is_weekend = 1 if day_of_week in [5, 6] else 0
    is_holiday = random.choice([0, 0, 0, 1])
    promotion = random.choice([0, 0, 1])

    temperature = round(random.uniform(35, 85), 2)
    current_stock = random.randint(50, 250)
    unit_price = 10.0
    expiry_days = random.randint(1, 10)

    base_demand = random.randint(30, 120)

    if is_weekend:
        base_demand += random.randint(5, 30)

    if is_holiday:
        base_demand += random.randint(10, 40)

    if promotion:
        base_demand += random.randint(10, 50)

    units_sold = min(current_stock, max(0, base_demand))

    waste_quantity = calculate_waste_quantity(
        current_stock=current_stock,
        units_sold=units_sold,
        expiry_days=expiry_days
    )

    return {
        "product_name": product_name,
        "category": category,
        "store_id": random.randint(1, 45),
        "date": today.strftime("%Y-%m-%d"),
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "promotion": promotion,
        "temperature": temperature,
        "current_stock": current_stock,
        "units_sold": units_sold,
        "unit_price": unit_price,
        "expiry_days": expiry_days,
        "waste_quantity": waste_quantity,
        "source": "simulated_live_event",
    }


def generate_shifted_food_sales_record():
    record = generate_food_sales_record()

    record["temperature"] = round(random.uniform(95, 110), 2)
    record["current_stock"] = random.randint(250, 500)
    record["units_sold"] = random.randint(5, 30)

    record["waste_quantity"] = calculate_waste_quantity(
        current_stock=record["current_stock"],
        units_sold=record["units_sold"],
        expiry_days=record["expiry_days"]
    )

    record["source"] = "shifted_drift_event"

    return record


if __name__ == "__main__":
    normal_record = generate_food_sales_record()
    shifted_record = generate_shifted_food_sales_record()

    print("Normal live event:")
    print(normal_record)

    print("\nShifted drift event:")
    print(shifted_record)