import os
import pandas as pd


# ============================================================
# 11 FUTURE FORECAST PIPELINE
# ============================================================
#
# Goal:
# Create customer-style future demand forecasts for forecastable parts.
#
# Important:
# Our experiments showed that for all 651 forecastable parts, the strongest
# overall method by MAE was the simple lag_1 baseline.
#
# lag_1 means:
#     next month prediction = previous month actual demand
#
# This script creates future forecasts from the latest real month in the data.
# Example:
#     if latest real month is 2026-01,
#     it creates predictions for 2026-02, 2026-03, ..., depending on horizon.
#
# Streamlit can read the Excel output from this script.


# -----------------------------
# 1. SETTINGS
# -----------------------------

file_path = "D:/Diploma_work/diploma/DB_31_24&25.xlsb"

output_dir = "outputs/xlsx"
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "11_future_forecasts.xlsx")

# Change this number if customer wants more or fewer future months.
# 6 means forecast next 6 months after the latest real month.
forecast_horizon_months = 6


# -----------------------------
# 2. LOAD RAW DATA
# -----------------------------

df = pd.read_excel(file_path, engine="pyxlsb")


# -----------------------------
# 3. CONVERT EXCEL DATE
# -----------------------------
#
# In the raw Excel file, Date is stored as a number like 45350.
# That number means "number of days since Excel's start date".
#
# pd.to_datetime converts it into a normal date.

df["Date Converted"] = pd.to_datetime(
    df["Date"],
    origin="1899-12-30",
    unit="D",
    errors="coerce",
)

df["Month"] = df["Date Converted"].dt.to_period("M").astype(str)


# -----------------------------
# 4. CREATE SOLD / RETURN QTY
# -----------------------------
#
# Qty can be positive or negative.
#
# Positive Qty = sold quantity
# Negative Qty = returned quantity / credit note
#
# net_qty is still the original Qty sum:
#     sold quantity - returned quantity

df["sold_qty"] = df["Qty"].clip(lower=0)
df["return_qty"] = df["Qty"].clip(upper=0).abs()


# -----------------------------
# 5. CREATE MONTHLY DEMAND TABLE
# -----------------------------
#
# We forecast monthly demand, not transaction-by-transaction demand.
# So we group rows by Month and Part Number.

monthly_part_demand = (
    df.groupby(["Month", "Part Number"], as_index=False)
    .agg(
        net_qty=("Qty", "sum"),
        sold_qty=("sold_qty", "sum"),
        return_qty=("return_qty", "sum"),
        transaction_count=("Qty", "count"),
        sale_value=("Sale val", "sum"),
        profit=("Profit", "sum"),
        description=("Description", "first"),
        fr=("FR", "first"),
    )
    .sort_values(["Part Number", "Month"])
)


# -----------------------------
# 6. SELECT FORECASTABLE PARTS
# -----------------------------
#
# Same rule we used earlier:
# - part appeared in at least 18 months
# - part had at least 50 transaction rows
# - part sold more than 0 units

part_summary = (
    monthly_part_demand.groupby("Part Number", as_index=False)
    .agg(
        active_months=("Month", "nunique"),
        total_net_qty=("net_qty", "sum"),
        total_sold_qty=("sold_qty", "sum"),
        total_return_qty=("return_qty", "sum"),
        total_transactions=("transaction_count", "sum"),
        avg_monthly_net_qty=("net_qty", "mean"),
        description=("description", "first"),
        fr=("fr", "first"),
    )
)

forecastable_parts = part_summary[
    (part_summary["active_months"] >= 18)
    & (part_summary["total_transactions"] >= 50)
    & (part_summary["total_sold_qty"] > 0)
].copy()


# -----------------------------
# 7. COMPLETE MONTHLY HISTORY
# -----------------------------
#
# Some parts may have no sales in some months.
# For forecasting, missing month should mean 0 demand, not "unknown".
#
# So we create every combination:
#     every forecastable part x every real month

all_months = sorted(monthly_part_demand["Month"].dropna().unique())
all_parts = sorted(forecastable_parts["Part Number"].unique())

full_index = pd.MultiIndex.from_product(
    [all_parts, all_months],
    names=["Part Number", "Month"],
)

complete_history = (
    monthly_part_demand.set_index(["Part Number", "Month"])
    .reindex(full_index)
    .reset_index()
)

numeric_cols = [
    "net_qty",
    "sold_qty",
    "return_qty",
    "transaction_count",
    "sale_value",
    "profit",
]

complete_history[numeric_cols] = complete_history[numeric_cols].fillna(0)

# Add stable part information back after reindexing.
part_info = forecastable_parts[["Part Number", "description", "fr"]].drop_duplicates()

complete_history = complete_history.drop(columns=["description", "fr"], errors="ignore")
complete_history = complete_history.merge(part_info, on="Part Number", how="left")


# -----------------------------
# 8. FUTURE MONTH LIST
# -----------------------------

latest_real_month = max(all_months)

future_months = pd.period_range(
    start=pd.Period(latest_real_month, freq="M") + 1,
    periods=forecast_horizon_months,
    freq="M",
).astype(str)


# -----------------------------
# 9. RECURSIVE FUTURE FORECAST
# -----------------------------
#
# Recursive means:
# - First future month uses real latest demand.
# - Second future month uses the previous predicted demand.
# - Third future month uses earlier predictions again.
#
# This lets us forecast multiple months ahead.
#
# But warning:
# The farther into the future we go, the less reliable the forecast becomes.

future_rows = []

for part_number in all_parts:
    part_history = complete_history[
        complete_history["Part Number"] == part_number
    ].sort_values("Month")

    description = part_history["description"].iloc[0]
    fr = part_history["fr"].iloc[0]

    # This list starts with real historical monthly demand.
    # Then we append predicted future demand month by month.
    demand_history = part_history["net_qty"].tolist()

    for step_number, forecast_month in enumerate(future_months, start=1):
        lag_1 = demand_history[-1] if len(demand_history) >= 1 else 0
        lag_2 = demand_history[-2] if len(demand_history) >= 2 else 0
        lag_3 = demand_history[-3] if len(demand_history) >= 3 else 0

        last_3_values = demand_history[-3:]
        rolling_mean_3 = sum(last_3_values) / len(last_3_values)

        # Best validated method from our all-651-parts test:
        # prediction = lag_1
        recommended_prediction = lag_1

        # Alternative predictions are saved too.
        # Streamlit can show them for comparison.
        pred_lag_1 = lag_1
        pred_rolling_mean_3 = rolling_mean_3
        pred_part_avg = part_history["net_qty"].mean()

        future_rows.append(
            {
                "Part Number": part_number,
                "description": description,
                "fr": fr,
                "forecast_month": forecast_month,
                "forecast_step": step_number,
                "pred_lag_1": round(pred_lag_1, 2),
                "pred_rolling_mean_3": round(pred_rolling_mean_3, 2),
                "pred_part_avg": round(pred_part_avg, 2),
                "recommended_model": "pred_lag_1",
                "recommended_prediction": round(recommended_prediction, 2),
                "latest_real_month": latest_real_month,
                "lag_1_used": round(lag_1, 2),
                "lag_2_used": round(lag_2, 2),
                "lag_3_used": round(lag_3, 2),
            }
        )

        # Add prediction to history so next future month can use it.
        demand_history.append(recommended_prediction)


future_forecasts = pd.DataFrame(future_rows)


# -----------------------------
# 10. CREATE CUSTOMER SEARCH COLUMNS
# -----------------------------
#
# Streamlit can use search_text to search by part number or description.

future_forecasts["search_text"] = (
    future_forecasts["Part Number"].astype(str)
    + " "
    + future_forecasts["description"].astype(str)
).str.lower()


# -----------------------------
# 11. SAVE OUTPUT FOR STREAMLIT
# -----------------------------

metadata = pd.DataFrame(
    {
        "metric": [
            "latest_real_month",
            "forecast_horizon_months",
            "forecastable_parts",
            "future_rows",
            "recommended_model",
            "important_warning",
        ],
        "value": [
            latest_real_month,
            forecast_horizon_months,
            len(all_parts),
            len(future_forecasts),
            "pred_lag_1",
            "Forecasts after the first future month are recursive, so uncertainty increases.",
        ],
    }
)

latest_history = complete_history[
    complete_history["Month"] == latest_real_month
].sort_values("net_qty", ascending=False)

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    future_forecasts.to_excel(writer, sheet_name="future_forecasts", index=False)
    latest_history.to_excel(writer, sheet_name="latest_real_month", index=False)
    forecastable_parts.to_excel(writer, sheet_name="forecastable_parts", index=False)
    metadata.to_excel(writer, sheet_name="metadata", index=False)


# -----------------------------
# 12. TERMINAL SUMMARY
# -----------------------------

print(f"Created: {output_file}")
print("Latest real month:", latest_real_month)
print("Forecast months:", ", ".join(future_months))
print("Forecastable parts:", len(all_parts))
print("Future forecast rows:", len(future_forecasts))
print("Recommended model: pred_lag_1")
print("Warning: multi-step forecasts are recursive, so accuracy decreases farther into the future.")
