import os
import pandas as pd


# ============================================================
# 1. FILE PATHS
# ============================================================

input_file = "results/xlsx/06b_feature_engineered_v4_no_lag24.csv"
output_folder = "results/xlsx"

os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(output_folder, "11_future_forecast_v2.xlsx")


# ============================================================
# 2. LOAD FEATURE-ENGINEERED DATA
# ============================================================

df = pd.read_csv(input_file)

df["Month Date"] = pd.to_datetime(df["Month Date"])
df["Month"] = df["Month Date"].dt.to_period("M").astype(str)

print("Loaded feature-engineered data.")
print("Rows:", len(df))
print("Unique parts:", df["Part Number"].nunique())


# ============================================================
# 3. FIND LATEST REAL MONTH
# ============================================================

latest_month_date = df["Month Date"].max()
latest_month = latest_month_date.to_period("M").strftime("%Y-%m")

print("Latest available real month:", latest_month)


# ============================================================
# 4. CREATE FUTURE MONTHS
# ============================================================

forecast_horizon = 6

future_month_dates = pd.date_range(
    start=latest_month_date + pd.DateOffset(months=1),
    periods=forecast_horizon,
    freq="MS"
)

future_months = [date.to_period("M").strftime("%Y-%m") for date in future_month_dates]

print("Forecast months:", ", ".join(future_months))


# ============================================================
# 5. GET LATEST DATA FOR EACH PART
# ============================================================

latest_part_rows = (
    df.sort_values(["Part Number", "Month Date"])
    .groupby("Part Number", as_index=False)
    .tail(1)
)

print("Forecastable parts:", latest_part_rows["Part Number"].nunique())


# ============================================================
# 6. CREATE RECURSIVE LAG-1 FUTURE FORECASTS
# ============================================================

future_rows = []

for _, row in latest_part_rows.iterrows():
    part_number = row["Part Number"]
    description = row["description"]
    fr = row["fr"]

    # The first future prediction uses the latest real net_qty.
    previous_prediction = max(row["net_qty"], 0)

    for step, future_date in enumerate(future_month_dates, start=1):
        future_month = future_date.to_period("M").strftime("%Y-%m")

        prediction = previous_prediction

        if step == 1:
            forecast_warning = "First forecast month uses latest real month as lag_1."
        elif step <= 3:
            forecast_warning = "Recursive forecast: uses previous forecast as lag_1."
        else:
            forecast_warning = "Longer recursive forecast: accuracy becomes weaker farther into the future."

        future_rows.append({
            "Part Number": part_number,
            "description": description,
            "fr": fr,
            "forecast_month": future_month,
            "forecast_step": step,
            "recommended_model": "pred_lag_1",
            "predicted_net_qty": prediction,
            "forecast_warning": forecast_warning,
        })

        # Next month uses this prediction as lag_1.
        previous_prediction = prediction


future_forecasts = pd.DataFrame(future_rows)


# ============================================================
# 7. CREATE SUMMARY TABLE
# ============================================================

summary = pd.DataFrame({
    "metric": [
        "latest_real_month",
        "forecast_horizon_months",
        "forecast_months",
        "forecastable_parts",
        "future_forecast_rows",
        "recommended_model",
        "important_warning"
    ],
    "value": [
        latest_month,
        forecast_horizon,
        ", ".join(future_months),
        latest_part_rows["Part Number"].nunique(),
        len(future_forecasts),
        "pred_lag_1",
        "Forecasts after the first future month are recursive, so accuracy usually decreases farther into the future."
    ]
})


# ============================================================
# 8. SAVE RESULTS
# ============================================================

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="summary", index=False)
    future_forecasts.to_excel(writer, sheet_name="future_forecasts", index=False)
    latest_part_rows.to_excel(writer, sheet_name="latest_real_part_rows", index=False)


print()
print("Created:", output_file)
print("Latest real month:", latest_month)
print("Forecast months:", ", ".join(future_months))
print("Forecastable parts:", latest_part_rows["Part Number"].nunique())
print("Future forecast rows:", len(future_forecasts))
print("Recommended model: pred_lag_1")
print("Warning: multi-step forecasts are recursive, so accuracy decreases farther into the future.")