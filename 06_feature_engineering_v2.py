import os
import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD RAW DATA
# ============================================================

file_path = "D:/Diploma_work/diploma/DB_31_24&25.xlsb"

df = pd.read_excel(file_path, engine="pyxlsb")

df["Date Converted"] = pd.to_datetime(
    df["Date"],
    origin="1899-12-30",
    unit="D",
    errors="coerce"
)

df["Month"] = df["Date Converted"].dt.to_period("M").astype(str)

df["sold_qty"] = df["Qty"].clip(lower=0)
df["return_qty"] = df["Qty"].clip(upper=0).abs()

os.makedirs("outputs", exist_ok=True)


# ============================================================
# 2. CREATE MONTHLY PART DEMAND
# ============================================================

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
        fr=("FR", "first")
    )
    .sort_values(["Part Number", "Month"])
)


# ============================================================
# 3. CREATE FORECASTABLE PART LIST
# ============================================================

part_summary = (
    monthly_part_demand.groupby("Part Number", as_index=False)
    .agg(
        active_months=("Month", "nunique"),
        total_net_qty=("net_qty", "sum"),
        total_sold_qty=("sold_qty", "sum"),
        total_return_qty=("return_qty", "sum"),
        total_transactions=("transaction_count", "sum"),
        avg_monthly_net_qty=("net_qty", "mean"),
        max_monthly_net_qty=("net_qty", "max"),
        total_sale_value=("sale_value", "sum"),
        total_profit=("profit", "sum"),
        description=("description", "first"),
        fr=("fr", "first")
    )
)

forecastable_parts = part_summary[
    (part_summary["active_months"] >= 18) &
    (part_summary["total_transactions"] >= 50) &
    (part_summary["total_sold_qty"] > 0)
].sort_values(
    ["active_months", "total_transactions", "total_sold_qty"],
    ascending=False
)

top_50_parts = forecastable_parts.head(50)["Part Number"]

model_data = monthly_part_demand[
    monthly_part_demand["Part Number"].isin(top_50_parts)
].copy()


# ============================================================
# 4. COMPLETE MONTHLY GRID
# ============================================================

all_months = sorted(monthly_part_demand["Month"].unique())
all_parts = sorted(top_50_parts.unique())

complete_index = pd.MultiIndex.from_product(
    [all_parts, all_months],
    names=["Part Number", "Month"]
)

model_data = (
    model_data.set_index(["Part Number", "Month"])
    .reindex(complete_index)
    .reset_index()
)

numeric_fill_columns = [
    "net_qty",
    "sold_qty",
    "return_qty",
    "transaction_count",
    "sale_value",
    "profit"
]

model_data[numeric_fill_columns] = model_data[numeric_fill_columns].fillna(0)

part_info = (
    monthly_part_demand.groupby("Part Number", as_index=False)
    .agg(
        description=("description", "first"),
        fr=("fr", "first")
    )
)

model_data = model_data.drop(columns=["description", "fr"])
model_data = model_data.merge(part_info, on="Part Number", how="left")


# ============================================================
# 5. DATE FEATURES
# ============================================================

model_data["Month Date"] = pd.to_datetime(model_data["Month"] + "-01")

model_data["year"] = model_data["Month Date"].dt.year
model_data["month_number"] = model_data["Month Date"].dt.month
model_data["quarter"] = model_data["Month Date"].dt.quarter

model_data["month_sin"] = np.sin(2 * np.pi * model_data["month_number"] / 12)
model_data["month_cos"] = np.cos(2 * np.pi * model_data["month_number"] / 12)


# ============================================================
# 6. SORT BEFORE TIME FEATURES
# ============================================================

model_data = model_data.sort_values(["Part Number", "Month Date"])


# ============================================================
# 7. LAG FEATURES
# ============================================================

model_data["lag_1"] = model_data.groupby("Part Number")["net_qty"].shift(1)
model_data["lag_2"] = model_data.groupby("Part Number")["net_qty"].shift(2)
model_data["lag_3"] = model_data.groupby("Part Number")["net_qty"].shift(3)

model_data["sold_qty_lag_1"] = model_data.groupby("Part Number")["sold_qty"].shift(1)
model_data["return_qty_lag_1"] = model_data.groupby("Part Number")["return_qty"].shift(1)
model_data["transaction_count_lag_1"] = model_data.groupby("Part Number")["transaction_count"].shift(1)


# ============================================================
# 8. SAFE ROLLING FEATURES
# ============================================================

# IMPORTANT:
# transform keeps rolling calculations inside each part only.
# shift(1) prevents using the current month target in its own features.

model_data["rolling_mean_3"] = (
    model_data.groupby("Part Number")["net_qty"]
    .transform(lambda x: x.shift(1).rolling(window=3).mean())
)

model_data["rolling_mean_6"] = (
    model_data.groupby("Part Number")["net_qty"]
    .transform(lambda x: x.shift(1).rolling(window=6).mean())
)

model_data["rolling_std_3"] = (
    model_data.groupby("Part Number")["net_qty"]
    .transform(lambda x: x.shift(1).rolling(window=3).std())
)


# ============================================================
# 9. RETURN FEATURES
# ============================================================

model_data["return_ratio"] = np.where(
    model_data["sold_qty"] > 0,
    model_data["return_qty"] / model_data["sold_qty"],
    0
)

model_data["return_ratio_lag_1"] = (
    model_data.groupby("Part Number")["return_ratio"].shift(1)
)


# ============================================================
# 10. PART-LEVEL FEATURES
# ============================================================

part_level_features = (
    model_data.groupby("Part Number", as_index=False)
    .agg(
        part_avg_monthly_qty=("net_qty", "mean"),
        part_max_monthly_qty=("net_qty", "max"),
        part_total_qty=("net_qty", "sum"),
        part_avg_transactions=("transaction_count", "mean")
    )
)

model_data = model_data.merge(part_level_features, on="Part Number", how="left")


# ============================================================
# 11. HIGH-DEMAND FLAG FOR ANALYSIS ONLY
# ============================================================

# This flag is useful for analysis, but should NOT be used as a model feature
# because it is created from the target variable net_qty.
high_demand_threshold = model_data["net_qty"].quantile(0.99)

model_data["is_high_demand_month_analysis_only"] = (
    model_data["net_qty"] >= high_demand_threshold
).astype(int)


# ============================================================
# 12. REMOVE ROWS WITHOUT ENOUGH HISTORY
# ============================================================

required_features = [
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
    "rolling_mean_6",
    "sold_qty_lag_1",
    "return_qty_lag_1",
    "transaction_count_lag_1",
    "return_ratio_lag_1"
]

model_ready_data = model_data.dropna(subset=required_features).copy()

model_ready_data["rolling_std_3"] = model_ready_data["rolling_std_3"].fillna(0)


# ============================================================
# 13. TRAIN/TEST SPLIT BY TIME
# ============================================================

test_months = sorted(model_ready_data["Month"].unique())[-5:]

train_data = model_ready_data[
    ~model_ready_data["Month"].isin(test_months)
].copy()

test_data = model_ready_data[
    model_ready_data["Month"].isin(test_months)
].copy()


# ============================================================
# 14. SAVE OUTPUTS
# ============================================================

output_file = "outputs/xlsx/06_feature_engineering_v2_report.xlsx"

feature_summary = pd.DataFrame({
    "metric": [
        "selected_parts",
        "all_months",
        "model_rows_before_dropna",
        "model_rows_after_dropna",
        "train_rows",
        "test_rows",
        "test_months",
        "high_demand_threshold_analysis_only"
    ],
    "value": [
        len(all_parts),
        len(all_months),
        len(model_data),
        len(model_ready_data),
        len(train_data),
        len(test_data),
        ", ".join(test_months),
        high_demand_threshold
    ]
})

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    model_data.to_excel(writer, sheet_name="model_data_before_dropna", index=False)
    model_ready_data.to_excel(writer, sheet_name="model_ready_data", index=False)
    train_data.to_excel(writer, sheet_name="train_data", index=False)
    test_data.to_excel(writer, sheet_name="test_data", index=False)
    feature_summary.to_excel(writer, sheet_name="feature_summary", index=False)
    forecastable_parts.to_excel(writer, sheet_name="forecastable_parts", index=False)

print(f"Created: {output_file}")
print("Selected parts:", len(all_parts))
print("All months:", len(all_months))
print("Model rows before dropna:", len(model_data))
print("Model rows after dropna:", len(model_ready_data))
print("Train rows:", len(train_data))
print("Test rows:", len(test_data))
print("Test months:", test_months)
print("High demand threshold analysis only:", high_demand_threshold)