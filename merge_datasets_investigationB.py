import pandas as pd

# ============================
# 1. Load cleaned datasets
# ============================
df1 = pd.read_csv("dataset1_clean.csv")
df2 = pd.read_csv("dataset2_clean.csv")

# ============================
# 2. Merge datasets
# ============================
merged = pd.merge(
    df1[[
        "date",
        "hours_after_sunset",
        "risk",
        "reward",
        "bat_landing_to_food",
        "habit",
        "seconds_after_rat_arrival",
        "season"
    ]],
    df2[[
        "date",
        "hours_after_sunset",
        "food_availability",
        "rat_arrival_number",
        "bat_landing_number"
    ]],
    on=["date", "hours_after_sunset"],
    how="left"
)

# ============================
# 3. Export merged dataset
# ============================
merged.to_csv("merged_dataset.csv", index=False)

print("Merged dataset has been created successfully")
