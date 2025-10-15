
import pandas as pd

# ========================
# 1. Load raw datasets
# ========================
df1 = pd.read_csv("dataset1.csv")
df2 = pd.read_csv("dataset2.csv")

# ========================
# 2. Convert time columns to datetime format
# ========================
df1["start_time"] = pd.to_datetime(df1["start_time"], format="%d/%m/%Y %H:%M", errors="coerce")
df2["time"] = pd.to_datetime(df2["time"], format="%d/%m/%Y %H:%M", errors="coerce")

# Create 'date' column for daily-level analysis
df1["date"] = df1["start_time"].dt.date
df2["date"] = df2["time"].dt.date

# ========================
# 3. Convert numeric columns to proper numeric type
# ========================
df1_numeric = ["bat_landing_to_food", "seconds_after_rat_arrival", "risk", "reward", "hours_after_sunset"]
df2_numeric = ["hours_after_sunset", "bat_landing_number", "food_availability", "rat_minutes", "rat_arrival_number"]

df1[df1_numeric] = df1[df1_numeric].apply(pd.to_numeric, errors="coerce")
df2[df2_numeric] = df2[df2_numeric].apply(pd.to_numeric, errors="coerce")

# ========================
# 4. Normalize time and label values
# ========================
# Round 'hours_after_sunset' to 0.5-hour intervals
df1["hours_after_sunset"] = (df1["hours_after_sunset"] * 2).round() / 2
df2["hours_after_sunset"] = (df2["hours_after_sunset"] * 2).round() / 2

# Fix typos and normalize text labels in 'habit'
df1["habit"] = df1["habit"].replace({
    "bat_figiht": "bat_fight",
    "rat attack": "rat_attack"
})

# ========================
# 5. Apply logical constraints
# ========================
# Keep only valid values (0 or 1) for risk, reward, and season
df1 = df1[df1["risk"].isin([0, 1]) | df1["risk"].isna()]
df1 = df1[df1["reward"].isin([0, 1]) | df1["reward"].isna()]
df1 = df1[df1["season"].isin([0, 1]) | df1["season"].isna()]

# ========================
# 6. Remove noise and invalid text entries
# ========================
noise_values = [
    "other", "others", "other_bats", "other bat",
    "other directions", "not_sure_rat", "all_pick",
    "bowl_out", "no_food"
]
df1 = df1[~df1["habit"].isin(noise_values)]

# Clean and normalize text in 'habit' column
habit_clean = (
    df1["habit"].astype(str)
    .str.strip()
    .str.replace(r"\s*;\s*", ",", regex=True)   # replace ';' with ','
    .str.replace(r"\s+", "", regex=True)        # remove whitespace
)

# ========================
# 7. Remove invalid numeric-only 'habit' values
# ========================
# Identify and exclude purely numeric entries
habit_split = habit_clean.str.split(",")
is_num = habit_split.apply(lambda x: all(i.replace('.', '', 1).isdigit() for i in x))
df1 = df1[~is_num]

# Drop rows with missing 'habit'
df1 = df1.dropna(subset=["habit"])

# ========================
# 8. Keep valid range for 'bat_landing_to_food'
# ========================
df1 = df1[(df1["bat_landing_to_food"] >= 0) & (df1["bat_landing_to_food"] <= 60)]

# ========================
# 9. Remove duplicates and standardize types
# ========================
df1 = df1.drop_duplicates()
df2 = df2.drop_duplicates()

# Convert 'habit' to categorical type
df1["habit"] = df1["habit"].astype(str).str.strip().str.lower().astype("category")

# ========================
# 10. Export cleaned datasets
# ========================
df1.to_csv("dataset1_clean.csv", index=False)
df2.to_csv("dataset2_clean.csv", index=False)

print("Cleaned datasets have been created successfully: dataset1_clean.csv, dataset2_clean.csv")
