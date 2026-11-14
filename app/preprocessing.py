
import pandas as pd
import numpy as np
from pathlib import Path

# PROJECT PATHS
PROJECT_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_DIR / "outputs"


# MODEL FEATURES

MODEL_FEATURES = [
    "Store",
    "DayOfWeek",
    "Open",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "Promo2",
    "CompetitionOpenDateMissing",
    "CompetitionDistance_Original",
    "Year",
    "Month",
    "Day",
    "IsWeekend",
    "WeekOfYear",
    "Quarter",
    "MonthPeriod",
    "IsMonthStart",
    "IsMonthEnd",
    "IsYearStart",
    "IsYearEnd",
    "Days_To_StateHoliday",
    "Days_After_StateHoliday",
    "CompetitionOpenMonths",
    "Promo2Active",
    "CompetitionMonths_Missing"
]


# 1. PREPROCESS STORE DATA

def preprocess_store_data(store_df):

    store = store_df.copy()

    # Fill missing CompetitionDistance
    store["CompetitionDistance"] = (
        store["CompetitionDistance"].fillna(
            store["CompetitionDistance"].median()
        )
    )

    # Handle Promo2 missing information
    mask = store["Promo2"] == 0

    store.loc[mask, "Promo2SinceWeek"] = 0
    store.loc[mask, "Promo2SinceYear"] = 0
    store.loc[mask, "PromoInterval"] = "None"

    # Competition opening date missing indicator
    store["CompetitionOpenDateMissing"] = (
        store["CompetitionOpenSinceMonth"].isnull()
        |
        store["CompetitionOpenSinceYear"].isnull()
    ).astype(int)

    # Fill missing competition opening information
    store["CompetitionOpenSinceMonth"] = (
        store["CompetitionOpenSinceMonth"].fillna(0)
    )

    store["CompetitionOpenSinceYear"] = (
        store["CompetitionOpenSinceYear"].fillna(0)
    )

    # Fill missing PromoInterval as done in Task 2
    store["PromoInterval"] = (
        store["PromoInterval"].fillna("No Promo2")
    )

    # Preserve distance after median imputation,
    # but before IQR capping
    store["CompetitionDistance_Original"] = (
        store["CompetitionDistance"]
    )

    # Calculate IQR bounds
    Q1 = store["CompetitionDistance"].quantile(0.25)
    Q3 = store["CompetitionDistance"].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Apply the same IQR capping used in Task 1
    store["CompetitionDistance"] = (
        store["CompetitionDistance"].clip(
            lower=lower_bound,
            upper=upper_bound
        )
    )

    return store


# 2. CREATE DATE FEATURES

def create_date_features(df):

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])

    # Basic date features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

    # Weekday
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    # Weekend indicator
    df["IsWeekend"] = (
        df["DayOfWeek"].isin([5, 6]).astype(int)
    )

    # Week of year
    df["WeekOfYear"] = (
        df["Date"].dt.isocalendar().week.astype(int)
    )

    # Quarter
    df["Quarter"] = df["Date"].dt.quarter

    # Month period
    df["MonthPeriod"] = np.select(
        [
            df["Day"] <= 10,
            df["Day"].between(11, 20)
        ],
        [
            "Beginning",
            "Middle"
        ],
        default="End"
    )

    # Month start and end
    df["IsMonthStart"] = (
        df["Date"].dt.is_month_start.astype(int)
    )

    df["IsMonthEnd"] = (
        df["Date"].dt.is_month_end.astype(int)
    )

    # Year start and end
    df["IsYearStart"] = (
        df["Date"].dt.is_year_start.astype(int)
    )

    df["IsYearEnd"] = (
        df["Date"].dt.is_year_end.astype(int)
    )

    return df


# 3. HOLIDAY DISTANCE FEATURES

def add_holiday_distance_features(df, holiday_dates):

    df = df.copy()

    dates = pd.to_datetime(df["Date"]).values

    holiday_dates = np.sort(
        np.unique(
            pd.to_datetime(holiday_dates).values.astype(
                "datetime64[ns]"
            )
        )
    )

    # Days until next holiday
    next_idx = np.searchsorted(
        holiday_dates,
        dates,
        side="left"
    )

    days_to = np.full(len(df), np.nan)

    valid_next = next_idx < len(holiday_dates)

    days_to[valid_next] = (
        holiday_dates[next_idx[valid_next]] -
        dates[valid_next]
    ).astype("timedelta64[D]").astype(int)

    # Days since previous holiday
    previous_idx = np.searchsorted(
        holiday_dates,
        dates,
        side="right"
    ) - 1

    days_after = np.full(len(df), np.nan)

    valid_previous = previous_idx >= 0

    days_after[valid_previous] = (
        dates[valid_previous] -
        holiday_dates[previous_idx[valid_previous]]
    ).astype("timedelta64[D]").astype(int)

    df["Days_To_StateHoliday"] = days_to
    df["Days_After_StateHoliday"] = days_after

    return df


# 4. COMPETITION FEATURES

def add_competition_features(df):

    df = df.copy()

    df["CompetitionOpenSince"] = pd.to_datetime(
        dict(
            year=df["CompetitionOpenSinceYear"],
            month=df["CompetitionOpenSinceMonth"],
            day=1
        ),
        errors="coerce"
    )

    df["CompetitionOpenMonths"] = (
        (df["Date"].dt.year -
         df["CompetitionOpenSince"].dt.year) * 12
        +
        (df["Date"].dt.month -
         df["CompetitionOpenSince"].dt.month)
    )

    # Preserve missing competition opening dates
    df.loc[
        df["CompetitionOpenSince"].isna(),
        "CompetitionOpenMonths"
    ] = np.nan

    df.drop(
        columns=["CompetitionOpenSince"],
        inplace=True
    )

    # Missing competition month indicator
    df["CompetitionMonths_Missing"] = (
        df["CompetitionOpenMonths"].isna().astype(int)
    )

    return df


# 5. PROMO2 ACTIVE FEATURE

def add_promo2_active(df):

    df = df.copy()

    iso = df["Date"].dt.isocalendar()

    current_year = iso.year.astype(int)
    current_week = iso.week.astype(int)

    started = (
        (current_year > df["Promo2SinceYear"])
        |
        (
            (current_year == df["Promo2SinceYear"])
            &
            (current_week >= df["Promo2SinceWeek"])
        )
    )

    active_month = df.apply(
        lambda row: (
            row["Date"].strftime("%b")
            in str(row["PromoInterval"]).split(",")
        ),
        axis=1
    )

    df["Promo2Active"] = (
        (df["Promo2"] == 1)
        & started
        & active_month
    ).astype(int)

    return df


# 6. MAIN PREPROCESSING FUNCTION

def prepare_input_data(raw_df, store_df):

    df = raw_df.copy()

    # Validate required input columns
    required_columns = [
        "Date",
        "Open",
        "Promo",
        "StateHoliday",
        "SchoolHoliday"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing input columns: {missing_columns}"
        )

    # Convert dates
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    if df["Date"].isna().any():
        raise ValueError(
            "Invalid or missing dates found in uploaded data."
        )

    # Store ID must be available
    if "Store" not in df.columns:
        raise ValueError(
            "Store column is missing. Please provide Store ID."
        )

    # Remove metadata columns if already supplied
    store_metadata = [
        "StoreType",
        "Assortment",
        "CompetitionDistance",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2",
        "Promo2SinceWeek",
        "Promo2SinceYear",
        "PromoInterval",
        "CompetitionOpenDateMissing",
        "CompetitionDistance_Original"
    ]

    df.drop(
        columns=[
            col for col in store_metadata
            if col in df.columns
        ],
        inplace=True
    )

    # Prepare store data
    stores = preprocess_store_data(store_df)

    # Merge store information
    df = df.merge(
        stores,
        on="Store",
        how="left",
        validate="many_to_one"
    )

    if df["StoreType"].isna().any():
        raise ValueError(
            "Some Store IDs were not found in store.csv."
        )

    # Handle missing Open values consistently with Task 1
    df["Open"] = df["Open"].fillna(1)

    # Ensure holiday column has string representation
    df["StateHoliday"] = (
        df["StateHoliday"].astype("string")
    )

    # Load saved holiday calendar
    holiday_path = OUTPUT_DIR / "holiday_dates.csv"

    if not holiday_path.exists():
        raise FileNotFoundError(
            "holiday_dates.csv not found in outputs folder."
        )

    holiday_df = pd.read_csv(
        holiday_path,
        parse_dates=["Date"]
    )

    holiday_dates = holiday_df["Date"].dropna()

    # Apply feature engineering
    df = create_date_features(df)

    df = add_holiday_distance_features(
        df,
        holiday_dates
    )

    df = add_competition_features(df)

    df = add_promo2_active(df)

    # Verify required model features
    missing_features = [
        col for col in MODEL_FEATURES
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    # Arrange features in model order
    df = df[["Date"] + MODEL_FEATURES]

    return df
