df_train = df_train.sort_values(by=["Гостиница", "Дата бронирования"])

df_train["prev_booking_date"] = df_train.groupby("Гостиница")["Дата бронирования"].shift(
    1
)
df_train["prev_cost"] = df_train.groupby("Гостиница")["Стоимость"].shift(1)
df_train["prev_num_rooms"] = df_train.groupby("Гостиница")["Номеров"].shift(1)
df_train["prev_category"] = df_train.groupby("Гостиница")["Категория номера"].shift(1)

df_train["time_between_bookings"] = (
    df_train["Дата бронирования"] - df_train["prev_booking_date"]
).dt.total_seconds() / (60 * 60 * 24)
df_train["similar_booking"] = (
    (df_train["time_between_bookings"] < 1)
    & (df_train["Стоимость"] == df_train["prev_cost"])
    & (df_train["Номеров"] == df_train["prev_num_rooms"])
    & (df_train["Категория номера"] == df_train["prev_category"])
)

df_train["similar_booking"] = df_train["similar_booking"].fillna(False)
df_train["similar_booking"] = df_train["similar_booking"].apply(int)


df_train = df_train.drop(
    [
        "prev_booking_date",
        "prev_cost",
        "prev_num_rooms",
        "prev_category",
        "time_between_bookings",
    ],
    axis=1,
)


# cat_columns = X_train.select_dtypes(include=["object"]).columns


# encoder = OneHotEncoder(drop="first", sparse_output=False)

# X_train = pd.concat(
#     [
#         X_train.drop(cat_columns, axis=1),
#         pd.DataFrame(
#             encoder.fit_transform(X_train[cat_columns]),
#             columns=encoder.get_feature_names_out(),
#         ),
#     ],
#     axis=1,
# )
# X_test_full = pd.concat(
#     [
#         X_test_cat.drop(cat_columns, axis=1),
#         pd.DataFrame(
#             encoder.transform(X_test_cat[cat_columns]),
#             columns=encoder.get_feature_names_out(),
#         ),
#     ],
#     axis=1,
# )


# from sklearn.datasets import make_friedman1

# from sklearn.feature_selection import RFECV

# from sklearn.svm import SVR


# selector = RFECV(RandomForestClassifier(), step=1, cv=StratifiedKFold(n_splits=3))

# selector = selector.fit(X_train, y_train)


def get_grouped_features(df):
    df = df.copy()
    df_hotel_grouped = (
        df_train.groupby("Гостиница")
        .agg(
            mean_cost_per_booking=("Стоимость", "mean"),
            median_cost_per_booking=("Стоимость", "median"),
            cost_per_booking_q10=("Стоимость", lambda x: x.quantile(0.1)),
            cost_per_booking_q90=("Стоимость", lambda x: x.quantile(0.9)),
            mean_nights_per_booking=("Ночей", "mean"),
            median_nights_per_booking=("Ночей", "median"),
            nights_per_booking_q10=("Ночей", lambda x: x.quantile(0.1)),
            nights_per_booking_q90=("Ночей", lambda x: x.quantile(0.9)),
            mean_prepayment=("Внесена предоплата", "mean"),
            median_prepayment=("Внесена предоплата", "median"),
            prepayment_q10=("Внесена предоплата", lambda x: x.quantile(0.1)),
            prepayment_q90=("Внесена предоплата", lambda x: x.quantile(0.9)),
        )
        .reset_index()
    )

    # Grouping by source of booking
    df_source_grouped = (
        df_train.groupby("Источник")
        .agg(
            mean_cost_per_booking=("Стоимость", "mean"),
            median_cost_per_booking=("Стоимость", "median"),
            cost_per_booking_q10=("Стоимость", lambda x: x.quantile(0.1)),
            cost_per_booking_q90=("Стоимость", lambda x: x.quantile(0.9)),
            mean_guests_per_booking=("Гостей", "mean"),
            median_guests_per_booking=("Гостей", "median"),
            guests_per_booking_q10=("Гостей", lambda x: x.quantile(0.1)),
            guests_per_booking_q90=("Гостей", lambda x: x.quantile(0.9)),
        )
        .reset_index()
    )

    # Grouping by room category
    df_room_category_grouped = (
        df_train.groupby("Категория номера")
        .agg(
            mean_cost_per_booking=("Стоимость", "mean"),
            median_cost_per_booking=("Стоимость", "median"),
            cost_per_booking_q10=("Стоимость", lambda x: x.quantile(0.1)),
            cost_per_booking_q90=("Стоимость", lambda x: x.quantile(0.9)),
            mean_nights=("Ночей", "mean"),
            median_nights=("Ночей", "median"),
            nights_q10=("Ночей", lambda x: x.quantile(0.1)),
            nights_q90=("Ночей", lambda x: x.quantile(0.9)),
        )
        .reset_index()
    )

    # Grouping by payment method
    df_payment_grouped = (
        df_train.groupby("Способ оплаты")
        .agg(
            mean_cost_per_booking=("Стоимость", "mean"),
            median_cost_per_booking=("Стоимость", "median"),
            cost_per_booking_q10=("Стоимость", lambda x: x.quantile(0.1)),
            cost_per_booking_q90=("Стоимость", lambda x: x.quantile(0.9)),
        )
        .reset_index()
    )

    # Grouping by number of rooms
    df_rooms_grouped = (
        df_train.groupby("Номеров")
        .agg(
            mean_cost_per_booking=("Стоимость", "mean"),
            median_cost_per_booking=("Стоимость", "median"),
            cost_per_booking_q10=("Стоимость", lambda x: x.quantile(0.1)),
            cost_per_booking_q90=("Стоимость", lambda x: x.quantile(0.9)),
            mean_guests=("Гостей", "mean"),
            median_guests=("Гостей", "median"),
            guests_q10=("Гостей", lambda x: x.quantile(0.1)),
            guests_q90=("Гостей", lambda x: x.quantile(0.9)),
        )
        .reset_index()
    )

    # Grouping by number of guests
    df_guests_grouped = (
        df_train.groupby("Гостей")
        .agg(
            mean_cost_per_booking=("Стоимость", "mean"),
            median_cost_per_booking=("Стоимость", "median"),
            cost_per_booking_q10=("Стоимость", lambda x: x.quantile(0.1)),
            cost_per_booking_q90=("Стоимость", lambda x: x.quantile(0.9)),
        )
        .reset_index()
    )

    return (
        df_hotel_grouped,
        df_source_grouped,
        df_room_category_grouped,
        df_payment_grouped,
        df_rooms_grouped,
        df_guests_grouped,
    )


def apply_groupby_features(
    df,
    df_hotel_grouped,
    df_source_grouped,
    df_room_category_grouped,
    df_payment_grouped,
    df_rooms_grouped,
    df_guests_grouped,
):
    df = df.copy()
    df = df.merge(df_hotel_grouped, on="Гостиница", how="left", suffixes=("", "_hotel"))
    df = df.merge(df_source_grouped, on="Источник", how="left", suffixes=("", "_source"))
    df = df.merge(
        df_room_category_grouped,
        on="Категория номера",
        how="left",
        suffixes=("", "_room"),
    )
    df = df.merge(
        df_payment_grouped, on="Способ оплаты", how="left", suffixes=("", "_payment")
    )
    df = df.merge(df_rooms_grouped, on="Номеров", how="left", suffixes=("", "_rooms"))
    df = df.merge(df_guests_grouped, on="Гостей", how="left", suffixes=("", "_guests"))

    return df
