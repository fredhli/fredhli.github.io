import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

dfn_year = pd.read_csv(
    "C:/Users/lihou/Box/PoliticsCentralBanking/estimation/topic-models/2024-10-25/df_step3.csv"
)

sns.set_palette("colorblind")
# plt.rcParams['font.family'] = 'Georgia'

dfn_year.index = pd.to_datetime(dfn_year.index, format="%Y")
dfn_year = dfn_year.sort_index()
time_series = dfn_year.index.year

last_year = 2023
last_year_data = dfn_year.loc[str(last_year)]

sorted_columns = last_year_data.mean().sort_values(ascending=False).index

plt.figure(figsize=(30, 15))

for col in sorted_columns:
    plt.plot(time_series, dfn_year[col], label=col)

plt.xlabel("Year")
plt.ylabel("Number of articles")
plt.title(
    "Proportion of articles mentioning specific topics over time",
    fontdict={"fontsize": 20, "fontweight": "bold"},
)

plt.xticks(time_series, time_series.astype(int), rotation=45)

plt.legend(
    title="Topics (freq from high to low)", loc="upper left", bbox_to_anchor=(1, 0.7)
)

plt.show()
