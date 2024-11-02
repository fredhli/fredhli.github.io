# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# # Set a clean style
# sns.set_style("whitegrid")

# # Convert index to datetime and sort it
# dfn_year.index = pd.to_datetime(dfn_year.index, format="%Y")
# dfn_year = dfn_year.sort_index()
# time_series = dfn_year.index.year

# num_labels = len(definitions.keys())
# num_rows = (num_labels + 1) // 2  # Calculate the number of rows needed

# fig, axs = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))

# # Add a title
# fig.suptitle(
#     "Frequency of Economic Terms in Speeches Over Time", fontsize=20, fontweight="bold"
# )

# # range definition keys by how frequent they are
# definitions_new = {
#     k: v
#     for k, v in sorted(
#         definitions.items(),
#         key=lambda item: dfn_year[f"{item[0]}_pct"].sum(),
#         reverse=True,
#     )
# }


# for i, label in enumerate(definitions_new.keys()):
#     row = i // 2
#     col = i % 2

#     # Use a color palette for different plots
#     sns.lineplot(
#         x=time_series,
#         y=dfn_year[f"{label}_pct"],
#         ax=axs[row, col],
#         color=sns.color_palette("tab10")[i % 10],
#     )
#     axs[row, col].set_title(
#         f"[Redacted Label {i+1}] Percentage Over Time", fontsize=14, fontweight="bold"
#     )
#     axs[row, col].set_xlabel("Year", fontsize=12)
#     axs[row, col].set_ylabel(f"[Redacted Label {i+1}] Percentage", fontsize=12)

#     # Add vertical dotted lines for years with a 500% increase within the next 2 years
#     for j in range(len(dfn_year) - 2):  # Ensure we can look 2 years ahead
#         all_time_high = dfn_year[f"{label}_pct"].max()
#         current_value = dfn_year[f"{label}_pct"].iloc[j]
#         future_value_1 = dfn_year[f"{label}_pct"].iloc[j + 1]
#         future_value_2 = dfn_year[f"{label}_pct"].iloc[j + 2]
#         year_j = dfn_year.iloc[j]["year"].year

#         cond1 = (
#             future_value_2 >= 5 * current_value
#             and future_value_1 >= 0.3 * all_time_high
#         )
#         cond2 = (
#             future_value_1 >= 5 * current_value
#             and future_value_2 >= 0.3 * all_time_high
#         )
#         cond3 = (
#             future_value_2 >= 10 * current_value
#             and current_value >= 0.1 * all_time_high
#         )
#         cond4 = (
#             future_value_1 >= 10 * current_value
#             and current_value >= 0.1 * all_time_high
#         )

#         # Special arrangements
#         cond5 = (i == 5) and ((year_j == 2005) or (year_j == 2003))
#         cond6 = (i == 7) and (year_j == 2007)
#         cond7 = (i == 6) and (year_j == 2001)
#         cond8 = (i == 3) and (year_j == 2008)
#         cond9 = (i == 2) and (year_j == 1992)

#         if (
#             cond1
#             or cond2
#             or cond3
#             or cond4
#             or cond5
#             or cond6
#             or cond7
#             or cond8
#             or cond9
#         ):
#             year_to_mark = dfn_year.index[j].year
#             axs[row, col].axvline(
#                 year_to_mark, color="salmon", linestyle="--", linewidth=1
#             )
#             # Mark years at the top left of dotted lines with slight offset for readability
#             axs[row, col].text(
#                 year_to_mark,
#                 axs[row, col].get_ylim()[1] * 0.95,
#                 str(year_to_mark),
#                 verticalalignment="top",
#                 horizontalalignment="left",
#                 color="black",
#                 fontsize=10,
#                 bbox=dict(
#                     facecolor="white",
#                     alpha=0.6,
#                     edgecolor="none",
#                     boxstyle="round,pad=0.3",
#                 ),
#             )

# # Remove any empty subplots
# for j in range(i + 1, num_rows * 2):
#     fig.delaxes(axs.flatten()[j])


# # add small text at bottom right, saying because of confidentiality, labels are redacted
# plt.figtext(
#     0.99,
#     0.01,
#     "Because of confidentiality, labels are redacted.",
#     horizontalalignment="right",
#     verticalalignment="bottom",
#     fontsize=12,
#     fontstyle="italic",
# )

# plt.tight_layout(rect=[0, 0, 1, 0.98])
# plt.show()
