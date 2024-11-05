# **Research Experience**

Here is a summary of my contributions to ongoing research projects.

## **Text-based Analysis using Novel Dataset**

We used a novel dataset of **U.S. congressional hearings**. After rigorously addressing *OCR reading errors* and [mapping database speakers](text-based-codes.md) to real-world congressmen, our next step is to train a **BERT topic model** to condense hundreds of thousands of speeches in the dataset into *500 topics*. 

### **Plot 1**
Following fine-tuning (including *embedding pretraining*, *vectorizer model*, *UMAP model*, *HDBSCAN model*, *c-TF-IDF model*, and *representation model*), this is an interactive map of the **top 20 most frequent topics**. Feel free to explore the data:

<iframe src="/assets/plots/topic_500.html" width="100%" height="1000px" style="border:none;"></iframe>

### **Plot 2**
Here is another plot depicting the frequency of how congressmen mention a certain ropix or terminology over time, as identified by AI-Large Language Models (LLMs).

Due to confidentiality, the labels are redacted, but the trends corresponding to major global or U.S. incidents are still visible. This dataset is crucial for future political economy analysis.

<div style="margin-top: 20px;">

  <figure style="text-align: center;">
    <img src="/assets/plots/text_based_plot.png" alt="Plot 2" style="width: 100%; border-radius: 8px;">
    <figcaption></figcaption>
  </figure>

</div>


## **The Casual Impact of Fiscal Shock**

This research investigates how government deficits and public debt levels influence asset prices and macroeconomic conditions. Using event study methodology, we analyze periods where UK budget deficit news emerged independently of economic conditions, addressing the omitted variable bias challenge (OVB). We employ **Large Language Models (LLMs)** to extract **budget surprises** from news data and examine their impact on financial markets.

We obtained `q1_ratio` by utilizing an AI LLMs to assess news articles related to budget announcements. For each article, the model answered two specific questions about whether the news would cause the budget deficit to increase or decrease. The responses were categorized as `up`, `down`, or `unsure`. The `q1_ratio` is calculated as a normalized measure using the proportions of `up` and `down` responses:

$$\text{q1_ratio} = \frac{\text{q1_up} - \text{q1_down}}{\text{q1_up} + \text{q1_down}}$$

Then, we conducted following regression analysis:

```
q1_ratio ~ 10.0_n_d + gbpusd_o_d + 10.0_n_d * gbpusd_o_d
```

- `10.0_n_d`: *change* in UK 10-year bond yield between announcement date `t+5` and `t-1`  

- `gbpusd_o_d`: *change* in GBP/USD Exchange Rate between announcement date `t+5` and `t-1`  

Here are the preliminary regression results:

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:               10.0_n_d   R-squared:                       0.470
Model:                            OLS   Adj. R-squared:                  0.447
Method:                 Least Squares   F-statistic:                     20.41
Date:                Sun, 03 Nov 2024   Prob (F-statistic):           1.42e-09
Time:                        09:37:37   Log-Likelihood:                 49.884
No. Observations:                  73   AIC:                            -91.77
Df Residuals:                      69   BIC:                            -82.61
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
===============================================================================
                  coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------
const           0.0448      0.020      2.254      0.027*      0.005       0.084
q1_ratio        0.1729      0.047      3.643      0.001***    0.078       0.268
gbpusd_o_d     -3.5446      0.638     -5.554      0.000***   -4.818      -2.272
interaction    -6.6575      1.594     -4.177      0.000***   -9.837      -3.478
===============================================================================
Omnibus:                        0.374    Durbin-Watson:                   1.495
Prob(Omnibus):                  0.829    Jarque-Bera (JB):                0.182
Skew:                           0.122    Prob(JB):                        0.913
Kurtosis:                       3.021    Cond. No.                         114.
===============================================================================
```

The regression results indicate a **significant positive relationship** between `q1_ratio` and the change in UK ten-year government bond yields (`10.0_n_d`). Specifically, a higher `q1_ratio` — implying **stronger public expectation of an increasing budget deficit** — is associated with an **increase in bond yields**. The coefficient for `q1_ratio` is positive and statistically significant at the **0.1% level**. Additionally, the negative coefficients for the change in the GBP/USD exchange rate (`gbpusd_o_d`) and the interaction term suggest that currency appreciation and its interplay with budget expectations also influence bond yields. The regression model explains approximately **47% of the variance** (`R squared`) in the change in bond yields.

These preliminary findings support the hypothesis that anticipated fiscal expansions lead investors to demand higher yields on government bonds, reflecting the pricing-in of budget deficit expectations into asset prices.