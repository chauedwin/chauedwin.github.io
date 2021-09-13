---
layout: post
title: "Portfolio Analysis with Statistics"
description: ""
preview: "While there are many methods of optimizing a stock portfolio, one such method is to maximize its excess return to volatility ratio. Such a model aims to find a middle ground between profitability and stability."
category: "interesting"
tags: []
---
{% include JB/setup %}

The method we will utilize aims to maximize the excess expected return of a portfolio over its volatility. The model and its 
derivations are based on the approach of 
**[Elton et al.](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1976.tb03217.x)**
It is motivated by the single index model, which we will first cover. The **[IPython notebook](https://github.com/chauedwin/PortfolioAnalysis/blob/main/PortfolioAnalysis.ipynb)** 
and **[code](https://github.com/chauedwin/PortfolioAnalysis/blob/main/pipeline.py)** are both available.

## Single Index Model

In short, the single index model(SIM) assumes that an asset's return is a dependent variable of the market's return, that is 

$$\begin{align}
R_i = \alpha_i + \beta_i R_m + \epsilon_i, \quad cov(R_m, \epsilon_i) = 0, \quad E(\epsilon_i\epsilon_j) = 0
\end{align}$$

where $$R_i$$ is the return of the $$i^{th}$$ asset, $$R_m$$ is the return of the market, and $$\epsilon_i$$ is random noise with 
mean $$0$$ and variance $$\sigma_{\epsilon_i}^2$$. The alpha of asset $$i$$, or $$\alpha_i$$ represents the "base" return of that 
asset (independent of the market) while the beta, or $$\beta_i$$, is the amount that asset changes for each unit of market 
change. Under this model, assets with positive betas follow the market, those with near-zero betas are largely unaffected 
by the market, and those with negative betas rise when the market falls. 

Additional notation includes variance of the return of a portfolio $$\sigma_p^2$$ and variance of the return of the market 
$$\sigma_m^2$$. Expected values will be represented by a bar, for instance, $$\overline{R}_m$$ is the expected return of the market.

## Finding an optimal portfolio(without short sales)

As stated before, this particular analysis aims to maximize the excess expected return of a portfolio over its volatility, or 

\begin{align}
\theta = \frac{\overline{R}_p - R_f}{\sigma_p^2}
\end{align}

$$\overline{R}_p$$ is the expected return of portfolio $$p$$, which can also be expressed as a weighted average of the expected 
returns of each asset in portfolio $$p$$, $$\displaystyle\sum_{i=1}^{N} x_i\overline{R}_i$$. Also, $$R_f$$ is a risk-free rate of 
return, such as the interest rate of a savings account. In other words, we want the portfolio with the maximal ratio of 
return (over a risk-free rate) to variance/volatility. Portfolios with very high risk will be penalized as well as 
portfolios with too low a rate of return despite stability. 

Before differentiating and solving, however, we need to impose two additional constraints. First, each asset weight $$x_i$$ 
must be larger than or equal to $$0$$ because we are not allowing short selling. Second, the sum of weights must total to 1, 
as it would not make sense otherwise. Altogether, we have the following optimization problem:

$$\begin{align}
\text{max}& \quad \theta = \frac{\overline{R}_p - R_f}{\sigma_p^2} \\
\text{s.t.}& \quad \displaystyle\sum_{i=1}^{N} x_i = 1 \\
&\quad x_i \geq 0
\end{align}$$

Setting aside the constraints, we can rewrite $$\theta$$ in terms of $$x_i$$. First the numerator,

$$\begin{align}
\overline{R}_p - R_f &= \displaystyle\sum_{i=1}^{N} x_i \overline{R}_i - R_f \\
&= \sum_{i=1}^{N} x_i \overline{R}_i - \sum_{i=1}^{N} x_i R_f \\
&= \sum_{i=1}^{N} x_i (\overline{R}_i - R_f)
\end{align}$$

We can rewrite $$\sigma_p^2$$ by utilizing the historical covariances of the assets in the portfolio. Suppose $$\sigma_i^2$$ 
is the variance of the $$i^{th}$$ asset and $$\sigma_{ij}$$ is the covariance between the $$i^{th}$$ and $$j^{th}$$ assets. 
Then the variance of the portfolio is the weighted average of each $$\sigma_i^2$$ and $$\sigma_{ij}$$:

$$\begin{align}
\sigma_p^2 = \displaystyle\sum_{i=1}^N \sum_{j=1}^N x_i x_j \sigma_{ij} = \sum_{i=1}^N x_i^2 \sigma_i^2 + \sum_{i=1}^N \sum_{j=1, j\neq i}^N x_i x_j \sigma_{ij}
\end{align}$$

Using the SIM model, we also have the following:

$$\begin{align}
\sigma_i^2 = var(R_i) &= var(\alpha_i + \beta_i R_m + \epsilon_i) \\ 
&= var(\beta_i R_m) + var(\epsilon_i) + 2\beta_i cov(R_m, \epsilon_i) \\
&= \beta_i^2 \sigma_m^2 + \sigma_{\epsilon_i}^2 \\
\sigma_{ij} = cov(R_i, R_j) &= cov(\alpha_i + \beta_i R_m + \epsilon_i, \alpha_j + \beta_j R_m + \epsilon_j) \\
&= \beta_i \beta_j cov(R_m, R_m) + \beta_i cov(R_m, \epsilon_j) + \beta_j cov(R_m, \epsilon_i) + cov(\epsilon_i, \epsilon_j) \\
&= \beta_i \beta_j \sigma_m^2
\end{align}$$

Plugging these two results into our expression for $$\sigma_p^2$$, we get

$$\begin{align}
\sigma_p^2 &= \displaystyle \sum_{i=1}^N x_i^2 \beta_i^2 \sigma_m^2 + \sum_{i=1}^N x_i^2 \sigma_{\epsilon_i} + \sum_{i=1}^N \sum_{j=1,j\neq i}^N x_i x_j \beta_i \beta_j \sigma_m^2 \\
\implies \theta &= \frac{\displaystyle\sum_{i=1}^{N} x_i (\overline{R}_i - R_f)}{\sqrt{\displaystyle \sum_{i=1}^N x_i^2 \beta_i^2 \sigma_m^2 + \sum_{i=1}^N x_i^2 \sigma_{\epsilon_i} + \sum_{i=1}^N \sum_{j=1,j\neq i}^N x_i x_j \beta_i \beta_j \sigma_m^2}}
\end{align}$$

Moving onto the optimizing, we can also notice that $$\theta$$ is not only 
**[homogenous](https://en.wikipedia.org/wiki/Homogeneous_function)**, but it is homogenous with degree 0. In other words, 
if we multiply each $$x_i$$ by a factor of $$r$$, this actually does not change the value of $$\theta$$. Thus we can safely 
ignore the equality constraint(but must still consider the inequality/nonnegative constraints), and our Lagrangian is 

$$\begin{align}
\mathcal{L} = \frac{\displaystyle\sum_{i=1}^{N} x_i (\overline{R}_i - R_f)}{\sigma_p} + \sum_{i=1}^N \mu_i x_i
\end{align}$$

Differentiating w.r.t $$x_i$$ and equating to zero(to avoid vector calculus), 

$$\begin{align}
\frac{\partial \mathcal{L}}{\partial x_i} &= \frac{(\overline{R}_i - R_f)\sigma_p - \frac{\partial \sigma_p}{\partial x_i} \cdot \displaystyle\sum_{i=1}^N x_i (R_i - R_f)}{\sigma_p^2} = \frac{\overline{R}_i - R_f}{\sigma_p} - \frac{\partial \sigma_p}{\partial x_i} \cdot \frac{\displaystyle\sum_{i=1}^N x_i (R_i - R_f)}{\sigma_p^2} + \mu_i\\
\frac{\partial \sigma_p}{\partial x_i} &= \frac{1}{2}\left(\displaystyle \sum_{i=1}^N x_i^2 \beta_i^2 \sigma_m^2 + \sum_{i=1}^N x_i^2 \sigma_{\epsilon_i} + \sum_{i=1}^N \sum_{j=1,j\neq i}^N x_i x_j \beta_i \beta_j \sigma_m^2\right)^{-\frac{1}{2}}\left(2x_i\beta_i^2 \sigma_m^2 + 2x_i \sigma_{\epsilon_i}^2 + 2\sum_{j=1,j\neq i}^N x_j \beta_i \beta_j \sigma_m^2\right) \\
&= \frac{x_i\beta_i^2 \sigma_m^2 + x_i \sigma_{\epsilon_i}^2 + \sum_{j=1,j\neq i}^N x_j \beta_i \beta_j \sigma_m^2}{\sigma_p} \\
\implies \frac{\partial \mathcal{L}}{\partial x_i} &= \frac{\overline{R}_i - R_f}{\sigma_p} - \frac{\displaystyle\sum_{i=1}^N x_i (R_i - R_f)}{\sigma_p^3} \left(x_i\beta_i^2 \sigma_m^2 + x_i \sigma_{\epsilon_i}^2 + \sum_{j=1,j\neq i}^N x_j \beta_i \beta_j \sigma_m^2\right) + \mu_i = 0 \\
\implies& \frac{\overline{R}_i - R_f}{\sigma_p} - \frac{\displaystyle\sum_{i=1}^N x_i (R_i - R_f)}{\sigma_p^2} \left(x_i\beta_i^2 \sigma_m^2 + x_i \sigma_{\epsilon_i}^2 + \sum_{j=1,j\neq i}^N x_j \beta_i \beta_j \sigma_m^2\right) + \mu_i = 0 \\
\end{align}$$

Now define $$z_i = \frac{\overline{R}_i - R_f}{\sigma_p^2} x_i$$. Solving the above expression for $$z_i$$(and leaving out 
boring algebra) yields 

$$\begin{align}
z_i = \frac{\overline{R}_i - R_f}{\sigma_{\epsilon_i}^2} - \frac{\beta_i}{\sigma_{\epsilon_i}^2} \sigma_m^2 \displaystyle\sum_{j=1}^N \beta_j z_j + \mu_i
\end{align}$$

with the KKT conditions $$z_i \geq 0$$, $$\mu_i \geq 0$$, $$z_i \mu_i \geq 0$$. Suppose we have already found the $$k$$ assets 
belonging in the optimal portfolio, and note that their corresponding $$\mu_1, \ldots \mu_k = 0$$ in order to minimize the 
Lagrangian. The assets that are not part of this optimal portfolio have negative $$z_i$$'s, thus it is also true that 

$$\begin{align}
\sum_{i=1}^N z_i \mu_i = \sum_{j=1}^k z_j \mu_j
\end{align}$$

Muliplying each $$z_i$$ by $$\beta_i$$ and summing from $$1$$ to $$k$$(or equivalently to $$N$$), we have

$$\begin{align}
&\sum_{i=1}^k z_i \beta_i = \sum_{i=1}^k \frac{\overline{R}_i - R_f}{\sigma_{\epsilon_i}} \beta_i - \sum_{i=1}^k \left(\frac{\beta_i}{\sigma_{\epsilon_i}} \sigma_m^2 \sum_{j=1}^k z_j \beta_j\right) \beta_i \\
\implies &\sum_{i=1}^k z_i \beta_i + \left(\sum_{i=1}^k \frac{\beta_i}{\sigma_{\epsilon_i}} \sigma_m^2\right)\left(\sum_{j=1}^k z_j \beta_j\right) = \sum_{i=1}^k \frac{\overline{R}_i - R_f}{\sigma_{\epsilon_i}} \beta_i \\
\implies &\displaystyle\sum_{i=1}^k z_i \beta_i = \frac{\displaystyle\sum_{i=1}^k \frac{\overline{R}_i - R_f}{\sigma_{\epsilon_i}} \beta_i}{1 + \sigma_m^2 \displaystyle\sum_{j=1}^k \frac{\beta_j}{\sigma_{\epsilon_j}}}
\end{align}$$

We can now rewrite the original $$z_i$$ equation 

$$\begin{align}
z_i &= \frac{\beta_i}{\sigma_{\epsilon_i}} \left(\frac{\overline{R}_i - R_f}{\beta_i} - \phi_k\right) + \mu_i
\end{align}$$

where $$\phi_k = \sigma_m^2 \displaystyle\sum_{i=1}^k z_i \beta_i$$. Regarding $$\mu_i$$, the KKT conditions imply that 
positive $$z_i$$ values will have zero $$\mu_i$$, and negative $$z_i$$ values will have zero $$\mu_i$$ as well.

Finding the assets that belong to the optimal portfolio is now relatively simple. We just need to calculate and rank each 
asset according to their excess return to beta ratio, or $$\frac{\overline{R}_i - R_f}{\beta_i}$$, and weed out those that 
have ratios less than $$\phi_k$$. The positive $$z_i$$'s can then be normalized by dividing by their sum total, yielding the 
weights of each asset to be included. 

It is important to note that the described method assumes that all chosen assets have positive betas. While this may 
often be true, small tweaks to this process must be made to incorporate assets with negative betas. More details can be 
found in the paper by Elton et al. **[here](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1976.tb03217.x)**.


## Data ETL 

To access stock data, we will use the **[yfinance library](https://github.com/ranaroussi/yfinance)** 
(one could also scrape prices from a webpage or query them from a database, though formatting may differ). 
We first download the historical monthly stock prices for the chosen stocks/tickers(slightly modifying the code in the 
library tutorial). The data comes in the form of a pandas dataframe with multi-level headers, so we also unstack the 
levels for simpler access.
___

```python

	import yfinance as yf
	import numpy as np
	import scipy.stats as stats
	import pandas as pd
	
	ticks = ["AMD", "MSFT", "TSLA"]
	data = yf.download(tickers = ['^GSPC'] + ticks, start = '2016-06-01', period = '1mo', interval = '1mo', 
					   group_by = 'ticker', auto_adjust = True, prepost = False, threads = True, proxy = None)

	# unstack
	stocks_raw = data.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
	stocks_raw = stocks_raw.groupby('Ticker').filter(lambda x: len(x) > 36)
	stocks_raw = stocks_raw.sort_values(by=['Date', 'Ticker'])
	print(stocks_raw)
```

___

<p align="center">
  <img src="{{site.baseurl}}/img/portfolioanalysis/stocks_raw_output.png">
</p>

Since we only want one stock price per month, we filter out the last row of each asset if it does not fall on the first 
day of the month. We also only keep the last 5 years of data to maintain an accurate representation of each company's 
relevant returns(profitability in the 1990s does not entail profitability in 2020s). Because we require the previous 
month's price to compute the current month's return, we need to keep an extra month(a total of 61 months)
___

```python

	recent = stocks_raw.index[-1] - pd.DateOffset(day = 1)
	stocks = stocks_raw.loc[stocks_raw.index <= recent].copy()
	
```

To compute the (percent) return of a specified observation, we subtract the current price with last month's price and 
divide by last month's price. We can easily vectorize this by subtracting an array of the (open) prices without the last 
observation from an array of the (open) prices without the first observation. We then divide by the former.

To make calculating parameters easier, we can pivot the dataframe such that each ticker's percent returns form individual 
columns. Note that we need to mask the data matrix to ignore NaN values.  

```python

	pivoted = stocks.pivot(columns = 'Ticker', values = 'Open').dropna(axis=1, how='all')
	print(pivoted)
	spprices = pivoted['^GSPC'].to_numpy()
	pivoted = pivoted.drop(columns = ['^GSPC'])
	prices = pivoted.to_numpy()
	returnarr = prices[1:, :] - prices[:(prices.shape[0] - 1), :]
	returnarr = returnarr /  prices[:(prices.shape[0] - 1), :]
	spreturns = (spprices[1:] - spprices[:(len(spprices) - 1)]) / spprices[:(len(spprices) - 1)]
```
___ 

<p align="center">
  <img src="{{site.baseurl}}/img/portfolioanalysis/pivoted_output.png">
</p>


## SIM and weight calculations

Next we compute the single index model parameters and store them in a dataframe, which includes $$\phi_k$$ values. Finally, 
we determine the assets above the cutoff point and compute the portfolio weights. We will assume the risk-free rate to be zero.

___
```python

	betas = np.zeros(returnarr.shape[1])
	alphas = np.zeros(returnarr.shape[1])
	unsyserr = np.zeros(returnarr.shape[1])
	for i in np.arange(returnarr.shape[1]):
		treturn = returnarr[:,i]
		tnonan = treturn[np.logical_not(np.isnan(treturn))]
		spmatch = spreturns[(len(spreturns) - len(tnonan)):]
		try:
			betas[i], alphas[i], r, p, se = stats.linregress(spmatch, tnonan)
		except: 
			print(returnarr[:,i])
		unsyserr[i] = np.sum((tnonan - alphas[i] - betas[i]*spmatch)**2) / (len(spmatch) - 2)
		
	returnmean = returnarr.mean(axis = 0)
	simdf = pd.DataFrame(data = {'alpha': alphas, 'beta': betas, 'eps': unsyserr, 'rmean': returnmean}, index = pivoted.columns.values)
	simdf['excess'] = simdf['rmean'] / simdf['beta']
	simdf = simdf.sort_values(by=['excess'], ascending = False)
	simdf = simdf.loc[(simdf['excess'] > 0) & (simdf['beta'] > 0)]
	print(simdf)
```
___

<p align="center">
  <img src="{{site.baseurl}}/img/portfolioanalysis/simdf_output.png">
</p>

___
```python

	num = simdf['rmean'] * simdf['beta'] / simdf['eps']
	den = simdf['beta']**2 / simdf['eps']
	simdf['phi'] = spreturns.var() * num.cumsum() / (1 + spreturns.var() * den.cumsum())
	cutoff = simdf.loc[simdf['phi'] < simdf['excess']]
	z = (cutoff['beta'] / cutoff['eps']) * (cutoff['excess'] - cutoff['phi'])
	weights = z.sort_values(ascending = False) / z.sum()
	weights.head()
```
___

<p align="center">
  <img src="{{site.baseurl}}/img/portfolioanalysis/weights_output.png">
</p>
