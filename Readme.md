Data_Ingestion.py -
Used for fetching data from yahoo finance. 
It fetches data for ranging from 2018 to 2026
Cleans the data and fills the missing values by forward filling
Converts the data values from INR to USD prices
Converts the prices to log returns
Saves log returns to a csv ("market_log_returns.csv")


User_Archetype_Generator.py -
Used for generating specified number of user archetypes (1M by default) based on the allocation of funds in various categories
It uses dirichlet distribution to generate weights of assets in the portfolio allocation
It also calculates CVaR (Conditional Value at Risk) for each user archetype




Moving_Window_Converter.py -
Used for converting the user archetypes into moving window format for training the model.
It calculates the following features for each user archetype:
- Context_Momentum: Total return of Nifty in the past year
- Context_Vol: Volatility of Nifty in the past year
- Target_Return: Total return of the portfolio in the next 4 years
- Target_Drawdown: Maximum drawdown of the portfolio in the next 4 years
- Target_Volatility: Volatility of the portfolio in the next 4 years

dynamic_XGBoost.py -
Used for training the XGBoost model for predicting the target variables.