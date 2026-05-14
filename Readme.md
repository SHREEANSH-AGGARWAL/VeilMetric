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



