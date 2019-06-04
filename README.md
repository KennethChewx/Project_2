# Project 2 - Ames Housing Data and Kaggle Challenge

# **Problem Statement**
To predict housing sale prices in Ames from a matrix of 79 features provided in the dataset.

# **Data Cleaning and data dictionary**
Most of the data imported into pandas had null columns because they were initially filled with NA. As a result, there was a need to convert them to an appropriate string so that it will not be recognised as null. As chosen, I have decided to replace the NaNs as string None, as most columns, the NaN represents an absence of the item. Importantly however, there are related categories within the columns that have null, for example Garage type, garage area etc. It was noted that some of the null values did not add up. For example, garage type null count is 157, but all other garage related columns had null count as 159. Upon investigation, it seems that 2 rows were not filled up, and as a result, had null in their rows despite having a garage type. For these two rows, the related categories are filled up with the mode in these selected category of garage type. Similar cases appeared in utilities and electrical columns, and basement related columns.

An unusual data was spotted in the year garage built column, where the maximum value was 2207 year. I believe that this was keyed in wrongly, when the correct value should have been 2007. Thus, this data was manually changed. As for the rest of the features, looking at the descriptives and the data dictionary proved to be adequate.

Importantly however, is that there are ordinal columns witin the dataset. As informed by the data dictionary, these columns are changed into category type and replaced with their category code; according to their ordinal level. In addition to changing the ordinal data to numeric, there are also 2 columns namely central air and street, where it can be changed into numeric binary column to represent one category or another. In central air, 0 for no central air and 1 for present of central air. In Street, 0 for pavement and 1 for gravel.

With regards to outliers, a simple plot of scatterplot reveals that for numeric columns, there are a lot of outliers present within most columns. However, it is important to note that these datapoints are not impossible and should not be dropped too soon, as they represent possible housing sales in Ames. According to the data dictionary however, there are 5 data points where General living area greater than 4000, which are true outliers. It is not obvious if those other outliers are entered incorrectly, and they seem to be adequate, so only 5 outliers are dropped in total.

# **Exploratory Data Analysis**
Looking at all the columns, it would be wise to visualise data according to their category as provided by the data dictionary. Namely, the data include 23 nominal, 23 ordinal, 14 discrete, and 20 continuous variables. Taking an overall outlook by only looking at just string and numeric columns might mask certain trends, and thus, I have created a list for data to visualise in these categories. I have also added an additional category of time, to look at time-sensitive data pertaining to year and date. Most of these distributions are plotted against sale price, if not against the number of counts of each column. Lastly, a distribution of saleprice itself as the target variable is plotted.

### Sale price distribution
In the sale price distribution, we can see that the sale price is quite normally distributed, althought it is slightly skewed towards the left, as there are some outliers whose sale price is extremely high. Also from the distribution, the mean of sale price is between 150000, where the mode is around too. Most sale prices range between 100000 to 300000.

### Distribution of continuous features against price
All houses show a general positive trend, where when the feature size increase, so does the sale price. For example, more Gn Liv Area equates to higher sale price. Furthermore, the slope for Lot area feature seems to have the strongest effect on sale price as a small increase in lot area seems to show a greater increase in sale price than other distributions.

However, there are several scatterplots that show how the distributions were heavily skewed towards the lower end of the scale. For example, in the 'Pool Area' columns, we see that most houses do not possess a pool except a handful of a few houses (can be counted; 8). These distributions show that in general, most houses do not possess these miscellaneous features and do not have much effect on the final sale price. 

As for the rest of the distributions, we can see that they generally follow a linear, if not slight exponential distribution against sale price. These features will be considered to possess a stronger relationship with the target column.<br>

### Ordinal columns distribution

These distribution plots show us distinctively which level of quality/condition the houses are. In general, the higher the score, the better the house. Thus, skewness will allow us to determine how well the house perform on these houses. Mainly, we see that if scores are concentrated on the left, the quality or score is low; Opposite is true for scores on the right since this is an ordinal feature.

Importantly to note, most ordinal features then to have modes that are highly dominated in one score, for example in Paved Drive, we see that from the data dictionary, we can tell that most houses have a proper paved driveway, and this number greatly overwhelms the partially paved and no pavement subcategory of 1 and 0 respectively. 

From an overall perspective, we see that the overall condition and overall quality seems to be normally distributed as it takes into account the average of all these other ordinal features. Thus, it might be more reasonable to forgot the other ordinal scales and to just use the overall ordinal feature for building a model.

### Distribution of nominal columns in counts

As seen from the distributions above, most of the graphs are highly biased towards one sub-category within each category of distribution plot. The lack of variance in theses nominal columns suggest that it is unlikely that these features will impact the target value of sales price. Thus, only categories with enough variance will be selected as features for the model. 

In general just to name a few, we can see from glace at these distributions, the most houses sold neighbourhood is NAmes (North Ames), most houses have RL (Residential Low Density) zoning classification, have Regular lot shape, and also have CentralAir.

### In addition to distribution of nominal columns: Boxplot of neighbourhood against sale price
From this figure, we can see that in the cheapest neighborhoods houses sell for a median price of around 100,000, and in the most expensive neighborhoods houses sell for around 300,000. We can also see that for some neighborhoods, dispersion between the prices is very low, meaning that all the prices are close to each other. In the most expensive neighborhood NridgHt, however, we see a large box — there is large dispersion in the distribution of prices.

The red line demarkates the distribution of houses with lower and higher than mean sale prices, and we can see the neighbourhoods that are more expensive or relatively cheaper.

### Distribution of discrete features

A general outlook of these discrete features shows a clear distinct mode for most of them. For example, houses do not have basement bathrooms, have 2 full bathrooms, 2 garage cars, 3 bedrooms, 1 kitchen and 0/1 fireplace. Also, most houses are sold in the month of June.

We can clearly see a dip in sales of houses in 2010, which may be a mis-representation if the data cuts off on the year 2010. Thus, year sold might be a biased feature.

### General trend of across the years of sale prices

Over the years, sale prices has increased of housing has increased. On average from the start of 122k in 1872 to 267k in 2010.

### Feature correlation with target

Some of these features have high correlation to the sale price, and might affect the overall model training if these said features are being used with training of the model. Thus, it is likely that these features are not to be considered during feature engineering. A threshold that might be reasonable would be having the absolute correlation lesser than 0.7.

### Correlation between features

As for the correlation between each feature, having similar features will only serve to over-emphasize on its effect on the target. One of either feature which correlates highly with each other should be kept. The feature being kept should have higher variance so that when considered into the model, it will be more generalisable in predicting future sale prices.

A threshold for similar features should be < 0.7

# **Preprocessing and feature engineering**

Several new features are engineered, as they are strongly related by relevance to each other. For example, I group the number of bathroom facilities there are present in each house by adding up the full baths and half baths. Furthermore, it is possible to generate a score for each of the facilities such as fireplace or kitchen. Note that it is only possible to consider ordinal scores pertaining to one facility, as distance between ordinal levels are kind of arbituary and not equal for a reasonable comparison to take place. Otherwise, other combined features may include the ease of parking a car based on the steepness of the slope on the ordinal scale 'Land Slope' and whether the driveway is paved or gravel. The higher the score, the easier the parking and the more expensive one's house sale price would be. The last feature I engineered would be the age of the house since its' last remodelling or last built. This might be more informative to a buyer as the house's premium usually goes to newer houses, whereas houses that have been on the market for a long time tend to be more worn down and thus cheaper.

Thus, I've selected these features that I have engineered, together with features that are important based on my own research and domain knowledge.

Before using these features for modeling, I have to check for multi collinearity and also correlation with the target variable sale price. High correlations needs to be reconsidered if it should be included as they may indicate data leakage. Subsequently then I create dummy variables for categorical variables and scale the whole data frame. 

One important point to note is that since the data has already been split into train and test set at the start, it is important to only do train-test-split with the training set, and to treat the test csv set as the holdout set for submission of scores.

# **Modeling and evaluation**

3 models are being selected for testing, mainly linear regression, lassoCV and ridgeCV. The best model which have high R-square score and lowest mean square error would be LassoCV. 

Looking at the coefficients of the features in the LassoCV model, it is not surprising to see that the total number of rooms matter in the sale price of the Ames house. This is also relavant to the number of total bathrooms both basement and above ground. Total number of rooms and facilities correlate with the overall square feet of the house and thus the larger the square feet, the more expensive the house would be. There is however a strong point for houses built in certain neighbourhoods, for example North Ridge Heights and also Stone Brooks. We do see from our previous plot in EDA that by neighbourhoor, these 2 neighbourhoods are actually the highest in median sale price with the highest price variance.

Interestingly, the fire_score also proved to be a strong factor in predicting sale prices. This is somewhat unexpected since one would likely to choose utilities, central air, heating quality etc over fireplace. The preference over fireplace than modern methods of heating might be again assigned to the affluent, who prefer to enjoy the finer things in life.

As expected, as the age of the house since its last remodelling time, the lower the sale price, as premium would often go to those houses that are newly renovated. As do the housing subclass of 120 and 160, which showcase houses from planned unit development from 1946 onwards. Sale prices also further depreciate when the shape of the lot which the house is built on becomes more irregular. 

Perhaps, to further improve the model, we could also look at interaction between lot shape and neighbourhood. This might be interesting because houses build around expensive neighbourhoods can sometimes be more irregular than houses built on flat land. We see houses built on hills or ridges have to follow the slope contours to provide strong foundation for the house. This lot shape and neighbourhood interaction might prove to complement or contradict the current direction of lot shape feature on sale price.