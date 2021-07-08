# Airline_Passenger_satisfaction
# Overview
The Dataset contains an Airline Satisfaction Passenger survey from kaggle dataset [https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction](https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction). The purpose of this analysis is to get the insight about what factors are highly correlated to a satisfied (or dissatisfied) passenger, and also can the data predict the passenger satisfaction. 
From Data Analysis result, The Airlines passenger satisfaction are dominant by the older people within range of the age between 40 - 60 for the business travel using business class wihtin long flying range distance, which supported and reinforced by good services score for older people like seat comfort, on-board service and leg room. and also in Machine Learning modeling, model got the 96% precision to predict passenger satisfaction using Support Vector Classifier.
# Dataset
The Dataset contains 25 columns as follows:
1. Gender: Gender of the passengers (Female, Male).
2. Customer Type: The customer type (Loyal customer, disloyal customer).
3. Age: The actual age of the passengers.
4. Type of Travel: Purpose of the flight of the passengers (Personal Travel, Business Travel).
5. Class: Travel class in the plane of the passengers (Business, Eco, Eco Plus).
6. Flight distance: The flight distance of this journey.
7. Inflight wifi service: Satisfaction level of the inflight wifi service (0:Not Applicable;1-5).
8. Departure/Arrival time convenient: Satisfaction level of Departure/Arrival time convenient.
9. Ease of Online booking: Satisfaction level of online booking.
10. Gate location: Satisfaction level of Gate location.
11. Food and drink: Satisfaction level of Food and drink.
12. Online boarding: Satisfaction level of online boarding.
13. Seat comfort: Satisfaction level of Seat comfort.
14. Inflight entertainment: Satisfaction level of inflight entertainment.
15. On-board service: Satisfaction level of On-board service.
16. Leg room service: Satisfaction level of Leg room service.
17. Baggage handling: Satisfaction level of baggage handling.
18. Check-in service: Satisfaction level of Check-in service.
19. Inflight service: Satisfaction level of inflight service.
20. Cleanliness: Satisfaction level of Cleanliness.
21. Departure Delay in Minutes: Minutes delayed when departure.
22. Arrival Delay in Minutes: Minutes delayed when Arrival.
23. Satisfaction: Airline satisfaction level(Satisfaction, neutral or dissatisfaction).
# Exploratory Data Analysis
To get understand the data, I do some Data Analysis as follows:
- Imbalance check within satified and not satisfied dataset.
- Not Available data analysis.
- Discrete Numeric Columns and Continuous Numeric Columns Analysis.
- Categorical columns analysis.
- Correlation between columns analysis.

![](https://github.com/RodzanIskandar/Airline_Passenger_satisfaction/blob/main/images/Data_Analysis1.png)
![](https://github.com/RodzanIskandar/Airline_Passenger_satisfaction/blob/main/images/Data_Analysis2.png)
![](https://github.com/RodzanIskandar/Airline_Passenger_satisfaction/blob/main/images/Data_Analysis4.png)
# Feature Engineering
- Fill the NA columns with median of NA_columns.
- Transform the 0 value (Not Applicable) with modus of the columns.
- Transform not normally distributed data to normally distributed.
- Encode string categorical column into numeric ordered by the sum of satisfaction within one categoy in each columns.
- Scale the dataset using MinMAxScaler'.
# Feature Selection
In the Feature Selection, Iam using filter methods to get the essential features to the model.
- I drop 'Departure Delay in Minutes' based on f-score in cotinuous columns and because its too corelated with 'Arrival Delay in Minutes'.
- In Categorical columns, Iam using chi squared as score function.
# Machine Learning Modeling
- Split the dataset to 80% training and 20% test.
- Compare the classification models in default setting and pick the top 3 performance using precision score in clasification problems.
- Check the performance of the best models using confusion_matrix and ROC_AUC
- Check the model fitting
- Hyperparameter tuning the model using GridSearchCV and RandomizedSearchCV
- Compare the two default models and models after tuning as the final recap.
![](https://github.com/RodzanIskandar/Airline_Passenger_satisfaction/blob/main/images/ML1.png)
![](https://github.com/RodzanIskandar/Airline_Passenger_satisfaction/blob/main/images/ML2.png)
![](https://github.com/RodzanIskandar/Airline_Passenger_satisfaction/blob/main/images/ML3.png)
![](https://github.com/RodzanIskandar/Airline_Passenger_satisfaction/blob/main/images/ML4.png)
