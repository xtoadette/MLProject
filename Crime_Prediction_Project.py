#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import necessary packages
import sys
from easygui import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# Load the dataset
data = pd.read_csv('2022.csv')


# check the shape of data
# print('Shape of data:',data.shape)

# check the unique categories of crime
# print(data["text_general_code"].value_counts())


# Drop rows with missing values if any
data.dropna(inplace=True)


# Encode the 'text_general_code' column to numeric labels for crime prediction
crime_label_encoder = LabelEncoder()
data['text_general_code'] = crime_label_encoder.fit_transform(data['text_general_code'])


# Create a mapping of original street names to numeric values for safety rating
location_block_rating_mapping = data.groupby('location_block')['text_general_code'].mean().round(2).to_dict()


# Calculate the minimum and maximum safety ratings
min_rating = min(location_block_rating_mapping.values())
max_rating = max(location_block_rating_mapping.values())


# Convert safety ratings to a scale between 1 and 10
for location_block, rating in location_block_rating_mapping.items():
    new_rating = 1 + (rating - min_rating) * 9 / (max_rating - min_rating)
    location_block_rating_mapping[location_block] = new_rating


# Create a new DataFrame for training the safety rating model
safety_rating_data = pd.DataFrame({'location_block': list(location_block_rating_mapping.keys()),
                                   'safety_rating': list(location_block_rating_mapping.values())})

# Split the data into training and testing sets
X = safety_rating_data['location_block']
y = safety_rating_data['safety_rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Convert 'location_block' to numeric labels for model training
location_block_label_encoder = LabelEncoder()
location_block_label_encoder.fit(X_train)

X_train_encoded = location_block_label_encoder.transform(X_train)


# Train the Random Forest Regressor for safety rating prediction
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train_encoded.reshape(-1, 1), y_train)


# Function to predict safety rating based on 'location_block'
def predict_safety_rating(street_name):
    
    # Preprocess the user input
    street_name = street_name.strip()

    # Encode the user input for prediction
    X_user_input_encoded = location_block_label_encoder.transform([street_name])

    # Predict the safety rating using the trained model
    predicted_safety_rating = regressor.predict(X_user_input_encoded.reshape(-1, 1))[0]

    return predicted_safety_rating


# Function to assign safety recommendation based on the safety rating
def assign_safety_recommendation(safety_rating):
    
    if safety_rating >= 9:
        return 'Safe'
    
    elif safety_rating >= 7.5:
        return 'Neutral'
    
    elif safety_rating >= 5.5:
        return 'Unsafe'
    
    else:
        return 'Dangerous'


# Function to display the breakdown of crime for a given location
def display_breakdown_of_crime(street_name):
    
    # Preprocess the user input
    street_name = street_name.strip()

    # Fill missing values in the 'location_block' column with an empty string
    data['location_block'].fillna('', inplace=True)

    # Filter data for the given street
    street_data = data[data['location_block'].str.contains(street_name, case=False)]

    street_data['text_general_code'] = crime_label_encoder.inverse_transform(street_data['text_general_code'])

#     print(street_data)

    if not street_data.empty:
        
        # Calculate crime breakdown
        crime_breakdown = street_data['text_general_code'].value_counts()

        # Plot a pie chart for crime breakdown
        plt.figure(figsize=(10, 6))
        
        plt.pie(crime_breakdown, labels=crime_breakdown.index, autopct='%1.1f%%', startangle=140,
                textprops={'fontsize': 12})  # Set fontsize for the breakdown percentages
        
        plt.axis('equal')
        
        plt.title('Crime Breakdown for ' + street_name)
        
        plt.show()

    else:
        print(f"\nNo data available for the given street...")


# Main function
def main():

    while True:
        message = "Enter the exact name of the street e.g  5700 BLOCK ASHLAND AV , and press OK"
        title = "Crime Predictions"
        d_text = ""
        output = enterbox(message, title, d_text)
        street_name = output.strip()

        if street_name:
            try:

                safety_rating = predict_safety_rating(street_name)

                if safety_rating > 0:

                    safety_recommendation = assign_safety_recommendation(safety_rating)

                    first_line = "Safety rating for " + street_name + ": " + str(safety_rating) + "\nSafety recommendation for " + street_name + ": " + str(safety_recommendation)

                    msg = msgbox(first_line, "Results")

                    display_breakdown_of_crime(street_name)

                else:
                    print(f"\nNo data available for the given street!!")

            except:
                print(f"\nNo data available for the given street!!")
        else:
            sys.exit()

            
if __name__ == "__main__":
    main()

