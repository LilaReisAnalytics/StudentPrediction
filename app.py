from flask import Flask, request, render_template, jsonify
import sys
import pandas as pd
from os import path
import joblib
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Folder and models
project_folder = r'C:\Users\User\OneDrive - Centennial College\Documents\Centennial\third semester\Neural\Group Pj\Test01'

model_path = path.join(project_folder, "saved_model.joblib")
# Load your pre-trained model
loaded_model = joblib.load(model_path)


student_df = pd.read_csv(path.join(project_folder, "df_student_cleaned.csv"))


print(loaded_model.summary())

# Extract the feature names from the columns of the DataFrame
feature_names = student_df.columns.tolist()


# Define the target column
target_column = 'First_Year_Persistence'

# Extract features (X) and target variable (y)
X = student_df.drop(target_column, axis=1)
y = student_df[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Assuming 'scaler' is the scaler used during model training
scaler = StandardScaler()

# Assuming 'X_train' is your training set
scaler.fit(X_train)

#Define the columns for one-hot encoding
columns_for_onehot_encoding = ['First_Language', 'Funding', 'Gender', 'Previous_Education', 'Age_Group', 'English_Grade']




@app.route('/')
def home():
    return render_template('index.html', feature_names=feature_names)

@app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received form data:", data)
        query_data = [data.get(feature_name) for feature_name in feature_names]
        print("Received form data:", query_data)  

        # Create a DataFrame from the received data
        query = pd.DataFrame([query_data], columns=feature_names)
        
          # Apply one-hot encoding to the new data
        query_encoded = pd.get_dummies(query, columns=columns_for_onehot_encoding, dtype=float)
        
        X_query = query_encoded.drop(target_column, axis=1)

        # Scale the new data
        X_query_scaled = scaler.transform(X_query)  

       

        # Make predictions using your model
        prediction = loaded_model.predict(X_query_scaled)

        print("Raw Prediction:", prediction)  # Add this line to print the raw prediction
        
        # Format the prediction for better readability
        formatted_prediction = "Will Persist" if prediction[0] == 1 else "Will Not Persist"


        print("Formatted Prediction:", formatted_prediction)  # Add this line to print the formatted prediction

        return jsonify({"prediction_firstYearPersistence": formatted_prediction})


    except Exception as e:
        # Log the error in a production environment
        print(f"Error: {str(e)}")
        return jsonify({'error': "An error occurred. Please try again."})




@app.route("/scores", methods=['GET', 'POST'])
def scores():
    try:
        # Assuming X_test and y_test are your test set
        loss, accuracy = loaded_model.evaluate(X_test, y_test)

        return jsonify({"accuracy": accuracy, "loss": loss})

    except Exception as e:
        # Log the error in a production environment
        print(f"Error: {str(e)}")
        return jsonify({'error': "An error occurred while calculating scores."})


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 12411

    # Load your pre-trained model
    loaded_model = joblib.load(model_path)

 

    app.run(port=port, debug=True)
