# Disaster Response Pipeline Project
This project first sets up a ETL pipeline to read and clean the disaster responses data. It then uses both NLP and ML algorithms to train a ML model to predict the category of a disaster response message. Lastly, it builds a web app to showcase this model.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

### Notes:
1. This classifier (based on a random forest model) can easily have a large file size even after compression. We need to make a tradeoff between performance and model size.

2. This repository only provides a minimal model. To achieve the best performance, better to do a more thorough grid search.

3. If we do need to store a large model, [this service](https://git-lfs.github.com/) can come in handy.