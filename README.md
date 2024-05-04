# Disaster-Response-Pipeline

### Instructions:
This project is a component of the Udacity Data Scientist Nanodegree. Its objective is to construct a Natural Language Processing (NLP) model capable of categorizing messages dispatched during disaster situations. The model is trained on a dataset comprising authentic messages transmitted during past disaster occurrences. Subsequently, the model is deployed to classify incoming messages during similar events.


1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Visualization
   ![screencapture-192-168-2-21-3001-2024-05-04-10_54_11](https://github.com/anhtran192/Disaster-Response-Pipeline/assets/147739264/d833dab6-a567-4356-af6b-7b030a107e61)
