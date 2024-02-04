# Databricks notebook source
# Necessary libraries are being installed silently without verbose output
%pip install -q "accelerate>=0.16.0,<1" "transformers[torch]>=4.28.1,<5" "torch>=1.13.1,<2" mlflow langchain xformers pyngrok torchvision > /dev/null

# COMMAND ----------

# Standard library imports
import os
import platform

# Necessary imports
import pandas as pd
import mlflow
from mlflow import MlflowClient, pyfunc
from mlflow.models.signature import infer_signature
from pyngrok import ngrok

# Logging: Start logging events for the current script

import warnings
warnings.filterwarnings("ignore")


import logging

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root_logger.addHandler(handler)


# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

# for handler in logger.handlers:
#     handler.setLevel(logging.INFO)
#     handler.format('%(asctime)s - %(levelname)s - %(message)s')


# Third party imports
import pandas as pd
import mlflow
from transformers import pipeline
from mlflow import MlflowClient, pyfunc
from mlflow.models.signature import infer_signature
from pyngrok import ngrok

class DataLoader:
    """
    A class used to represent a DataLoader

    ...

    Attributes
    ----------
    file_path : str
        a string representing the path to the data file
    sample : int
        an integer representing the number of samples to take from the DataFrame

    Methods
    -------
    load_data():
        Function to load data from a file and return a random sample from the DataFrame.
    """

    def __init__(self, file_path, sample):
        """
        Parameters
        ----------
        file_path : str
            The path to the data file
        sample : int
            The number of samples to take from the DataFrame
        """

        self.file_path = file_path
        self.sample = sample

    def load_data(self):
        """
        Function to load data from a file.

        Returns
        -------
        DataFrame
            A DataFrame containing the sampled data.
        """

        # Logging the beginning of data loading
        print(f"Starting data loading from {self.file_path}")

        # Asserting the existence of the file at given path
        assert os.path.exists(self.file_path), f"File not found at {self.file_path}"

        # Load the file into a pandas DataFrame
        df = pd.read_csv(self.file_path)

        # Log the completion of data loading
        print("Data loading completed")

        # Return a random sample from the DataFrame
        return df.sample(self.sample)


class Summarizer:
    """
    A class used to represent a Text Summarizer.

    ...

    Attributes
    ----------
    model_name : str
        a formatted string to determine which Hugging Face model to use
    min_length : int
        an integer to specify minimum length of summarized text
    max_length : int
        an integer to specify maximum length of summarized text
    truncation : bool
        a boolean to decide if text needs to be truncated for summarization
    do_sample : bool
        a boolean to decide if the summarization will use sampling
    cache_dir : str
        a formatted string to specify where models will be cached

    Methods
    -------
    summarize_text(text: str)
        Returns the summarized version of the input text
    summarize_reviews(reviews: list)
        Returns a list of summarized text from a list of reviews
    """

    def __init__(self, model_name="t5-small", min_length=20, max_length=50, truncation=True, do_sample=True, cache_dir="/model_cache"):
        """
        Parameters:
        model_name (str): The name of the Hugging Face model to be used for summarization.
        min_length (int): The minimum length of the summary.
        max_length (int): The maximum length of the summary.
        truncation (bool): Whether to truncate sequences to max length.
        do_sample (bool): Whether to do sampling.
        cache_dir (str): The directory for caching models.
        """

        self.model_name = model_name
        self.min_length = min_length
        self.max_length = max_length
        self.truncation = truncation
        self.do_sample = do_sample
        self.cache_dir = cache_dir

        print("Loading Summarizer model")

        # Initializing the Hugging Face pipeline for text summarization
        self.summarizer_model = pipeline(
            task="summarization",
            model=self.model_name,
            min_length=self.min_length,
            max_length=self.max_length,
            truncation=self.truncation,
            do_sample=self.do_sample,
            model_kwargs={"cache_dir": self.cache_dir},
        )

        print("Summarizer model loaded")

    def summarize_text(self, text):
        """
        Function to summarize a given text.

        Parameters:
        text (str): The text to be summarized.

        Returns:
        str: The summarized version of the text.
        """
        print(f"Summarizing the text: {text[:50]}...")  # Printing the first 50 characters for brevity

        # Summarize the text
        summary = self.summarizer_model(text)[0]["summary_text"]

        print("Text summarized")

        return summary
    
    def summarize_reviews(self, reviews):
        """
        Function to summarize multiple reviews.

        Parameters:
        reviews (list): A list of reviews to be summarized.

        Returns:
        list: A list of summarized reviews.
        """
        print(f"Summarizing {len(reviews)} reviews")

        # Apply the summarize_text method to each review in the list of reviews
        summaries = [self.summarize_text(review) for review in reviews]

        print("Reviews summarized")

        return summaries




# Define MLflowHandler class
class MLflowHandler:
     # Initialize the class with optional experiment_name parameter
    def __init__(self, experiment_name=f"/Users/raj.20332@iimtrichy.ac.in/LLMOps"):
        """
        Create an instance of the MLflowHandler class.

        Parameters
        ----------
        experiment_name : str, optional
            Name of the experiment, by default "LLM - MLflow experiment"

        Returns
        -------
        None
        """
        # Log the initialization process
        logging.info('Initializing MLflowHandler')
        
        # Set experiment_name and run_id attributes
        self.experiment_name = experiment_name
        self.run_id = None

    # Define method to start a new run in MLflow experiment
    def start_run(self, model):
        """
        Start a new run in MLflow experiment.

        Parameters
        ----------
        model : class instance
            An instance of a model class with necessary attributes.

        Returns
        -------
        None
        """
        # Log the start of MLflow run

        logging.info('Starting MLflow run')
        
         # Check if all necessary attributes exist in model
        assert all(hasattr(model, attr) for attr in ("model_name", "min_length", "max_length", "truncation", "do_sample"))
        
        # Set the current experiment
        mlflow.set_experiment(self.experiment_name)
        
         # Start a new MLflow run and log parameters
        with mlflow.start_run(nested=True):
            # self.run_id = mlflow.active_run().info.run_id
            
            mlflow.log_params(
                {
                    "hf_model_name": model.model_name,
                    "min_length": model.min_length,
                    "max_length": model.max_length,
                    "truncation": model.truncation,
                    "do_sample": model.do_sample,
                }
            )

# Define method to log predictions and corresponding inputs to the MLflow experiment
    def log_predictions(self, inputs, outputs):
        """
        Log predictions and corresponding inputs to the MLflow experiment.

        Parameters
        ----------
        inputs : list
            List of input data.
        outputs : list
            List of output data.

        Returns
        -------
        None
        """
        
        # Log the prediction process
        logging.info('Logging predictions')
        
         # Start a new MLflow run and log parameters

        mlflow.log_params(
            {
                "average_input_length": sum(len(inp) for inp in inputs) / len(inputs),
                "average_output_length": sum(len(out) for out in outputs) / len(outputs),
                "max_input_length": max(len(inp) for inp in inputs),
                "max_output_length": max(len(out) for out in outputs),
                "min_input_length": min(len(inp) for inp in inputs),
                "min_output_length": min(len(out) for out in outputs),
            }
        )
        # Log predictions using the LLM library
        mlflow.llm.log_predictions(
            inputs=inputs, 
            outputs=outputs, 
            prompts=["summarization of reviews" for _ in outputs]
        )

    # Define method to log the model to the MLflow experiment
    def log_model(self, model, artifact_path, review):
        """
        Log the model to the MLflow experiment.

        Parameters
        ----------
        model : class instance
            An instance of a model class with necessary attributes.
        artifact_path : str
            Path for the artifact.
        review : str
            Sample review for logging with the model.

        Returns
        -------
        dict
            Information about the logged model.
        """
        
        # Log the model logging process
        logging.info('Logging model')

        # Create dataframes for input and output
        input_df = pd.DataFrame({"review": [review]})
        output_df = pd.DataFrame({"summary": model.summarize_reviews([review])})

        inputs=input_df["review"].tolist()
        outputs = output_df["summary"].tolist()

        # Infer signature from input and output
        signature = infer_signature(input_df, output_df)

         # Log parameters

        mlflow.log_params(
            {
                'Model Signature': str(signature),
                'artifact_path': artifact_path,
                'Chain_Type': "Context Chain"
            }
        )
        
        # Define inference configuration and conda environment
        inference_config = {
            "min_length": model.min_length,
            "max_length": model.max_length,
            "truncation": model.truncation,
            "do_sample": model.do_sample,
        }

        conda_env = {
            'channels': ['defaults'],
            'dependencies': [
                f'python={platform.python_version()}',
                'pip',
                {
                    'pip': [
                        'mlflow',
                        'torch',
                        'transformers',
                    ]
                }
            ],
            'name': 'mlflow-env'
        }

        # Log the model with transformers
        model_info = mlflow.transformers.log_model(
            transformers_model=model.summarizer_model,
            artifact_path=artifact_path,
            task="summarization",
            inference_config=inference_config,
            signature=signature,
            input_example="This is an example of a long news article which this pipeline can summarize for you.",
        )
        return model_info

    # Define method to register the model to the MLflow registry
    def register_model(self, model_name, model_info):
        """
        Register the model to the MLflow registry.

        Parameters
        ----------
        model_name : str
            Name of the model.
        model_info : dict
            Information about the model.

        Returns
        -------
        None
        """
        
        # Log the model registration process
        logging.info('Registering model')

        # Format model_name for registration
        model_name = model_name.replace("/", "_").replace(".", "_").replace(":", "_")
        mlflow.register_model(model_uri=model_info.model_uri, name=model_name)

    # Define method to transition a registered model to a specified stage
    def transition_model_to_stage(self, model_name, model_version, stage):
        """
        Transition a registered model to a specified stage.

        Parameters
        ----------
        model_name : str
            Name of the model.
        model_version : str
            Version of the model.
        stage : str
            The desired stage for the model.

        Returns
        -------
        None
        """
        
         # Log the model transition process
        logging.info('Transitioning model to stage')

        # Create an instance of MlflowClient
        mlflow_client = mlflow.tracking.MlflowClient()
        
        # Try to transition model version to the specified stage
        try:
            mlflow_client.transition_model_version_stage(name=model_name, version=model_version, stage=stage)
        except Exception as e:
            logging.error(f"Error transitioning model version to stage: {e}")
            raise RuntimeError(f"Error transitioning model version to stage: {e}")

    # Define method to load a model from the MLflow model registry
    def load_model(self, model_name, model_version=None, stage=None):
        """
        Load a model from the MLflow model registry.

        Parameters
        ----------
        model_name : str
            Name of the model.
        model_version : str, optional
            Version of the model, by default None
        stage : str, optional
            Stage of the model, by default None

        Returns
        -------
        class
            Class of the loaded model.
        """
        
         # Log the model loading process
        logging.info('Loading model')

        # Try to load the model from the specified stage or version
        try:
            if stage:
                return pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")
            else:
                return pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise RuntimeError(f"Error loading model: {e}")

    # Define method to search for registered models in the MLflow model registry
    def search_registered_models(self, model_name):
        """
        Search for registered models in the MLflow model registry.

        Parameters
        ----------
        model_name : str
            Name of the model.

        Returns
        -------
        list
            List of registered models.
        """
        
        # Log the model search process
        logging.info('Searching registered models')

        # Create an instance of MlflowClient
        mlflow_client = mlflow.tracking.MlflowClient()
        
        # Try to search for registered models by model_name
        try:
            return mlflow_client.search_registered_models(filter_string=f"name='{model_name}'")
        except Exception as e:
            logging.error(f"Error searching registered models: {e}")
            raise RuntimeError(f"Error searching registered models: {e}")



def run_pipeline(reviews):
    """
    This function executes a DevOps/MLOps pipeline with MLflow.

    Parameters
    ----------
    reviews : list
        List of reviews to process.
    """

    # Initialize MLflow handler
    mlflow_handler = MLflowHandler()
    
    logging.info('Pipeline initiated')

    # Create an instance of your model (change this according to your model class)
    model = Summarizer()

    # Start an MLflow run and log the model parameters
    mlflow_handler.start_run(model)
    
    logging.info('MLflow run started and model parameters logged')

    # Log the model
    artifact_path = "_pipeline/summarizer"
    model_info = mlflow_handler.log_model(model, artifact_path, reviews)
    
    logging.info('Model logged')

    # Register the model
    mlflow_handler.register_model("model summarizer", model_info)
    
    logging.info('Model registered')

    # Search for registered models
    model_results = mlflow_handler.search_registered_models("model summarizer")
    
    logging.info('Registered models searched')

    # Load model (version 1) from the model registry
    modelpyV1 = mlflow_handler.load_model("model summarizer", model_version=1)
    
    logging.info('Model Version 1 loaded')

    # Predict using the model
    modelpyV1.predict(pd.DataFrame({"review": [reviews[0]]}))
    
    logging.info('Prediction executed with Model Version 1')

    # Transition the model to 'Staging' stage
    mlflow_handler.transition_model_to_stage('model summarizer', 1, "Staging")
    
    logging.info('Model transitioned to Staging')

    # Load model from 'Staging' stage
    modelStaged = mlflow_handler.load_model("model summarizer", model_version=1, stage="Staging")
    
    logging.info('Model loaded from Staging')

    # Predict using the model from 'Staging'
    modelStaged.predict(pd.DataFrame({"review": [reviews[0]]}))
    
    logging.info('Prediction executed with Staging model')

    # Transition the model to 'Production' stage
    mlflow_handler.transition_model_to_stage('model summarizer', 1, "Production")
    
    logging.info('Model transitioned to Production')

    # Load model from 'Production' stage
    modelProduction = mlflow_handler.load_model("model summarizer", model_version=1, stage="Production")
    
    logging.info('Model loaded from Production')

    # Predict using the model from 'Production'
    predictions = modelProduction.predict(pd.DataFrame({"review": reviews}))

    mlflow_handler.log_predictions(reviews, predictions)
    
    logging.info('Prediction executed with Production model')

    logging.info('Pipeline completed successfully')



# COMMAND ----------

reviews = ['Oh dear me! Rarely has a "horror" film bored me, or made me laugh, as much as this one. After a spirited start with an intriguing premise, it descends into not much more than a slasher flick, with some supernatural and sexual asides. The usually excellent Alice Krige is wasted in this one, and the plot twists are ludicrous. Don\'t bother unless you\'re really desperate. Rating: 3/10.',
 'SPOILERS A Jewish Frodo? Yep, that\'ll be Elijah Wood again.<br /><br />Ever since the concluding part of "Lord of the Rings", Elijah Wood as Frodo has found it increasingly difficult to get away from that major role. Playing a football hooligan, a psychopath and now a young Jewish American, Wood has tried any route he can to escape this typecasting. Now, with "Everything Is Illuminated" he might finally have achieved this. Playing a role which isn\'t as radical as other efforts, he truly gets to the soul of his character. Still, it isn\'t like Wood does this alone. Aided by a magnificent adaptation by first time directer Liev Schreiber and a wonderful performance by newcomer Eugene Hutz, Wood has found a magnificent production to spread his wings. "Everything is Illuminated" is a magnificent, moving piece of cinema.<br /><br />Jonathan Safran Foer (Wood), a young American Jew, sets out to the Ukraine to find the mysterious girl who rescued his grandfather and helped him get to America. Arriving in the country, Jonathan meets the all talking, all dancing Alex (Hutz) and his racist grandfather (Boris Leskin). Travelling across the country, the three slowly learn more and more about the history and relations that Alex and Jonathan never knew existed.<br /><br />It\'s a strange feeling when the film progresses into it\'s second chapter (it is actually divided into four overall). The first part, whilst occasionally a bit funny, is mostly serious and intense. So when we are given a brief history of Alex and his family in the second part, to switch from serious to hilarious is a weird step. It doesn\'t quite work, but as the film progresses, it definitely learns it\'s lesson as this mix of humour and sadness merges finer as time passes.<br /><br />To the ultimate credit of everyone involved, as the story does continue, so do we begin to fall for the characters more and more. Elijah Wood is magnificent, Boris Leskin is so intense and strong that it raises questions why Hollywood has never properly noticed him. Most notable of all however is newcomer Eugene Hutz. Playing an intensely troubled character, Hutz is absolutely brilliant. He shows the split between his relatives and the real world with almost perfect skill, and when his character is communicating with Wood, you genuinely connect with him on a deeper level. Without Hutz, the story is so strong that the film would still be magnificent, but with him, it hits the next level.<br /><br />As a debut work for actor turned director Liev Schreiber, the story is also a brilliant piece to start. A work of passion (Schreiber\'s grandfather himself an immigrant to America), he manages to truly embrace the emotion of the content, and by presenting us with some truly beautiful scenery and some magnificent shots, he manages to really hit home. The final half hour in particular is so beautifully created, that it\'s a challenge for a tear not to form in any viewers eye. It is a moving story, and with Schreiber\'s help, it becomes even more powerful.<br /><br />Constructed with love from a passionate director, "Everything is Illuminated" is a beautiful piece. A road story with a difference, it is magnificently acted and wonderfully written. It\'s a film that everyone should see, and it is the perfect way for Elijah Wood to finally lay Frodo to rest.',
 'It\'s out of question that the real Anna Anderson was NOT Princess Anastasia. Apart from very distinctive differences in physical appearance(Anderson\'s eyes are perceivably larger, lips thicker, nose larger and turned up at the end....etc), Anderson\'s unable to speak Russian was a ridiculous tell......That\'s why I detest Anna Anderson and her confederates so much. Not a lot of swindlers have the audacity and endurance to scam for 60+ years with such a blatantly untenable scheme.<br /><br />Yet to some extent I have sympathy for Anna Anderson. Life must have been hard for a young Polish peasant worker in those days. And to impersonate another woman for 60+ years is an arduous task for anybody.She had to hold back her fleshy lips all the time to mimic the thin lips of Anastasia\'s, and had to occasionally go lunatic to make people believe all her chaotic memory was just a result of mental problem.<br /><br />Anna Anderson was an awesome woman on a wrong track. Had she put her good-looks, learned elegance, endurance, acting skills into proper use, she could of made a first-class actress.<br /><br />On a side note: Some main characters of this two-parter seem to be loosely based on real figures. Prince Erich could be a mixture of Gleb Botkin(believed by many the most possible brain behind the whole scheme), Duke George and Dmitri of Leuchtenberg, and several other figures. And Darya Romanoff seem to be based on the gorgeous Princess Xenia Georgievna Romanova. But unlike the real confederates, Prince Erich was motiveless in this show and supported Anna out of love for and sincere belief in her, which is touching.<br /><br />On the whole this is a great show. Fictionalised a bit but still remains faithful to the reality. The power of Amy Irving\'s acting lies in that she successfully represented Anderson\'s self-assuredness, the mixture of impersonating others and being herself is intriguing. Just as Princess Xenia said about Anderson:"She was herself at all times and never gave the slightest impression of acting a part." Highly recommended.',
 'This 1925 film narrates the story of the mutiny on board battleship Potemkin at the port of Odessa. The movie celebrated the 20th anniversary of the uprising of 1905, which was seen as a direct precursor to the October Revolution of 1917. Following his montage theory, Eisenstein plays with scenes, their duration and the way they combine to emphasize his message, besides he uses different camera shot angles and revolutionary illumination techniques. The "Odessa Steps" sequence in Potemkin is one of the most famous in the history of cinema. The baby carriage coming loose down the steps after its mother has been shot was later recreated in Brian d\' Palma\'s The Untouchables. It is clear that the film is one of the best ever made considering its time and how innovative it was though you need a little bit of patience and to be a real movie enthusiast to go through its 70 minutes.',
 "Redo the Oscars from 1992, and this film might get nominated, or even win. It was SO good at capturing its era and dual cultures that it belongs in American and Japanese time capsules. If you wanted to know what living here or there was like back then, this film will show you. As an American, you'll feel like you tagged along for an extended Japanese vacation, and by the end of the film, you'll be a die-hard Dragons fan, as you accept the injection of Japanese tradition and culture into their baseball, much as we have done with our culture in our own game.<br /><br />Jack Elliot (Tom Selleck) is a slumping, aging Detroit Tigers' slugger who is traded to the Dragons, perennial runners-up to the dynastic Yomuri Giants, Japan's answer to the Yankees. The Giants are admired for their success, yet that success also has everyone wanting to surpass them, something which is rarely done. The Dragons' manager recruits Jack as the final piece of the pennant-winning puzzle, and we're left with what could have been Gung Ho on a baseball field, but instead was much more.<br /><br />The casting was outstanding: Selleck proved that with a good script and a character that suits him, he can carry a film as well as he did his television show, and the Japanese cast was equally good, down to Mr. Takagi from Die Hard back as the image-conscious owner. The other actors, including the one who plays the love interest (also the manager's daughter), strong and independent yet simultaneously a believer in Japanese traditions, beyond what was forced on her. She is a proper and supportive girlfriend for Jack. Even her father never tells her not to see him, almost sympathizing with Jack for what he endures from her, and a bit relieved he at least knows the man she has chosen to love.<br /><br />The baseball scenes are great, bolstered immensely by a pre-fame Dennis Haysbert as another American ex-patriate and Jack's western mentor. The usual fish-out-of-water elements are there, and you can almost feel yourself stumbling right along with Jack to fit into a country that doesn't speak our language, and doesn't practice our ways, yet copies everything we do, including our national pastime. one of the funnier scenes occurs when Jack, clutching a magazine, informs his manager that he has learned of the tradition in Japan where you can get drunk and tell off your boss, and it can't be used against you, and exercises that right very humorously. The plots and subplots are tied up neatly at the end, but not too neatly, and nothing concludes unrealistically.<br /><br />To call this a comedy is misguided: it's a pure comedy-drama, or even a drama with good humor. The plot is too deep to dismiss it the way it was by critics as an actor out of his league trying to carry a lightweight film. The situations were amusing, but in their place against a far more serious, profound, and precisely detailed backdrop that results in one of the best films I've ever seen. The baseball cinematography rivals that of For Love Of The Game, for realism.<br /><br />Some say the film is about baseball, or about Japan, but more than anything it seems to be about the workplace, and how people arrive at work from totally different origins, with different agendas, and somehow have to put their differences aside for the good of the company, or the team.<br /><br />A truly great film that never should have had to apologize for itself the way it did when it was in theaters."]

# COMMAND ----------

run_pipeline(reviews)

# COMMAND ----------


