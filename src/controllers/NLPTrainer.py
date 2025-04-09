# from src.helpers.config import STORAGE_FOLDER_PATH
# import joblib
# import os
# import json
# from typing import List , Union , Dict
# from datetime import datetime
# from threading import Thread , get_ident
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import classification_report 
# from sklearn.linear_model import LogisticRegression




# class NLPTrainer :
#     def __init__(self) -> None :
#         self._storage_path = STORAGE_FOLDER_PATH
#         if not os.path.exists(self._storage_path) :
#             os.makedirs(self._storage_path)

#         self._status_path = os.path.join(self._storage_path , "model_status.json")
#         self._model_path = os.path.join(self._storage_path , "model_pickle.joblib")

#         # check for status file 
#         if os.path.exists(self._status_path) :
#             with open(self._status_path) as f :
#                 self._model_status = json.load(f)

        
#         else :
#             self._model_status = {"status" : "No Model Found " , 
#                                   "timestamp" : datetime.now().isoformat() , 
#                                   "Classes" : [] , 
#                                   "Evaluation" : {} }

#         # Check Model 
#         if os.path.exists(self._model_path) :
#             self.model = joblib.load(self._model_path)

#         else :
#             self.model = None


#         self._running_threads = []
#         self._pipeline = None





    # def _update_status(self , status : str , Classes : List[str] = [] , Evaluation : Dict = {}) -> None :
    #     self.model_status["status"] = status
    #     self.model_status["timestamp"] = datetime.now().isoformat()
    #     self.model_status["Classes"] = Classes
    #     self.model_status["Evaluation"] = Evaluation

    #     with open(self._status_path , "w+") as f :
    #         json.dump(self.model_status , f , indent=2)



    # def _train_job(self , X_train : List[str] , y_train : List[Union[str , int]] , 
    #                X_test : List[str] , y_test :  List[Union[str , int]]) :
        

    #     self._pipeline.fit(X_train , y_train)
    #     report = classification_report(y_test , self._pipeline.predict(X_test) , output_dict=True , zero_division=0)
    #     classes = self._pipeline.classes_.tolist()


    #     self._update_status(status = "Model Ready" , classes = classes , Evaluation = report)
    #     joblib.dump(self._pipeline , self._model_path , compress=9)


    #     self.model = self._pipeline
    #     self._pipeline = None


    #     # Remove Completed Thread 
    #     thread_id = get_ident()
    #     for i , t in enumerate(self._running_threads) :
    #         if t.ident == thread_id :
    #             self._running_threads.pop(i)
    #             break



    # def train(self , texts : List[str] , labels :List[Union[str , int]]) -> None :

    #     # Split & Train 
    #     X_train , X_test , y_train , y_test = train_test_split(texts , labels)
    #     clf = LogisticRegression()
    #     vec = TfidfVectorizer(stop_words="english" , 
    #                           min_df=0.01 , max_df=0.35 , ngram_range=(1,2))
        
    #     self._pipeline = make_pipeline(vec , clf)
    #     self.model = None
    #     self._update_status(status = "Training")



    #     t = Thread(target=self._train_job , args=(X_train , X_test , y_train , y_test))
    #     self._running_threads.append(t)
    #     t.start()




    # def predict(self , texts : List[str]) -> List[Dict] :
    #     response = []
    #     if self.model :
    #         probs = self.model.predict_proba(texts) 
    #         for i ,row in enumerate(probs) :
    #             row_pred = {}
    #             row_pred["text"] = texts[i]
    #             row_pred["predictions"] = {cls : round(float(prob) , 3) for cls , prob in zip(self.model_status["calsses"])}
    #             response.append(row_pred)

    #     else :
    #         raise Exception("NO Thread Model Was Found .")
    #     return response



    # def get_status(self) -> Dict:
    #     return self.model_status

























from src.helpers.config import STORAGE_FOLDER_PATH
import joblib
import os
import json
from typing import List, Union, Dict
from datetime import datetime
from threading import Thread, get_ident
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression





class NLPTrainer:
    def __init__(self) -> None:
        self._storage_path = STORAGE_FOLDER_PATH
        if not os.path.exists(self._storage_path):
            os.makedirs(self._storage_path)

        self._status_path = os.path.join(self._storage_path, "model_status.json")
        self._model_path = os.path.join(self._storage_path, "model_pickle.joblib")

        # Check for status file
        if os.path.exists(self._status_path):
            with open(self._status_path) as f:
                self._model_status = json.load(f)
        else:
            self._model_status = {
                "status": "No Model Found",
                "timestamp": datetime.now().isoformat(),
                "Classes": [],
                "Evaluation": {}
            }

        # Check Model
        if os.path.exists(self._model_path):
            self.model = joblib.load(self._model_path)
        else:
            self.model = None

        self._running_threads = []
        self._pipeline = None




    def _update_status(self, status: str, Classes: List[str] = [], Evaluation: Dict = {}) -> None:
        """Update the model status and save it to the JSON file."""
        self._model_status["status"] = status
        self._model_status["timestamp"] = datetime.now().isoformat()
        self._model_status["Classes"] = Classes
        self._model_status["Evaluation"] = Evaluation

        with open(self._status_path, "w+") as f:
            json.dump(self._model_status, f, indent=2)




    def _train_job(self, X_train: List[str], y_train: List[Union[str, int]],
                   X_test: List[str], y_test: List[Union[str, int]]) -> None:
        """Train the model in a separate thread."""
        self._pipeline.fit(X_train, y_train)
        report = classification_report(
            y_test, self._pipeline.predict(X_test), output_dict=True, zero_division=0
        )
        classes = self._pipeline.classes_.tolist()

        self._update_status(status="Model Ready", Classes=classes, Evaluation=report)
        joblib.dump(self._pipeline, self._model_path, compress=9)

        self.model = self._pipeline
        self._pipeline = None

        # Remove completed thread
        thread_id = get_ident()
        self._running_threads = [t for t in self._running_threads if t.ident != thread_id]




    def train(self, texts: List[str], labels: List[Union[str, int]]) -> None:
        
        X_train, X_test, y_train, y_test = train_test_split(texts, labels)
        clf = LogisticRegression()
        vec = TfidfVectorizer(stop_words="english", min_df=0.01, max_df=0.35, ngram_range=(1, 2))

        self._pipeline = make_pipeline(vec, clf)
        self.model = None
        self._update_status(status="Training")

        t = Thread(target=self._train_job, args=(X_train, y_train, X_test, y_test))
        self._running_threads.append(t)
        t.start()





    def predict(self, texts: List[str]) -> List[Dict]:

        if self.model is None:
            raise Exception("No trained model is available. Please train the model first.")

        response = []
        probs = self.model.predict_proba(texts)
        for i, row in enumerate(probs):
            row_pred = {
                "text": texts[i],
                "predictions": {cls: round(float(prob), 3) for cls, prob in zip(self._model_status["Classes"], row)}
            }
            response.append(row_pred)

        return response




def get_status(self) -> Dict:
    
        return self._model_status