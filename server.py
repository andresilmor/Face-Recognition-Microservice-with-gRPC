# server.py
# we will use asyncio to run our service
import asyncio
import grpc

import cv2
import numpy as np

from deepface import DeepFace as df
from models import Age, Emotion, Gender, Race
from detectors import FaceDetector
import common.functions as functions
import common.distance as dst

# from the generated grpc server definition, import the required stuff
from ms_faceRecognition_pb2_grpc import FaceRecognitionService, add_FaceRecognitionServiceServicer_to_server
# import the requests and reply types
from ms_faceRecognition_pb2 import FaceRecognitionRequest, FaceRecognitionInferenceReply, Box

import logging

from io import BytesIO 
from PIL import Image
import os
import pickle

import pandas as pd
from tqdm import tqdm

import logging
from time import perf_counter


logging.basicConfig(level=logging.INFO)

class FaceRecognitionService(FaceRecognitionService):
    def __init__(self) -> None:
        self.ageModel = functions.build_model("Age")
        self.emotionModel = functions.build_model("Emotion")
        self.genderModel = functions.build_model("Gender")
        self.raceModel = functions.build_model("Race")
        self.opencvModel = FaceDetector.build_model("opencv")
        self.arcFaceModel = functions.build_model("ArcFace")
        super().__init__()

    async def Represent(self, img_path, model_name = 'VGG-Face', model = None, enforce_detection = True, detector_backend = 'opencv', align = True, normalization = 'base'):

        """
        This function represents facial images as vectors. The function uses convolutional neural networks models to generate vector embeddings.
        Parameters:
            img_path (string): exact image path. Alternatively, numpy array (BGR) or based64 encoded images could be passed.
            enforce_detection (boolean): If any face could not be detected in an image, then verify function will return exception. Set this to False not to have this exception. This might be convenient for low resolution images.
            detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd, dlib or mediapipe
            normalization (string): normalize the input image before feeding to model
        Returns:
            Represent function returns a multidimensional vector. The number of dimensions is changing based on the reference model. E.g. FaceNet returns 128 dimensional vector; VGG-Face returns 2622 dimensional vector.
        """
        resp_objs = []

        #---------------------------------
        # we started to run pre-process in verification. so, this can be skipped if it is coming from verification.
        if detector_backend != "skip":
            target_size = await functions.find_target_size(model_name=model_name)

            img_objs = await functions.extract_faces(img = img_path, 
                                    target_size = target_size,
                                    face_detector=self.opencvModel,
                                    detector_backend = detector_backend, 
                                    grayscale = False, 
                                    enforce_detection = enforce_detection, 
                                    align = align)
        else: # skip
            if type(img_path) == str:
                img = await functions.load_image(img_path)
            elif type(img_path).__module__ == np.__name__:
                img = img_path.copy()
            else:
                raise ValueError(f"unexpected type for img_path - {type(img_path)}")
            
            img_region = [0, 0, img.shape[1], img.shape[0]]
            img_objs = [(img, img_region, 0)]
        #---------------------------------

        for img, region, confidence in img_objs:
            #custom normalization
            img = await functions.normalize_input(img = img, normalization = normalization)
            
            #represent
            if "keras" in str(type(model)):
                #new tf versions show progress bar and it is annoying
                embedding = model.predict(img, verbose=0)[0].tolist()
            else:
                #SFace and Dlib are not keras models and no verbose arguments
                embedding = model.predict(img)[0].tolist()
            
            resp_obj = {}
            resp_obj["embedding"] = embedding
            resp_obj["facial_area"] = region
            resp_objs.append(resp_obj)
        return resp_objs

    async def Find(self, img_path, db_path, model = None, model_name ='VGG-Face', distance_metric = 'cosine', enforce_detection = True, detector_backend = 'opencv', align = True, normalization = 'base', silent=False):
        #-------------------------------
        if os.path.isdir(db_path) != True:
            raise ValueError("Passed db_path does not exist!")
        else:
            target_size = await functions.find_target_size(model_name=model_name)

            #---------------------------------------

            file_name = "representations_%s.pkl" % (model_name)
            file_name = file_name.replace("-", "_").lower()

            if os.path.exists(db_path+"/"+file_name):

                if not silent:
                    print("WARNING: Representations for images in ",db_path," folder were previously stored in ", file_name, ". If you added new instances after this file creation, then please delete this file and call find function again. It will create it again.")

                f = open(db_path+'/'+file_name, 'rb')
                representations = pickle.load(f)

                if not silent:
                    print("There are ", len(representations)," representations found in ",file_name)

            else: #create representation.pkl from scratch
                employees = []

                for r, d, f in os.walk(db_path): # r=root, d=directories, f = files
                    for file in f:
                        if ('.jpg' in file.lower()) or ('.jpeg' in file.lower()) or ('.png' in file.lower()):
                            exact_path = r + "/" + file
                            employees.append(exact_path)

                if len(employees) == 0:
                    raise ValueError("There is no image in ", db_path," folder! Validate .jpg or .png files exist in this path.")

                #------------------------
                #find representations for db images

                representations = []

                #for employee in employees:
                pbar = tqdm(range(0,len(employees)), desc='Finding representations', disable = True if silent == True else False)
                for index in pbar:
                    employee = employees[index]

                    img_objs = await functions.extract_faces(img = employee, 
                        target_size = target_size, 
                        detector_backend = detector_backend, 
                        grayscale = False, 
                        enforce_detection = enforce_detection, 
                        align = align
                    )

                    for img_content, img_region, img_confidence in img_objs:
                        embedding_obj = await self.Represent(img_path = img_content
                            , model = model
                            , model_name = model_name
                            , enforce_detection = enforce_detection
                            , detector_backend = "skip"
                            , align = align
                            , normalization = normalization
                            )
                        
                        img_representation = embedding_obj[0]["embedding"]

                        instance = []
                        instance.append(employee) 
                        instance.append(img_representation)
                        representations.append(instance)

                #-------------------------------

                f = open(db_path+'/'+file_name, "wb")
                pickle.dump(representations, f)
                f.close()

                if not silent: 
                    print("Representations stored in ",db_path,"/",file_name," file. Please delete this file when you add new identities in your database.")

            #----------------------------
            #now, we got representations for facial database
            df = pd.DataFrame(representations, columns = ["identity", f"{model_name}_representation"])

            # img path might have move than once face
            target_objs = await functions.extract_faces(img = img_path, 
                        target_size = target_size, 
                        face_detector = self.opencvModel,
                        detector_backend = detector_backend, 
                        grayscale = False, 
                        enforce_detection = enforce_detection, 
                        align = align
                    )
            
            resp_obj = []
    	    
            for target_img, target_region, target_confidence in target_objs:
                target_embedding_obj = await self.Represent(img_path = target_img
                            , model = model
                            , model_name = model_name
                            , enforce_detection = enforce_detection
                            , detector_backend = "skip"
                            , align = align
                            , normalization = normalization
                            )
                
                target_representation = target_embedding_obj[0]["embedding"]

                result_df = df.copy() #df will be filtered in each img
                result_df["source_x"] = target_region["x"]
                result_df["source_y"] = target_region["y"]
                result_df["source_w"] = target_region["w"]
                result_df["source_h"] = target_region["h"]
                
                
                distances = []
                for index, instance in df.iterrows():
                    source_representation = instance[f"{model_name}_representation"]

                    if distance_metric == 'cosine':
                        distance = dst.findCosineDistance(source_representation, target_representation)
                    elif distance_metric == 'euclidean':
                        distance = dst.findEuclideanDistance(source_representation, target_representation)
                    elif distance_metric == 'euclidean_l2':
                        distance = dst.findEuclideanDistance(dst.l2_normalize(source_representation), dst.l2_normalize(target_representation))
                    else:
                        raise ValueError(f"invalid distance metric passes - {distance_metric}")

                    distances.append(distance)

                    #---------------------------

                result_df[f"{model_name}_{distance_metric}"] = distances

                threshold = dst.findThreshold(model_name, distance_metric)
                result_df = result_df.drop(columns = [f"{model_name}_representation"])
                result_df = result_df[result_df[f"{model_name}_{distance_metric}"] <= threshold]
                result_df = result_df.sort_values(by = [f"{model_name}_{distance_metric}"], ascending=True).reset_index(drop=True)

                resp_obj.append(result_df)

            # -----------------------------------
            
            return resp_obj

    async def Analyze(self, img_path, actions = ('emotion', 'age', 'gender', 'race') , enforce_detection = True, detector_backend = 'opencv', align = True, silent = False):
        #---------------------------------
        # validate actions 
        if type(actions) == str:
            actions = (actions,)

        actions = list(actions)
        #---------------------------------
        # build models
        models = {}
        if 'emotion' in actions:
            models['emotion'] = self.emotionModel

        if 'age' in actions:
            models['age'] = self.ageModel

        if 'gender' in actions:
            models['gender'] = self.genderModel

        if 'race' in actions:
            models['race'] = self.raceModel
        #---------------------------------
        resp_objects = []
        img_objs = await functions.extract_faces(img=img_path, target_size=(224, 224), face_detector=self.opencvModel, detector_backend=detector_backend, grayscale = False, enforce_detection=enforce_detection, align=align)
        
        
        for img_content, img_region, img_confidence in img_objs:
            if img_content.shape[0] > 0 and img_content.shape[1] > 0:
                obj = {}
                #facial attribute analysis
                pbar = tqdm(range(0, len(actions)), desc='Finding actions', disable = silent)
                for index in pbar:
                    action = actions[index]
                    #pbar.set_description("Action: %s" % (action))

                    if action == 'emotion':
                        img_gray = cv2.cvtColor(img_content[0], cv2.COLOR_BGR2GRAY)
                        img_gray = cv2.resize(img_gray, (48, 48))
                        img_gray = np.expand_dims(img_gray, axis = 0)
                        
                        
                        emotion_predictions = models['emotion'].predict(img_gray, verbose=0)[0,:]

                        sum_of_predictions = emotion_predictions.sum()

                        obj["emotion"] = {}
                        emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

                        for i in range(0, len(emotion_labels)):
                            emotion_label = emotion_labels[i]
                            emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
                            obj["emotion"][emotion_label] = emotion_prediction

                        obj["dominant_emotion"] = emotion_labels[np.argmax(emotion_predictions)]

                    elif action == 'age':
                        age_predictions = models['age'].predict(img_content, verbose=0)[0,:]
                        apparent_age = Age.findApparentAge(age_predictions)
                        obj["age"] = int(apparent_age) #int cast is for the exception - object of type 'float32' is not JSON serializable

                    elif action == 'gender':
                        gender_predictions = models['gender'].predict(img_content, verbose=0)[0,:]
                        gender_labels = ["Woman", "Man"]
                        obj["gender"] = {}
                        for i, gender_label in enumerate(gender_labels):
                            gender_prediction = 100 * gender_predictions[i]
                            obj["gender"][gender_label] = gender_prediction

                        obj["dominant_gender"] = gender_labels[np.argmax(gender_predictions)]

                    elif action == 'race':
                        race_predictions = models['race'].predict(img_content, verbose=0)[0,:]
                        sum_of_predictions = race_predictions.sum()

                        obj["race"] = {}
                        race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']
                        for i in range(0, len(race_labels)):
                            race_label = race_labels[i]
                            race_prediction = 100 * race_predictions[i] / sum_of_predictions
                            obj["race"][race_label] = race_prediction

                        obj["dominant_race"] = race_labels[np.argmax(race_predictions)]

                    #-----------------------------
                    # mention facial areas
                    obj["region"] = img_region
                    
                resp_objects.append(obj)
        
        return resp_objects	

    async def Inference(self, request: FaceRecognitionRequest, context) -> FaceRecognitionInferenceReply:
        start = perf_counter()

        npimg = cv2.imdecode(np.frombuffer(request.image, np.uint8), -1)

        cv2.rectangle(npimg, ( 0, 0 ), (200, 856) , (0, 0, 0), -1) # |o
        cv2.rectangle(npimg, ( 1300, 0 ), (1504, 856) , (0, 0, 0), -1) # o|
        cv2.rectangle(npimg, ( 200, 720 ), (1300, 856) , (0, 0, 0), -1) # _

        #cv2.imwrite(os.getcwd() + "/frame4.png",npimg)
        
        try: 
            resp = await self.Find(img_path=npimg[request.personBox['y1']:request.personBox['y2'], request.personBox['x1']:request.personBox['x2']],
                       db_path='database/', model= self.arcFaceModel, model_name='ArcFace', enforce_detection=True, silent=True)
            
        except Exception as ex:
            print(ex)
            return FaceRecognitionInferenceReply(uuid=None, box=None)

        try: 
            obj = await self.Analyze(img_path=npimg[request.personBox['y1']:request.personBox['y2'], request.personBox['x1']:request.personBox['x2']], 
                    actions = ['emotion'],
                    silent=True
            )     
        except Exception as ex:
            print(ex)
            return FaceRecognitionInferenceReply(uuid=None, box=None)

        x1 = request.personBox['x1'] + obj[0]["region"]["x"]
        x2 = x1 + obj[0]["region"]["w"]
        y1 = request.personBox['y1'] + obj[0]["region"]["y"]
        y2 = y1 + obj[0]["region"]["h"]
        identity = str(resp[0].iloc[:1]['identity'])
        id = identity[identity.find("/") + 1: identity.rfind("/")] 
            
        #faceCenterX =  int(request.personBox['x1'] + ((x2 - x1) / 2))

        #faceCenterY = int(request.personBox['y1'] + ((y2 - y1) / 2))
        #cv2.rectangle(npimg, (x1, y1,), (x2, y2), (0,0,255), 2)
        #cv2.imwrite(os.getcwd() + "/frame2.png", npimg)

        #im = np.array(Image.open(BytesIO(bytearray(request.image))))
        #im = Image.fromarray(im)
        #im.save(os.getcwd() + "/frame2.png")
        #npimg = cv2.imdecode(np.frombuffer(request.image, np.uint8), -1)

        logging.info(
            f"[✅] In {(perf_counter() - start) * 1000:.2f}ms"
        )


        return FaceRecognitionInferenceReply(uuid=id, box=Box(x1=x1, x2=x2, y1=y1, y2=y2))



async def serve():
    server = grpc.aio.server()
    add_FaceRecognitionServiceServicer_to_server(FaceRecognitionService(), server)
    # using ip v6
    adddress = "[::]:50054"
    server.add_insecure_port(adddress)
    logging.info(f"[📡] Starting server on {adddress}")
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())