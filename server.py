# server.py
# we will use asyncio to run our service
import asyncio
import grpc

import cv2
import numpy as np


from deepface import DeepFace as df
from deepface.detectors import FaceDetector as fd
import tensorflow as tf

import detectors.RetinaFaceWrapper as RetinaFaceWrapper
import detectors.OpenCvWrapper as OpenCvWrapper


# --------------------------------------------------
# configurations of dependencies

tf_version = tf.__version__
tf_major_version = int(tf_version.split(".", maxsplit=1)[0])
tf_minor_version = int(tf_version.split(".")[1])

if tf_major_version == 1:
    from keras.preprocessing import image
elif tf_major_version == 2:
    from tensorflow.keras.preprocessing import image

# --------------------------------------------------

from pymongo import MongoClient

# from the generated grpc server definition, import the required stuff
from ms_faceRecognition_pb2_grpc import FaceRecognitionService, add_FaceRecognitionServiceServicer_to_server
# import the requests and reply types
from ms_faceRecognition_pb2 import FaceRecognitionRequest, FaceRecognitionInferenceReply, FaceRecognitionWithRectListRequest, FaceRect, FaceRecognized

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

connection = "mongodb+srv://andresilmor:project-greenhealth@carexr-mongodb-cluster.mcnxmz7.mongodb.net/?retryWrites=true&w=majority"

class FaceRecognitionService(FaceRecognitionService):
    def __init__(self) -> None:
        super().__init__()
        self.dbClient = MongoClient(connection)
        model = df.build_model("ArcFace")
        self.arcFaceModel = model
        self.retinafaceDetector = fd.build_model("retinaface")
        self.opencvDetector = fd.build_model("opencv")
        self.database = "FaceDB"


        

# -------------------------------------------------------------------------------------------------
#                               DeepFace modified
# -------------------------------------------------------------------------------------------------

    async def NormalizeInput(self, img):

        # @trevorgribble and @davedgd contributed this feature
        # restore input in scale of [0, 255] because it was normalized in scale of
        # [0, 1] in preprocess_face
        img = img * 255

        img = img - 127.5
        img = img / 128
    
        return img

    async def ExtractFaces(self, img, target_size=(224, 224), detectionModel=None, grayscale=False, enforce_detection=True, align=True):
        print("ExtractFaces")
        # this is going to store a list of img itself (numpy), it region and confidence
        extracted_faces = []     
        
       
        # img might be path, base64 or numpy array. Convert it to numpy whatever it is.
        
        img_region = [0, 0, img.shape[1], img.shape[0]]

        face_objs = []
        if detectionModel == None:
            face_objs = [(img, img_region, 0)]
        else:
            if (detectionModel == self.retinafaceDetector):
                print("Its equal to retinaface")
                face_objs = await RetinaFaceWrapper.detect_face(detectionModel, img, align)
            elif (detectionModel == self.opencvDetector):
                face_objs = await OpenCvWrapper.detect_face(detectionModel, img, align)

   
        # in case of no face found
        if len(face_objs) == 0 and enforce_detection is True:
            raise ValueError(
                "Face could not be detected. Please confirm that the picture is a face photo "
                + "or consider to set enforce_detection param to False."
            )

        if len(face_objs) == 0 and enforce_detection is False:
            face_objs = [(img, img_region, 0)]

        for current_img, current_region, confidence in face_objs:
            if current_img.shape[0] > 0 and current_img.shape[1] > 0:

                if grayscale is True:
                    current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

                # resize and padding
                if current_img.shape[0] > 0 and current_img.shape[1] > 0:
                    factor_0 = target_size[0] / current_img.shape[0]
                    factor_1 = target_size[1] / current_img.shape[1]
                    factor = min(factor_0, factor_1)

                    dsize = (int(current_img.shape[1] * factor), int(current_img.shape[0] * factor))
                    current_img = cv2.resize(current_img, dsize)

                    diff_0 = target_size[0] - current_img.shape[0]
                    diff_1 = target_size[1] - current_img.shape[1]
                    if grayscale is False:
                        # Put the base image in the middle of the padded image
                        current_img = np.pad(
                            current_img,
                            (
                                (diff_0 // 2, diff_0 - diff_0 // 2),
                                (diff_1 // 2, diff_1 - diff_1 // 2),
                                (0, 0),
                            ),
                            "constant",
                        )
                    else:
                        current_img = np.pad(
                            current_img,
                            ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)),
                            "constant",
                        )

                # double check: if target image is not still the same size with target.
                if current_img.shape[0:2] != target_size:
                    current_img = cv2.resize(current_img, target_size)

                # normalizing the image pixels
                img_pixels = image.img_to_array(current_img)  # what this line doing? must?
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255  # normalize input in [0, 1]

                # int cast is for the exception - object of type 'float32' is not JSON serializable
                region_obj = {
                    "x": int(current_region[0]),
                    "y": int(current_region[1]),
                    "w": int(current_region[2]),
                    "h": int(current_region[3]),
                }

                extracted_face = [img_pixels, region_obj, confidence]
                extracted_faces.append(extracted_face)

        if len(extracted_faces) == 0 and enforce_detection == True:
            raise ValueError(
                f"Detected face shape is {img.shape}. Consider to set enforce_detection arg to False."
            )
        
        return extracted_faces

    async def Represent(self, npimg, recognitionModel, enforce_detection=True, detectionModel=None, align=True, normalization="base"):

        resp_objs = []
        print("Represent")
        target_size = (112, 112) # ArcFace, on change, check functions.find_target_size
        
        if detectionModel != None:
            img_objs = await self.ExtractFaces(
                img=npimg,
                target_size=target_size,
                detectionModel=detectionModel,
                grayscale=False,
                enforce_detection=enforce_detection,
                align=align,
            )
            
        else:  # skip
            img = npimg.copy()
            # --------------------------------
            if len(img.shape) == 4:
                img = img[0]  # e.g. (1, 224, 224, 3) to (224, 224, 3)
            if len(img.shape) == 3:
                img = cv2.resize(img, target_size)
                img = np.expand_dims(img, axis=0)
            # --------------------------------
            img_region = [0, 0, img.shape[1], img.shape[0]]
            img_objs = [(img, img_region, 0)]
        # ---------------------------------
        print(img_objs)
        for img, region, _ in img_objs:
            # custom normalization
            img = await self.NormalizeInput(img=img)
            print(type(img))
            # represent
            if "keras" in str(type(recognitionModel)):
                # new tf versions show progress bar and it is annoying
                embedding = recognitionModel.predict(img, verbose=0)[0].tolist()
            else:
                # SFace and Dlib are not keras models and no verbose arguments
                embedding = recognitionModel.predict(img)[0].tolist()

            resp_obj = {}
            resp_obj["embedding"] = embedding
            resp_obj["facial_area"] = region
            resp_objs.append(resp_obj)
        
        return resp_objs


# -------------------------------------------------------------------------------------------------
#                               "Utility"
# -------------------------------------------------------------------------------------------------


    async def QueryDatabase(self, collection, targets):
        return self.dbClient[self.database][collection].aggregate([
            {
                "$addFields": {
                    "target_embeddings": targets
                }
            },
            {
                "$unwind": "$embeddings"
            },
            {
                "$unwind": "$target_embeddings"
            },
            {
                "$addFields": {
                    "dot_product": {
                        "$reduce": {
                            "input": {
                                "$zip": {
                                    "inputs": ["$embeddings.embedding", "$target_embeddings.embedding"]
                                }
                            },
                            "initialValue": 0,
                            "in": {
                                "$add": [
                                    "$$value",
                                    {
                                        "$multiply": [
                                            { "$arrayElemAt": ["$$this", 0] },
                                            { "$arrayElemAt": ["$$this", 1] }
                                        ]
                                    }
                                ]
                            }
                        }
                    },
                    "magnitude_product": {
                        "$sqrt": {
                            "$multiply": [
                                { "$sqrt": { "$sum": { "$map": { "input": "$embeddings.embedding", "as": "e", "in": { "$pow": ["$$e", 2] } } } } },
                                { "$sqrt": { "$sum": { "$map": { "input": "$target_embeddings.embedding", "as": "t", "in": { "$pow": ["$$t", 2] } } } } }
                            ]
                        }
                    }
                }
            },
            {
                "$addFields": {
                    "similarity": {
                        "$divide": ["$dot_product", "$magnitude_product"]
                    }
                }
            },
            {
                "$match": {
                    "similarity": { "$gte": 0.75 }
                }
            },
            {
                "$sort": {
                    "similarity": -1
                }
            },
            {
                "$group": {
                    "_id": {
                        "uuid": "$uuid",
                        "target_id": "$target_embeddings.id"
                    },
                    "max_similarity": { "$first": "$similarity" },
                    "count": { "$sum": 1 }
                }
            },
            {
                "$match": {
                    "count": { "$gt": 2 }
                }
            },
            {
                "$group": {
                    "_id": None,
                    "results": {
                        "$push": {
                            "uuid": "$_id.uuid",
                            "target_id": "$_id.target_id"
                        }
                    }
                }
            },
            {
                "$unwind": "$results"
            },
            {
                "$replaceRoot": {
                    "newRoot": "$results"
                }
            },
            {
                "$project": {
                    "_id": 0
                }
            }
        ])

# -------------------------------------------------------------------------------------------------
#                               Services
# -------------------------------------------------------------------------------------------------

    async def AccurateInferenceWithDetection(self, request: FaceRecognitionRequest, context) -> FaceRecognitionInferenceReply:
        start = perf_counter()


        npimg = cv2.imdecode(np.frombuffer(request.image, np.uint8), -1)

        cv2.rectangle(npimg, ( 0, 0 ), (200, 856) , (0, 0, 0), -1) # |o
        cv2.rectangle(npimg, ( 1300, 0 ), (1504, 856) , (0, 0, 0), -1) # o|
        cv2.rectangle(npimg, ( 200, 720 ), (1300, 856) , (0, 0, 0), -1) # _

        #cv2.imwrite(os.getcwd() + "/frame4.png",npimg)
        
        reply = FaceRecognitionInferenceReply()
        
        try: 
            
            for collection in request.collections:
  
                target_embedding = await self.Represent(npimg = npimg, recognitionModel = self.arcFaceModel, detectionModel = self.retinafaceDetector, enforce_detection = False)
                targets = []
                count = 0
                index = 0
                result = []
                faces = {}
                embeddings = {}
                total = 0
                for target in target_embedding:
                    faces[index] = target["facial_area"]
                    embeddings[index] =  target["embedding"]
                    cv2.rectangle(npimg, (target["facial_area"]["x"], target["facial_area"]["y"]), (target["facial_area"]["x"]+target["facial_area"]["w"], target["facial_area"]["y"]+target["facial_area"]["h"]), (0, 0, 255), 2)

                    if count == 5:
                        targets.append({"id": index, "embedding" : target["embedding"]})
                        query = await self.QueryDatabase(collection="arcface_retinaface", targets=targets)
        
                        for i in query:
                            i["rect"] = faces[i["target_id"]]
                            #print( embeddings[i["target_id"]])
                            result.append(i)
                    
                        total += len(targets)
                        targets = []
                        
                        count = 0

                    else:
                        count += 1

                        #print(target)
                        targets.append({"id": index, "embedding" : target["embedding"]})

                    index += 1
                    

                total += len(targets)


                query = await self.QueryDatabase(collection=collection, targets=[])
                for i in query:
                    i["rect"] = faces[i["target_id"]]
                    #print( embeddings[i["target_id"]])
                    result.append(i)

            

            cv2.imwrite('sample_retinaFace.jpg',npimg)  

            print(result)

            
        except Exception as ex:
            print(ex)
            return reply

        for recognition in result:
            faceRect = FaceRect(x1=recognition["rect"]["x"], x2=recognition["rect"]["x"] + recognition["rect"]["w"], y1=recognition["rect"]["y"], y2=recognition["rect"]["y"]+recognition["rect"]["h"])
            faceRecon = FaceRecognized(uuid=recognition["uuid"], faceRect=faceRect) 
         
            reply.recognitions.append(faceRecon)

        print(reply)
        logging.info(
            f"[✅] In {(perf_counter() - start) * 1000:.2f}ms"
        )


        return reply
    
    async def FastInferenceWithDetection(self, request: FaceRecognitionRequest, context) -> FaceRecognitionInferenceReply:
        start = perf_counter()


        npimg = cv2.imdecode(np.frombuffer(request.image, np.uint8), -1)

        cv2.rectangle(npimg, ( 0, 0 ), (200, 856) , (0, 0, 0), -1) # |o
        cv2.rectangle(npimg, ( 1300, 0 ), (1504, 856) , (0, 0, 0), -1) # o|
        cv2.rectangle(npimg, ( 200, 720 ), (1300, 856) , (0, 0, 0), -1) # _

        #cv2.imwrite(os.getcwd() + "/frame4.png",npimg)
        
        reply = FaceRecognitionInferenceReply()
        
        try: 
            #print(request.collections)
            #print(type(self.arcFaceModel))

            for collection in request.collections:
                
                target_embedding = await self.Represent(npimg = npimg, recognitionModel = self.arcFaceModel, detectionModel = self.opencvDetector, enforce_detection = False)
                print("Detected (" + collection +"): " + str(len(target_embedding)))

                targets = []
                count = 0
                index = 0
                result = []
                faces = {}
                total = 0
                for target in target_embedding:
                    faces[index] = target["facial_area"]

                    cv2.rectangle(npimg, (target["facial_area"]["x"], target["facial_area"]["y"]), (target["facial_area"]["x"]+target["facial_area"]["w"], target["facial_area"]["y"]+target["facial_area"]["h"]), (0, 0, 255), 2)

                    print(target["facial_area"])
                    if count == 5:
                        targets.append({"id": index, "embedding" : target["embedding"]})
                        query = await self.QueryDatabase(collection="arcface_retinaface", targets=targets)
        
                        for i in query:
                            i["rect"] = faces[i["target_id"]]
                            result.append(i)
                    
                        total += len(targets)
                        targets = []
                        
                        count = 0

                    else:
                        count += 1

                        #print(target)
                        targets.append({"id": index, "embedding" : target["embedding"]})

                    index += 1
                    

                total += len(targets)


                query = await self.QueryDatabase(collection=collection, targets=[])
                for i in query:
                    i["rect"] = faces[i["target_id"]]
                    result.append(i)

                print(".")

            cv2.imwrite('sample.jpg',npimg)  
            print(result)

            
        except Exception as ex:
            print(ex)
            return reply

        for recognition in result:
            faceRect = FaceRect(x1=recognition["rect"]["x"], x2=recognition["rect"]["x"] + recognition["rect"]["w"], y1=recognition["rect"]["y"], y2=recognition["rect"]["y"]+recognition["rect"]["h"])
            faceRecon = FaceRecognized(uuid=recognition["uuid"], faceRect=faceRect) 
            reply.recognitions.append(faceRecon)

        print(reply)
        logging.info(
            f"[✅] In {(perf_counter() - start) * 1000:.2f}ms"
        )


        return reply
    
    async def InferenceWithoutDetection(self, request: FaceRecognitionWithRectListRequest, context) -> FaceRecognitionInferenceReply:
        start = perf_counter()

        print("Here")

        npimg = cv2.imdecode(np.frombuffer(request.image, np.uint8), -1)

        cv2.rectangle(npimg, ( 0, 0 ), (200, 856) , (0, 0, 0), -1) # |o
        cv2.rectangle(npimg, ( 1300, 0 ), (1504, 856) , (0, 0, 0), -1) # o|
        cv2.rectangle(npimg, ( 200, 720 ), (1300, 856) , (0, 0, 0), -1) # _
        target_size = (112, 112)
        #cv2.imwrite(os.getcwd() + "/frame4.png",npimg)
        reply = FaceRecognitionInferenceReply()

        
        resp_objs = []
        grayscale = False
        target_embedding = []
        extracted_faces = []
        count = 0 
      
        for rect in request.rectList:
            print("Shape " + str(npimg.shape[1]))
            print(type(npimg))
            npimg = Image.fromarray(npimg[rect.y1:rect.y1 + (rect.y2 - rect.y1) , rect.x1:rect.x1+(rect.x2 - rect.x1), :])
            npimg.save("your_file_" + str(count) + ".jpeg")
            print(type(npimg))
            npimg = np.array(npimg)
            print(type(npimg))
            count += 1
       
            npimg = npimg.copy()

            if len(npimg.shape) == 4:
                npimg = npimg[0]  # e.g. (1, 224, 224, 3) to (224, 224, 3)
            if len(npimg.shape) == 3:
                npimg = cv2.resize(npimg, target_size)
                npimg = np.expand_dims(npimg, axis=0)

            
            img_region = [0, 0, npimg.shape[1], npimg.shape[0]]
            img_objs = [(npimg, img_region, 0)]

            for img, region, confidence in img_objs:
                # custom normalization
                img = await self.NormalizeInput(img=img)
                model = self.arcFaceModel
                print(type(img))
                
                # represent
                if "keras" in str(type(model)):
                    # new tf versions show progress bar and it is annoying
                    embedding = model.predict(img, verbose=0)[0].tolist()
                else:
                    # SFace and Dlib are not keras models and no verbose arguments
                    embedding = model.predict(img)[0].tolist()

                resp_obj = {}
                resp_obj["embedding"] = embedding
                resp_obj["facial_area"] = region
                resp_obj["face_confidence"] = confidence
                target_embedding.append(resp_obj)
      
                
        for collection in request.collections:

                targets = []
                count = 0
                index = 0
                result = []
                faces = {}
                embeddings = {}
                total = 0
                for target in target_embedding:
                    faces[index] = target["facial_area"]
                    embeddings[index] =  target["embedding"]
                    if count == 5:
                        targets.append({"id": index, "embedding" : target["embedding"]})
                        query = await self.QueryDatabase(collection=collection, targets=targets)
        
                        for i in query:
                            i["rect"] = faces[i["target_id"]]
                            print( embeddings[i["target_id"]])
                            result.append(i)
                    
                        total += len(targets)
                        targets = []
                        
                        count = 0

                    else:
                        count += 1

                        #print(target)
                        targets.append({"id": index, "embedding" : target["embedding"]})

                    index += 1
                    

                total += len(targets)


                query = await self.QueryDatabase(collection=collection, targets=[])
                for i in query:
                    i["rect"] = faces[i["target_id"]]
                    print( embeddings[i["target_id"]])
                    result.append(i)

            


        print(result)


        '''

            for current_img, current_region, confidence in face_objs:
                if current_img.shape[0] > 0 and current_img.shape[1] > 0:
                    if grayscale is True:
                        current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

                    # resize and padding
                    if current_img.shape[0] > 0 and current_img.shape[1] > 0:
                        factor_0 = target_size[0] / current_img.shape[0]
                        factor_1 = target_size[1] / current_img.shape[1]
                        factor = min(factor_0, factor_1)

                        dsize = (
                            int(current_img.shape[1] * factor),
                            int(current_img.shape[0] * factor),
                        )
                        current_img = cv2.resize(current_img, dsize)

                        diff_0 = target_size[0] - current_img.shape[0]
                        diff_1 = target_size[1] - current_img.shape[1]
                        if grayscale is False:
                            # Put the base image in the middle of the padded image
                            current_img = np.pad(
                                current_img,
                                (
                                    (diff_0 // 2, diff_0 - diff_0 // 2),
                                    (diff_1 // 2, diff_1 - diff_1 // 2),
                                    (0, 0),
                                ),
                                "constant",
                            )
                        else:
                            current_img = np.pad(
                                current_img,
                                (
                                    (diff_0 // 2, diff_0 - diff_0 // 2),
                                    (diff_1 // 2, diff_1 - diff_1 // 2),
                                ),
                                "constant",
                            )

                    # double check: if target image is not still the same size with target.
                    if current_img.shape[0:2] != target_size:
                        current_img = cv2.resize(current_img, target_size)

                    # normalizing the image pixels
                    # what this line doing? must?
                    img_pixels = image.img_to_array(current_img)
                    img_pixels = np.expand_dims(img_pixels, axis=0)
                    img_pixels /= 255  # normalize input in [0, 1]

                    # int cast is for the exception - object of type 'float32' is not JSON serializable
                    region_obj = {
                        "x": int(current_region[0]),
                        "y": int(current_region[1]),
                        "w": int(current_region[2]),
                        "h": int(current_region[3]),
                    }

                    extracted_face = [img_pixels, region_obj, confidence]
                    extracted_faces.append(extracted_face)
            '''


            

        print(resp_objs)
        return reply
            
        

        for recognition in result:
            faceRect = FaceRect(x1=recognition["rect"]["x"], x2=recognition["rect"]["x"] + recognition["rect"]["w"], y1=recognition["rect"]["y"], y2=recognition["rect"]["y"]+recognition["rect"]["h"])
            faceRecon = FaceRecognized(uuid=recognition["uuid"], faceRect=faceRect) 
            reply.recognitions.append(faceRecon)

        print(reply)
        logging.info(
            f"[✅] In {(perf_counter() - start) * 1000:.2f}ms"
        )


        return reply

    '''
    async def Inference(self, request: FaceRecognitionRequest, context) -> FaceRecognitionInferenceReply:
        start = perf_counter()

        npimg = cv2.imdecode(np.frombuffer(request.image, np.uint8), -1)

        cv2.rectangle(npimg, ( 0, 0 ), (200, 856) , (0, 0, 0), -1) # |o
        cv2.rectangle(npimg, ( 1300, 0 ), (1504, 856) , (0, 0, 0), -1) # o|
        cv2.rectangle(npimg, ( 200, 720 ), (1300, 856) , (0, 0, 0), -1) # _

        #cv2.imwrite(os.getcwd() + "/frame4.png",npimg)
        
        try: 
            resp = await self.Find(img_path=npimg[request.personBox['y1']:request.personBox['y2'], request.personBox['x1']:request.personBox['x2']],
                       db_path='database/', model= self.arcFaceModel, model_name='ArcFace', enforce_detection=False, silent=True)
            
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

        print(obj)

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
    '''


async def serve():

    server = grpc.aio.server()
    add_FaceRecognitionServiceServicer_to_server(FaceRecognitionService(), server)
    # using ip v6
    adddress = "[::]:50080"
    server.add_insecure_port(adddress)
    logging.info(f"[📡] Starting server on {adddress}")
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())
