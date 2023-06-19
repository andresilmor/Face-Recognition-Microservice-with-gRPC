# server.py
# we will use asyncio to run our service
import asyncio
import grpc

import cv2
import numpy as np

from deepface import DeepFace as df


from pymongo import MongoClient

# from the generated grpc server definition, import the required stuff
from ms_faceRecognition_pb2_grpc import FaceRecognitionService, add_FaceRecognitionServiceServicer_to_server
# import the requests and reply types
from ms_faceRecognition_pb2 import FaceRecognitionRequest, FaceRecognitionInferenceReply, FaceRecognitionWithRectListRequest, FaceRect

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

connection = "mongodb+srv://andresilmor:project-greenhealth>@carexr-mongodb-cluster.mcnxmz7.mongodb.net/?retryWrites=true&w=majority"

class FaceRecognitionService(FaceRecognitionService):
    def __init__(self) -> None:
        super().__init__()
        self.dbClient = MongoClient(connection)




    async def AccurateInferenceWithDetection(self, request: FaceRecognitionRequest, context) -> FaceRecognitionInferenceReply:
        start = perf_counter()

        print("yo")

        npimg = cv2.imdecode(np.frombuffer(request.image, np.uint8), -1)

        cv2.rectangle(npimg, ( 0, 0 ), (200, 856) , (0, 0, 0), -1) # |o
        cv2.rectangle(npimg, ( 1300, 0 ), (1504, 856) , (0, 0, 0), -1) # o|
        cv2.rectangle(npimg, ( 200, 720 ), (1300, 856) , (0, 0, 0), -1) # _

        #cv2.imwrite(os.getcwd() + "/frame4.png",npimg)
        
        try: 
             print("yo")
            
        except Exception as ex:
            print(ex)
            return FaceRecognitionInferenceReply(None)

    


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


        return FaceRecognitionInferenceReply(None)

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
    adddress = "[::]:50054"
    server.add_insecure_port(adddress)
    logging.info(f"[📡] Starting server on {adddress}")
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())