# server.py
# we will use asyncio to run our service
import asyncio
import grpc

import cv2
import numpy as np

from deepface import DeepFace as df

# from the generated grpc server definition, import the required stuff
from ms_faceRecognition_pb2_grpc import FaceRecognitionServer, add_FaceRecognitionServerServicer_to_server
# import the requests and reply types
from ms_faceRecognition_pb2 import FaceRecognitionRequest, FaceRecognitionInferenceReply

import logging


logging.basicConfig(level=logging.INFO)

class FaceRecognitionService(FaceRecognitionServer):
    async def inference(self, request: FaceRecognitionRequest, context) -> FaceRecognitionInferenceReply:
        npimg = cv2.imdecode(np.frombuffer(request.image, np.uint8), -1)
        
        try: 
            resp = df.find(img_path=npimg[request.personBox['y1']:request.personBox['y2'], request.personBox['x1']:request.personBox['x2']],
                       db_path='database/', model_name='ArcFace', enforce_detection=True, prog_bar=False, silent=True)
        except:
            return FaceRecognitionInferenceReply(uuid=None, faceCenterX=None, faceCenterY=None)

        try: 
            obj = df.analyze(img_path=npimg[request.personBox['y1']:request.personBox['y2'], request.personBox['x1']:request.personBox['x2']], 
                    actions = ['age', 'gender', 'race', 'emotion']
            )
            print(obj)
        
        except:
            return FaceRecognitionInferenceReply(uuid=None, faceCenterX=None, faceCenterY=None)


        x1 = request.personBox['x1'] + obj["region"]["x"]
        x2 = x1 + obj["region"]["w"]
        y1 = request.personBox['y1'] + obj["region"]["y"]
        y2 = y1 + obj["region"]["h"]

        identity = str(resp.iloc[:1]['identity'])
        id = identity[identity.find("/") + 1: identity.rfind("/")] 
            
        faceCenterX =  int(request.personBox['x1'] + ((x2 - x1) / 2))

        faceCenterY = int(request.personBox['y1'] + ((y2 - y1) / 2))

        return FaceRecognitionInferenceReply(uuid=id, faceCenterX=faceCenterX, faceCenterY=faceCenterY)



async def serve():
    server = grpc.aio.server()
    add_FaceRecognitionServerServicer_to_server(FaceRecognitionService(), server)
    # using ip v6
    adddress = "[::]:50054"
    server.add_insecure_port(adddress)
    logging.info(f"[📡] Starting server on {adddress}")
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())