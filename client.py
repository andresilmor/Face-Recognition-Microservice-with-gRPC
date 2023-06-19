import asyncio
from io import BytesIO

import grpc
from PIL import Image
import cv2
from ms_faceRecognition_pb2 import FaceRecognitionRequest, FaceRecognitionInferenceReply, FaceRecognitionWithRectListRequest
from ms_faceRecognition_pb2_grpc import FaceRecognitionServiceStub
import logging
from pprint import pformat
import numpy as np
import json

async def main():
   

    async with grpc.aio.insecure_channel("[::]:50054") as channel:
        stub = FaceRecognitionServiceStub(channel)


        #   QRCode Decode
        '''

        img = Image.open("test_images/qrCode_multiple.png")
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG', subsampling=0, quality=100)
        img_byte_arr = img_byte_arr.getvalue()

        res: QRCodeContent = await stub.QRCodeDecode(
            InferenceRequest(image=img_byte_arr)
        )
        print(res.content)
        json_res = json.loads(res.content)

        print(json_res)
        '''
        img = Image.open("double.jpg")
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG', subsampling=0, quality=100)
        img_byte_arr = img_byte_arr.getvalue()
        npimg = cv2.imdecode(np.frombuffer(img_byte_arr, np.uint8), -1)
        res: FaceRecognitionInferenceReply = await stub.FastInferenceWithDetection(
            FaceRecognitionRequest(image=img_byte_arr)
        )

        print(res) 
if __name__ == "__main__":
    asyncio.run(main())