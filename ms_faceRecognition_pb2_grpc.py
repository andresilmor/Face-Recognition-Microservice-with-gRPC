# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import ms_faceRecognition_pb2 as ms__faceRecognition__pb2


class FaceRecognitionServiceStub(object):
    """This tells gRPC we have an InferenceServer service with an inference function, notice that we need to specify the type of the messages: InferenceRequest and InferenceReply

    "repeated" means list of

    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Inference = channel.unary_unary(
                '/faceRecognition.FaceRecognitionService/Inference',
                request_serializer=ms__faceRecognition__pb2.FaceRecognitionRequest.SerializeToString,
                response_deserializer=ms__faceRecognition__pb2.FaceRecognitionInferenceReply.FromString,
                )
        self.InferenceWithoutDetection = channel.unary_unary(
                '/faceRecognition.FaceRecognitionService/InferenceWithoutDetection',
                request_serializer=ms__faceRecognition__pb2.FaceRecognitionWithRectListRequest.SerializeToString,
                response_deserializer=ms__faceRecognition__pb2.FaceRecognitionInferenceReply.FromString,
                )
        self.FastInferenceWithDetection = channel.unary_unary(
                '/faceRecognition.FaceRecognitionService/FastInferenceWithDetection',
                request_serializer=ms__faceRecognition__pb2.FaceRecognitionRequest.SerializeToString,
                response_deserializer=ms__faceRecognition__pb2.FaceRecognitionInferenceReply.FromString,
                )
        self.AccurateInferenceWithDetection = channel.unary_unary(
                '/faceRecognition.FaceRecognitionService/AccurateInferenceWithDetection',
                request_serializer=ms__faceRecognition__pb2.FaceRecognitionRequest.SerializeToString,
                response_deserializer=ms__faceRecognition__pb2.FaceRecognitionInferenceReply.FromString,
                )


class FaceRecognitionServiceServicer(object):
    """This tells gRPC we have an InferenceServer service with an inference function, notice that we need to specify the type of the messages: InferenceRequest and InferenceReply

    "repeated" means list of

    """

    def Inference(self, request, context):
        """Whatever
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def InferenceWithoutDetection(self, request, context):
        """Does inference on image based on the faces detected bounding box from a third party
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def FastInferenceWithDetection(self, request, context):
        """Does face detection and then face recognition, uses a fast face detection mode (ex: OpenCV)
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def AccurateInferenceWithDetection(self, request, context):
        """Does face detection and then face recognition, uses a accurate face detection mode, way slower than Opencv (ex: Retinaface)
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_FaceRecognitionServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Inference': grpc.unary_unary_rpc_method_handler(
                    servicer.Inference,
                    request_deserializer=ms__faceRecognition__pb2.FaceRecognitionRequest.FromString,
                    response_serializer=ms__faceRecognition__pb2.FaceRecognitionInferenceReply.SerializeToString,
            ),
            'InferenceWithoutDetection': grpc.unary_unary_rpc_method_handler(
                    servicer.InferenceWithoutDetection,
                    request_deserializer=ms__faceRecognition__pb2.FaceRecognitionWithRectListRequest.FromString,
                    response_serializer=ms__faceRecognition__pb2.FaceRecognitionInferenceReply.SerializeToString,
            ),
            'FastInferenceWithDetection': grpc.unary_unary_rpc_method_handler(
                    servicer.FastInferenceWithDetection,
                    request_deserializer=ms__faceRecognition__pb2.FaceRecognitionRequest.FromString,
                    response_serializer=ms__faceRecognition__pb2.FaceRecognitionInferenceReply.SerializeToString,
            ),
            'AccurateInferenceWithDetection': grpc.unary_unary_rpc_method_handler(
                    servicer.AccurateInferenceWithDetection,
                    request_deserializer=ms__faceRecognition__pb2.FaceRecognitionRequest.FromString,
                    response_serializer=ms__faceRecognition__pb2.FaceRecognitionInferenceReply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'faceRecognition.FaceRecognitionService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class FaceRecognitionService(object):
    """This tells gRPC we have an InferenceServer service with an inference function, notice that we need to specify the type of the messages: InferenceRequest and InferenceReply

    "repeated" means list of

    """

    @staticmethod
    def Inference(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/faceRecognition.FaceRecognitionService/Inference',
            ms__faceRecognition__pb2.FaceRecognitionRequest.SerializeToString,
            ms__faceRecognition__pb2.FaceRecognitionInferenceReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def InferenceWithoutDetection(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/faceRecognition.FaceRecognitionService/InferenceWithoutDetection',
            ms__faceRecognition__pb2.FaceRecognitionWithRectListRequest.SerializeToString,
            ms__faceRecognition__pb2.FaceRecognitionInferenceReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def FastInferenceWithDetection(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/faceRecognition.FaceRecognitionService/FastInferenceWithDetection',
            ms__faceRecognition__pb2.FaceRecognitionRequest.SerializeToString,
            ms__faceRecognition__pb2.FaceRecognitionInferenceReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def AccurateInferenceWithDetection(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/faceRecognition.FaceRecognitionService/AccurateInferenceWithDetection',
            ms__faceRecognition__pb2.FaceRecognitionRequest.SerializeToString,
            ms__faceRecognition__pb2.FaceRecognitionInferenceReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
