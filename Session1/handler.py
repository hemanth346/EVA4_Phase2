try:
    import unzip_requirements
except ImportError:
    pass

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import boto3
import os
import io
import json
import base64

from requests_toolbelt.multipart import decoder

# lambda console log
print('Imported packages - Logger')


# define env variables
S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'molibenetv2-tensorclan'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'mobilenet_v2.pt'

# lambda console log
print('Downloaded model - Logger')


s3 = boto3.client('s3')

try:
    if os.path.isfile(MODEL_PATH) != True:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        # lambda console log
        print('Creating bytesteam - Logger')
        bytesteam = io.BytesIO(obj['Body'].read())

        print('Loading Model - Logger')
        model = torch.jit.load(bytesteam)
        print('Model loaded- Logger')

except Exception as e:
    # lambda console log
    print(repr(e) + ' - Logger')
    raise(e)


def transform_image(image_bytes):
    try:
        transformations = transforms.Compose([
            transforms.Resize(255),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
            ])
        image = Image.open(io.BytesIO(image_bytes))
        return transformations(image).unsqueeze(0)
    except Exception as e:
        # lambda console log
        print(repr(e) + ' - Logger')
        raise(e)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    return model(tensor).argmax().item()

def classify_image(event, context):
    try:
        content_type_header = event['headers']['content-type']
        # lambda console log
        print(event['body'][:10] + ' - Logger')
        body = base64.b64decode(event['body'])
        print('Body loader - Logger')

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        prediction = get_prediction(image_bytes=picture.content)
        print(prediction, ' - Logger')

        
        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename) < 4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]

        return {
            "statusCode": 200,
            "headers" : {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            "Access-Control-Allow-Credentials": True
            },
            'body': json.dumps({'file': filename.replace('"',''), 'predicted': prediction})
        }

    except Exception as e:
        print(repr(e), '- Logger')
        return {
            "statusCode": 500,
            "headers" : {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            "Access-Control-Allow-Credentials": True
            },
            'body': json.dumps({'error': repr(e)})        
        }
