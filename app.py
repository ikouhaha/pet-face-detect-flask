"""
Run a rest API exposing the yolov5s object detection model
"""



import torch
from flask import Flask, request, make_response,jsonify,Response
from PIL import Image
import json
import io


app = Flask(__name__)

def createJsonResponse(code,result):
     data = {}
    #  resp.status_code = code
     data["result"] = result
     if code == 200:
        data["success"] = True
     else:
        data["success"] = False
     response = make_response(data, code)
     response.headers["Content-Type"] = 'application/json'
     return response

@app.route("/")
def hello():
    return "home page"

@app.route("/detect", methods=["POST"])
def predict():
    
    response = None
    print("hi")
    try:
        model = torch.hub.load('ultralytics_yolov5_master', 'custom', path='best.pt', source='local') # force_reload to recache
        if not request.method == "POST":
            return

        if request.files.get("image"):
            image_file = request.files["image"]
            image_bytes = image_file.read()
            img = Image.open(io.BytesIO(image_bytes))
            results = model(img) 
            resultJson = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
            response = createJsonResponse(200,resultJson)
        else:
            response = createJsonResponse(500,"no image")
            
            
    except Exception as e:
       response = createJsonResponse(500,str(e))
    finally:
       return response
    

   


if __name__ == "__main__":
    
    #parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    #parser.add_argument("--port", default=5000, type=int, help="port number")
    #model = torch.hub.load('ultralytics/yolov5','yolov5s')
    #model = torch.hub.load('ikouhaha/pet_face_yolov5','custom', path='best.pt', force_reload=True)  # force_reload to recache
    app.run(host='0.0.0.0',port=5000)  # debug=True causes Restarting with stat
