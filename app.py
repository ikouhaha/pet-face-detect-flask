"""
Run a rest API exposing the yolov5s object detection model
"""



import cv2
import torch
from flask import Flask, request, make_response,render_template
from PIL import Image
import json
import io
import numpy as np
import Util



global model 
def init():
    global model 
    model = torch.hub.load('ultralytics_yolov5_master', 'custom', path='best.pt', source='local') # force_reload to recache


app = Flask(__name__)

init()


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
    serverUrl = request.environ["PATH_INFO"]
    return "home page"

@app.route("/test")
def test():
    serverUrl = Util.getServerUrl(request.url)
    return render_template("test.html",serverUrl=serverUrl)

# @app.route("/detect", methods=["POST"])
# def predict():
    
#     response = None
    
#     try:
#         if(model is None):
#             model = initModel()
#         if not request.method == "POST":
#             return

#         if request.files.get("image"):
#             image_file = request.files["image"]
#             image_bytes = image_file.read()
#             img = Image.open(io.BytesIO(image_bytes))
#             results = model(img) 
#             resultJson = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
#             response = createJsonResponse(200,resultJson)
#         else:
#             response = createJsonResponse(500,"no image")
            
            
#     except Exception as e:
#        response = createJsonResponse(500,str(e))
#     finally:
#        return response

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
      result = detectBase64()
      j = result.json
      jr = j["result"]
      if(j["success"]==False):
          return 'file upload failed' + jr
      #f.save(secure_filename(f.filename))
    
      return render_template("result.html",result=jr)

#only support 1 image 1 face 
@app.route("/detectBase64", methods=["POST"])
def detectBase64():
    global model 
    response = None
    data = {}
    img_size = 224 #resize to 224x224
    
    try:
        if(model is None):
            response = createJsonResponse(500,"no model")
            return 
            
        model.eval()
        if not request.method == "POST":
            return

        if request.files.get("image"):
            
            image_file = request.files["image"]
            image_bytes = image_file.read()
            image = np.asarray(bytearray(image_bytes), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            ori_img = image.copy()
            new_im,ratio,top,left = Util.resize_img(image,img_size) #resize to 224x224
            cv2.imwrite("test.jpg", new_im) #test
            new_im_copy = new_im.copy()
            #annotator = Annotator(cvImg, line_width=3, example=str("cat"))
            #img = Image.open(io.BytesIO(image_bytes))
            results = model(new_im,size=img_size) 
            #resultJson = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
            results.render() 

            if(len(results.pandas().xyxy)==0):
                response = createJsonResponse(500,"cant detect")
                return
            if(len(results.pandas().xyxy[0])==0):
                response = createJsonResponse(500,"cant detect")
                return
            if(len(results.pandas().xyxy[0])>1):
                response = createJsonResponse(500,"cant detect more than 1")
                return

            for xyxy in results.pandas().xyxy:
                xyJson = json.loads(xyxy.to_json(orient="records"))
                data["xyxy"] = xyJson[0]
            #only 1 image
            for img in results.imgs:
                xmin,xmax,ymin,ymax = Util.readOriginalBB(data["xyxy"],ratio,top,left)
                img_str = Util.getBase64FromImage(img)
                data["labelImg"] = img_str 
                crop = ori_img[ymin:ymax, xmin:xmax]     
                crop_img_str = Util.getBase64FromImage(crop)
                data["cropImg"] = crop_img_str 

                ####test area
                
                ####test area

            #im0 = annotator.result()
            
            response = createJsonResponse(200,data)
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
