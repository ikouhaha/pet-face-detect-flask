"""
Run a rest API exposing the yolov5s object detection model
"""



import base64
from distutils import extension
from http.client import HTTPResponse
import cv2
import torch
from flask import Flask, request, make_response,render_template, send_file
from PIL import Image
import json
import io
import numpy as np
import os
import urllib3
import Util
import zipfile
from flask_cors import CORS
from pathlib import Path



from siamese.PetFinder import PetFinder


global model,dataPath
def init() :
    global model,dataPath
    # dataPath = "/app/data"
    dataPath = "C:/Users/ikouh/Documents/fyp/data"
    model = torch.hub.load('ultralytics_yolov5_master', 'custom', path=dataPath+'/best.pt', source='local') # force_reload to recache

    # cat_folder = ShareServiceClient.from_connection_string(os.environ["file_store_string"],os.environ["share_name"],"cat")
    # service_client.create_share()
app = Flask(__name__)
CORS(app) 

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


def detectImageResult(rImage,img_size):
    data = {}
    image_file = rImage
    image_bytes = image_file.read()
    image = np.asarray(bytearray(image_bytes), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    ori_img = image.copy()
    new_im,ratio,top,left = Util.resize_img(image,224) #resize to 224x224
    results = model(new_im,size=img_size) 
    
    results.render() 

    if(len(results.pandas().xyxy)==0):
        raise Exception('cant detect')
    if(len(results.pandas().xyxy[0])==0):
        raise Exception('cant detect')
    #if(len(results.pandas().xyxy[0])>1):
    #    raise Exception('cant detect more than 1 object per image')

    cropList = []
    cropStrList = []
    name = ""
    for xyxy in results.pandas().xyxy:
        xyJson = json.loads(xyxy.to_json(orient="records"))
        data["xyxys"] = xyJson
       
    #only 1 image
    for img in results.imgs:
        for xyxy in data["xyxys"]:
            xmin,xmax,ymin,ymax = Util.readOriginalBB(xyxy,ratio,top,left)
            name = xyxy["name"]
            crop = ori_img[ymin:ymax, xmin:xmax]  
            cropList.append(crop)
    
    for crop in cropList:
        
        resizeCrop,ratio,top,left = Util.resize_img(crop,100)
        img_str = Util.encodeBase64FromImage(resizeCrop)
        cropStrList.append(img_str)


    img_str = Util.encodeBase64FromImage(img)

    data["name"] = name
    data["labelImg"] = img_str  
    data["cropImgs"] = cropStrList 
        
        
    return data    





@app.route("/")
def hello():
    serverUrl = request.environ["PATH_INFO"]
    return "home page"

@app.route("/test")
def test():
    serverUrl = Util.getServerUrl(request.environ["HTTP_HOST"])
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

@app.route('/download', methods = ['POST'])
def downloadFile ():
    #For windows you need to use drive name [ex: F:/Example.pdf]
    requestList = request.form["results"]
    requestList = json.loads(requestList.replace('\'',"\""))
    
    imgList = []
    i = 0
    zip_buffer = io.BytesIO()
    zf = zipfile.ZipFile(zip_buffer, mode="w",compression=zipfile.ZIP_DEFLATED)
    for r in requestList:
          name = r["name"]
          for base64img in r["cropImgs"]:
             imgName = name + str(i+1)+".jpg"
             img =  Util.decodeBase64ToImage(base64img)
             #for file_name, data in [('1.txt', io.BytesIO(b'111')), ('2.txt', io.BytesIO(b'222'))]:
             zf.writestr(imgName, img)
             i = i+1
    zf.close()
    zip_buffer.seek(0)
    
    response = make_response(zip_buffer.read())
    response.headers.set('Content-Type', 'zip')
    response.headers.set('Content-Disposition', 'attachment', filename='download.zip' )
    return response        
    #return send_file(zip, as_attachment=True)

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
      serverUrl = Util.getServerUrl(request.environ["HTTP_HOST"])
      result = detectBase64(1)
      j = result.json
      jrs = j["result"]
      errMsg = ""
      if(j["success"]==False):
          for jr in jrs:
              errMsg+=jr
          return 'file upload failed ' + errMsg
      #f.save(secure_filename(f.filename))
    
      return render_template("result.html",results=jrs,serverUrl=serverUrl)

#only support 1 image 1 face 
@app.route("/detectBase64/<isSaveResult>", methods=["POST"])
def detectBase64(isSaveResult):
    isSaveResult = bool(int(isSaveResult))
    global model,dataPath
    response = None
    data = {}
    img_size = 224 #resize to 224x224
    file_size = 0
    
    try:
        if(model is None):
            response = createJsonResponse(500,"no model")
            return 
            
        model.eval()
        if not request.method == "POST":
            response = createJsonResponse(500,"only support POST method")
            return 

        file_size = Util.getFileSize(request.content_length,"MB")
        if(file_size) > 100 :
            response = createJsonResponse(500,"the size of files is too big ({:.5f}MB)".format(file_size))
            return 
        
        if(isSaveResult == True and request.form.__contains__("name")==False):
            response = createJsonResponse(500,"no name")
            return
        else:
            name = request.form["name"]
            catPath = dataPath+"/cat/face/"+name 
            dogPath = dataPath+"/dog/face/"+name 


        if(request.files["image"]):
            dataList = []
            for requestFile in request.files.getlist("image"):
                extension =  Path(requestFile.filename).suffix
                data = detectImageResult(requestFile,img_size)
                dataList.append(data)
                if(isSaveResult==True):
                    if(data["name"]=="cat"):
                        path = catPath
                    else:
                        path = dogPath

                    for crop in data["cropImgs"]:
                        with open(path+extension, "wb") as fh:
                            fh.write(base64.b64decode(crop))


            #im0 = annotator.result()


            
            response = createJsonResponse(200,dataList)
        else:
            response = createJsonResponse(500,"no image")
            
            
    except Exception as e:
       response = createJsonResponse(500,str(e))
    finally:
       return response

@app.route("/detectByBase64", methods=["POST"])
def detectByBase64():
    global model 
    response = None
    data = {}
    img_size = 224 #resize to 224x224
    file_size = 0
    
    try:
        if(model is None):
            response = createJsonResponse(500,"no model")
            return 
            
        model.eval()
        if not request.method == "POST":
            response = createJsonResponse(500,"only support POST method")
            return 
        
        file_size = Util.getFileSize(request.content_length,"MB")
        if(file_size) > 100 :
            response = createJsonResponse(500,"the size of files is too big ({:.5f}MB)".format(file_size))
            return 
            
        
        if(request.files["image"]):
            dataList = []
            for requestFile in request.files.getlist("image"):
                data = detectImageResult(requestFile,img_size)
                dataList.append(data)

            #im0 = annotator.result()


            
            response = createJsonResponse(200,dataList)
        else:
            response = createJsonResponse(500,"no image")
            
            
    except Exception as e:
       response = createJsonResponse(500,str(e))
    finally:
       return response
   
@app.route("/findSimilarity", methods=["POST"])
def findSimilarity():
    try:
        imageBase64 = request.json["imageBase64"]
        type = request.json["type"]
        if(imageBase64 is None or imageBase64==""):
            response = createJsonResponse(500,"no image")
        else:
            #im0 = annotator.result()
            results = PetFinder().find(dataPath+"/"+type,dataPath+"/{0}net.pt".format(type),imageBase64)   
            response = createJsonResponse(200,results)   
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
