import cv2
from PIL import Image
import base64
import io
import matplotlib.pyplot as plt

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show() 

def getServerUrl(host):
  schema = ""
  if "localhost" in host:
    schema ="http"
  else:
    schema ="https"
  
  return schema+"://"+host

def getFileSize(byte,type="MB"):
  if type=="MB":
    return byte/1024/1024
  if type=="KB":
    return byte/1024


def resize_img(im,img_size):
  old_size = im.shape[:2] # old_size is in (height, width) format
  ratio = float(img_size) / max(old_size)
  new_size = tuple([int(x*ratio) for x in old_size])
  # new_size should be in (width, height) format
  im = cv2.resize(im, (new_size[1], new_size[0]))
  delta_w = img_size - new_size[1]
  delta_h = img_size - new_size[0]
  top, bottom = delta_h // 2, delta_h - (delta_h // 2)
  left, right = delta_w // 2, delta_w - (delta_w // 2)
  new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
      value=[0, 0, 0])
  return new_im, ratio, top, left  

def encodeBase64FromImage(img):
  buffered = io.BytesIO()
  img_array = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  img_array.save(buffered, format="JPEG")
  img_str = base64.b64encode(buffered.getvalue())
  return img_str.decode("utf-8") 

def decodeBase64ToImage(imgstring):
  imgdata = base64.b64decode(imgstring)
  return imgdata

def readOriginalBB(xyxy,ratio,top,left):
  xmin = int(abs(xyxy["xmin"]-left)/ratio)
  xmax = int(abs(xyxy["xmax"]-left)/ratio)
  ymin = int(abs(xyxy["ymin"]-top)/ratio)
  ymax = int(abs(xyxy["ymax"]-top)/ratio)
  return xmin,xmax,ymin,ymax

def getBase64ImageAndExtension(imgstring):
  imgdata = base64.b64decode(imgstring)
  img = Image.open(io.BytesIO(imgdata))
  return img,img.format
# var base64Data = str64.replace(/^data:image\/(png|jpeg|jpg|gif);base64,/, '')
#     const image = Buffer.from(base64Data, "base64")
    
#     var type 
#     switch (base64Data.charAt(0)) {
#         case "/":
#             type="image/jpg"
#             break;
#         case "R":
#             type="image/png"
#             break;
#         case "i":
#             type="image/gif"
#             break;
#         case "P":
#             type="image/jpeg"
#             break;
#         default:
#             break;
#     }

#     return { type, image }





 
  