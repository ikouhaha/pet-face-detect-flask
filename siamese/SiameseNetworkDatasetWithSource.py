from .package import *

class SiameseNetworkDatasetWithSource(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=True,sourceImgBase64=""):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        self.sourceImgBase64 = sourceImgBase64
        
    def __getitem__(self,index):
        img1_tuple = self.imageFolderDataset.imgs[index]
       

        img0 = Image.open(BytesIO(base64.b64decode(self.sourceImgBase64)))
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0,img1,torch.from_numpy(np.array([int(True)],dtype=np.float32)), img1_tuple[0] 
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)