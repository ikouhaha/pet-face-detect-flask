
from .config import Config
from .contrastiveLoss import ContrastiveLoss
from .siameset import SiameseNetwork
from .SiameseNetworkDatasetWithSource import SiameseNetworkDatasetWithSource
from .package import *
import os


class PetFinder():
    
    def find(self,path,netpath,imageBase64):        
        results = []
        folder_dataset_test = dset.ImageFolder(root=path)
        siamese_dataset = SiameseNetworkDatasetWithSource(
                                                imageFolderDataset=folder_dataset_test
                                                ,transform=transforms.Compose([transforms.Resize((100,100)),
                                                                            transforms.ToTensor()
                                                                            ])
                                            ,should_invert=False
                                            ,sourceImgBase64=imageBase64)
                                            

        test_dataloader = DataLoader(siamese_dataset,num_workers=0,batch_size=1,shuffle=True)
        dataiter = iter(test_dataloader)

        
        net = SiameseNetwork()
        net.load_state_dict(torch.load(netpath))
        
        for i in range(0,siamese_dataset.__len__()):
            x0,x1,label2,filename = next(dataiter)
            print(os.path.basename(filename[0]))
            #concatenated = torch.cat((x0,x1),0)
            output1,output2 = net(Variable(x0),Variable(x1))
            euclidean_distance = F.pairwise_distance(output1, output2)
            results.append({
                "filename":os.path.basename(filename[0]),
                "distance":round(euclidean_distance.item(),2)
            })
            
        #sort by distance
        newlist = sorted(results, key=lambda x: x["distance"], reverse=False)
        print(newlist)
        return newlist

