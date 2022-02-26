import os

#os.system("docker container prune -a --force") --manually run
#os.system("docker image prune -a --force") --manually run
os.system("docker build -t ikouhaha888/pet-face-detect:0.1 .")
#os.system("docker push  containerappservicedennis.azurecr.io/pet-face-detect:0.1")


