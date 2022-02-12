import os
import sys
os.system("docker image prune -a --force")
os.system("docker build -t containerappservicedennis.azurecr.io/pet-face-detect:0.1 .")
os.system("docker push  containerappservicedennis.azurecr.io/pet-face-detect:0.1")


