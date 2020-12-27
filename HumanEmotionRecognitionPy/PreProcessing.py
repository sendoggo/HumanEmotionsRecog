import glob
from shutil import copyfile
import re
import os

emotions = ["NE", "AN", "DI", "FE", "HA", "SA", "SU"] #Define emotion order
participants = glob.glob("JAFFE\\*") #Returns a list of all folders with participant numbers

for x in participants:
    filename = os.path.basename(x)
    if (os.path.basename(x).split('.')[1][:-1] == "NE"):
        print ("JAFFEset\neutral\\%s" % os.path.basename(x)) #Do same for emotion containing image
        
    elif (os.path.basename(x).split('.')[1][:-1] == "AN"):
        print ("JAFFEset\anger\\%s" % os.path.basename(x)) #Do same for emotion containing image
        
    
    