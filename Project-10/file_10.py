
import os, sys
from os import listdir

 #Open a file
f= open("my_dataset_train.txt","w+")
filename=['Industrial','River','Forest','AnnualCrop','HerbaceousVegetation','Highway','Pasture','PermanentCrop','Residential','SeaLake']
j=0
for p in filename:
	path = "EuroSAT_Train/"+p
	dirs = os.listdir( path )
	i=0;
	# This would print all the files and directories
	for file in dirs:
		i=i+1
	   	f.write ("/home/shubgupta/EuroSAT_Train/"+path+'/'+file+" "+str(j)+"\n")
	print j
	j=j+1
