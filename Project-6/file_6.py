
import os, sys
from os import listdir

# Open a file
f= open("my_dataset_test.txt","w+")
filename=['Forest','Industrial','Highway','Pasture','PermanentCrop','Residential']
j=0
for p in filename:
	path = "EuroSAT_Test/"+p
	dirs = os.listdir( path )
	i=0;
	# This would print all the files and directories
	for file in dirs:
		i=i+1
	   	f.write ("/home/ndixitasohan/EuroSAT_Test/"+path+'/'+file+" "+str(j)+"\n")
	print j
	j=j+1
#images = listdir('EuroSAT_Train/')

#print(images)
