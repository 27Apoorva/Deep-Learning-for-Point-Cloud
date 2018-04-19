import matplotlib.pyplot as plt
import numpy as np
import cv2

x_g = []
y_g = []
z_g = []
x_e = []
y_e = []
z_e = []

def parseFile():
	fp = open('output_file', "r")
	for line in fp:
		split = line.split()
		x_g.append(split[0])
		y_g.append(split[1])
		z_g.append(split[2])
	fp.close()
	fp = open('estimated', "r")
	for line in fp:
		split = line.split()
		x_e.append(split[0])
		y_e.append(split[1])
		z_e.append(split[2])
	fp.close()
	ground, = plt.plot(x_g, z_g, label = 'ground')
	est , = plt.plot(x_e, z_e, label = 'estimate')
        plt.axis('equal')
	plt.legend([ground, est], ['ground', 'estimate'])
        plt.show(block=False)
	fp = open('img_file_names', "r")
	for line in fp:
		split = line.split()
		img = cv2.imread(split[0],0)
		cv2.imshow('image',img)
		cv2.waitKey(3)
	cv2.destroyAllWindows()
	plt.show()

if __name__ == "__main__":
    parseFile()

