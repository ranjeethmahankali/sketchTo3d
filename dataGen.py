from Rhino.Display import *
import rhinoscriptsyntax as rs
import scriptcontext as sc
from System.Drawing import *
import random
import pickle

sampleNum = int(input('Enter number of samples:'))

currentView = sc.doc.Views.ActiveView
picSize = Size(32,24)
xLim = [-11,12]
yLim = [-11,12]
zLim = [0,15]

def writeToFile(data, name):
	with open(str(name)+'.pkl', 'wb') as output:
		pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

class scene:
	def __init__(self):
		self.objects = []
		return
	
	#create a random scene 
	def populate(self, objNumRange):
		self.reset()
		objNum = random.randint(objNumRange[0], objNumRange[1])
		
		for i in range(objNum):
			x = random.sample(range(xLim[0], xLim[1]),2)
			y = random.sample(range(yLim[0], yLim[1]),2)
			z = random.sample(range(zLim[0], zLim[1]),2)
			x.sort()
			y.sort()
			z.sort()
			
			vert = [
				[x[0], y[0], z[0]],
				[x[0], y[1], z[0]],
				[x[1], y[1], z[0]],
				[x[1], y[0], z[0]],
				[x[0], y[0], z[1]],
				[x[0], y[1], z[1]],
				[x[1], y[1], z[1]],
				[x[1], y[0], z[1]]
			]
			
			boxID = None
			try:
				boxID = rs.AddBox(vert)
			except:
				print(x,y,z)
			
			self.objects.append(boxID)
		
		return
		
	#delete all objects, empty the list and done
	def reset(self):
		num = None
		try:
			num = rs.DeleteObjects(self.objects)
		except:
			print(len(self.objects))
		if num != len(self.objects):
			print('Scene reset was not clean')
		
		self.objects = []
		return
	
	#gets the image of the view as a 2d array
	def getView(self):
		#return the view as a byte array
		img = RhinoView.CaptureToBitmap(currentView, picSize)
		
		arr = [[n for n in range(picSize.Width)] for m in range(picSize.Height)]
		for y in range(picSize.Height):
			for x in range(picSize.Width):
				color = img.GetPixel(x,y)
				val = float((color.R + color.G + color.B)/3)/255
				arr[y][x] = 1-val
				
		return arr
	
	#converts 3d array into voxels and returns it as a point cloud
	def getVoxels(self, bytes):
		ptList = []
		for x in range(len(bytes)):
			for y in range(len(bytes[x])):
				for z in range(len(bytes[x][y])):
					if bytes[x][y][z] == 1:
						ptList.append([x+xLim[0],y+yLim[0],z])
		
		cloudID = rs.AddPointCloud(ptList)
		return cloudID
						
	
	#return voxels as a 3d array
	def getVoxelBytes(self):
		xR = xLim[1] - xLim[0] + 1
		yR = yLim[1] - yLim[0] + 1
		zR = zLim[1] - zLim[0] + 1
		
		arr = [[[0 for z in range(zR)] for y in range(yR)] for x in range(xR)]
		
		x = xLim[0]
		while x <= xLim[1]:
			y = yLim[0]
			while y <= yLim[1]:
				z = zLim[0]
				while z <= zLim[1]:
					#check if point inside
					for surf in self.objects:
						if rs.IsPointInSurface(surf, [x,y,z]):
							if arr[x-xLim[0]][y-yLim[0]][z] == 0:arr[x-xLim[0]][y-yLim[0]][z] = 1
							break
					z += 1
				y += 1
			x += 1
		
		return arr
	
scn = scene()
#views = []
#models = []
#rs.EnableRedraw(False)
#scn.populate([1,3])
#rs.EnableRedraw(True)
#for i in range(sampleNum):
#	img_data = scn.getView()
#	voxel_data = scn.getVoxelBytes()
#	
#	views.append(img_data)
#	models.append(voxel_data)
#
##scn.reset()
#
#fileName = input('Enter File Name:')
#writeToFile([views, models], 'results/test')

with open('results/vox.pkl', 'rb') as inp:
	vox = pickle.load(inp)

cld = scn.getVoxels(vox[0])