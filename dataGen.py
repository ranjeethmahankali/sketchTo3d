from Rhino.Display import *
import rhinoscriptsyntax as rs
import scriptcontext as sc
from System.Drawing import *
import random
import pickle
import math

currentView = sc.doc.Views.ActiveView
picSize = Size(64,48)
voxelSize = 1.0
xLim = [-11,12]
yLim = [-11,12]
zLim = [0,15]

def writeToFile(data, path):
	with open(path, 'wb') as output:
		pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

def makeCube(center, size):
	if size <= 0:
		print('cube size invalid')
		return None
	
	s = size/2
	vert = [
		[center[0]-s, center[1]-s, center[2]-s],
		[center[0]-s, center[1]+s, center[2]-s],
		[center[0]+s, center[1]+s, center[2]-s],
		[center[0]+s, center[1]-s, center[2]-s],
		[center[0]-s, center[1]-s, center[2]+s],
		[center[0]-s, center[1]+s, center[2]+s],
		[center[0]+s, center[1]+s, center[2]+s],
		[center[0]+s, center[1]-s, center[2]+s]
	]
	
	cubeID = rs.AddBox(vert)
	rs.SurfaceIsocurveDensity(cubeID, 0)
	return cubeID

#this makes a scene with boxes
class box_scene:
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
	
	#converts 3d array into voxels and returns it as a group of cubes
	def getVoxels(self, bytes):
		ptList = []
		for x in range(len(bytes)):
			for y in range(len(bytes[x])):
				for z in range(len(bytes[x][y])):
					if bytes[x][y][z] == 1:
						ptList.append([x+xLim[0],y+yLim[0],z])
		
		boxes = []
		for pt in ptList:
			boxes.append(makeCube(pt, voxelSize))
		
		group = rs.AddGroup('voxels')
		rs.AddObjectsToGroup(boxes, 'voxels')
			
		return group
							
	#return voxels as a 3d binary array
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

#this makes a scene with balls
class ball_scene:
	def __init__(self):
		self.objects = []
		return
	
	#populate the scene randomly
	def populate(self, objNumRange = [1,1], sizeRange = [2,8]):
		self.reset()
		objNum = random.randint(objNumRange[0], objNumRange[1])
		
		isTree = None
		for i in range(objNum):
			size = random.randint(sizeRange[0], sizeRange[1])
			isTree = random.sample([True, False],1)[0]
			print(isTree)
			
			rad = size/2
			rad_i = int(math.ceil(rad))
			x = random.sample(range(xLim[0]+rad_i, xLim[1]-rad_i),1)[0]
			y = random.sample(range(yLim[0]+rad_i, yLim[1]-rad_i),1)[0]
			
			z = None
			if isTree:
				z = zLim[1] - rad
				zBase = 0
				trunkID = rs.AddLine([x,y,zBase],[x,y,z])
				self.objects.append(trunkID)
			else:
				z = zLim[0] + rad
			
			try:
				ballID = rs.AddSphere([x,y,z], rad)
				rs.SurfaceIsocurveDensity(ballID, 0)
				self.objects.append(ballID)
			except:
				print([x,y,z])
			
		return
	
	#delete all objects, empty the list and done
	def reset(self):
		num = None
		try:
			num = rs.DeleteObjects(self.objects)
		except:
			print('Something went wrong while resetting the scene')
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
	
	#converts 3d array into voxels and returns it as a group of cubes
	def getVoxels(self, bytes):
		ptList = []
		for x in range(len(bytes)):
			for y in range(len(bytes[x])):
				for z in range(len(bytes[x][y])):
					if bytes[x][y][z] == 1:
						ptList.append([x+xLim[0],y+yLim[0],z])
		
		boxes = []
		for pt in ptList:
			boxes.append(makeCube(pt, voxelSize))
		
		group = rs.AddGroup('voxels')
		rs.AddObjectsToGroup(boxes, 'voxels')
			
		return group
	
	#return voxels as a 3d binary array
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

def makeBoxDataset(num, path):
	scn = box_scene()
	views = []
	models = []
	for i in range(num):
		rs.EnableRedraw(False)
		scn.populate([1,2])
		rs.EnableRedraw(True)
		img_data = scn.getView()
		voxel_data = scn.getVoxelBytes()
		
		views.append(img_data)
		models.append(voxel_data)
	
		scn.reset()
	writeToFile([views, models], path)

def BoxResult():
	with open('results/19.pkl', 'rb') as inp:
		vox = pickle.load(inp)
	
	rs.EnableRedraw(False)
	model = scn.getVoxels(vox[0])
	rs.EnableRedraw(True)

#sampleNum = int(input('Enter number of samples:'))
#fileName= input('Enter the fileName:')
#filePath = 'data/%s.pkl'%fileName
#makeBoxDataset(sampleNum, filePath)
#BoxResult()

scn = ball_scene()
rs.EnableRedraw(False)
scn.populate([1,3])
rs.EnableRedraw(True)
#inp = input('waiting...')
#rs.EnableRedraw(False)
#bytes = scn.getVoxelBytes()
##scn.reset()
#vox = scn.getVoxels(bytes)
#rs.EnableRedraw(True)