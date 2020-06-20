from support import *
filename=input('enter filename')
data=np.load('facedata/'+filename+'.npy')
render(data)