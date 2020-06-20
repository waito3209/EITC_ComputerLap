from support import *
filename=input('enter filename')
data=np.load('enterdata/'+filename+'.npy')
render(data)