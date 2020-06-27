from support import *
while True:
    filename=input('enter filename')
    data=np.load('facedata/'+filename+'.npy')
    render(data)
    for i in listdir('facedata'):
        data = np.load('facedata/' + i )
        print(data.shape)
