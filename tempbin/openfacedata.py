from support import *
while True:

    for i in listdir('../facedata'):
        data = np.load('facedata/' + i )
        print(data.shape)
        render(data
               )
