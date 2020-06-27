from support import *

while True:
    filename=input('enter filename')
    data=np.load(filename)
    render(data)
    path=input('file')
    for i in listdir(path):
        print(i)
        data = np.load(path + '/'+i )
        print(data.shape)
    for i in listdir(path):
        data = np.load(path + '/'+i )
        render(data)