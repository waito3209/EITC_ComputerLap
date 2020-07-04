import PySimpleGUI as sg
import cv2
import numpy as np
from support import detail

"""
Demo program that displays a webcam using OpenCV
"""


def main():
    sg.theme('Dark Blue 3')

    # define the window layout
    layout = [[sg.Text('WebCam Image', size=(40, 1), justification='center', font='Helvetica 20')],
              [sg.Image(filename='', key='image')],
              [sg.Button('Record', size=(10, 1), font='Helvetica 14'),
               sg.Button('Cap', size=(10, 1), font='Any 14'),
               sg.Button('Exit', size=(10, 1), font='Helvetica 14'), ]]

    # create the window and show it without the plot
    window = sg.Window('Face Recognition', layout, location=(800, 400))  # location=(800, 400)

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    cam = cv2.VideoCapture(0)
    recording = True

    while True:
        event, values = window.read(timeout=5)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break

        elif event == 'Record':
            recording = True

        elif event == 'Cap':
            recording = False
            #img = np.full((480, 640), 255)  #np.zeros((img_h, img_w, 3), dtype=np.uint8)
            # this is faster, shorter and needs less includes
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()
            window['image'].update(data=imgbytes)
            detail(frame)

        if recording:
            ret, frame = cam.read()
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()
            window['image'].update(data=imgbytes)

    cam.release()
    window.close()


main()
