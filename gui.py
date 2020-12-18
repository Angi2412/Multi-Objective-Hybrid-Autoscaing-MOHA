# Copyright (c) 2020 Angelina Horn
import os

import PySimpleGUI as sg

import sandbox

sg.theme('Reddit')

layout = [
    [sg.Text('Microservice configuration')],
    [sg.Text('Name:', size=(15, 1)), sg.InputText()],
    [sg.Text('Port:', size=(15, 1)), sg.Spin([i for i in range(0000, 9999)], initial_value=0000, size=(4, 1))],
    [sg.Text('Dockerfile:', size=(15, 1)), sg.In(size=(35, 1)), sg.FileBrowse()],
    [sg.Text('API information')],
    [sg.Text('Route:', size=(15, 1)), sg.InputText()],
    [sg.Text('Attribute:', size=(15, 1)), sg.InputText()],
    [sg.Text('Testfile:', size=(15, 1)), sg.In(size=(35, 1)),
     sg.FileBrowse(file_types=(("Text Files", "*.txt"),), initial_folder=os.getcwd())],
    [sg.Submit(), sg.Cancel()]
]

window = sg.Window('PodAutoScaler', layout)
event, values = window.read()
sandbox.execute(name=values[0], port=values[1], docker_path=values[2], route=values[3], input_type=values[4],
                testfile_path=values[5])
window.close()
