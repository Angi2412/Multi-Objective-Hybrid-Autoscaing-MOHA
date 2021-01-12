# Copyright (c) 2020 Angelina Horn
from gevent import monkey

monkey.patch_all()
import os
from pathlib import Path

import PySimpleGUI as sg

from sandbox import deployment, benchmark

sg.theme('Reddit')

layout = [
    [sg.Text('Microservice configuration')],
    [sg.Text('Name:', size=(15, 1)), sg.InputText()],
    [sg.Text('Port:', size=(15, 1)), sg.Spin([i for i in range(0000, 9999)], initial_value=0000, size=(4, 1))],
    [sg.Text('Dockerfile:', size=(15, 1)), sg.In(size=(35, 1)), sg.FileBrowse()],
    [sg.Text('API information')],
    [sg.Text('Route:', size=(15, 1)), sg.InputText()],
    [sg.Text('Testfile:', size=(15, 1)), sg.In(size=(35, 1)),
     sg.FileBrowse(file_types=(("Text Files", "*.txt"),),
                   initial_folder=os.path.join(os.getcwd(), "data", "loadtest"))],
    [sg.Text('Benchmark')],
    [sg.Text('Users:', size=(15, 1)), sg.Spin([i for i in range(0000, 9999)], initial_value=1, size=(4, 1))],
    [sg.Text('Spawn rate:', size=(15, 1)), sg.Spin([i for i in range(0000, 9999)], initial_value=10, size=(4, 1))],
    [sg.Submit(), sg.Cancel()]
]

window = sg.Window('PodAutoScaler', layout)
event, values = window.read()
# deployment(name=values[0], port=int(values[1]), docker_path=values[2])
benchmark(route=values[3], testfile=Path(values[4]).name, users=int(values[5]), spawn_rate=int(values[6]))
window.close()
