# Copyright (c) 2020 Angelina Horn
from gevent import monkey

monkey.patch_all()
import os
from pathlib import Path

import PySimpleGUI as sg

from benchmark import start_run

sg.theme('Reddit')

layout = [
    [sg.Text('Benchmark')],
    [sg.Text('Users:', size=(15, 1)), sg.Spin([i for i in range(0000, 9999)], initial_value=100, size=(4, 1))],
    [sg.Text('Spawn rate:', size=(15, 1)), sg.Spin([i for i in range(0000, 9999)], initial_value=10, size=(4, 1))],
    [sg.Text('Number of runs:', size=(15, 1)), sg.Spin([i for i in range(0000, 9999)], initial_value=5, size=(4, 1))],
    [sg.Text('Expressions:', size=(15, 1)), sg.Spin([i for i in range(0000, 9999)], initial_value=5, size=(4, 1))],
    [sg.Text('Step:', size=(15, 1)), sg.Spin([i for i in range(0000, 9999)], initial_value=50, size=(4, 1))],
    [sg.Text('Pod limit:', size=(15, 1)), sg.Spin([i for i in range(0000, 9999)], initial_value=5, size=(4, 1))],
    [sg.Submit(), sg.Cancel()]
]

window = sg.Window('PodAutoScaler', layout)
event, values = window.read()
start_run("teastore", values[0], values[1], values[2], values[3], values[4], values[5])
window.close()
