# Copyright (c) 2020 Angelina Horn
from gevent import monkey

monkey.patch_all()

import PySimpleGUI as sg

from benchmark import start_run, p
from k8s_tools import k
import logging
import numpy as np

sg.theme('Reddit')

layout = [
    [sg.Text('Load')],
    [sg.Text('Max Users:', size=(15, 1)), sg.Spin([i for i in range(0000, 9999)], initial_value=10, size=(4, 1))],
    [sg.Text('Spawn rate:', size=(15, 1)), sg.Spin([i for i in range(0000, 9999)], initial_value=1, size=(4, 1))],
    [sg.Text('_' * 30)],
    [sg.Text('Duration')],
    [sg.Text('HH:', size=(15, 1)), sg.Spin([i for i in range(0000, 9999)], initial_value=0, size=(2, 1))],
    [sg.Text('MM:', size=(15, 1)), sg.Spin([i for i in range(0000, 9999)], initial_value=2, size=(2, 1))],
    [sg.Text('_' * 30)],
    [sg.Text('Diversity')],
    [sg.Text('Number of runs:', size=(15, 1)), sg.Spin([i for i in range(0000, 9999)], initial_value=5, size=(4, 1))],
    [sg.Text('Expressions:', size=(15, 1)), sg.Spin([i for i in range(0000, 9999)], initial_value=5, size=(4, 1))],
    [sg.Text('Step:', size=(15, 1)), sg.Spin([i for i in range(0000, 9999)], initial_value=50, size=(4, 1))],
    [sg.Text('Pod limit:', size=(15, 1)), sg.Spin([i for i in range(0000, 9999)], initial_value=5, size=(4, 1))],
    [sg.Text('_' * 30)],
    [sg.Text('Load Testing')],
    [sg.Text('Spawn Pattern:', size=(15, 1)), sg.Combo(['Linear', 'Double Wave'], default_value="Linear")],
    [sg.Text('Tool:', size=(15, 1)), sg.Combo(['Locust', 'JMeter'], default_value="Locust"),
     sg.Checkbox('Locust History')],
    [sg.Output(size=(65, 15), key='log', )],
    [sg.Button("Start", key="start"), sg.Exit()]
]

window = sg.Window('Benchmark', layout, finalize=True)
logger_benchmark = p
logger_k8s = k
formatter = logging.Formatter("%(levelname)s: %(message)s")

viewHandler = logging.StreamHandler(window["log"].tk_out)
viewHandler.setFormatter(formatter)
logger_benchmark.addHandler(viewHandler)
logger_k8s.addHandler(viewHandler)
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED or event == 'Exit':
        break
    if event == "start":
        locust = True
        custom = False
        if values[9] == "JMeter":
            locust = False
        if values[8] == "Double Wave":
            custom = True
        load = np.arange(1, values[0], 1).tolist()
        start_run(name="teastore", load=load, spawn_rate=values[1], expressions=values[5], step=values[6],
                  pods_limit=values[7], runs=values[4], custom_shape=custom, history=values[10], sample=False,
                  locust=locust)
