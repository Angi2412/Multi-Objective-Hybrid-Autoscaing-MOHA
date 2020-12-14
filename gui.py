# This will be the gui for the sandbox
import PySimpleGUI as sg

sg.theme('Reddit')

layout = [
    [sg.Text('Please enter your microservice information')],
    [sg.Text('Name:', size=(15, 1)), sg.InputText()],
    [sg.Text('Port:', size=(15, 1)), sg.InputText()],
    [sg.Text('Dockerfile:', size=(15, 1)), sg.In(size=(35, 1)), sg.FileBrowse()],
    [sg.Submit(), sg.Cancel()]
]

window = sg.Window('PodAutoScaler', layout)
event, values = window.read()
window.close()
print(event, values[0], values[1], values[2])  # the input data looks like a simple list when auto numbered
