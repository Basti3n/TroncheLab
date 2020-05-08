import tkinter
from tkinter import PhotoImage
from tkinter.ttk import Frame, Button, Style

from main.src.process.essai_process import EssaiProcess
from main.src.process.main_process import MainProcess


class Application(Frame):
    master: tkinter
    width: tkinter
    height: tkinter
    quit: Button
    start_main_process: Button
    start_essai_process: Button

    def __init__(self, width: int, height: int, master: tkinter = None ):
        super().__init__(master)
        self.width, self.height = width, height
        self.master = master
        self.pack()

        self.master.style = Style()
        self.master.style.theme_use("clam")

        self._add_data_to_app()
        self._create_widgets()

    def _create_widgets(self) -> None:
        self.start_main_process = Button(self, text='Start Main Process', command=MainProcess.run)
        self.start_essai_process = Button(self, text='Start Essai Process', command=EssaiProcess.run)
        self.start_main_process.grid(column=0, row=0, pady=10)
        self.start_essai_process.grid(column=0, row=1, pady=10)
        self.create_leave_button()


    def _add_data_to_app(self) -> None:
        self.master.title('TroncheLab')
        self.master.geometry(f'{self.width}x{self.height}')
        self.master.iconbitmap('../resources/image/icon.ico')
        self.master.tk.call('wm', 'iconphoto', self.master._w, PhotoImage(file='../resources/image/icon.png'))

    def create_leave_button(self) -> None :
        style = Style()
        style.configure('W.TButton', font=('calibri', 10, 'bold', 'underline'), foreground='red')
        self.quit = Button(self, text='QUIT', command=self.master.destroy, style='W.TButton')
        self.quit.grid(column=0, row=2, pady=10)

