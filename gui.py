import tkinter as tk
from tkinter import Tk, Label, Button, StringVar, Frame
from tkinter.filedialog import askopenfilename
import os
from DataComp import zip_file, unzip_file

root = Tk()

class MyGUI:

	tempdir = ""

	def __init__(self, master):
		self.master = master
		master.title("File Compression")

		self.master.geometry("400x230+300+200")
		self.master.config(bg='#fff')
		#self.master.resizable(0,0)

		self.label = Label(master, bg = '#fff', text="Choose a File to compress")
		self.label.pack(pady = 10)

		self.open_file_button = Button(master, text="Compress File", command=self.c_file)
		self.open_file_button.pack(pady = 10)

		self.open_file_button = Button(master, text="Decompress File", command=self.d_file)
		self.open_file_button.pack(pady = 10)


		#self.compress_file_button = Button(master, text="Compress File", command=self.compress_file)
		#self.compress_file_button.pack()

		self.close_button = Button(master, text="Close", command=master.quit)
		self.close_button.pack(pady = 10)


		
		
		

	def c_file(self):
		print("get location of file here")
		#filename = askopenfilename()
		currdir = os.getcwd()
		self.tempdir = askopenfilename(parent=root, initialdir=currdir, title='Please select a directory')
		if len(self.tempdir) > 0:
			print("You chose %s" % self.tempdir)
			self.label.configure(text="You chose %s" % self.tempdir)
			#self.compress_file_button = Button(self.master, text="Compress File", command=self.compress_file(self.tempdir)).pack()
			#compress_file(self.tempdir)
			zip_file(self.tempdir)

	def compress_file(self, dir):
		print("enter command to compress the file %s" % dir)
		#f = open(dir, "rb")
		#byteArr = map(ord, f.read())
		#byteArr = map(lambda x : '{0:08b}'.format(x),byteArr)
		#print(byteArr)
		zip_file(dir)

	def d_file(self):
		print("get location of file to decompress here")
		#filename = askopenfilename()
		currdir = os.getcwd()
		self.tempdir = askopenfilename(parent=root, initialdir=currdir, title='Please select a directory')
		if len(self.tempdir) > 0:
			print("You chose %s" % self.tempdir)
			self.label.configure(text="You chose %s" % self.tempdir)
			#decompress_file(self.tempdir)
			unzip_file(self.tempdir)
		

	def decompress_file(self, dir):
		print("enter comand to decompress the file %s" % dir)
		unzip_file(dir)

my_gui = MyGUI(root)
root.mainloop()
