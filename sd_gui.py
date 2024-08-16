import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Button, Label
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sum_diff import *


root = tk.Tk()
root.title('GNR 607 Sum - Diff algorithm')
root.geometry("600x600")

class res():
    def __init__(self) :
        self.grayscale_im_pil = None
        self.asm_pil = None
        self.idm_pil = None
        self.con_pil = None
        self.ent_pil = None
        self.ori_pil = None
        self.tk_img = None

varobj = res()

# label1 = "Select an Image:"
image_label = tk.Label(root, text="Select an Image:")
image_label.pack()
global file_path
file_path = None
stem = None
global tk_img, im, panel
global grayscale_im_pil, asm_pil, idm_pil, con_pil, ent_pil, ori_pil

def sel_image():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif")])

    if file_path is None:
        # messagebox.
        messagebox.showerror("Error", f"An error occurred: No such file exists")
    print(file_path, "HA")
    stem = os.path.splitext(file_path)[0]
    image_label['text'] = stem


sel_button = tk.Button(root, text="Select", command=sel_image)
sel_button.pack()

# Numerical inputs for d1, d2 and num_clusters
d1_lab = tk.Label(root, text="D1 parameter:")
d1_lab.pack()
e_d1 = ttk.Entry(root)
e_d1.pack()

d2_lab = tk.Label(root, text="D2 parameter:")
d2_lab.pack()
e_d2 = ttk.Entry(root)
e_d2.pack()

n_lab = tk.Label(root, text="Vertical Window Size 'N':")
n_lab.pack()
e_n = ttk.Entry(root)
e_n.pack()

m_lab = tk.Label(root, text="Horizontal Window Size 'M':")
m_lab.pack()
e_m = ttk.Entry(root)
e_m.pack()

num_cluster_lab = tk.Label(root, text="Desired Number of Clusters 'k':")
num_cluster_lab.pack()
e_num_cluster = ttk.Entry(root)
e_num_cluster.pack()

def save_file(img):
    filename = filedialog.asksaveasfile(mode='w', defaultextension=".jpg")
    if not filename:
        return
    im = Image.fromarray(img)
    im.save(filename)
grayscale_im_pil = None
asm_pil = None
idm_pil = None
con_pil = None
ent_pil = None
ori_pil = None

def open_img(im_name):
    global grayscale_im_pil, asm_pil, idm_pil, con_pil, ent_pil, ori_pil
    global tk_img, im, panel
    # print(im_name, "WOW")
    if im_name == "clustered":
        # print(grayscale_im_pil.shape)
        global grayscale_im_pil
        print(grayscale_im_pil)
        nwin = tk.Toplevel()
        nwin.title("Clustered Image")
        tk_img = ImageTk.PhotoImage(image=Image.fromarray(grayscale_im_pil))
        panel = Label(nwin, image=tk_img)
        panel.image = tk_img
        im = grayscale_im_pil 
        panel.pack() 
        savebtn = Button(nwin, text='Save As', command=lambda : save_file(im))
        savebtn.pack()
        nwin.mainloop()
    elif im_name== "original":
        # print("Hereee")
        global ori_pil
        nwin = tk.Toplevel()
        nwin.title("Original Image")
        tk_img = ImageTk.PhotoImage(image=Image.fromarray(ori_pil))
        panel = Label(nwin, image=tk_img)
        panel.image = tk_img
        im = ori_pil
        panel.pack()
        savebtn = Button(nwin, text='Save As', command=lambda : save_file(im))
        savebtn.pack()
        nwin.mainloop()
    elif im_name== "asm":
        # print(asm_pil.shape)
        global asm_pil
        # print(asm_pil)
        varobj.tk_img = ImageTk.PhotoImage(image=Image.fromarray(varobj.asm_pil))
        # print(tk_img)
        nwin = tk.Toplevel()
        nwin.title("Angular Second Moment Image")
        # panel = Label(nwin, image=varobj.tk_img)
        panel = Label(nwin, image=ImageTk.PhotoImage(image=Image.fromarray(varobj.asm_pil)))
        panel.image = varobj.tk_img
        im = varobj.asm_pil
        savebtn = Button(nwin, text='Save As', command=lambda : save_file(im))
        savebtn.pack()
        nwin.mainloop()
    elif im_name == "idm":
        global idm_pil
        idm_p2 = idm_pil
        nwin = tk.Toplevel()
        nwin.title("Inverse Difference Moment Image")
        # tk_img = ImageTk.PhotoImage(image=Image.fromarray(idm_pil))
        tk_img = ImageTk.PhotoImage(image=Image.fromarray(idm_p2))
        panel = Label(nwin, image=tk_img)
        panel.image = tk_img
        im = idm_p2
        savebtn = Button(nwin, text='Save As', command=lambda : save_file(im))
        savebtn.pack()
        nwin.mainloop()
    elif im_name == "con":
        global con_pil
        nwin = tk.Toplevel()
        nwin.title("Contrast Image")
        tk_img = ImageTk.PhotoImage(image=Image.fromarray(con_pil))
        panel = Label(nwin, image=tk_img)
        panel.image = tk_img
        im = con_pil
        savebtn = Button(nwin, text='Save As', command=lambda : save_file(im))
        savebtn.pack()
        nwin.mainloop()
    elif im_name == "ent":
        global ent_pil
        nwin = tk.Toplevel()
        nwin.title("Entropy Image")
        tk_img = ImageTk.PhotoImage(image=Image.fromarray((ent_pil)))
        panel = Label(nwin, image=tk_img)
        panel.image = tk_img
        im = ent_pil
        savebtn = Button(nwin, text='Save As', command=lambda : save_file(im))
        savebtn.pack()
        nwin.mainloop()

    # savebtn = Button(root, text='save image', command=lambda : save_file(im))
    # savebtn.pack()
    # print("Hehe")


def proc(file_path, d1, d2,n, m, c):
    global grayscale_im_pil, asm_pil, idm_pil, con_pil, ent_pil, ori_pil
    global varobj
    grayscale_im_pil, asm_pil, idm_pil, con_pil, ent_pil, ori_pil = process_image(file_path, d1, d2, n, m, c)
    varobj.asm_pil = asm_pil
    varobj.con_pil = con_pil
    varobj.grayscale_im_pil = grayscale_im_pil
    varobj.idm_pil = idm_pil
    varobj.ori_pil = ori_pil
    varobj.ent_pil = ent_pil
    print(file_path)
    btn1 = Button(root, text='Original Image', command=lambda:open_img("original"))
    # btn2 = Button(root, text='ASM', command=lambda:open_img("asm"))
    # btn3 = Button(root, text='IDM', command=lambda:open_img("idm"))
    # btn4 = Button(root, text='CON', command=lambda:open_img("con"))
    # btn5 = Button(root, text='ENT', command=lambda:open_img("ent"))
    # btn2 = Button(root, text='Angular Second Moment Image', command=lambda:save_file(asm_pil))
    # btn3 = Button(root, text='Save Inverse Difference Moment Image', command=lambda:save_file(idm_pil))
    # btn4 = Button(root, text='Save Contrast Image', command=lambda:save_file(con_pil))
    # btn5 = Button(root, text='Save Entropy Image', command=lambda:save_file(ent_pil))
    btn6 = Button(root, text='Clustered Image', command=lambda:open_img("clustered"))
    btn1.pack()
    # btn2.pack()
    # btn3.pack()
    # btn4.pack()
    # btn5.pack()
    btn6.pack()


proc_button = tk.Button(root, text="Process", command = lambda: proc(file_path, int(e_d1.get()), int(e_d2.get()), int(e_n.get()), int(e_m.get()), int(e_num_cluster.get())))
proc_button.pack()


# # stem = os.path.splitext(file_path)[0]
# image = cv2.imread(file_path)

# # Convert the image to grayscale
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Convert the grayscale image to a NumPy array
# gray_array = np.array(gray_image)
# array = np.array(gray_array, dtype=np.int16)

# shape = gray_array.shape
# width = shape[1]
# height = shape[0]
# pixel_range = (0, 255)
# m=10 #horizontal j
# n=11 #vertical i
# d1 = e_d1 # horizontal
# d2 = e_d2 # vertical

# arr1 = {i: 0 for i in range(2*pixel_range[1]+1)}
# arr2 = {i: 0 for i in range(-pixel_range[1], pixel_range[1]+1)}
# temp_sum = {i: 0 for i in range(2*pixel_range[1]+1)}
# temp_diff = {i: 0 for i in range(-pixel_range[1], pixel_range[1]+1)}

# asm = np.array([[0 for j in range(width-m)] for i in range(height-n)],dtype=float)
# idm = np.array([[0 for j in range(width-m)] for i in range(height-n)],dtype=float)
# con = np.array([[0 for j in range(width-m)] for i in range(height-n)],dtype=float)
# ent = np.array([[0 for j in range(width-m)] for i in range(height-n)],dtype=float)


# for k in range(height-n):
#   for l in range(width-m):
#     if(l==0):
#       for i in range(n-d2):
#         for j in range(m-d1):
#           arr1[array[k+i][l+j]+array[k+i+d2][l+j+d1]] = arr1[array[k+i][l+j]+array[k+i+d2][l+j+d1]] + 1

#           arr2[array[k+i][l+j]-array[k+i+d2][l+j+d1]] = arr2[array[k+i][l+j]-array[k+i+d2][l+j+d1]] + 1

#           if(j==0):
#             temp_sum[array[k+i][l+j]+array[k+i+d2][l+j+d1]] = temp_sum[array[k+i][l+j]+array[k+i+d2][l+j+d1]] + 1
#             temp_diff[array[k+i][l+j]-array[k+i+d2][l+j+d1]] = temp_diff[array[k+i][l+j]-array[k+i+d2][l+j+d1]] + 1

#       S = norm(np.array(list(arr1.values())))
#       D = norm(np.array(list(arr2.values())))
#       print(np.array(list(arr1.values())), S, np.array(list(arr2.values())), D)
#       asm[k][l]=float(ASM(S,D))
#       idm[k][l]=float(IDM(S,D))
#       con[k][l]=float(CON(S,D))
#       ent[k][l]=float(ENT(S,D))

#     else:
#       arr1 = {key: arr1[key] - temp_sum[key] for key in arr1 if key in temp_sum}
#       arr2 = {key: arr2[key] - temp_diff[key] for key in arr2 if key in temp_diff}
#       temp_sum = {key: 0 for key in temp_sum}
#       temp_diff = {key: 0 for key in temp_diff}
#       for i in range(n-d2):
#         temp_sum[array[k+i][l+m-d1-1]+array[k+i+d2][l+m-1]] = temp_sum[array[k+i][l+m-d1-1]+array[k+i+d2][l+m-1]] + 1
#         temp_diff[array[k+i][l+m-d1-1]-array[k+i+d2][l+m-1]] = temp_diff[array[k+i][l+m-d1-1]-array[k+i+d2][l+m-1]] + 1
#       arr1 = {key: arr1[key] + temp_sum[key] for key in arr1 if key in temp_sum}
#       arr2 = {key: arr2[key] + temp_diff[key] for key in arr2 if key in temp_diff}
#       S = norm(np.array(list(arr1.values())))
#       D = norm(np.array(list(arr2.values())))
#       asm[k][l]=float(ASM(S,D))
#       idm[k][l]=float(IDM(S,D))
#       con[k][l]=float(CON(S,D))
#       ent[k][l]=float(ENT(S,D))


# data ={}
# df = pd.DataFrame(data)


# # Create a 2D NumPy array
# array_2d = asm

# # Flatten the array
# flattened_array_asm = array_2d.ravel()

# # Create a Pandas Series from the flattened array
# asm_s = pd.Series(flattened_array_asm)
# df['asm'] = asm_s


# # Create a 2D NumPy array
# array_2d = idm

# # Flatten the array
# flattened_array_idm = array_2d.ravel()

# # Create a Pandas Series from the flattened array
# idm_s = pd.Series(flattened_array_idm)
# df['idm'] = idm_s


# # Create a 2D NumPy array
# array_2d = con

# # Flatten the array
# flattened_array_con = array_2d.ravel()

# # Create a Pandas Series from the flattened array
# con_s = pd.Series(flattened_array_con)
# df['con'] = con_s

# # Create a 2D NumPy array
# array_2d = ent

# # Flatten the array
# flattened_array_ent = array_2d.ravel()

# # Create a Pandas Series from the flattened array
# ent_s = pd.Series(flattened_array_ent)
# df['ent'] = ent_s

# # save plot in a file
# # X = df.iloc[:,2]
# # y = df.iloc[:,1]
# # plt.scatter(X,y)

# scaled_data = df
# kmeans = KMeans(n_clusters=e_num_cluster,max_iter=1000)
# kmeans.fit(scaled_data)

# cl = kmeans.predict(scaled_data)

# # X = df.iloc[:,1]
# # y = df.iloc[:,-1]
# # plt.scatter(X,y,c=cl)

# # fitting KMeans
# kmeans = KMeans(n_clusters=e_num_cluster)
# kmeans.fit(scaled_data)

# final_cl = kmeans.predict(scaled_data)

# max_v = max_arr(final_cl)
# final_cl = final_cl*(255/max_v)

# final_cl = final_cl.reshape(ent.shape[0],ent.shape[1])

# count = 0
# for i in final_cl:
#   for j in i:
#     count+=j

# count = count/255


# df_new = pd.DataFrame(final_cl)
# sample_data_path

# grayscale_image = cv2.merge((final_cl,final_cl,final_cl))
# grayscale_im_pil = cv2.cvtColor(grayscale_image, cv2.COLOR_BGR2RGB)
# asm_pil = cv2.cvtColor(asm, cv2.COLOR_BGR2RGB)
# idm_pil = cv2.cvtColor(idm, cv2.COLOR_BGR2RGB)
# con_pil = cv2.cvtColor(con, cv2.COLOR_BGR2RGB)
# ent_pil = cv2.cvtColor(ent, cv2.COLOR_BGR2RGB)
# ori_pil = cv2.cvtColor(gray_array, cv2.COLOR_BGR2RGB)




# btn1 = Button(root, text='Original Image', command=lambda:open_img("original"))
# btn2 = Button(root, text='ASM', command=lambda:open_img("asm"))
# btn3 = Button(root, text='IDM', command=lambda:open_img("idm"))
# btn4 = Button(root, text='CON', command=lambda:open_img("con"))
# btn5 = Button(root, text='ENT', command=lambda:open_img("ent"))


root.mainloop()


# Save the grayscale image to a file (optional)
# cv2.imwrite('output.png', grayscale_image)
# df_new.to_csv('sample_data.csv', index=True)