import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Button
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

def preproc_image(file_path):
    # Pre Process the image and return the grayscal image

    # Load the image
    image = cv2.imread(file_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to a NumPy array
    gray_array = np.array(gray_image)

    return gray_image

def norm(ar):
    sum = np.sum(ar)
    norm_data = ar / sum
    return norm_data

def max_arr(arr):
    return np.max(arr)

def ASM(S,D):
    Psum=0
    Pdiff=0

    for i in range(0,len(S)):
        Psum += (S[i])**2

    for i in range(0,len(S)):
        Pdiff += (D[i])**2

    return Psum*Pdiff

def IDM(S,D):
    Sum=0
    L= (len(D)+1)/2
    for j in range(0,len(D)):
        Sum += D[j]/(1+(j-L+1)**2)

    return Sum

def CON(S,D):
    Sum=0
    L= (len(D)+1)/2
    for j in range(0,len(D)):
        Sum += ((j-L+1)**2)*D[j]

    return Sum

def ENT(S,D):
    SumS=0
    SumD=0

    for i in range(0,len(S)):
        if(S[i]!=0):
            SumS += S[i]*(math.log10(S[i]))
        if(D[i]!=0):
            SumD += D[i]*(math.log10(D[i]))

    return -SumS-SumD

def process_image(file_path, e_d1, e_d2, e_n, e_m, e_num_cluster):
    image = cv2.imread(file_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to a NumPy array
    gray_array = np.array(gray_image)
    array = np.array(gray_array, dtype=np.int16)

    shape = gray_array.shape
    width = shape[1]
    height = shape[0]
    pixel_range = (0, 255)
    m=e_n #horizontal j
    n=e_m #vertical i
    d1 = e_d1 # horizontal
    d2 = e_d2 # vertical

    arr1 = {i: 0 for i in range(2*pixel_range[1]+1)}
    arr2 = {i: 0 for i in range(-pixel_range[1], pixel_range[1]+1)}
    temp_sum = {i: 0 for i in range(2*pixel_range[1]+1)}
    temp_diff = {i: 0 for i in range(-pixel_range[1], pixel_range[1]+1)}

    asm = np.array([[0 for j in range(width-m)] for i in range(height-n)],dtype=float)
    idm = np.array([[0 for j in range(width-m)] for i in range(height-n)],dtype=float)
    con = np.array([[0 for j in range(width-m)] for i in range(height-n)],dtype=float)
    ent = np.array([[0 for j in range(width-m)] for i in range(height-n)],dtype=float)


    for k in range(height-n):
      for l in range(width-m):
        arr1 = {i: 0 for i in range(2*pixel_range[1]+1)}
        arr2 = {i: 0 for i in range(-pixel_range[1], pixel_range[1]+1)}
        for i in range(n-d2):
          for j in range(m-d1):
            arr1[array[k+i][l+j]+array[k+i+d2][l+j+d1]] = arr1[array[k+i][l+j]+array[k+i+d2][l+j+d1]] + 1

            arr2[array[k+i][l+j]-array[k+i+d2][l+j+d1]] = arr2[array[k+i][l+j]-array[k+i+d2][l+j+d1]] + 1

            if(j==0):
              temp_sum[array[k+i][l+j]+array[k+i+d2][l+j+d1]] = temp_sum[array[k+i][l+j]+array[k+i+d2][l+j+d1]] + 1
              temp_diff[array[k+i][l+j]-array[k+i+d2][l+j+d1]] = temp_diff[array[k+i][l+j]-array[k+i+d2][l+j+d1]] + 1

        S = norm(np.array(list(arr1.values())))
        D = norm(np.array(list(arr2.values())))
        # print(np.array(list(arr1.values())), S, np.array(list(arr2.values())), D)
        asm[k][l]=float(ASM(S,D))
        idm[k][l]=float(IDM(S,D))
        con[k][l]=float(CON(S,D))
        ent[k][l]=float(ENT(S,D))



    data ={}
    df = pd.DataFrame(data)


    # Create a 2D NumPy array
    array_2d = asm

    # Flatten the array
    flattened_array_asm = array_2d.ravel()

    # Create a Pandas Series from the flattened array
    asm_s = pd.Series(flattened_array_asm)
    df['asm'] = asm_s


    # Create a 2D NumPy array
    array_2d = idm

    # Flatten the array
    flattened_array_idm = array_2d.ravel()

    # Create a Pandas Series from the flattened array
    idm_s = pd.Series(flattened_array_idm)
    df['idm'] = idm_s


    # Create a 2D NumPy array
    array_2d = con

    # Flatten the array
    flattened_array_con = array_2d.ravel()

    # Create a Pandas Series from the flattened array
    con_s = pd.Series(flattened_array_con)
    df['con'] = con_s

    # Create a 2D NumPy array
    array_2d = ent

    # Flatten the array
    flattened_array_ent = array_2d.ravel()

    # Create a Pandas Series from the flattened array
    ent_s = pd.Series(flattened_array_ent)
    df['ent'] = ent_s

    # save plot in a file
    # X = df.iloc[:,2]
    # y = df.iloc[:,1]
    # plt.scatter(X,y)

    scaled_data = df
    kmeans = KMeans(n_clusters=e_num_cluster,max_iter=1000)
    kmeans.fit(scaled_data)

    cl = kmeans.predict(scaled_data)

    # X = df.iloc[:,1]
    # y = df.iloc[:,-1]
    # plt.scatter(X,y,c=cl)

    # fitting KMeans
    kmeans = KMeans(n_clusters=e_num_cluster)
    kmeans.fit(scaled_data)

    final_cl = kmeans.predict(scaled_data)

    max_v = max_arr(final_cl)
    final_cl = final_cl*(255/max_v)

    final_cl = final_cl.reshape(ent.shape[0],ent.shape[1])

    count = 0
    for i in final_cl:
      for j in i:
        count+=j

    count = count/255


    df_new = pd.DataFrame(final_cl)
    final_cl = final_cl.astype(np.uint8)
    # sample_data_path

    grayscale_image = cv2.merge((final_cl,final_cl,final_cl))
    grayscale_im_pil = cv2.cvtColor(grayscale_image, cv2.COLOR_BGR2RGB)
    asm = asm.astype(np.uint8)
    idm = idm.astype(np.uint8)
    con = con.astype(np.uint8)
    ent = ent.astype(np.uint8)
    gray_array = gray_array.astype(np.uint8)
    asm_pil = cv2.cvtColor(asm, cv2.COLOR_BGR2RGB)
    cv2.imwrite("ASM.png", asm_pil)
    idm_pil = cv2.cvtColor(idm, cv2.COLOR_BGR2RGB)
    cv2.imwrite("IDM.png", idm_pil)
    con_pil = cv2.cvtColor(con, cv2.COLOR_BGR2RGB)
    cv2.imwrite("CON.png", con_pil)
    ent_pil = cv2.cvtColor(ent, cv2.COLOR_BGR2RGB)
    cv2.imwrite("ENT.png", ent_pil)
    ori_pil = cv2.cvtColor(gray_array, cv2.COLOR_BGR2RGB)
    return grayscale_im_pil, asm_pil, idm_pil, con_pil, ent_pil, ori_pil

