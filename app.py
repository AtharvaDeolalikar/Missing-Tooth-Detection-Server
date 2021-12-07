import tensorflow as tf
import glob
import cv2
import tensorflow, keras
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
#from google.colab.patches import cv2_imshow
import requests
import random
from io import BytesIO
from PIL import Image
# %matplotlib inline

""" from google.colab import drive
drive.mount('/content/drive') """

model=tf.keras.models.load_model("missingtoothdetectionmodel.h5")

def Numbering(coordinate, img, img_name):
  #asceding all the teeth
  new_cor=[]
  temp_teeth=[]
  for i in coordinate['Teeth']:
    temp_teeth.append(i[0])
  for j in np.sort(temp_teeth):
    for i in coordinate['Teeth']:
      if j==i[0] and i not in new_cor:
        new_cor.append(i)

  coordinate['Teeth']=new_cor

  #seperate the upper and lower teeth by calcualting the mean of y cordiate , based on x1cordinate as point of perception
  upper_teeth=[]
  lower_teeth=[]

  #get the array cordinate as values 
  x1_cordinate=[]
  y1_cordinate=[]
  height=[]
  width=[]
  for i in coordinate['Teeth']:
    x1_cordinate.append(i[0])
    y1_cordinate.append(i[1])
    height.append(i[2])
    width.append(i[3])

  #getting mean values
  middle_x1=np.mean(x1_cordinate)
  middle_y1=np.mean(y1_cordinate)
  mean_height=np.mean(height)
  mean_width=np.mean(width)

  #seperating upper and lower teeth based on mean height and get their centers
  u=[]
  l=[]
  u_centers=[]
  l_centers=[]
  for j in coordinate['Teeth']:
    if j[1]>middle_y1:
      if j[0] not in l:
        lower_teeth.append(j)
        l.append(j[0])
        l_centers.append((int(j[0]+j[2]/2),int(j[1]+j[3]/2)))
    else:
      if j[0] not in u:
        upper_teeth.append(j)
        u.append(j[0])
        u_centers.append((int(j[0]+j[2]/2),int(j[1]+j[3]/2)))


  #check the teeth in each quadrant based on middle line
  lr=0
  ll=0
  tr=0
  tl=0
  for k,i in enumerate(upper_teeth):
    if i[0]<int(middle_x1):
      #i.append("TL")
      tl=tl+1
    else:
      #i.append("TR")
      tr=tr+1

  for k,i in enumerate(lower_teeth):
    if i[0]<int(middle_x1):
      #i.append("LL")
      ll=ll+1
    else:
      #i.append("LR")
      lr=lr+1

  import math
  def distance_(p1,p2):
    return int(math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) ))
  def middle_cordinate(P,Q):
    return (int((P[0]+Q[0])/2),int((P[1]+Q[1])/2))



  U_dist=[]
  L_dist=[]
  no_of_u_teeth=len(upper_teeth)
  no_of_l_teeth=len(lower_teeth)
  for i in range(no_of_u_teeth):
    U_dist.append(distance_(u_centers[i],u_centers[i-1]))
  del U_dist[0]
  sort_U_dist=np.sort(U_dist)

  for i in range(no_of_l_teeth):
    L_dist.append(distance_(l_centers[i],l_centers[i-1]))
  del L_dist[0]

  sort_L_dist=np.sort(L_dist)

  #draw circle in between the cordinate
  def circle_(P,Q,img_demo):
    R=middle_cordinate(P,Q)
    img_demo =cv2.circle(img_demo, R, 2, (0,0,255), 2)
    return img_demo


  #point out missing teeth
  missing_no=[]
  def between_missing_teeth_prediction(img_demo, no_of_l_teeth, ):
    #global no_of_l_teeth,no_of_u_teeth,lower_teeth,upper_teeth
    count_=0
    for m in range(16-no_of_l_teeth):
      if sort_L_dist[-1+1*m]/np.median(L_dist)>1.5:
        img_demo=circle_(l_centers[L_dist.index(sort_L_dist[-1+1*m])],l_centers[L_dist.index(sort_L_dist[-1+1*m])+1],img_demo)
        count_=count_+1
        for k in range(no_of_l_teeth):
          if middle_cordinate(l_centers[L_dist.index(sort_L_dist[-1+1*m])],l_centers[L_dist.index(sort_L_dist[-1+1*m])+1])[0]<lower_teeth[k][0]:
            a=list(l_centers[L_dist.index(sort_L_dist[-1])])
            #a[0]=a[0]+int(mean_width/2)
            #a[1]=a[1]+int(mean_height/2)
            a[0]=a[0]+10
            a[1]=a[1]+10
            a.append(int(mean_height))
            a.append(int(mean_width/2))
            lower_teeth.insert(k,a)
            missing_no.append(32-k)
            break
    no_of_l_teeth=no_of_l_teeth+count_
    count_=0
    for n in range(16-no_of_u_teeth):
      if sort_U_dist[-1+1*n]/np.median(U_dist)>1.5:

        img_demo=circle_(u_centers[U_dist.index(sort_U_dist[-1+1*n])],u_centers[U_dist.index(sort_U_dist[-1+1*n])+1],img_demo)
        count_=count_+1
        for k in range(no_of_u_teeth):
          if middle_cordinate(u_centers[U_dist.index(sort_U_dist[-1+1*n])],u_centers[U_dist.index(sort_U_dist[-1+1*n])+1])[0]<upper_teeth[k][0]:
            a=list(u_centers[U_dist.index(sort_U_dist[-1])])
            #a[0]=a[0]+int(mean_width/2)
            #a[1]=a[1]+int(mean_height/2)
            a[0]=a[0]+10
            a[1]=a[1]+10
            a.append(int(mean_height))
            a.append(int(mean_width/2))
            upper_teeth.insert(k,a)
            print("Missing tooth at",k+1)
            missing_no.append(k+1)
            break
    no_of_l_teeth=no_of_l_teeth+count_
    return img_demo,no_of_l_teeth,no_of_u_teeth,lower_teeth,upper_teeth


  plt.rcParams['figure.figsize'] =[15,15]
  img_demo=cv2.merge((img,img, img))

  for j,i in enumerate(upper_teeth):
    img_demo=cv2.circle(img_demo,u_centers[j] , 2, (255,0,255), 2)
    img_demo = cv2.line(img_demo,u_centers[j-1],u_centers[j], (0, 0, 255), 1)
    img_demo=cv2.putText(img_demo, '{}'.format(U_dist[j-1]), middle_cordinate(u_centers[j-1],u_centers[j]), cv2.FONT_HERSHEY_SIMPLEX,  0.3,  (0, 0, 255), 1, cv2.LINE_AA)
    
  for k,i in enumerate(lower_teeth):
    img_demo=cv2.circle(img_demo,l_centers[k] , 2, (255,0,255), 2)
    img_demo = cv2.line(img_demo,l_centers[k-1],l_centers[k],(0,0,255),1)
    img_demo=cv2.putText(img_demo, '{}'.format(L_dist[k-1]),middle_cordinate(l_centers[k-1],l_centers[k]), cv2.FONT_HERSHEY_SIMPLEX,  0.3,  (0, 0, 255), 1, cv2.LINE_AA)

  img_demo,no_of_l_teeth,no_of_u_teeth,lower_teeth,upper_teeth=between_missing_teeth_prediction(img_demo, no_of_l_teeth)

  plt.imshow(img_demo)


  #check the teeth in each quadrant based on middle line
  lr=0
  ll=0
  tr=0
  tl=0
  for k,i in enumerate(upper_teeth):
    if i[0]<int(middle_x1):
      i.append("TL")
      tl=tl+1
    else:
      i.append("TR")
      tr=tr+1

  for k,i in enumerate(lower_teeth):
    if i[0]<int(middle_x1):
      i.append("LL")
      ll=ll+1
    else:
      i.append("LR")
      lr=lr+1


  #check the tooth in each quandrant , if the tooth is greater than 8 in one quandrant shift it to adjacent quandrant 
  def proper_quadrant_checker(upper_teeth,lower_teeth):
    lr=0
    ll=0
    tr=0
    tl=0
    for k,i in enumerate(upper_teeth):
      if i[4]=="TL":
        tl=tl+1
      else:
        tr=tr+1

    for k,i in enumerate(lower_teeth):
      if i[4]=="LL":
        ll=ll+1
      else:
        lr=lr+1

    #shift teeth between quandrant when one quandrant has more than 8 teeth
    if lr>8:
      ll=ll+lr%8
      lr=lr-lr%8
    if ll>8:
      lr=lr+ll%8
      ll=ll-ll%8
    if tr>8:
      tl=tl+tr%8
      tr=tr-tr%8
    if tr>8:
      tl=tl+tr%8
      tr=tr-tr%8
    
    '''
    temp=lr
    lr=lr-lr//8+ll//8
    ll=ll-ll//8+temp//8
    temp=tr
    tr=tr-tr//8+tl//8
    tl=tl-tl//8+temp//8'''

    return lr,ll,tr,tl

  lr,ll,tr,tl=proper_quadrant_checker(upper_teeth,lower_teeth)

  lr_=lr
  ll_=ll
  tr_=tr
  tl_=tl

  for i in range(16-len(upper_teeth)):
    if tl!=8:
      teeth_=[upper_teeth[0][0]-15,upper_teeth[0][1],15, 58,"TL"]
      upper_teeth.insert(0,teeth_)
      tl=tl+1
      missing_no.append(tl-tl_)
    if tr!=8:
      teeth_=[upper_teeth[-1][0]+15,upper_teeth[-1][1],15, 58,"TR"]
      upper_teeth.append(teeth_)
      tr=tr+1
      missing_no.append(8+tr)

  for i in range(16-len(lower_teeth)):
    if ll!=8:
      teeth_=[lower_teeth[0][0]-15,lower_teeth[0][1],15, 58,"LL"]
      lower_teeth.insert(0,teeth_)
      ll=ll+1
      missing_no.append(ll-ll_)
    if lr!=8:
      teeth_=[lower_teeth[-1][0]+15,lower_teeth[-1][1],15, 58,"LR"]
      lower_teeth.append(teeth_)
      lr=lr+1
      missing_no.append(8+lr)



  plt.rcParams['figure.figsize'] =[15,15]
  img_demo=cv2.merge((img,img, img))
  img_demo = cv2.line(img_demo, (int(middle_x1),100), (int(middle_x1),400), (0, 0, 255), 1)
  img_demo = cv2.line(img_demo, (100,int(middle_y1+mean_height)), (400,int(middle_y1+mean_height)), (0, 0, 255), 1)

  for j,i in enumerate(upper_teeth):
    if (j+1) in missing_no:
      img_demo = cv2.rectangle(img_demo,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(255,255,0),1)
      img_demo=cv2.putText(img_demo, '{}'.format(i[4]), (i[0],i[1]+i[3]-10), cv2.FONT_HERSHEY_SIMPLEX,  0.3,  (255,255,0), 1, cv2.LINE_AA)
      img_demo=cv2.putText(img_demo, '{}'.format(j+1), (i[0],i[1]+i[3]), cv2.FONT_HERSHEY_SIMPLEX,  0.3, (255,255,0), 1, cv2.LINE_AA)
    else:
      img_demo = cv2.rectangle(img_demo,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(0,255,0),1)
      img_demo=cv2.putText(img_demo, '{}'.format(i[4]), (i[0],i[1]+i[3]-10), cv2.FONT_HERSHEY_SIMPLEX,  0.3,  (0, 0, 255), 1, cv2.LINE_AA)
      img_demo=cv2.putText(img_demo, '{}'.format(j+1), (i[0],i[1]+i[3]), cv2.FONT_HERSHEY_SIMPLEX,  0.3,  (0, 0, 255), 1, cv2.LINE_AA)

  #img_demo = cv2.rectangle(img_demo,(upper_teeth[j-1][0],upper_teeth[j-1][1]),(upper_teeth[j-1][0]+upper_teeth[j-1][2],upper_teeth[j-1][1]+upper_teeth[j-1][3]),(255,255,0),1)
    
  for k,i in enumerate(lower_teeth):
    if (32-k) in missing_no:
      img_demo = cv2.rectangle(img_demo,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(255,255,0),1)
      img_demo=cv2.putText(img_demo, '{}'.format(i[4]), (i[0],i[1]+i[3]-10), cv2.FONT_HERSHEY_SIMPLEX,  0.3,  (255,255,0), 1, cv2.LINE_AA)
      img_demo=cv2.putText(img_demo, '{}'.format(32-k), (i[0],i[1]+i[3]), cv2.FONT_HERSHEY_SIMPLEX,  0.3, (255,255,0), 1, cv2.LINE_AA)
    else:
      img_demo = cv2.rectangle(img_demo,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(0,255,0),1)
      img_demo=cv2.putText(img_demo, '{}'.format(i[4]), (i[0],i[1]+i[3]-10), cv2.FONT_HERSHEY_SIMPLEX,  0.3,  (0, 0, 255), 1, cv2.LINE_AA)
      img_demo=cv2.putText(img_demo, '{}'.format(32-k), (i[0],i[1]+i[3]), cv2.FONT_HERSHEY_SIMPLEX,  0.3,  (0, 0, 255), 1, cv2.LINE_AA)
    

  plt.imshow(img_demo)
  plt.axis('off')
  plt.savefig('final-' + img_name, bbox_inches='tight')

#@title Old Numbering Function

'''
def drawRect(x,y,w,h,color,img):
  # img = cv2.rectangle(img, (boxes['bbox']['left'], boxes['bbox']['top']), (boxes['bbox']['left'] + boxes['bbox']['width'], boxes['bbox']['top'] + boxes['bbox']['height']), color, 2)
  cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
  return img
def missingTooth(coordinate, img_name):
  inp = "/content/" + img_name
  teeth_data = coordinate
  if inp=='exit' or inp=='break':
      pass
  else:
    all_cords_of_teeth = []
    
    img = cv2.imread(inp)
    img = cv2.resize(img, (512, 512))

    raw_img =  img.copy()
    img_copy = img.copy()

    plt.imshow(img)
    plt.show()
    plt.rcParams["figure.figsize"] = (50,10)

    upper_teeth = []
    bottom_teeth = []

    for cord in teeth_data['Teeth']:
        if(cord[1]>250):
            upper_teeth.append(cord)
        else:
            bottom_teeth.append(cord)
            
    new_temp_list = upper_teeth+bottom_teeth

    for singleObject in new_temp_list:
      try:
  #print('\n\n** Processing Tooth featureId : ',singleObject['featureId'])

        x,y,w,h = singleObject[0], singleObject[1], singleObject[2], singleObject[3]
  #print('** Box Cordinates are ',singleObject['bbox'])

        is_tooth_exists = 'yes'

        coord = (x,y,w,h)
        centerCoord = (int(coord[0]+(coord[2]/2)), int(coord[1]+(coord[3]/2)))
        if is_tooth_exists!='yes':
          continue
        else:
          all_cords_of_teeth.append(['Teeth_', coord , centerCoord , is_tooth_exists])

      except Exception as e:
          print('\n Sufficient data not available of featureId : ',singleObject['featureId']," Error : ",e)


    all_cords_of_teeth,bottom_teeth_cord,upper_teeth_cord = process_coordinates(all_cords_of_teeth)
    bottom_teeth_cord.reverse()
    upper_teeth_cord.reverse()
    print('\n\n******************')
    print('**** Before Processing Coordinates *****\n\n')

    for index in range(len(all_cords_of_teeth)):
      if all_cords_of_teeth[index][3]=='yes':
          img = drawRect(all_cords_of_teeth[index][1][0],all_cords_of_teeth[index][1][1],all_cords_of_teeth[index][1][2],all_cords_of_teeth[index][1][3],(0,255,0),img)
      else:
          img = drawRect(all_cords_of_teeth[index][1][0],all_cords_of_teeth[index][1][1],all_cords_of_teeth[index][1][2],all_cords_of_teeth[index][1][3],(0,0,255),img)

      img = cv2.line(img, all_cords_of_teeth[index][2], (all_cords_of_teeth[index][2][0]+5,all_cords_of_teeth[index][2][1]+5), (255, 0, 0), 5)

      cv2.putText(img, str(index+1), all_cords_of_teeth[index][2], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255),  1)

    img = cv2.resize(img, (1280, 720))
    print(img.shape)
    plt.imshow(img)
    plt.show()
    plt.rcParams["figure.figsize"] = (20,20)

    print('\n\n********************')
    print('**** After Processing Final Output Image *****\n\n')

    teeth_count = 1
    for index in range(len(upper_teeth_cord)):
      if upper_teeth_cord[index][3]=='yes':
          img_copy = drawRect(upper_teeth_cord[index][1][0],upper_teeth_cord[index][1][1],upper_teeth_cord[index][1][2],upper_teeth_cord[index][1][3],(0,255,0),img_copy)
      else:
          img_copy = drawRect(upper_teeth_cord[index][1][0],upper_teeth_cord[index][1][1],upper_teeth_cord[index][1][2],upper_teeth_cord[index][1][3],(0,0,255),img_copy)

      img_copy = cv2.line(img_copy, upper_teeth_cord[index][2], (upper_teeth_cord[index][2][0]+2,upper_teeth_cord[index][2][1]+2), (255, 0, 0), 2)

      cv2.putText(img_copy, str(teeth_count), upper_teeth_cord[index][2], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255),  1)

      teeth_count = teeth_count +1


    for index in range(len(bottom_teeth_cord)):
      if bottom_teeth_cord[index][3]=='yes':
          img_copy = drawRect(bottom_teeth_cord[index][1][0],bottom_teeth_cord[index][1][1],bottom_teeth_cord[index][1][2],bottom_teeth_cord[index][1][3],(0,255,0),img_copy)
      else:
          img_copy = drawRect(bottom_teeth_cord[index][1][0],bottom_teeth_cord[index][1][1],bottom_teeth_cord[index][1][2],bottom_teeth_cord[index][1][3],(0,0,255),img_copy)

      img_copy = cv2.line(img_copy, bottom_teeth_cord[index][2], (bottom_teeth_cord[index][2][0]+2,bottom_teeth_cord[index][2][1]+2), (255, 0, 0), 2)

      cv2.putText(img_copy, str(teeth_count), bottom_teeth_cord[index][2], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255),  1)

      teeth_count = teeth_count +1
    img_copy = cv2.resize(img_copy, (1280, 720))
    plt.imshow(img_copy)
    plt.show()
    plt.rcParams["figure.figsize"] = (20,20)
    plt.imshow(img_copy)
    plt.axis('off')
    plt.savefig('final-' + img_name, bbox_inches='tight')
    return True

  f = coordinate
  teeth_data = coordinate
  image_id = ''
  img_url = ''
  print("IMG Path", img_name)
  inp = "/content/" + img_name
  if inp=='exit' or inp=='break':
      pass
  else:
    all_cords_of_teeth = []
    img = cv2.imread(inp)
    img = cv2.resize(img, (512, 512))
    raw_img =  img.copy()
    img_copy = img.copy()

    for singleObject in teeth_data['Teeth']:
      try:
        x,y,w,h = singleObject[0], singleObject[1], singleObject[2], singleObject[3]
        is_tooth_exists = 'yes'
        coord = (x,y,w,h)
        centerCoord = (int(coord[0]+(coord[2]/2)), int(coord[1]+(coord[3]/2)))
        if is_tooth_exists!='yes':
          continue
        else:
          all_cords_of_teeth.append(['Teeth_', coord , centerCoord , is_tooth_exists])
      except Exception as e:
          print('\nSufficient data not available of featureId : ',singleObject['featureId']," Error : ",e)

    all_cords_of_teeth,upper_teeth_cord,bottom_teeth_cord = process_coordinates(all_cords_of_teeth)

    for index in range(len(all_cords_of_teeth)):
      if all_cords_of_teeth[index][3]=='yes':
          img = drawRect(all_cords_of_teeth[index][1][0],all_cords_of_teeth[index][1][1],all_cords_of_teeth[index][1][2],all_cords_of_teeth[index][1][3],(0,255,0),img)
      else:
          img = drawRect(all_cords_of_teeth[index][1][0],all_cords_of_teeth[index][1][1],all_cords_of_teeth[index][1][2],all_cords_of_teeth[index][1][3],(0,0,255),img)

      img = cv2.line(img, all_cords_of_teeth[index][2], (all_cords_of_teeth[index][2][0]+5,all_cords_of_teeth[index][2][1]+5), (255, 0, 0), 5)

      cv2.putText(img, str(index+1), all_cords_of_teeth[index][2], cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255),  2)
    img = cv2.resize(img, (1280, 720))
    teeth_count = 1

    for index in range(len(upper_teeth_cord)):
      if upper_teeth_cord[index][3]=='yes':
          img_copy = drawRect(upper_teeth_cord[index][1][0],upper_teeth_cord[index][1][1],upper_teeth_cord[index][1][2],upper_teeth_cord[index][1][3],(0,255,0),img_copy)
      else:
          img_copy = drawRect(upper_teeth_cord[index][1][0],upper_teeth_cord[index][1][1],upper_teeth_cord[index][1][2],upper_teeth_cord[index][1][3],(0,0,255),img_copy)

      img_copy = cv2.line(img_copy, upper_teeth_cord[index][2], (upper_teeth_cord[index][2][0]+5,upper_teeth_cord[index][2][1]+5), (255, 0, 0), 5)

      cv2.putText(img_copy, str(teeth_count), upper_teeth_cord[index][2], cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255),  2)

      teeth_count = teeth_count +1

    for index in range(len(bottom_teeth_cord)):
      if bottom_teeth_cord[index][3]=='yes':
          img_copy = drawRect(bottom_teeth_cord[index][1][0],bottom_teeth_cord[index][1][1],bottom_teeth_cord[index][1][2],bottom_teeth_cord[index][1][3],(0,255,0),img_copy)
      else:
          img_copy = drawRect(bottom_teeth_cord[index][1][0],bottom_teeth_cord[index][1][1],bottom_teeth_cord[index][1][2],bottom_teeth_cord[index][1][3],(0,0,255),img_copy)

      img_copy = cv2.line(img_copy, bottom_teeth_cord[index][2], (bottom_teeth_cord[index][2][0]+5,bottom_teeth_cord[index][2][1]+5), (255, 0, 0), 5)

      cv2.putText(img_copy, str(teeth_count), bottom_teeth_cord[index][2], cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255),  2)

      teeth_count = teeth_count +1

    img_copy = cv2.resize(img_copy, (2791, 1316))
    plt.rcParams["figure.figsize"] = (20,20)
    plt.imshow(img_copy)
    plt.axis('off')
    plt.savefig('final-' + img_name, bbox_inches='tight')
    #plt.show()
    return "Done!!!" '''

def toothDetection(img_name):
  img="/content/"+ img_name
  img = cv2.imread(img)       
  i1,i2,i3=cv2.split(img)     
  img = cv2.resize(i1, (512, 512))
  img=np.array(img)
  img=img.reshape([1,512,512,1])
  img=np.float32(img/255)
  predict_img=model.predict(img)
  img=img.reshape([512,512])
  predict=predict_img.reshape([512,512])
  img_=cv2.merge((img,img, img))

  predict_=predict>0.9
  mask=np.uint8(predict_)*255
  contours, _ = cv2.findContours(mask.reshape([512,512]), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  no_of_teeth=0
  coordinate={}
  list_=[]

  for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)
    if  cv2.contourArea(contour) < 100 :
      continue
    if  cv2.contourArea(contour) > 1100  :
      if h>100:
        img_ = cv2.rectangle(img_,(x,y),(x+w,y+int(h/2)),(0,255,0),1)
        img_ = cv2.rectangle(img_,(x,y+int(h/2)+1),(x+w,y+h),(0,255,0),1)
        list_.append([x,y,w,h])
        continue
      if w>30:
        img_ = cv2.rectangle(img_,(x,y),(x+int(w/2),y+h),(0,255,0),1)
        img_ = cv2.rectangle(img_,(x+int(w/2)+2,y),(x+w,y+h),(0,255,0),1)
        list_.append([x,y,w,h])
        continue

    img_ = cv2.rectangle(img_,(x,y),(x+w,y+h),(0,255,0),1)
    list_.append([x,y,w,h])
    no_of_teeth=no_of_teeth+1

  coordinate["Teeth"]=list_
  Numbering(coordinate, img, img_name)
  return coordinate

from flask import Flask, request, send_file
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['POST'])
def hello():
  if request.files:
    if os.path.exists("final.png"):
      os.remove("final.png")
    file = request.files['file']
    print(f"Received file {file.filename}")
    file.save(file.filename)
  coordinates= toothDetection(file.filename)
  return coordinates

@app.route("/image/<name>", methods=['POST', 'GET'])
def getImage(name):
  print("request for", name)
  if os.path.exists("final-" + name):
    return send_file("final-" + name)
  else:
    return "Image not found", 404

if __name__ == "__main__":
  app.run()