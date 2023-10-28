import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer  # Removing stem words
from sklearn.feature_extraction.text import CountVectorizer  # To convert text to numerical data
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from collections import defaultdict
import warnings  # disable python warnings
import requests
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import csv

warnings.filterwarnings("ignore")
import re  # check whether a given string matches a given pattern (using the match function), or contains such apattern (using the search function)
import tkinter.font as tkFont
from tkinter import *
from PIL import ImageTk, Image
from PIL import Image
from urllib.request import urlopen
from tkinter import Canvas

root = Tk()
root.title("Movieflix")
root.state("zoomed")
root.geometry("1920x1080")
root.configure(bg="black")


film_data = pd.read_csv("database//movies_metadata.csv", low_memory=False)



file_top_30 = open("database//top_30_movies_url_link.txt", "r")
data_top_30 = file_top_30.read()
data_top_30 = data_top_30.replace('\n', ' ').split(".jpg")
file_top_30.close() 


data_top_30_sts = data_top_30
data_top_9_rtg = []




data_sm = pd.read_csv("database//soup_data.csv", low_memory=False) # soup_data file is the file after cleaning the data
data_content_based = pd.read_csv("database//links_for_id.csv", low_memory=False) # links_for_id file contains links of movies poster
count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
count_matrix = count.fit_transform(data_sm['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)
indices = pd.Series(data_sm.index, index=data_sm['title'])  # Creating a mapping between movie and title and index
df_cosine = pd.DataFrame(cosine_sim)


data_rating = pd.read_csv("database//ratings_small.csv", low_memory=False)
data_rating = data_rating.drop(columns="timestamp")

# variable declaration
label1 = Label()
label2 = Label()
label3 = Label()
num1 = Label()
num2 = Label()
num3 = Label()
frame1 = Frame()
frame2 = Frame()
frame3 = Frame()
forward_button = Button()
backward_button = Button()







def home_tab():

  def forward(first_ind,val): # Ensures the forward navigation of the movies poster
    global label1
    global label2
    global label3
    global forward_button
    global backward_button
    global num1
    global num2
    global num3

    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    ending = 27

    if val == 1:
      data_top = data_top_30
      x1 = 1335
      x2 = 1415
      y1 = 410
      y2 = 411
    elif val == 3:
      data_top = data_top_30_sts
      x1 = 1328
      x2 = 1419
      y1 = 468
      y2 = 468
    elif val == 4:
      data_top = data_top_9_rtg
      x1 = 1335
      x2 = 1415
      y1 = 410
      y2 = 411
      ending = 6

    # Destroying the elements of the current list of 3 movies in order to move to next 3 movies
    label1.pack_forget()
    label2.pack_forget()
    label3.pack_forget()
    forward_button.pack_forget()
    backward_button.pack_forget()
    num1.pack_forget()
    num2.pack_forget()
    num3.pack_forget()

    # making labels
    num1 = Label(frame1, text=first_ind + 1, font="satoshi 15 bold", fg="white", bg="black")
    num1.pack()
    num2 = Label(frame2, text=first_ind + 2, font="satoshi 15 bold", fg="white", bg="black")
    num2.pack()
    num3 = Label(frame3, text=first_ind + 3, font="satoshi 15 bold", fg="white", bg="black")
    num3.pack()

    # getting movie poster links from the data_top file
    link1 = f"{data_top[first_ind]}.jpg"
    u1 = urlopen(link1)
    rg1 = u1.read()
    u1.close()

    link2 = f"{data_top[first_ind + 1]}.jpg"
    u2 = urlopen(link2)
    rg2 = u2.read()
    u2.close()

    link3 = f"{data_top[first_ind + 2]}.jpg"
    u3 = urlopen(link3)
    rg3 = u3.read()
    u3.close()

    # placing movies poster on the labels created above
    photo1 = ImageTk.PhotoImage(data=rg1)
    label1 = Label(frame1, image=photo1, width=400, height=500, bg="black")
    label1.image = photo1
    photo2 = ImageTk.PhotoImage(data=rg2)
    label2 = Label(frame2, image=photo2, width=400, height=500, bg="black")
    label2.image = photo2
    photo3 = ImageTk.PhotoImage(data=rg3)
    label3 = Label(frame3, image=photo3, width=400, height=500, bg="black")
    label3.image = photo3

    backward_button = Button(root, text="<<", font="satoshi 15 bold", bg="black", fg="white",
                                 command=lambda: backward(first_ind - 3, val),cursor="hand2")
    forward_button = Button(root, text=">>", font="satoshi 15 bold", fg="white", bg="black",
                            command=lambda: forward(first_ind + 3, val),cursor="hand2")

    if first_ind == ending:
      forward_button = Button(root, text=">>", font="satoshi 15 bold", fg="white", bg="black", state=DISABLED)

    label1.pack()
    label2.pack()
    label3.pack()
    backward_button.place(x=x1, y=y1, width=54, height=50)
    forward_button.place(x=x2, y=y2, width=54, height=50)

  def backward(first_ind,val): # Ensures the backward navigation of the movies poster
    global label1
    global label2
    global label3
    global forward_button
    global backward_button
    global num1
    global num2
    global num3

    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0

    if val == 1:
      data_top = data_top_30
      x1 = 1335
      x2 = 1415
      y1 = 410
      y2 = 411
    elif val == 3:
      data_top = data_top_30_sts
      x1 = 1328
      x2 = 1419
      y1 = 468
      y2 = 468
    elif val == 4:
      data_top = data_top_9_rtg
      x1 = 1335
      x2 = 1415
      y1 = 410
      y2 = 411

    # Destroying the elements of the current list of 3 movies in order to move to next 3 movies
    label1.pack_forget()
    label2.pack_forget()
    label3.pack_forget()
    forward_button.pack_forget()
    backward_button.pack_forget()
    num1.pack_forget()
    num2.pack_forget()
    num3.pack_forget()

    # making labels
    num1 = Label(frame1, text=first_ind + 1, font="satoshi 15 bold", fg="white", bg="black")
    num1.pack()
    num2 = Label(frame2, text=first_ind + 2, font="satoshi 15 bold", fg="white", bg="black")
    num2.pack()
    num3 = Label(frame3, text=first_ind + 3, font="satoshi 15 bold", fg="white", bg="black")
    num3.pack()

    # getting movie poster links from the data_top file
    link1 = f"{data_top[first_ind]}.jpg"
    u1 = urlopen(link1)
    rg1 = u1.read()
    u1.close()

    link2 = f"{data_top[first_ind + 1]}.jpg"
    u2 = urlopen(link2)
    rg2 = u2.read()
    u2.close()

    link3 = f"{data_top[first_ind + 2]}.jpg"
    u3 = urlopen(link3)
    rg3 = u3.read()
    u3.close()

    # placing movies poster on the labels created above
    photo1 = ImageTk.PhotoImage(data=rg1)
    label1 = Label(frame1, image=photo1, width=400, height=500, bg="black")
    label1.image = photo1
    photo2 = ImageTk.PhotoImage(data=rg2)
    label2 = Label(frame2, image=photo2, width=400, height=500, bg="black")
    label2.image = photo2
    photo3 = ImageTk.PhotoImage(data=rg3)
    label3 = Label(frame3, image=photo3, width=400, height=500, bg="black")
    label3.image = photo3

    backward_button = Button(root, text="<<", font="satoshi 15 bold", bg="black", fg="white",
                                 command=lambda: backward(first_ind - 3, val))
    forward_button = Button(root, text=">>", font="satoshi 15 bold", fg="white", bg="black",
                            command=lambda: forward(first_ind + 3, val))

    if first_ind == 0:
      backward_button = Button(root, text="<<", font="satoshi 15 bold", fg="white", bg="black", state=DISABLED)

    label1.pack()
    label2.pack()
    label3.pack()
    backward_button.place(x=x1, y=y1, width=54, height=50)
    forward_button.place(x=x2, y=y2, width=54, height=50)


  def init(idx_):
    global label1
    global label2
    global label3
    global num1
    global num2
    global num3
    global frame1
    global frame2
    global frame3
    global forward_button
    global backward_button
    data_top = []
    x1=0
    x2=0
    x3=0
    x4=0
    x5=0
    y1=0
    y2=0
    y3=0
    y4=0
    y5=0

    if idx_ == 1:
      data_top = data_top_30
      x1 = 62
      x2 = 493
      x3 = 924
      x4 = 1335
      x5 = 1415
      y1 = 223
      y2 = 223
      y3 = 223
      y4 = 410
      y5 = 411
    elif idx_ == 2:
      data_top = data_top_30_gen
      x1 = 80
      x2 = 500
      x3 = 918
      x4 = 1332
      x5 = 1410
      y1 = 220
      y2 = 220
      y3 = 220
      y4 = 400
      y5 = 400
    elif idx_ == 3:
      data_top = data_top_30_sts
      x1 = 70
      x2 = 495
      x3 = 919
      x4 = 1328
      x5 = 1419
      y1 = 270
      y2 = 270
      y3 = 270
      y4 = 468
      y5 = 468
    elif idx_ == 4:
      data_top = data_top_9_rtg
      x1 = 62
      x2 = 493
      x3 = 924
      x4 = 1335
      x5 = 1415
      y1 = 223
      y2 = 223
      y3 = 223
      y4 = 410
      y5 = 411

    # getting movie poster links from the data_top file
    link1 = f"{data_top[0]}.jpg"
    u1 = urlopen(link1)
    rg1 = u1.read()
    u1.close()

    link2 = f"{data_top[1]}.jpg"
    u2 = urlopen(link2)
    rg2 = u2.read()
    u2.close()

    link3 = f"{data_top[2]}.jpg"
    u3 = urlopen(link3)
    rg3 = u3.read()
    u3.close()

    # making labels and placing movies poster on the created labels
    frame1 = Frame(root, bg="black")
    frame1.place(x=x1, y=y1, width=400, height=500)
    num1 = Label(frame1, text="1", font="satoshi 15 bold", fg="white", bg="black")
    num1.pack()
    photo1 = ImageTk.PhotoImage(data=rg1)
    label1 = Label(frame1, image=photo1, width=400, height=500, bg="black")
    label1.image = photo1
    label1.pack()

    frame2 = Frame(root, bg="black")
    frame2.place(x=x2, y=y2, width=400, height=500)
    num2 = Label(frame2, text="2", font="satoshi 15 bold", fg="white", bg="black")
    num2.pack()
    photo2 = ImageTk.PhotoImage(data=rg2)
    label2 = Label(frame2, image=photo2, width=400, height=500, bg="black")
    label2.image = photo2
    label2.pack()

    frame3 = Frame(root, bg="black")
    frame3.place(x=x3, y=y3, width=400, height=500)
    num3 = Label(frame3, text="3", font="satoshi 15 bold", fg="white", bg="black")
    num3.pack()
    photo3 = ImageTk.PhotoImage(data=rg3)
    label3 = Label(frame3, image=photo3, width=400, height=500, bg="black")
    label3.image = photo3
    label3.pack()

    # making and placing forward and backward button
    backward_button = Button(root, text="<<", font="satoshi 15 bold", bg="black", fg="white", command=backward,
                              state=DISABLED)
    backward_button.place(x=x4, y=y4, width=54, height=50)
    forward_button = Button(root, text=">>", font="satoshi 15 bold", fg="white", bg="black",
                            command=lambda: forward(3, idx_))
    forward_button.place(x=x5, y=y5, width=54, height=50)

  def hometab_destroy():#to destroy the home page
    background_img_label.destroy()
    mybutton1.destroy()
    mybutton2.destroy()
  def similar_to_search(): # executes when user clicks on Similar to Search button on homepage
    global data_top_30_sts
    data_top_30_sts = data_top_30
    hometab_destroy()

    def get_recommendations(title):
      idx_ = indices[title]  # movie id corrosponding to the given title
      sim_score_ = list(enumerate(cosine_sim[idx_]))  # list of cosine similarity scores value along the given index
      sim_score_ = sorted(sim_score_, key=lambda x: x[1],reverse=True)  # sorting the given scores in ascending order
      sim_score_ = sim_score_[0:31]  # top 30 scores
      movie_indices = [i[0] for i in sim_score_]  # Finding the indices of 30 most similar movies
      return data_sm['title'].iloc[movie_indices]

    def back(): # for Back_to_Home button
      Label_background.destroy()
      my_entry.destroy()
      my_list_.destroy()
      button1_.destroy()
      button2_.destroy()
      frame1.destroy()
      frame2.destroy()
      frame3.destroy()
      home_tab()

    def search_sts(): # recommends top30 movies from user input
      global data_top_30_sts
      curr_list = get_recommendations(my_entry.get()).head(30)
      curr_link_list_ = []
      for it in range(30):
        id = curr_list.keys()[it]
        curr_link_list_.append(data_content_based['link'][id])
      data_top_30_sts = curr_link_list_
      global label1
      global label2
      global label3
      global num1
      global num2
      global num3
      global frame1
      global frame2
      global frame3
      global forward_button
      global backward_button
      label1.pack_forget()
      label2.pack_forget()
      label3.pack_forget()
      forward_button.pack_forget()
      backward_button.pack_forget()
      num1.pack_forget()
      num2.pack_forget()
      num3.pack_forget()
      # making label
      num1 = Label(frame1, text=1, font="satoshi 15 bold", fg="white", bg="black")
      num1.pack()
      num2 = Label(frame2, text=2, font="satoshi 15 bold", fg="white", bg="black")
      num2.pack()
      num3 = Label(frame3, text=3, font="satoshi 15 bold", fg="white", bg="black")
      num3.pack()
      # making and placing forward and backward button
      backward_button = Button(root, text="<<", font="satoshi 15 bold", bg="black", fg="white", command=backward,state=DISABLED)
      forward_button = Button(root, text=">>", font="satoshi 15 bold", fg="white", bg="black",command=lambda: forward(3,3))

      backward_button.place(x=1328, y=468, width=54, height=50)
      forward_button.place(x=1419, y=468, width=54, height=50)
      # getting movie poster links from the data_top_30_sts file
      link1 = f"{data_top_30_sts[0]}.jpg"
      u1 = urlopen(link1)
      rg1 = u1.read()
      u1.close()

      link2 = f"{data_top_30_sts[1]}.jpg"
      u2 = urlopen(link2)
      rg2 = u2.read()
      u2.close()

      link3 = f"{data_top_30_sts[2]}.jpg"
      u3 = urlopen(link3)
      rg3 = u3.read()
      u3.close()
      # placing movies poster on the labels created above
      photo1 = ImageTk.PhotoImage(data=rg1)
      label1 = Label(frame1, image=photo1, width=400, height=500, bg="black")
      label1.image = photo1
      photo2 = ImageTk.PhotoImage(data=rg2)
      label2 = Label(frame2, image=photo2, width=400, height=500, bg="black")
      label2.image = photo2
      photo3 = ImageTk.PhotoImage(data=rg3)
      label3 = Label(frame3, image=photo3, width=400, height=500, bg="black")
      label3.image = photo3

      label1.pack(pady=4)
      label2.pack(pady=4)
      label3.pack(pady=4)
    def update_list(data):
      my_list_.delete(0, END)
      for item in data:
        my_list_.insert(END, item)
    def fill_out(e):
      my_entry.delete(0, END)
      my_entry.insert(0, my_list_.get(ACTIVE))
    def check(e): # suggest movies by prefix matching of user input
      typed = my_entry.get()
      if typed == '':
        data = starting_list
      else:
        data = []
        for item in data_sm['title']:
          if (re.match(typed, item, re.IGNORECASE)):
            data.append(item)
      update_list(data)
    
    # making background_image
    background_image = PhotoImage(file="images//similar_to_search.png")
    Label_background = Label(root, image=background_image)
    Label_background.image = background_image
    Label_background.place(x=0, y=0)

    def on_enter_1_(e): # hovering effects for button1 while entering the button
      button1_['background'] = "#d2010b"
      button1_['foreground'] = "white"
    
    def on_leave_1_(e): # hovering effects for button1 while leaving the button
      button1_['background'] = "#E50914"
      button1_['foreground'] = "white"
    
    # creating Back to Home button
    button1_ = Button(text="Back To Home", font="aerial 12 bold", borderwidth=0, highlightthickness=0,relief="flat", fg="white", bg="#E50914", activeforeground="white",activebackground="#b3030c", command=back,cursor="hand2")
    button1_.bind("<Enter>", on_enter_1_) # hovering effect while entering
    button1_.bind("<Leave>", on_leave_1_) # hovering effect while leaving
    button1_.place(x=3, y=3.4, width=141, height=48)

    my_entry = Entry(bd=0, bg="#d9d9d9", highlightthickness=0, font="Helvetica 15")
    my_entry.place(x=495, y=82, width=405, height=33)
    
    def on_enter_2_(e): # hovering effects to the button2 while entering the button
      button2_['background'] = "#d2010b"
      button2_['foreground'] = "white"

    def on_leave_2_(e): # hovering effects to the button2 while leaving the button
      button2_['background'] = "#E50914"
      button2_['foreground'] = "white"
    
    # creating Search button 
    button2_ = Button(text="Search", font="aerial 15", borderwidth=0, highlightthickness=0, relief="flat",fg="white", bg="#E50914", activeforeground="white", activebackground="#b3030c",command=search_sts,cursor="hand2")
    button2_.bind("<Enter>", on_enter_2_) # hovering effect while entering
    button2_.bind("<Leave>", on_leave_2_) # hovering effect while leaving
    button2_.place(x=905, y=82, width=78, height=33)
    my_list_ = Listbox(root, relief="flat", fg="#ffffff", bg="#141414")
    my_list_.place(x=495, y=120, width=365, height=114)
    starting_list = ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 'Fight Club', 'Pulp Fiction','Forrest Gump']
    update_list(starting_list)
    my_list_.bind("<<ListboxSelect>>", fill_out)
    my_entry.bind("<KeyRelease>", check)
    # getting movie poster links from the data_top_30_sts file
    link1 = f"{data_top_30_sts[0]}.jpg"
    u1 = urlopen(link1)
    rg1 = u1.read()
    u1.close()

    link2 = f"{data_top_30_sts[1]}.jpg"
    u2 = urlopen(link2)
    rg2 = u2.read()
    u2.close()

    link3 = f"{data_top_30_sts[2]}.jpg"
    u3 = urlopen(link3)
    rg3 = u3.read()
    u3.close()

    global label1
    global label2
    global label3
    global num1
    global num2
    global num3
    global frame1
    global frame2
    global frame3
    global forward_button
    global backward_button
    init(3)

  def rate_to_get():
    hometab_destroy()
    def rate_to_get_results():
      # implementing the SVD algorithm for collaborative filtering 
            
      data_rating.isna().sum()  # checking for missing values
      movies = data_rating['movieId'].nunique() # nunique is similar to count but only takes unique values
      users = data_rating['userId'].nunique()
      columns = ['userId', 'movieId', 'rating'] # columns to use for training
      # create reader from surprise 
      # the rating should lie in the provided scale
      reader = Reader(rating_scale=(1, 5))
      taken_id_lists = [] # this list stores the movies id for which the user has given rating
      for it in final_list:
        df2 = film_data.loc[film_data['title'] == it, 'id'].iloc[0]
        taken_id_lists.append(df2)

      test_case = {'userId': [99999, 99999, 99999, 99999, 99999],
                    'movieId': [taken_id_lists[0], taken_id_lists[1], taken_id_lists[2], taken_id_lists[3], taken_id_lists[4]],
                    'rating': [(s1.get()), (s2.get()), (s3.get()), (s4.get()), (s5.get())]
                    }

      df = pd.DataFrame(test_case)
      frames = [data_rating, df]
      result = pd.concat(frames)
      
      data = Dataset.load_from_df(result[columns], reader) # create dataset from dataframe

      trainset = data.build_full_trainset() # create trainset

      testset = trainset.build_anti_testset() # create testset, here the anti_testset is testset

      model = SVD(n_epochs=25, verbose=True)
      cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True) # cv is the number of parts in which data will be                                                                                             divided

      prediction = model.test(testset) # prediction

      def get_top_n(prediction, n): # this function recommend the topn movies based on prediction using the surprise library
        # First map the predictions to each user
        top_n_ = defaultdict(list) 
        for uid, iid, true_r, est, _ in prediction:
          top_n_[uid].append((iid, est))
        # Then sort the predictions for each user and retrieve the n highest ones
        for uid, user_ratings in top_n_.items():
          user_ratings.sort(key=lambda x: x[1], reverse=True)
          top_n_[uid] = user_ratings[:n]

        return top_n_

      rcmnd_list = []
      top_n = get_top_n(prediction, n=30)
      # to store the imdb_id of top30 movies in the "rcmnd_list" list
      for uid, user_ratings in top_n.items():
        if uid==99999:
          for (iid, rating) in user_ratings:
            for i in range(film_data.shape[0]):
              if int(film_data['id'][i])==int(iid):
                rcmnd_list.append(film_data['imdb_id'][i])
        
          break
      data_top_9_rtg.clear()
      

      # to store the top9 movies poster link in "data_top_9_rtg" list
      for it in range(9):
        id = rcmnd_list[it]
        request_id = "http://www.omdbapi.com/?i=" + id + "&apikey=9bfb620b"
        json_data = requests.get(request_id).json()
        data_top_9_rtg.append(json_data['Poster'])
      # function calls for destroying the page where user has rated
      Label_background.destroy()
      button1_.destroy()
      l1.destroy()
      l2.destroy()
      l3.destroy()
      l4.destroy()
      l5.destroy()
      s1.destroy()
      s2.destroy()
      s3.destroy()
      s4.destroy()
      s5.destroy()
      my_entry.destroy()
      my_list_.destroy()
      button2_.destroy()
      button3_.destroy()
      button4_.destroy()
      
      def back_rtg(): # defining the function for "back" button, after the execution of this function user will reach to rating page 
        # function calls to destroy the elements of the "rate_to_get_results" page
        Label_background2.destroy()
        button1__.destroy()
        frame1.destroy()
        frame2.destroy()
        frame3.destroy()
        backward_button.destroy()
        forward_button.destroy()
        rate_to_get()

      background2_image = PhotoImage(file="images//rate_to_get_rec.png")
      Label_background2 = Label(root, image=background2_image)
      Label_background2.image = background2_image
      Label_background2.place(x=0, y=0)

      def on_enter_1__(e): # hovering effects while entering "back" button
        button1__['background'] = "#d2010b"
        button1__['foreground'] = "white"

      def on_leave_1__(e): # hovering effects while leaving "back" button
        button1__['background'] = "#E50914"
        button1__['foreground'] = "white"

      button1__ = Button(text="Back", font="aerial 12 bold", borderwidth=0, highlightthickness=0, relief="flat",
                          fg="white", bg="#E50914", activeforeground="white", activebackground="#b3030c",
                          command=back_rtg,cursor="hand2")
      button1__.bind("<Enter>", on_enter_1__) # hovering effect while entering "back" button
      button1__.bind("<Leave>", on_leave_1__) # hovering effect while leaving "back" button
      button1__.place(x=3, y=3.4, width=141, height=48)
      # getting movie poster links from the data_top_9_rtg file
      link1 = f"{data_top_9_rtg[0]}.jpg"
      u1 = urlopen(link1)
      rg1 = u1.read()
      u1.close()

      link2 = f"{data_top_9_rtg[1]}.jpg"
      u2 = urlopen(link2)
      rg2 = u2.read()
      u2.close()

      link3 = f"{data_top_9_rtg[2]}.jpg"
      u3 = urlopen(link3)
      rg3 = u3.read()
      u3.close()

      global label1
      global label2
      global label3
      global num1
      global num2
      global num3
      global frame1
      global frame2
      global frame3
      global forward_button
      global backward_button
      init(4) # call to "init" function
    def back():
      Label_background.destroy()
      button1_.destroy()
      l1.destroy()
      l2.destroy()
      l3.destroy()
      l4.destroy()
      l5.destroy()
      s1.destroy()
      s2.destroy()
      s3.destroy()
      s4.destroy()
      s5.destroy()
      my_entry.destroy()
      my_list_.destroy()
      button2_.destroy()
      home_tab()
    def update(data):
      my_list_.delete(0, END)

      for item in data:
        my_list_.insert(END, item)
    def fill_out(e): # updates "my_entry"
      my_entry.delete(0, END)
      my_entry.insert(0, my_list_.get(ACTIVE))

    def check(e): # prefix matching of user input
      typed = my_entry.get()
      if typed == '':
        data = starting_list
      else:
        data = []
        for item in film_data['original_title']:
          if (re.match(typed, item, re.IGNORECASE)):
            data.append(item)
      update(data)
    
    background_image=PhotoImage(file=f"images//rate_to_get.png")
    Label_background=Label(root,image=background_image)
    Label_background.image=background_image
    Label_background.place(x=0,y=0)


    add_cnt=0
    l1=Label()
    l2=Label()
    l3=Label()
    l4=Label()
    l5=Label()
    final_list=[]

    def add_list(): # Adds movies to movies_list section
      flag = 0
      for it in film_data['original_title']:
        if (it == my_entry.get()):
          flag = 1
      if flag == 0:
        return
      nonlocal add_cnt
      nonlocal l1, l2, l3, l4, l5
      final_list.append(my_entry.get())
      if add_cnt == 0:
        l1 = Label(root, text=my_entry.get(), font="aerial 15", bg="#25dae9")
        l1.place(x=455, y=530, width=450, height=35)
      elif add_cnt == 1:
        l2 = Label(root, text=my_entry.get(), bg='#ffcc66', font="aerial 15")
        l2.place(x=455, y=575, width=450, height=35)
      elif add_cnt == 2:
        l3 = Label(root, text=my_entry.get(), font="aerial 15", bg="#25dae9")
        l3.place(x=455, y=620, width=450, height=35)
      elif add_cnt == 3:
        l4 = Label(root, text=my_entry.get(), font="aerial 15", bg="#ffcc66")
        l4.place(x=455, y=665, width=450, height=35)
      elif add_cnt == 4:
        l5 = Label(root, text=my_entry.get(), font="aerial 15", bg="#25dae9")
        l5.place(x=455, y=710, width=450, height=35)
      add_cnt = add_cnt + 1
    
    def cross(): # removes last added movie in "movie_list" section
      nonlocal add_cnt
      if(add_cnt==1):
        l1.destroy()
      elif (add_cnt==2):
        l2.destroy()
      elif (add_cnt==3):
        l3.destroy()
      elif (add_cnt==4):
        l4.destroy()
      elif (add_cnt==5):
        l5.destroy()
      add_cnt=add_cnt-1

    # used spinbox to get rating from the user
    s1 = Spinbox(root, from_=1, to=5, bg="#25dae9", font="aerial 15")
    s1.place(x=940, y=530, width=100, height=35)

    s2 = Spinbox(root, from_=1, to=5, font="aerial 15", bg='#ffcc66')
    s2.place(x=940, y=575, width=100, height=35)

    s3 = Spinbox(root, from_=1, to=5, font="aerial 15", bg="#25dae9")
    s3.place(x=940, y=620, width=100, height=35)

    s4 = Spinbox(root, from_=1, to=5, font="aerial 15", bg="#ffcc66")
    s4.place(x=940, y=665, width=100, height=35)

    s5 = Spinbox(root, from_=1, to=5, font="aerial 15", bg="#25dae9")
    s5.place(x=940, y=710, width=100, height=35)


    def on_enter_1_(e): # hovering effects while entering the "back to home" button
      button1_['background'] = "#d2010b"
      button1_['foreground'] = "white"

    def on_leave_1_(e): # hovering effects while leavng the "back to home" button
      button1_['background'] = "#E50914"
      button1_['foreground'] = "white"

    button1_ = Button(text="Back To Home", font="aerial 12 bold", borderwidth=0, highlightthickness=0,
                      relief="flat", fg="white", bg="#E50914", activeforeground="white",
                      activebackground="#b3030c", command=back,cursor="hand2")
    button1_.bind("<Enter>", on_enter_1_)
    button1_.bind("<Leave>", on_leave_1_)
    button1_.place(x=3, y=3.4, width=141, height=48)
    my_entry = Entry(bd=0, bg="#d9d9d9", highlightthickness=0, font="Helvetica 15")
    my_entry.place(x=455, y=215, width=483, height=38)

    def on_enter_2_(e): # hovering effects while entering the "add in list" button
      button2_['background'] = "#d2010b"
      button2_['foreground'] = "white"

    def on_leave_2_(e): # hovering effects while leavng the "add in list" button
      button2_['background'] = "#E50914"
      button2_['foreground'] = "white"
    
    button2_ = Button(text="Add in List", font="aerial 15", borderwidth=0, highlightthickness=0, relief="flat",
                      fg="white", bg="#E50914", activeforeground="white", activebackground="#b3030c",
                      command=add_list,cursor="hand2")
    button2_.bind("<Enter>", on_enter_2_) # hovering effect while entering
    button2_.bind("<Leave>", on_leave_2_) # hovering effect while leaving
    button2_.place(x=940, y=215, width=98, height=40)

    my_list_ = Listbox(root, relief="flat", fg="#ffffff", bg="#141414")
    my_list_.place(x=455, y=258, width=388, height=118)
    
    starting_list = ['Mission: Impossible', 'The Godfather', 'The Dark Knight', 'Fight Club', 'Pulp Fiction', 'Inception'] # recommends default to user
    update(starting_list)
    my_list_.bind("<<ListboxSelect>>", fill_out)
    my_entry.bind("<KeyRelease>", check)

    remove_icon = PhotoImage(file="images//cross.png")

    def on_enter_3_(e): # function for giving hovering effects to the button while entering the button3
      button3_['background'] = "#d2010b"
      button3_['foreground'] = "white"

    def on_leave_3_(e): # function for giving hovering effects to the button while leaving the button3
      button3_['background'] = "#E50914"
      button3_['foreground'] = "white"

    button3_ = Button(font="aerial 15", borderwidth=0, highlightthickness=0, relief="flat", fg="white",
                      bg="#E50914", activeforeground="white", activebackground="#b3030c", command=cross,cursor="hand2")
    button3_.config(compound='right', image=remove_icon)
    button3_.image = remove_icon
    button3_.bind("<Enter>", on_enter_3_) # hovering effect while entering
    button3_.bind("<Leave>", on_leave_3_) # hovering effect while leaving
    button3_.place(x=940, y=755, width=35, height=35)

    def on_enter_4_(e): # function for giving hovering effects to the button while entering the button4
      button4_['background'] = "#d2010b"
      button4_['foreground'] = "white"

    def on_leave_4_(e): # function for giving hovering effects to the button while leaving the button4
      button4_['background'] = "#E50914"
      button4_['foreground'] = "white"
    
    button4_ = Button(text="Submit", font="aerial 15", borderwidth=0, highlightthickness=0, relief="flat",
                      fg="white", bg="#E50914", activeforeground="white", activebackground="#b3030c",
                      command=rate_to_get_results,cursor="hand2")
    button4_.bind("<Enter>", on_enter_4_) # hovering effect while entering
    button4_.bind("<Leave>", on_leave_4_) # hovering effect while leaving
    button4_.place(x=1000, y=755, width=100, height=35)
  

  logo_photo = PhotoImage(file="images//MF_LOGO.png")
  root.iconphoto(False, logo_photo)
  
  background_img_home= PhotoImage(file=f"images//bgimg.png")
  background_img_label = Label(root, image=background_img_home, bg="#ffffff", height=840, width=1544, bd=0, highlightthickness=0, relief="ridge")
  background_img_label.image = background_img_home
  background_img_label.place(x=-7,y=0)
  
  def on_enter_1(e): # hovering effects while entering button1
    mybutton1['background'] = '#d2010b'
    mybutton1['foreground'] = "white"

  def on_leave_1(e): # hovering effects while leaving button1
    mybutton1['background'] = "#E50914"
    mybutton1['foreground'] = 'white'

  mybutton1 = Button(text="Similar To Search", font="aerial 12 bold",borderwidth=0,highlightthickness=0,relief="flat",fg="white",bg="#E50914",activebackground="#b3030c",activeforeground="white",command=similar_to_search,cursor="hand2")

  mybutton1.bind("<Enter>", on_enter_1) # hovering effect while entering
  mybutton1.bind("<Leave>", on_leave_1) # hovering effect while leaving
  mybutton1.place(x=1139, y=27, width=181, height=47)

  def on_enter_2(e): # hovering effects to the button while entering the button2
    mybutton2['background'] = "#d2010b"
    mybutton2['foreground'] = "white"

  def on_leave_2(e): # hovering effects leaving the button2
    mybutton2['background'] = "#E50914"
    mybutton2['foreground'] = 'white'

  mybutton2 = Button(text="Rate And Get", font="aerial 12 bold", borderwidth=0, highlightthickness=0,
  relief="flat", fg='white', bg="#E50914", activeforeground="white",
  activebackground='#b3030c',command=rate_to_get,cursor="hand2")
  
  mybutton2.bind("<Enter>", on_enter_2) # hovering effect while entering
  mybutton2.bind("<Leave>", on_leave_2) # hovering effect while leaving
  mybutton2.place(x=1334, y=27, width=181, height=47)

home_tab()

root.mainloop()