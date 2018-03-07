from tkinter import *
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np

# Fictionele data om een aantal functies van tkinter en matplotlib te demonstreren
data={"Verbs":10,"Nouns":50,"Articles":18,"Adjective":25}


# Functies voor een Pie chart, bar graph, data tonen en een exit functie die het window sluit
def chart(data):
    x=[]
    y=[]
    for key,value in data.items():
        x.append(key)
        y.append(value)

    labels = x
    sizes = y


    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')

    plt.show()

def bargraph(data):
    x=[]
    y=[]
    for key,value in data.items():
        x.append(key)
        y.append(value)
    objects = x
    y_pos = np.arange(len(objects))
    performance = y

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('# occurrences')
    plt.title('Word Type')

    plt.show()

def displaydata(data):
    output.delete(1.0,END)
    output.insert(END,data)


def quit():
    window.destroy()
    exit()




#specificaties van het canvas opzetten (titel,vorm,kleur,...)
window=Tk()
window.geometry("500x270")
window.title("GUI test")


#drop-down menu aanmaken waaraan ook functies kunnen gebonden worden
menu=Menu(window)
window.config(menu=menu)
subMenu=Menu(menu)
menu.add_cascade(label="File",menu=subMenu)
subMenu.add_command(label="Quit",command=quit)

#Toolbar aanmaken waaraan ook functies kunnen gebonden worden
toolbar=Frame(window,bg="blue")
DataButt=Button(toolbar,text="Display data",command=lambda:displaydata(data))
DataButt.pack(side=LEFT)
insertButt=Button(toolbar,text="exit",command=quit)
insertButt.pack(side=LEFT,padx=2,pady=2)
toolbar.pack(side=TOP,fill=X)

#statusbar onderaan aanmaken
statusbar=Label(window,text="this is a status bar",bd=1,relief=SUNKEN,anchor=W)
statusbar.pack(side=BOTTOM,fill=X)

#Labels en tekst die op het canvas geprint kan worden
datatitle=Label(window,text="DATA",)
datatitle.pack(side=TOP,padx=1,pady=1,anchor="w")

#output window voor tekst, er kan ook geprint worden in de command prompt
output=Text(window,width=60,height=3,wrap=WORD,bg="white")
output.pack(side=TOP,anchor="w")

#knoppen waaraan verschillende functies gebonden kunnen worden
Button(window,text="Graph (bar)",width=10,command=lambda:bargraph(data)).pack(side=TOP,anchor="w")
Button(window,text="Pie Chart",width=10,command=lambda:chart(data)).pack(side=TOP,anchor="w")
Button(window,text="exit",width=10,command=quit).pack(side=TOP,anchor="w")

#canvas openhouden
window.mainloop()
