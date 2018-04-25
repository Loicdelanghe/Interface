from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool,OpenURL, TapTool
from bokeh.layouts import column,row,widgetbox
from bokeh.models.widgets import Panel, Tabs, DataTable, DateFormatter, TableColumn, Div, PreText
from bokeh.io import curdoc
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from statistics import median
from bokeh.models import CheckboxGroup, CustomJS
import numpy as np
import pandas as pd
from bokeh.palettes import Dark2_5 as palette




output_file("sampleClausElsschot.html")


df = pd.read_csv('Claus_data2.csv')
df2 = pd.read_csv('Elsschot_data2.csv')

datasets=[]
datasets.append(df)
datasets.append(df2)
authors=["Claus","Elsschot"]


def extract_score(df):
    score=[]
    for item in df.loc[:,"scores"]:
        score.append(item)
    return score

def extract_doc(df):
    doc=[]
    for item in df.loc[:,"Roman"]:
        doc.append(item)
    return doc

def extract_year(df):
    year=[]
    for item in df.loc[:,"Jaartal"]:
        year.append(item)
    return year


def extract_age(df):
    age=[]
    for item in df.loc[:,"leeftijd"]:
        age.append(item)
    return age

Score_data=[]
age_data=[]
year_data=[]
doc_data=[]

for item in datasets:
    Score_data.append(extract_score(item))
    age_data.append(extract_age(item))
    year_data.append(extract_year(item))
    doc_data.append(extract_doc(item))



def trendlineslinear(Scores,age):
    Scores=np.asarray(Scores)
    age2=np.asarray(age)
    Scores= Scores.reshape(-1,1)
    age2 = age2.reshape(-1,1)
    regr = linear_model.LinearRegression()
    regr.fit(age2, Scores)
    prediction = regr.predict(age2)
    prediction=prediction.tolist()

    source_trendline = ColumnDataSource(data=dict(
        x=[age[0],age[-1]],
        y=[prediction[0],prediction[-1]],
    ))


    Trendline=p2.line('x','y',line_width=3,source=source_trendline,color="red",legend="Trendline")

    return Trendline


def datatables(titels,scores):
    data = dict(
            roman=titels,
            score=scores,
        )
    source_tabel1 = ColumnDataSource(data)


    columns = [
            TableColumn(field="roman", title="Roman"),
            TableColumn(field="score", title="Score"),
        ]
    data_table = DataTable(source=source_tabel1, columns=columns, width=300, height=400)
    return data_table


def data_plot_1(leeftijd,scores,titels,authors,a):
    sourceline = ColumnDataSource(data=dict(
        x=leeftijd,
        y=scores,
        Titel=titels,
    ))


    datapunten=p1.circle('x', 'y', size=20, name=authors[a], source=sourceline,color=colors[a],legend=authors[a])
    return datapunten


def data_plot_2(leeftijd,scores,titels,authors,a):
    sourceline = ColumnDataSource(data=dict(
        x=leeftijd,
        y=scores,
        Titel=titels,
    ))


    datapunten=p2.circle('x', 'y', size=20, name=authors[a], source=sourceline,color=colors[a],legend=authors[a])
    return datapunten



def msq(Scores,leeftijd,jaartallen,a):
    Scores=np.asarray(Scores[a])
    jaartallen=np.asarray(leeftijd[a])
    Scores= Scores.reshape(-1,1)
    jaartallen = jaartallen.reshape(-1,1)
    regr = linear_model.LinearRegression()
    regr.fit(jaartallen, Scores)
    prediction = regr.predict(jaartallen)
    prediction=prediction.tolist()
    msq=mean_squared_error(Scores, prediction)
    return msq



def info(authors,Score_data,a,predicted_vals):
    mean=sum(Score_data[a])/len(Score_data[a])
    std=np.std(Score_data[a])
    Median=median(Score_data[a])
    msq=predicted_vals[a]
    pre2 = PreText(text="""                        {0}:

                            Mean: {1}
                            Standard dev: {2}
                            Median: {3}
                            Mean squared error = {4}""".format(authors[a],mean,std,Median,msq),
    width=500, height=100)
    return pre2

if __name__ == '__main__':
    colors=[]
    [colors.append(x) for x in palette]
    colors=colors[:len(authors)]


    tools = "pan,wheel_zoom,box_zoom,reset,save,tap".split(',')
    hover = HoverTool(tooltips=[
        ("Titel","@Titel"),
        ("Score","@y")
    ])
    tools.append(hover)



    p1 = figure(plot_width=1000, plot_height=400, tools=tools,
               title="Romans Claus/Elsschot")
    p1.background_fill_color = "#dddddd"
    p1.yaxis.axis_label = "Markers 1000/words"
    p1.yaxis.axis_label_standoff = 20
    for item in authors:
        a=authors.index(item)
        data_plot_1(year_data[a],Score_data[a],doc_data[a],authors,a)

    p1.legend.location = "top_left"
    p1.legend.click_policy="hide"



    p3 = figure(x_range=authors,plot_width=500, plot_height=400, title="Mean",
           toolbar_location=None, tools="")
    p3.background_fill_color = "#dddddd"
    p3.vbar(x=authors, top=[30,32], width=0.3,color=colors)

    p4 = figure(x_range=authors,plot_width=500, plot_height=400, title="Mean sentence length (words)",
           toolbar_location=None, tools="")
    p4.vbar(x=authors, top=[22,18], width=0.3,color=colors)
    p4.background_fill_color = "#dddddd"
    p4.yaxis.axis_label = "Mean sentence length"
    p4.yaxis.axis_label_standoff = 20
    p4.xaxis.axis_label = "Author"




    p2 = figure(plot_width=1000, plot_height=400, tools="pan,wheel_zoom,box_zoom,reset,save".split(','),title="Leeftijd")
    p2.background_fill_color = "#dddddd"
    p2.yaxis.axis_label = "markers 1000/words"
    p2.yaxis.axis_label_standoff = 20
    p2.xaxis.axis_label = "Leeftijd"
    for item in authors:
        a=authors.index(item)
        data_plot_2(age_data[a],Score_data[a],doc_data[a],authors,a)
    for item in authors:
        a=authors.index(item)
        trendlineslinear(Score_data[a],age_data[a])
    empty=[]
    for item in authors:
        n=authors.index(item)
        empty.append(datatables(doc_data[n],Score_data[n]))
    predicted_vals=[]
    for item in authors:
        a=authors.index(item)
        predicted_vals.append(msq(Score_data,age_data,year_data,a))
    infos=[]
    for item in authors:
        a=authors.index(item)
        infos.append(info(authors,Score_data,a,predicted_vals))
    p2.legend.location = "top_left"
    p2.legend.click_policy="hide"




    layout = row(column(row(p1),row(p3,p4)),row(widgetbox(empty)))
    layout2=row(p2,widgetbox(infos))
    curdoc().add_root(layout)
    curdoc().add_root(layout2)
    tab2 = Panel(child=layout2, title="Leeftijd")
    tab1 = Panel(child=layout, title="Tijd")


    tabs = Tabs(tabs=[ tab1, tab2 ])

    show(tabs)
