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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel




output_file("Timeline Comparison.html")

#set of functions to extract the data from csv files
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


#functions used to plot trendlines and data

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


def gauss_trend(Score,age):

    L=np.array(age)
    X = L.reshape(-1,1)
    y = Score

    kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
        + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
    gp = GaussianProcessRegressor(kernel=kernel,
                              alpha=0.0).fit(X, y)
    X_ = np.linspace(20, 80)
    y_mean, y_cov = gp.predict(X_[:, np.newaxis], return_cov=True)

    sourceline = ColumnDataSource(data=dict(
        x=X_,
        y=y_mean,
        ))

    Trendline=p5.line('x','y',line_width=3,source=sourceline,color="green",legend="Trendline")
    return Trendline




def datatables(doc_name,scores):
    data = dict(
            Doc_name=doc_name,
            score=scores,
        )
    source_tabel1 = ColumnDataSource(data)


    columns = [
            TableColumn(field="Doc_name", title="Doc_name"),
            TableColumn(field="score", title="Score"),
        ]
    data_table = DataTable(source=source_tabel1, columns=columns, width=300, height=400)
    return data_table


def data_plot_1(age,scores,doc_name,authors,a):
    sourceline = ColumnDataSource(data=dict(
        x=age,
        y=scores,
        Titel=doc_name,
    ))


    data_circle=p1.circle('x', 'y', size=20, name=authors[a], source=sourceline,color=colors[a],legend=authors[a])
    return data_circle


def data_plot_2(age,scores,doc_name,authors,a):
    sourceline = ColumnDataSource(data=dict(
        x=age,
        y=scores,
        Titel=doc_name,
    ))


    data_circle=p2.circle('x', 'y', size=20, name=authors[a], source=sourceline,color=colors[a],legend=authors[a])
    return data_circle

def data_plot_3(age,scores,doc_name,authors,a):
    sourceline = ColumnDataSource(data=dict(
        x=age,
        y=scores,
        Titel=doc_name,
    ))


    data_circle=p5.circle('x', 'y', size=20, name=authors[a], source=sourceline,color=colors[a],legend=authors[a])
    return data_circle



#functions used to calculate additional metrics for the data analysis

def msq(Scores,age,a):
    Scores=np.asarray(Scores[a])
    new_age=np.asarray(age[a])
    Scores= Scores.reshape(-1,1)
    new_age = new_age.reshape(-1,1)
    regr = linear_model.LinearRegression()
    regr.fit(new_age, Scores)
    prediction = regr.predict(new_age)
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

    #get colors from the Bokeh palet depending on the number of authors
    colors=[]
    [colors.append(x) for x in palette]
    colors=colors[:len(authors)]

    index_authors = []
    [index_authors.append(authors.index(item)) for item in authors]

    #creates a set of empty lists that will be used to store additional metrics
    #(list format to facilitate organising the layout of the plots)
    empty=[]
    predicted_vals=[]
    infos=[]
    mean_score=[]

    #define toolset and hovertool
    tools = "pan,wheel_zoom,box_zoom,reset,save,tap".split(',')
    hover = HoverTool(tooltips=[
        ("Titel","@Titel"),
        ("Score","@y")
    ])
    tools.append(hover)


    #TAB1: Main plot, plots a timeline of all documents (x-axis: years, y-axis: scores)
    p1 = figure(plot_width=1000, plot_height=400, tools=tools,
               title="Document timeline")
    [data_plot_1(year_data[a],Score_data[a],doc_data[a],authors,a) for a in index_authors]
    p1.background_fill_color = "#dddddd"
    p1.yaxis.axis_label = "Markers 1000/words"
    p1.yaxis.axis_label_standoff = 20
    p1.legend.location = "top_left"
    p1.legend.click_policy="hide"

    #TAB1: subplots used to display additional information on the documents (e.g mean and mean sentence length)
    [mean_score.append(sum(Score_data[a])/len(Score_data[a])) for a in index_authors]

    p3 = figure(x_range=authors,plot_width=500, plot_height=400, title="Mean",
           toolbar_location=None, tools="")
    p3.vbar(x=authors, top=mean_score, width=0.3,color=colors)
    p3.background_fill_color = "#dddddd"
    p3.yaxis.axis_label = "Mean score"
    p3.yaxis.axis_label_standoff = 20
    p3.xaxis.axis_label = "Author"

    p4 = figure(x_range=authors,plot_width=500, plot_height=400, title="Mean sentence length (words)",
           toolbar_location=None, tools="")
    p4.vbar(x=authors, top=[22,18], width=0.3,color=colors)
    p4.background_fill_color = "#dddddd"
    p4.yaxis.axis_label = "Mean sentence length"
    p4.yaxis.axis_label_standoff = 20
    p4.xaxis.axis_label = "Author"



    #TAB2: plots linear trendline for the data (x-axis: author age, y-axis: document score)
    p2 = figure(plot_width=1000, plot_height=400, tools="pan,wheel_zoom,box_zoom,reset,save".split(','),title="Linear regression analysis")

    [data_plot_2(age_data[a],Score_data[a],doc_data[a],authors,a) for a in index_authors]
    [trendlineslinear(Score_data[a],age_data[a]) for a in index_authors]

    p2.background_fill_color = "#dddddd"
    p2.yaxis.axis_label = "markers 1000/words"
    p2.yaxis.axis_label_standoff = 20
    p2.xaxis.axis_label = "age"
    p2.legend.location = "top_left"
    p2.legend.click_policy="hide"



    #TAB2: plots gaussian trendline for the data (x-axis: author age, y-axis: document score)
    p5 = figure(plot_width=1000, plot_height=400, tools="pan,wheel_zoom,box_zoom,reset,save".split(','),title="Gaussian regression analysis")

    [data_plot_3(age_data[a],Score_data[a],doc_data[a],authors,a) for a in index_authors]
    [gauss_trend(Score_data[a],age_data[a]) for a in index_authors]

    p5.background_fill_color = "#dddddd"
    p5.yaxis.axis_label = "markers 1000/words"
    p5.yaxis.axis_label_standoff = 20
    p5.xaxis.axis_label = "age"
    p5.legend.location = "top_left"
    p5.legend.click_policy="hide"



    #constructs data tables and additional metrics for each author that can be displayed
    [empty.append(datatables(doc_data[a],Score_data[a])) for a in index_authors]
    [predicted_vals.append(msq(Score_data,age_data,a)) for a in index_authors]
    [infos.append(info(authors,Score_data,a,predicted_vals)) for a in index_authors]

    #organize tab and plot layout
    layout = row(column(row(p1),row(p3,p4)),row(widgetbox(empty)))
    layout2=column(row(p2,widgetbox(infos)),p5)
    curdoc().add_root(layout)
    curdoc().add_root(layout2)
    tab2 = Panel(child=layout2, title="age")
    tab1 = Panel(child=layout, title="Timeline")


    tabs = Tabs(tabs=[ tab1, tab2 ])

    show(tabs)
