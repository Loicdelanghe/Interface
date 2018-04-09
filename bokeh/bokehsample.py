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
import sampledata



output_file("sampleClausElsschot.html")

JaartallenC=sampledata.Claus_jaren
ScoresC=sampledata.Claus_getal
TitelsC=sampledata.CLaus_roman
LeeftijdC=sampledata.Claus_leeftijd
examplesC=sampledata.vbC

JaartallenE=sampledata.Elsschot_jaren
ScoresE=sampledata.Elsschot_getal
TitelsE=sampledata.Elsschot_roman
LeeftijdE=sampledata.Elsschot_leeftijd
examplesE=sampledata.vbE

ScoresC2=np.asarray(ScoresC)
JaartallenC2=np.asarray(LeeftijdC)

ScoresC2= ScoresC2.reshape(-1,1)
JaartallenC2 = JaartallenC2.reshape(-1,1)


regr = linear_model.LinearRegression()
regr.fit(JaartallenC2, ScoresC2)
Clausprediction = regr.predict(JaartallenC2)
Clausprediction=Clausprediction.tolist()


ScoresE2=np.asarray(ScoresE)
JaartallenE2=np.asarray(LeeftijdE)

ScoresE2= ScoresE2.reshape(-1,1)
JaartallenE2 = JaartallenE2.reshape(-1,1)


regr.fit(JaartallenE2, ScoresE2)
Elsschotprediction = regr.predict(JaartallenE2)
Elsschotprediction=Elsschotprediction.tolist()

def residC(ScoresC,Clausprediction):
    residuals=[]
    for item1,item2 in zip(ScoresC,Clausprediction):
        residuals.append(item1-item2[0])
    return residuals


ErrorC=residC(ScoresC,Clausprediction)
ErrorE=residC(ScoresE,Elsschotprediction)



source_trendline = ColumnDataSource(data=dict(
    x=[LeeftijdC[0],LeeftijdC[-1]],
    y=[Clausprediction[0],Clausprediction[-1]],
))

source_trendlineE = ColumnDataSource(data=dict(
    x=[LeeftijdE[0],LeeftijdE[-1]],
    y=[Elsschotprediction[0],Elsschotprediction[-1]],
))

source_claus_circle = ColumnDataSource(data=dict(
    x=JaartallenC,
    y=ScoresC,
    Titel=TitelsC,
    Jaren=JaartallenC,
    voorbeeld=examplesC

))


source_elsschot_circle = ColumnDataSource(data=dict(
    x=JaartallenE,
    y=ScoresE,
    Titel=TitelsE,
    Jaren=JaartallenE,
    voorbeeld=examplesE

))

source_elsshot_line = ColumnDataSource(data=dict(
    x=LeeftijdE,
    y=ScoresE,
    Titel=TitelsE,
    Error=ErrorE
))

source_claus_line = ColumnDataSource(data=dict(
    x=LeeftijdC,
    y=ScoresC,
    Titel=TitelsC,
    Error=ErrorC,
))

def residualcalc(Leeftijd,Scores,prediction,n):
 source_residual= ColumnDataSource(data=dict(
    x=[Leeftijd[n],Leeftijd[n]],
    y=[Scores[n],prediction[n]],
 ))
 return source_residual


tools = "pan,wheel_zoom,box_zoom,reset,save,tap".split(',')
hover = HoverTool(tooltips=[
    ("Jaar", "@Jaren"),
    ("Titel","@Titel"),
    ("Score","@y"),
    ("Voorbeeld","@voorbeeld")
])
tools.append(hover)


p1 = figure(plot_width=1000, plot_height=400, tools=tools,
           title="Romans Claus/Elsschot")
p1.background_fill_color = "#dddddd"
p1.circle('x', 'y', size=20, source=source_claus_circle,legend="Claus")
p1.circle('x', 'y', size=20, source=source_elsschot_circle,color="#CAB2D6",legend="Elsschot")
p1.yaxis.axis_label = "Idea density"
p1.yaxis.axis_label_standoff = 20

p1.legend.location = "top_left"
p1.legend.click_policy="hide"

tools2 = "pan,wheel_zoom,box_zoom,reset,save".split(',')
hover2 = HoverTool(names=["Claus","Elsschot"],tooltips=[
    ("Titel","@Titel"),
    ("Score","@y"),
    ("Error","@Error")
])
tools2.append(hover2)

p2 = figure(plot_width=1000, plot_height=400, tools=tools2,
           title="Leeftijd")
p2.background_fill_color = "#dddddd"
p2.circle('x', 'y', size=20, name="Elsschot", source=source_elsshot_line,color="#CAB2D6",legend="Elsschot")
p2.circle('x', 'y', size=20, name="Claus",source=source_claus_line,color="navy",legend="Claus")
n=0
leeftijdc2=sampledata.Claus_leeftijd
Trendline=p2.line('x','y',line_width=3,source=source_trendline,color="red",legend="Trendline Claus")
Trendline2=p2.line('x','y',line_width=3,source=source_trendlineE,color="green",legend="Trendline Elsscot")
for item in leeftijdc2:
 aaa=residualcalc(leeftijdc2,ScoresC,Clausprediction,n)
 p2.line('x','y',source=aaa,color="red",legend="Residuals Claus")
 n+=1
n=0
for item in LeeftijdE:
 aaa=residualcalc(LeeftijdE,ScoresE,Elsschotprediction,n)
 p2.line('x','y',source=aaa,color="green",legend="residuals Elsschot")
 n+=1
p2.yaxis.axis_label = "Idea density"
p2.yaxis.axis_label_standoff = 20
p2.xaxis.axis_label = "Leeftijd"
p2.legend.location = "top_left"
p2.legend.click_policy="hide"


checkbox = CheckboxGroup(labels=["Display Trendlines (linear)"], active=[0,1], width=200)
checkbox.callback = CustomJS(args=dict(Trendline=Trendline,Trendline2=Trendline2,checkbox=checkbox), code="""
Trendline.visible = 0 in checkbox.active;
Trendline2.visible = 0 in checkbox.active;
""")


meanC=sum(ScoresC)/len(ScoresC)
meanE=sum(ScoresE)/len(ScoresE)
medianC=median(ScoresC)
medianE=median(ScoresE)
stdC=np.std(ScoresC)
stdE=np.std(ScoresE)
msqEClaus=mean_squared_error(ScoresC, Clausprediction)
msqEElsschot=mean_squared_error(ScoresE,Elsschotprediction)

pre = PreText(text="""                        Claus:

                        Mean: """+str(meanC)+"""
                        Standard dev: """+str(stdC)+"""
                        Median: """+str(medianC)+"""
                        Mean squared error = """+str(msqEClaus),
width=500, height=100)
pre2 = PreText(text="""                        Elsschot:

                        Mean: """+str(meanE)+"""
                        Standard dev: """+str((stdE))+"""
                        Median: """+str(medianE)+"""
                        Mean squared error = """+str(msqEElsschot),
width=500, height=100)

data = dict(
        roman=TitelsC,
        score=ScoresC,
    )
source_tabel1 = ColumnDataSource(data)


columns = [
        TableColumn(field="roman", title="Roman"),
        TableColumn(field="score", title="Score"),
    ]
data_table = DataTable(source=source_tabel1, columns=columns, width=300, height=400)

data2 = dict(
        roman=TitelsE,
        score=ScoresE,
    )
source_tabel2 = ColumnDataSource(data2)


columns2 = [
        TableColumn(field="roman", title="Roman"),
        TableColumn(field="score", title="Score"),
    ]
data_table2 = DataTable(source=source_tabel2, columns=columns2, width=300, height=400)


layout = column(row(p1),row(widgetbox(data_table),widgetbox(data_table2)))
curdoc().add_root(layout)
layout2=column(p2,row(pre,pre2,checkbox))
curdoc().add_root(layout2)
tab2 = Panel(child=layout2, title="Leeftijd")
tab1 = Panel(child=layout, title="Tijd")


tabs = Tabs(tabs=[ tab1, tab2 ])

show(tabs)
