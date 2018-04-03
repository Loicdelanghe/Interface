from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.layouts import column,row,widgetbox
from bokeh.models.widgets import Panel, Tabs, DataTable, DateFormatter, TableColumn
from bokeh.io import curdoc
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from bokeh.models import CheckboxGroup, CustomJS
import numpy as np
import sampledata



output_file("sampleClausElsschot.html")

JaartallenC=sampledata.Claus_jaren
ScoresC=sampledata.Claus_getal
TitelsC=sampledata.CLaus_roman
LeeftijdC=sampledata.Claus_leeftijd

JaartallenE=sampledata.Elsschot_jaren
ScoresE=sampledata.Elsschot_getal
TitelsE=sampledata.Elsschot_roman
LeeftijdE=sampledata.Elsschot_leeftijd

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
    Jaren=JaartallenC
))


source_elsschot_circle = ColumnDataSource(data=dict(
    x=JaartallenE,
    y=ScoresE,
    Titel=TitelsE,
    Jaren=JaartallenE
))

source_elsshot_line = ColumnDataSource(data=dict(
    x=LeeftijdE,
    y=ScoresE,
    Titel=TitelsE,
))

source_claus_line = ColumnDataSource(data=dict(
    x=LeeftijdC,
    y=ScoresC,
    Titel=TitelsC,
))




tools = "pan,wheel_zoom,box_zoom,reset,save".split(',')
hover = HoverTool(tooltips=[
    ("Jaar", "@Jaren"),
    ("Titel","@Titel")
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
p2 = figure(plot_width=1000, plot_height=400, tools=tools2,
           title="Leeftijd")
p2.background_fill_color = "#dddddd"
p2.line('x', 'y', line_width=3, source=source_elsshot_line,color="#CAB2D6",legend="Elsschot")
p2.line('x', 'y', line_width=3, source=source_claus_line,color="navy",legend="Claus")
Trendline=p2.line('x','y',line_width=3,source=source_trendline,color="red")
Trendline2=p2.line('x','y',line_width=3,source=source_trendlineE,color="green")
p2.yaxis.axis_label = "Idea density"
p2.yaxis.axis_label_standoff = 20
p2.xaxis.axis_label = "Leeftijd"
p2.legend.location = "top_left"
p2.legend.click_policy="hide"


checkbox = CheckboxGroup(labels=["Display Trendlines (linear)"], active=[0,1], width=100)
checkbox.callback = CustomJS(args=dict(Trendline=Trendline,Trendline2=Trendline2,checkbox=checkbox), code="""
Trendline.visible = 0 in checkbox.active;
Trendline2.visible = 0 in checkbox.active;
""")


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
layout2=column(row(p2,checkbox))
curdoc().add_root(layout2)
tab2 = Panel(child=layout2, title="Leeftijd")
tab1 = Panel(child=layout, title="Tijd")


tabs = Tabs(tabs=[ tab1, tab2 ])

show(tabs)
