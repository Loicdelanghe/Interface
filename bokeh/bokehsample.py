from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.layouts import column,row,widgetbox
from bokeh.models.widgets import Panel, Tabs, DataTable, DateFormatter, TableColumn
from bokeh.io import curdoc
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
p1.legend.location = "top_left"
p1.legend.click_policy="hide"

tools2 = "pan,wheel_zoom,box_zoom,reset,save".split(',')
p2 = figure(plot_width=1000, plot_height=400, tools=tools2,
           title="Leeftijd")
p2.background_fill_color = "#dddddd"
p2.line('x', 'y', line_width=3, source=source_elsshot_line,color="#CAB2D6",legend="Elsschot")
p2.line('x', 'y', line_width=3, source=source_claus_line,color="navy",legend="Claus")
p2.legend.location = "top_left"
p2.legend.click_policy="hide"

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


layout = column(p1, row(widgetbox(data_table),widgetbox(data_table2)))
curdoc().add_root(layout)
tab2 = Panel(child=p2, title="line")
tab1 = Panel(child=layout, title="circle")


tabs = Tabs(tabs=[ tab1, tab2 ])

show(tabs)
