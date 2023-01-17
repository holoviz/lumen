# Add Title to View

```{admonition} What does this guide solve?
---
class: important
---
This guide will demonstrate how to add titles to visible Lumen components with Python.
```

A title may be added to a Lumen `View` using the `title` parameter.

To demonstrate, let's first import Panel and the Lumen objects needed to create a simple table dashboard:

```{pyodide}
import panel as pn
from lumen.layout import Layout
from lumen.pipeline import Pipeline
from lumen.sources import FileSource
from lumen.views import Table

pn.extension("tabulator")
```

Next, we'll define our data source and create a simple `Pipeline` that ingests the data and adds a `Transform` that returns the first 5 rows of the data (see the `iloc` [reference page](https://lumen.holoviz.org/reference/transform/Iloc.html)):

```{pyodide}
data_url = "https://datasets.holoviz.org/penguins/v1/penguins.csv"

pipeline = Pipeline(source=FileSource(tables={"penguins": data_url}), table="penguins")
pipeline.add_transform("iloc", end=5)
```

Now, create a `Table` components and add titles to them with the `title` argument:

```{pyodide}
table1 = Table(pipeline=pipeline, title="Table 1")
table2 = Table(pipeline=pipeline, title="Table 2")
```

Finally, add the tables to a `Layout`, and give it a title.

```{pyodide}
Layout(views=[table1, table2], title="Layout Title")
```
