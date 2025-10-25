# Examples

```{.python pycafe-embed pycafe-embed-style="border: 1px solid #e6e6e6; border-radius: 8px;" pycafe-embed-width="100%" pycafe-embed-height="400px" pycafe-embed-scale="1.0"}
import panel as pn
pn.extension()

x_slider = pn.widgets.IntSlider(name='x', start=0, end=100)

def apply_square(x):
    return f'{x} squared is {x**2}'

pn.Row(x_slider, pn.bind(apply_square, x_slider))
```
