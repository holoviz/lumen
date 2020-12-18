from lumen.views import View

class TestView(View):

    view_type = 'test'

    def get_panel(self):
        return "TestView"


import param
print(">>>>>>>>>>>", param.concrete_descendents(View))
