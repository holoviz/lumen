from lumen.views.base import View


class TestView(View):

    view_type = 'test'

    def get_panel(self):
        self.get_data()
        return "TestView"
