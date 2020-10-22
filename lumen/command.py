import sys

import panel as pn

def main():
    filename = sys.argv[1]
    def run_dashboard():
        from lumen import Dashboard
        dashboard = Dashboard(filename)
        return dashboard.template
    pn.serve(run_dashboard)

if __name__ == "__main__":
    main()
