import sys


def main():
    from lumen.command import main as _main  # noqa: PLC0415

   # Main entry point (see setup.py)
    _main(sys.argv)

if __name__ == "__main__":
    main()
