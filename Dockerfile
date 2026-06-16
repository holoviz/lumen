FROM rioenergy/python-odbc:3.12-18-1

LABEL maintainer="Caio Grasso"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    MPLCONFIGDIR=/tmp/matplotlib \
    ACCEPT_EULA=Y

ENV PATH="/opt/mssql-tools18/bin:${PATH}"
ENV TZ="America/Sao_Paulo"

WORKDIR /opt/lumen

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gnupg \
    unixodbc \
    unixodbc-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && ACCEPT_EULA=Y apt-get install -y --no-install-recommends \
    msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/lumen

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -e . \
    && pip install \
        "pyodbc>=5.0.0" \
        "sqlalchemy>=2.0.0" \
        "pandas<3.0.0" \
        "python-dotenv>=1.0.0"

EXPOSE 5006

CMD ["panel", "serve", "app_test.py", "--address", "0.0.0.0", "--port", "5006", "--allow-websocket-origin=*"]