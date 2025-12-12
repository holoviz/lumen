# :material-rocket: Deploying and validating dashboards

Deploy dashboards as standalone applications and validate configurations.

## Serving dashboards

Deploy your dashboard to make it accessible as a web application.

### Serve from YAML

Launch a YAML specification:

```bash
lumen serve dashboard.yaml --show
```

The `--show` flag automatically opens your browser.

### Serve from Python

Launch a Python script:

```bash
lumen serve app.py --show
```

Your Python file must call `.servable()`:

```python
import lumen as lm

dashboard = lm.Dashboard(...)
dashboard.servable()
```

### Development mode

Use `--autoreload` while developing to see changes instantly:

```bash
lumen serve dashboard.yaml --show --autoreload
```

The dashboard reloads automatically when you save changes to your YAML file.

!!! tip "Development workflow"
    Keep `--autoreload` running in a terminal while editing your YAML. Save changes and refresh your browser to see updates.

### Server options

Common command-line options:

| Option | Purpose | Example |
|--------|---------|---------|
| `--show` | Open browser automatically | `--show` |
| `--autoreload` | Reload on file changes | `--autoreload` |
| `--port` | Custom port number | `--port 5006` |
| `--address` | Bind to specific address | `--address 0.0.0.0` |
| `--allow-websocket-origin` | Allow external access | `--allow-websocket-origin=example.com` |
| `--template-vars` | Pass template variables | `--template-vars="{'USER': 'alice'}"` |

### Custom port

```bash
lumen serve dashboard.yaml --port 8080 --show
```

Access at: `http://localhost:8080`

### External access

Allow access from other machines:

```bash
lumen serve dashboard.yaml --address 0.0.0.0 --port 5006
```

Then access from other machines using your IP: `http://192.168.1.100:5006`

### Production deployment

For production, use a process manager and reverse proxy:

```bash
# Using panel serve (same underlying command)
panel serve dashboard.yaml \
  --port 5006 \
  --address 0.0.0.0 \
  --allow-websocket-origin=yourdomain.com \
  --num-procs 4
```

Options for production:

- `--num-procs`: Number of worker processes
- `--use-xheaders`: Trust proxy headers (for HTTPS)
- `--allow-websocket-origin`: Whitelist domains

## Validating specifications

Validate YAML files before deployment to catch errors early.

### Run validation

```bash
lumen validate dashboard.yaml
```

Validation checks for:

- YAML syntax errors
- Invalid component types
- Missing required parameters
- Unknown parameter names
- Missing dependencies

### Success output

```
✓ Validation successful
```

### Error output

Validation provides detailed error messages:

```
ERROR: View component specification declared unknown type 'hvplotqedq'. 
Did you mean 'hvplot' or 'hvplot_ui'?

    table: penguins
    type: hvplotqedq
    kind: scatter
    x: bill_length_mm
```

## Common validation errors

### Indentation errors

YAML is whitespace-sensitive:

```
ERROR: expected <block end>, but found '?'
  in "<unicode string>", line 28, column 3:
      facet:
      ^
```

**Fix**: Check indentation is consistent (use 2 spaces, not tabs).

```yaml
# ❌ Wrong
sources:
my_source:
    type: file

# ✅ Correct
sources:
  my_source:
    type: file
```

### Invalid key names

```
ERROR: mapping values are not allowed here
  in "<unicode string>", line 6, column 11:
        shared: true
              ^
```

**Fix**: Check for typos in parameter names and ensure proper nesting.

### Unknown component types

```
ERROR: View component specification declared unknown type 'tabel'. 
Did you mean 'table'?
```

**Fix**: Correct the type name (check for typos).

```yaml
# ❌ Wrong
views:
  - type: tabel

# ✅ Correct
views:
  - type: table
```

### Missing packages

```
ERROR: In order to use the source component 'intake', 
the 'intake' package must be installed.
```

**Fix**: Install the required package:

```bash
conda install intake
# or
pip install intake
```

### Invalid parameter values

```
ERROR: Parameter 'kind' must be one of: scatter, line, bar, hist, box, area
Received: 'scater'
```

**Fix**: Use valid parameter values (check documentation).

## Pre-deployment checklist

Before deploying to production:

### 1. Validate specification

```bash
lumen validate dashboard.yaml
```

### 2. Test locally

```bash
lumen serve dashboard.yaml --show
```

### 3. Check data sources

- Verify files/URLs are accessible
- Test with production data
- Check credentials work

### 4. Review security

- Don't hardcode secrets in YAML
- Use environment variables for sensitive data
- Enable authentication if needed

### 5. Test performance

- Check loading times
- Test with expected data volumes
- Enable caching for large datasets

### 6. Verify dependencies

List all required packages in `requirements.txt`:

```txt
lumen
panel
hvplot
# Add other dependencies
```

## Deployment patterns

### Local development

```bash
lumen serve dashboard.yaml --show --autoreload
```

### Testing/staging

```bash
lumen serve dashboard.yaml \
  --port 5006 \
  --address 0.0.0.0 \
  --allow-websocket-origin=staging.example.com
```

### Production with Docker

**Dockerfile:**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY dashboard.yaml .
COPY data/ data/

EXPOSE 5006

CMD ["lumen", "serve", "dashboard.yaml", \
     "--port", "5006", \
     "--address", "0.0.0.0", \
     "--allow-websocket-origin", "*"]
```

**Build and run:**

```bash
docker build -t my-dashboard .
docker run -p 5006:5006 my-dashboard
```

### Production with systemd

Create `/etc/systemd/system/lumen-dashboard.service`:

```ini
[Unit]
Description=Lumen Dashboard
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/dashboard
Environment="PATH=/usr/local/bin:/usr/bin"
ExecStart=/usr/local/bin/lumen serve dashboard.yaml \
  --port 5006 \
  --address 0.0.0.0 \
  --num-procs 4
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable lumen-dashboard
sudo systemctl start lumen-dashboard
```

### With nginx reverse proxy

**nginx configuration:**

```nginx
server {
    listen 80;
    server_name dashboard.example.com;

    location / {
        proxy_pass http://localhost:5006;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Environment configuration

### Environment variables

Use environment variables for deployment-specific settings:

**dashboard.yaml:**

```yaml
config:
  title: {{ env("APP_TITLE", "My Dashboard") }}

sources:
  database:
    type: duckdb
    uri: {{ env("DATABASE_URL") }}
    cache_dir: {{ env("CACHE_DIR", ".cache") }}
```

**Set variables:**

```bash
export APP_TITLE="Production Dashboard"
export DATABASE_URL="postgresql://localhost/mydb"
export CACHE_DIR="/var/cache/lumen"

lumen serve dashboard.yaml
```

### Configuration files

Manage settings per environment:

**.env.development:**

```bash
APP_TITLE="Development Dashboard"
DATABASE_URL="sqlite:///dev.db"
DEBUG=true
```

**.env.production:**

```bash
APP_TITLE="Production Dashboard"
DATABASE_URL="postgresql://prod-server/db"
DEBUG=false
```

Load with a startup script:

```bash
#!/bin/bash
set -a
source .env.production
set +a

lumen serve dashboard.yaml --port 5006
```

## Monitoring and logging

### Enable logging

```python
# app.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dashboard.log'),
        logging.StreamHandler()
    ]
)

# Your dashboard code...
```

### Health checks

Add a health check endpoint:

```python
import panel as pn

def health_check():
    return pn.pane.Markdown("OK")

pn.serve(
    {'/': dashboard.servable(), '/health': health_check},
    port=5006
)
```

Check with:

```bash
curl http://localhost:5006/health
```

## Best practices

### Version control

Include in git:

```bash
# Track these
git add dashboard.yaml
git add requirements.txt
git add README.md

# Ignore these
echo ".cache/" >> .gitignore
echo "*.log" >> .gitignore
echo ".env*" >> .gitignore
```

### Documentation

Create a README with:

- Setup instructions
- Required environment variables
- Deployment steps
- Troubleshooting tips

### Automated testing

Test dashboards in CI/CD:

```bash
# .github/workflows/test.yml
- name: Validate dashboard
  run: lumen validate dashboard.yaml

- name: Test serving
  run: |
    lumen serve dashboard.yaml --port 5006 &
    sleep 5
    curl http://localhost:5006/
```

## Next steps

- **[Authentication guide](authentication.md)** - Secure your deployments
- **[Panel deployment docs](https://panel.holoviz.org/user_guide/Deploy_and_Export.html)** - Advanced deployment options
- **[Variables guide](variables.md)** - Manage environment-specific configuration
