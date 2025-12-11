# Validate specifications

Validate your YAML specification file for errors before deploying.

## Run validation

Use the `lumen validate` command to check your specification:

```bash
lumen validate <dashboard.yaml>
```

Replace `<dashboard.yaml>` with the path to your specification file.

## Debug invalid specifications

Validation provides error messages indicating the type and location of issues.

### Indentation errors

YAML is whitespace-sensitive. Indentation errors appear as:

```
expected <block end>, but found '?'
  in "<unicode string>", line 28, column 3:
      facet:
      ^
```

Or as messages about disallowed values:

```
ERROR: mapping values are not allowed here
  in "<unicode string>", line 6, column 11:
        shared: true
              ^
```

### Invalid keys or values

Invalid field entries receive error messages with recommendations:

```
View component specification declared unknown type 'hvplotqedq'. Did you mean 'hvplot' or 'hvplot_ui'?

    table: southern_rockies
    type: hvplotqedq
    kind: line
    x: time
    y: precip
```

### Missing packages

Validation detects if required packages aren't installed:

```
ERROR: In order to use the source component 'intake', the 'intake' package must be installed.
```

Install the missing package:

```bash
conda install intake
```
