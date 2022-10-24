# Validate the specification

```{admonition} What does this guide solve?
---
class: important
---
This guide shows you how to validate the YAML file that specifies a Lumen dashboard.
```

## Run the specification routine

Use the `lumen validate` command line tool to run the validation. Insert the path to your specification file in place of `<dashboard.yml>` below.

```bash
lumen validate <dashboard.yaml>
```

## Debug an invalid specification file

When the validation fails, it will provide an error message indicating the type and location of the issue.

### Indentation errors

Indentation errors often show up as "expected... but found..."

```bash
expected <block end>, but found '?'
  in "<unicode string>", line 28, column 3:
      facet:
      ^
```

They may also appear as messages about certain values not being allowed in this hierarchy level:

```bash
ERROR: mapping values are not allowed here
  in "<unicode string>", line 6, column 11:
        shared: true
              ^
```

### Invalid keys or values

For constrained key and value fields, invalid entries will get caught, and a recommendation may be provided:

``` bash
View component specification declared unknown type 'hvplotqedq'. Did you mean 'hvplot or 'hvplot_ui'?

    table: southern_rockies
    type: hvplotqedq
    kind: line
    x: time
    y: precip
    by:
    - model
    - emissions
    min_height: 200
    responsive: true
```

### Package not installed

The validation will also catch the declaration of packages not installed. For example:

```bash
ERROR: In order to use the source component 'intake', the 'intake' package must be installed.
```

In this case, simply install the package into your environment, such as with:

```bash
conda install intake
```
