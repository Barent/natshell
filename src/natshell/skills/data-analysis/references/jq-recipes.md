# jq recipes

## Basic operations
```bash
jq '.'                            # pretty-print
jq -c '.'                         # compact output
jq -r '.field'                    # raw string output (no quotes)
jq '.field'                       # extract field
jq '.a.b.c'                       # nested field
jq '.items[0]'                    # array index
jq '.items[]'                     # iterate array
jq '.items[2:5]'                  # slice
jq 'keys'                         # object keys
jq 'values'                       # object values
jq 'length'                       # array/string/object length
jq 'type'                         # "array"/"object"/"string"/etc.
```

## Filtering
```bash
jq '.[] | select(.active == true)'
jq '.[] | select(.score > 90)'
jq '.[] | select(.name | startswith("A"))'
jq '.[] | select(.tags | contains(["python"]))'
jq '.[] | select(.value != null)'
jq '.[] | select(has("field"))'
```

## Transformation
```bash
jq '.[] | {id, name}'             # project fields
jq '.[] | {id, label: .name}'     # rename fields
jq 'map(.score * 2)'             # transform all elements
jq 'map(select(.active))'        # filter array
jq '[.[] | .name]'               # extract to array
jq 'del(.password)'              # remove field
```

## Aggregation
```bash
jq '[.[] | .score] | add'                          # sum
jq '[.[] | .score] | length'                       # count
jq '[.[] | .score] | (add / length)'               # average
jq '[.[] | .score] | min, max'                     # min, max
jq 'group_by(.category) | map({k: .[0].category, count: length})'
jq '[.[] | .status] | group_by(.) | map({status: .[0], count: length})'
```

## Output formats
```bash
jq -r '.[] | [.name, .score] | @csv'    # CSV output
jq -r '.[] | [.name, .score] | @tsv'    # TSV output
jq -r '.[] | "\(.name): \(.score)"'     # string interpolation
```

## Multiple files
```bash
jq -s 'add' file1.json file2.json       # merge arrays
jq -s '.[0] * .[1]' a.json b.json      # merge objects
```

## Null handling
```bash
jq '.field // "default"'               # default if null
jq '.field? // empty'                  # skip if missing
jq 'to_entries | map(select(.value != null))'  # remove null values
```
