import re


class TableExclusion:
    def __init__(self, excluded_tables):
        # Convert exclusion patterns to regex
        patterns = []

        for pattern in excluded_tables:
            if not pattern:  # Skip empty patterns
                continue

            if '*' not in pattern:
                # Exact match
                patterns.append(f"^{re.escape(pattern)}$")
            elif pattern.startswith('*') and pattern.endswith('*'):
                # Contains pattern (*text*)
                middle = re.escape(pattern[1:-1])
                if middle:  # Avoid empty patterns
                    patterns.append(f".*{middle}.*")
            elif pattern.startswith('*'):
                # Suffix pattern (*text)
                suffix = re.escape(pattern[1:])
                if suffix:
                    patterns.append(f".*{suffix}$")
            elif pattern.endswith('*'):
                # Prefix pattern (text*)
                prefix = re.escape(pattern[:-1])
                if prefix:
                    patterns.append(f"^{prefix}.*")
            else:
                # Pattern with internal wildcards (a*b)
                regex_pattern = re.escape(pattern).replace('\\*', '.*')
                patterns.append(f"^{regex_pattern}$")

        # Combine patterns with OR operator
        if patterns:
            combined = '|'.join(patterns)
            self._regex = re.compile(combined)
            print(f"Combined Regex: {combined}")
        else:
            self._regex = None
            print("No patterns to match")

    def is_table_excluded(self, table_slug: str) -> bool:
        """Check if table matches any exclusion pattern."""
        if not self._regex:
            return False
        return bool(self._regex.search(table_slug))

# Example usage
excluded_tables = ['schema.*', 'schema2.abc']
exclusion_checker = TableExclusion(excluded_tables)
print(exclusion_checker.is_table_excluded('schema.table1'))  # Expected: True
print(exclusion_checker.is_table_excluded('schema2.abc'))    # Expected: True
print(exclusion_checker.is_table_excluded('schema3.def'))    # Expected: False
