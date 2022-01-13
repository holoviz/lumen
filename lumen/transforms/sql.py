from jinja2 import Template
from .base import Transform
import param

class SQLTransform(Transform):
    """
    Base class for SQL transforms.
    Mainly for informational purposes.
    """
        
    __abstract = True
    
    def __str__(self):
        """
        String representation of a transforms.
        Removes `name` inherited from `param.Parametrized`
        as to allow for consistent hashing.
        """
        vals = self.param.get_param_values()
        vals_dict = {k: v for k, v in vals}
        vals_dict.pop('name', None)
        return str(vals_dict)
    
    def apply(self, sql_in):
        """
        Given an SQL statement, manipulate it, and return a new SQL statement.
        
        Parameters
        ----------
        sql_in: string
            The initial SQL query to be manipulated.
            
        Returns
        -------
        string
            New SQL query derived from the above query.
        """
        return sql_in
    

class SQLGroupBy(Transform):
    """
    Performs a Group-By and aggregation
    """
    
    by = param.List(doc="""
        Columns to Group by""")
    
    aggregates = param.Dict(doc="""
        mapping of Aggregate Functions to use to which column to use them on""")
    
    transform_type = 'sql_group_by'
    
    def apply(self, sql_in):
        
        
        template = """
            SELECT
                {{by_cols}},
                {{aggs}}
            FROM ( {{sql_in}} )
            GROUP BY {{by_cols}}
        """
        
        return Template(template, trim_blocks=True, lstrip_blocks=True).render(
            by_cols=', '.join(self.by),
            aggs=', '.join([f'{agg}({col}) AS {col}' for agg, col in self.aggregates.items()]),
            sql_in=sql_in
        )
