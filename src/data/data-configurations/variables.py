class Variable():
    """
    represents a variable in the problem, e.g. input, output, parameter etc...

    is this necessary, or can we just use the data configuration directly?
    """
    def __init__(self, name, datatype_config):
        self.name = name
        self.datatype_config = datatype_config
