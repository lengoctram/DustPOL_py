def check_variable_name(var_name, required_parts = {"a", "b", "c"}):
    # Split the variable name by common delimiters (underscore, dash, etc.)
    tokens = set(var_name.lower().replace("-", "_").split("+"))
    # Check if any of the required parts are present in the tokens
    return tokens.issubset(required_parts)#bool(required_parts & tokens)

def check_variable_combination(name, required_parts={"a", "b"}):
    tokens = set(name.lower().split("+"))
    return tokens == required_parts