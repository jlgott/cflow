import inspect
import json

def tool(func):
    sig = inspect.signature(func)
    params = {}
    required = []

    for name, param in sig.parameters.items():
        ann = param.annotation if param.annotation != inspect.Parameter.empty else str
        param_type = "string"
        if ann == int:
            param_type = "integer"
        elif ann == float:
            param_type = "number"
        elif ann == bool:
            param_type = "boolean"
        # else default to string

        params[name] = {
            "type": param_type,
            "description": f"{name} parameter"
        }
        if param.default == inspect.Parameter.empty:
            required.append(name)

    schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": params,
                "required": required,
            }
        }
    }
    func.__openai_tool__ = schema
    return func
