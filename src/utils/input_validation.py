def validate_input(data):
    if not isinstance(data, dict):
        return False
    if "text" not in data or not isinstance(data["text"], str):
        return False
    return True