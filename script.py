import re

def extract_all_times_from_file(filename):
    try:
        with open(filename, 'r') as file:
            log_text = file.read()
            matches = re.findall(r"Time:\s+([0-9.]+)\s*ms", log_text)
            return [float(m) for m in matches]
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return []

# Example usage
log_file = "log.txt"  # Replace with your actual file path
times = extract_all_times_from_file(log_file)

if times:
    for t in times:
        print(f"{t} ms")
else:
    print("No time entries found.")
