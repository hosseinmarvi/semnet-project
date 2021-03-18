import psutil

print(psutil.virtual_memory().total / 1024**3)
