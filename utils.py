def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)