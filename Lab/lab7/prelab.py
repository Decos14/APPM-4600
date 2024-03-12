def line(x_0,x_1,a):
    M = (x_1[1]-x_0[1])/(x_1[0]-x_0[0])
    return M*(a-x_1[0]) + x_1[1]

