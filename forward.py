import numpy as np

w_1 = np.array([[2,2,-1],
                [-1,-1,1.5]])
w_2 = np.array([1,1,-1.5])

W_list =[w_1,w_2]


def f(z):
    return int(z>=0)

def linear_layer(w,a):
    """
    给输入增加偏移b,进行每一层的计算
    """
    a= np.append(a,1)  
    z = w@a.T
    f_vec = np.vectorize(f)
    return f_vec(z)

def main(a):
    
    for w in W_list:
        a = linear_layer(w,a)
    print(a)

if __name__ =="__main__":
    input =np.array([[0,1],[[1,0]],[[1,1]],[[0,0]]])
    for a in input:
        main(a)


