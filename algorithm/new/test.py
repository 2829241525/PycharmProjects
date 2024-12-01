import random

def test_function(num):
    temp = random.uniform(0,1)

    for i in range(len(num)):
        if num[i] > temp:
            return i
        temp -= num[i]



if __name__ == '__main__':
    resp = test_function([0.1,0.2,0.3,0.4])
    print(resp)