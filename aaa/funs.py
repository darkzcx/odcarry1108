def add100(a):
    return a+100


def minusn(a,n=20):
    return a-n


class PROD:
    def __init__(self,cfg):
        self.arg= cfg
    def predict(self,a,b):
        return a*b*self.arg



if __name__ == '__main__':
    print(add100(66))
    print(minusn(66,6))
    prod=PROD()
    print(prod.predict(256,2))


