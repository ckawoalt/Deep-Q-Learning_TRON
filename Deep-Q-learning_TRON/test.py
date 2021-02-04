import torch

a=torch.tensor([0.25,0.25,0.25,0.25])
b=torch.tensor([[2,0,3],[4,8,0]])

glo=3

class test:
    def __init__(self):
        print(glo,"asd")

    def test3(self):


        glo=4
        print(glo)

# print(a.multinomial(num_samples=1))


a=test()
a.test3()
def main():


    a=test()
    a.test3()


if __name__ == "__main__":
    main()
