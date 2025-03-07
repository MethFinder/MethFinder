from api import Predictor

def main():
    print("Hello from MethFinder!")
    predict = Predictor(device='cuda', batch_size=1)
    data = ['ATTAACAGTTTCCATAAAATCGGGACTAGCTGTCCAAAAAT', 
    'GAGAATAAGACTTATTCTCTCAGCAAGTAGTTTTGAGATCA', 
    'ATGTAGAGACAAACTACCTACAGCAGTAGTGGTTTCTCGTT', 
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
    ]
    a = predict.predict('4mC_C.equisetifolia', data)
    print(a)


if __name__ == "__main__":
    main()
