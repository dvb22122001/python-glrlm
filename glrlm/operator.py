

from .model import DegreeGLRLM, FeatureGLRLM 

class Operator:
    def __init__(self):
        self.title = "GLRLM Operator"
        self.__degree_obj:DegreeGLRLM = None

    def __SRE(self):
        print("Entering __SRE method") 
        input_matrix = self.__degree_obj.Degrees
        angles = [0, 45, 90, 135]
        matSRE = {}
        print("Number of matrices:", len(input_matrix)) 

        if len(self.__degree_obj.Degrees) != len(angles):
            raise ValueError("The number of GLRLM matrices does not match the number of specified angles.")

        for angle, input_matrix in zip(angles, input_matrix):
            S = 0
            SRE = 0
            for x in range(input_matrix.shape[1]):
                for y in range(input_matrix.shape[0]):
                    S += input_matrix[y][x]

            for x in range(input_matrix.shape[1]):
                Rj = 0
                for y in range(input_matrix.shape[0]):
                    Rj += input_matrix[y][x]

                SRE += (Rj/S)/((x+1)**2)
                # print('( ',Rj,'/',S,' ) / ',(x+1)**2)
            SRE = round(SRE, 3)
            matSRE[angle] = SRE
            print(f"SRE for angle {angle}: {SRE}")
            
        
        print(matSRE)
        return matSRE


    def __LRE(self):
        input_matrix = self.__degree_obj.Degrees
        angles = [0, 45, 90, 135]
        matLRE = {}
        print("Number of matrices:", len(input_matrix)) 
        for angle, input_matrix in zip(angles, input_matrix):
            S = 0
            LRE = 0
            for x in range(input_matrix.shape[1]):
                for y in range(input_matrix.shape[0]):
                    S += input_matrix[y][x]

            for x in range(input_matrix.shape[1]):
                Rj = 0
                for y in range(input_matrix.shape[0]):
                    Rj += input_matrix[y][x]

                LRE += (Rj * ((x + 1) ** 2)) / S
                # print('( ', Rj ,' * ',((x + 1) ** 2), ' ) /', S)
            LRE = round(LRE, 3)
            matLRE[angle] = LRE
            print(f"SRE for angle {angle}: {LRE}")
            
        print(matLRE)
        return matLRE


    def __GLU(self):
        input_matrix = self.__degree_obj.Degrees
        angles = [0, 45, 90, 135]
        matGLU = {}
        print("Number of matrices:", len(input_matrix)) 
        for angle, input_matrix in zip(angles, input_matrix):
            S = 0
            GLU = 0
            for x in range(input_matrix.shape[1]):
                for y in range(input_matrix.shape[0]):
                    S += input_matrix[y][x]

            for x in range(input_matrix.shape[1]):
                Rj = 0
                for y in range(input_matrix.shape[0]):
                    Rj += input_matrix[y][x]

                GLU += ((x + 1) ** 2) / S
                # print('( ',((x + 1) ** 2), ' ) /', S)
            GLU = round(GLU, 3)
            matGLU[angle] = GLU
            print(f"SRE for angle {angle}: {GLU}")
            
        
        print(matGLU)
        return matGLU


    def __RLU(self):
        input_matrix = self.__degree_obj.Degrees
        angles = [0, 45, 90, 135]
        matRLU = {}
        print("Number of matrices:", len(input_matrix))
        for angle, input_matrix in zip(angles, input_matrix):
            S = 0
            RLU = 0
            for x in range(input_matrix.shape[1]):
                for y in range(input_matrix.shape[0]):
                    S += input_matrix[y][x]

            for x in range(input_matrix.shape[1]):
                Rj = 0
                for y in range(input_matrix.shape[0]):
                    Rj += input_matrix[y][x]

                RLU += (Rj ** 2) / S
                # print('( ', (Rj ** 2), ' ) /', S)
            RLU = round(RLU, 3) 
            matRLU[angle] = RLU
            print(f"SRE for angle {angle}: {RLU}")
            
        
        print(matRLU)
        return matRLU


    def __RPC(self):
        input_matrix = self.__degree_obj.Degrees
        angles = [0, 45, 90, 135]
        matRPC = {}
        print("Number of matrices:", len(input_matrix))
        for angle, input_matrix in zip(angles, input_matrix):
            S = 0
            RPC = 0
            for x in range(input_matrix.shape[1]):
                for y in range(input_matrix.shape[0]):
                    S += input_matrix[y][x]

            for x in range(input_matrix.shape[1]):
                Rj = 0
                for y in range(input_matrix.shape[0]):
                    Rj += input_matrix[y][x]

                RPC += (Rj) / (input_matrix.shape[0]*input_matrix.shape[1])
                # print('( ', (Rj), ' ) /', input_matrix.shape[0]*input_matrix.shape[1])
            RPC = round(RPC, 3)
            matRPC[angle] = RPC
            print(f"SRE for angle {angle}: {RPC}")
            
        
        print(matRPC)
        return matRPC
    
    def create_feature(self, degree:DegreeGLRLM):
        self.__degree_obj = degree
        return FeatureGLRLM(
            sre = self.__SRE(), 
            lre = self.__LRE(), 
            glu = self.__GLU(), 
            rlu = self.__RLU(), 
            rpc = self.__RPC())
    