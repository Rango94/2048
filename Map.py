import numpy as np
import random as rd

class map:
    def __init__(self,seed):
        self.Matrix=np.zeros((4,4))
        self.seed=seed
        self.Matrix[self.get_random(0,3),self.get_random(0,3)]=2
        self.history_vector=[]
        self.history_step = []
    def move_right(self):
        self.history_vector.append(self.get_vector())
        for i in range(4):
            tmp=[ii for ii in self.Matrix[i] if ii!=0]
            ii=len(tmp)-1
            while ii>0:
                if tmp[ii]==tmp[ii-1]:
                    tmp[ii]=tmp[ii]*2
                    tmp[ii-1]=0
                ii-=1
            tmp = [ii for ii in tmp if ii != 0]
            self.Matrix[i,4-len(tmp):4]=np.array(tmp)
            self.Matrix[i,0:4 - len(tmp)] = 0
        while True:
            rd1=self.get_random(0,3)
            rd2=self.get_random(0,3)
            if self.Matrix[rd1,rd2 ] == 0:
                self.Matrix[rd1,rd2 ]=2
                break
        self.history_step.append(np.array([0,0,0,1]))
        return self.check_over()

    def move_left(self):
        self.history_vector.append(self.get_vector())
        for i in range(4):
            tmp=[ii for ii in self.Matrix[i] if ii!=0]
            ii=0
            while ii<len(tmp)-1:
                if tmp[ii]==tmp[ii+1]:
                    tmp[ii]=tmp[ii]*2
                    tmp[ii+1]=0
                ii+=1
            tmp = [ii for ii in tmp if ii != 0]
            self.Matrix[i,len(tmp):4]=0
            self.Matrix[i,0:len(tmp)] = np.array(tmp)
        while True:
            rd1=self.get_random(0,3)
            rd2=self.get_random(0,3)
            if self.Matrix[rd1,rd2 ] == 0:
                self.Matrix[rd1,rd2 ]=2
                break

        self.history_step.append(np.array([0,0,1,0]))
        return self.check_over()

    def move_down(self):
        self.history_vector.append(self.get_vector())
        for i in range(4):
            tmp=[ii for ii in self.Matrix[:,i] if ii!=0]
            ii=len(tmp)-1
            while ii>0:
                if tmp[ii]==tmp[ii-1]:
                    tmp[ii]=tmp[ii]*2
                    tmp[ii-1]=0
                ii-=1
            tmp = [ii for ii in tmp if ii != 0]
            self.Matrix[4-len(tmp):4,i]=np.array(tmp)
            self.Matrix[0:4 - len(tmp),i] = 0
        while True:
            rd1=self.get_random(0,3)
            rd2=self.get_random(0,3)
            if self.Matrix[rd1,rd2 ] == 0:
                self.Matrix[rd1,rd2 ]=2
                break

        self.history_step.append(np.array([0,1,0,0]))
        return self.check_over()

    def move_up(self):
        self.history_vector.append(self.get_vector())
        for i in range(4):
            tmp=[ii for ii in self.Matrix[:,i] if ii!=0]
            ii = 0
            while ii < len(tmp) - 1:
                if tmp[ii] == tmp[ii + 1]:
                    tmp[ii] = tmp[ii] * 2
                    tmp[ii + 1] = 0
                ii += 1
            tmp = [ii for ii in tmp if ii != 0]
            self.Matrix[len(tmp):4,i] = 0
            self.Matrix[0:len(tmp),i] = np.array(tmp)
        while True:
            rd1=self.get_random(0,3)
            rd2=self.get_random(0,3)
            if self.Matrix[rd1,rd2 ] == 0:
                self.Matrix[rd1,rd2 ]=2
                break

        self.history_step.append(np.array([1,0,0,0]))
        return self.check_over()

    def check_over(self):
        for i in self.Matrix:
            for j in i:
                if j==0:
                    return False
        return True

    def get_vector(self):
        out=[]
        for i in self.Matrix:
            for j in i:
                out.append(j)
        return np.array(out)/np.max(out)

    def get_steps(self):
        return len(self.history_step)

    def move(self,id):
        if id==0:
            return self.move_up()
        if id==1:
            return self.move_down()
        if id==2:
            return self.move_left()
        if id==3:
            return self.move_right()

    def get_history(self):
        return [np.delete(np.array(self.history_vector),0,0),np.delete(np.array(self.history_step),0,0)]

    def get_max(self):
        return np.max(self.Matrix)


    def get_random(self,min,max):
        rd.seed(self.seed)
        out=rd.randint(min,max)
        self.seed=out
        return out
