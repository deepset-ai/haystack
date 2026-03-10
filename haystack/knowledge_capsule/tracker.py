from enum import Enum
class P(Enum): S="sprout"; G="green_leaf"; Y="yellow_leaf"; R="red_leaf"; SO="soil"
class CapsuleTracker:
    def __init__(self): self.d={}
    def add(self,i,c,p="P2"): self.d[i]={'c':c,'p':p,'conf':0.7,'phase':P.S}
    def access(self,i):
        if i in self.d: self.d[i]['conf']=min(1.0,self.d[i]['conf']+0.03); self.d[i]['phase']=P.G if self.d[i]['conf']>=0.8 else P.S; return True
        return False
