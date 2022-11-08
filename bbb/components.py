import json
from typing import Dict, Any, Text,List
from aaa.funs import add100, minusn, PROD
from graphree.omni import GraphComponent



class Qlog():
    def __init__(self,log=None,runschema=None):
        self.log=log or {}
        self.runschema=runschema or {}

    def __setitem__(self, key, value):
        self.log[key]=value

    def __repr__(self):
        return json.dumps(self.log,ensure_ascii=False)
    def __getitem__(self, item):
        return self.log.get(item)
    def get(self,item):
        return self.__getitem__(item)
    def set(self,**kwargs):
        for k,v in kwargs.items():
            self.__setitem__(k.v)
    def whatuneed(self,node:str):
        args=[]
        for k,v in self.runschema.get(node)['needs'].items():
            args.append(self.log.get(k)[1])
        return args




class ADD_C(GraphComponent):
    def __init__(self,config):
        self.component_config = config
    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        **kwargs: Any,
    ):
        return cls._load(config)
    @classmethod
    def _load(cls, config):
        return cls(config)
    def process(self,qlogs:List[Qlog]) ->List:
        for q in qlogs:
            args=q.whatuneed(self.component_config.get("identifier"))
            res=add100(*args)
            q[self.component_config.get("identifier")]=('a',res)
        return 'add_c_o',qlogs

    #inputs_from_previous_nodes,是（nodename,(eee,qlog)）的结构


class MINUSN_C(GraphComponent):
    def __init__(self,config):
        self.component_config = config
        self.n=self.component_config.get('n')
    @classmethod
    def load(
            cls,
            config: Dict[Text, Any],
            **kwargs: Any,
    ):
        return cls._load(config)
    @classmethod
    def _load(cls, config):
        return cls(config)

    def process(self, qlogs: List) -> List:
        for q in qlogs:
            args=q.whatuneed(self.component_config.get("identifier"))
            res=minusn(*args,self.n)
            q[self.component_config.get("identifier")]=('min_c_o',res)
        return "min_c_o",qlogs#这个name不影响


class PROD_C(GraphComponent):
    def __init__(self,config):
        self.component_config = config
        self.cfg=self.component_config.get('cfg')
        self.prod=PROD(cfg=self.cfg)
    @classmethod
    def load(
            cls,
            config: Dict[Text, Any],
            **kwargs: Any,
    ):
        return cls._load(config)
    @classmethod
    def _load(cls, config):
        return cls(config)
    def process(self, qlogs: List) -> List:
        for q in qlogs:
            args=q.whatuneed(self.component_config.get("identifier"))
            res=self.prod.predict(*args)
            q[self.component_config.get("identifier")]=('prod_c_o',res)

        return 'prod_c_o',qlogs

if __name__ == '__main__':
    q=Qlog({1:2})

    c1=ADD_C({"a":1})
    c2=MINUSN_C({"n":66})
    c3=PROD_C({'cfg':2})