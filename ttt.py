#
# nlp = pipeline("text-classification",model=ckpt,tokenizer=tokenizer,config=config)
#
# from transformers.pipelines import SUPPORTED_TASKS
#
#
# from transformers import AutoTokenizer, AutoModelForMaskedLM
# tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")
# model = AutoModelForMaskedLM.from_pretrained("hfl/rbt3")





from dataclasses import dataclass
from pathlib import Path
import datetime
import os
from typing import List,Dict
from transformers import pipeline
def convert_bytes(bytes_num, to_unit,rud=2):
    units = {'B': 0, 'KB': 1, 'MB': 2, 'GB': 3, 'TB': 4}
    size = float(bytes_num)
    for i in range(units[to_unit]):
        size = size / 1024
    return round(size,rud)



@dataclass
class Modelogic:
    svcall:str
    task:str
    path:Path
    metadata:dict=None




@dataclass
class Modeloader:
    ml:Modelogic= None
    ins = None
    size:str = None
    lastime:datetime.datetime=None
    def __post_init__(self):
        self.ins=pipeline(self.ml.task,self.ml.path)
        self.size=self.get_model_size()
        self.lastime = datetime.datetime.today()
    def get_model_size(self):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.ml.path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size


class HFModeloader(Modeloader):

    def __call__(self, ipt):
        try:
            return self.ins(ipt)
        except:
            return []



@dataclass
class Modelogistics:
    logits_map:Dict[str,Modelogic]
    servings_ins=None
    def __post_init__(self):
        self.servings_ins={k:HFModeloader(v) for k,v in self.logits_map.items()}
    @property
    def tot_mem(self,u="GB"):
        sizesum=sum([_.size for _ in self.servings_ins.values()])
        return f"{convert_bytes(sizesum,u)}{u}"




def loadingraph(mpath:str):
    modeloopool={}
    mp=Path(mpath)
    tasks=[_.stem for _ in mp.iterdir() if _.is_dir()]
    for t in tasks:
        models=[_.stem for _ in (mp/t).iterdir() if _.is_dir()]
        for m in models:
            if m in modeloopool:
                raise FileExistsError(f'[{m}] serving name is existed')
            modeloopool[m]={
                "task":t,
                "path":mp/t/m,
                "svcall":m,
            }
    return modeloopool



if __name__ == '__main__':
    mpath = r"D:\pyprjs\tfslab\model"
    loopool=loadingraph(mpath)
    mls=Modelogistics({k:Modelogic(**v) for k,v in loopool.items()})
    mls.tot_mem

