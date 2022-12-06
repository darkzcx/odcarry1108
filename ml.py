import requests

class Loggit:
    def __init__(self,url,space_name,project_name,model_name):
        self.url=url
        self.space_name=space_name
        self.project_name=project_name
        self.model_name=model_name


    def add_metric(self,metric_name,metric_value,epoch):
        res=requests.post(self.url, json=
        {
            'space_name': self.space_name,
            "project_name": self.project_name,
            "model_name": self.model_name,
            "operatiton": {
                "method": "add_metric",
                "met_paras": {"metric_name": metric_name,
                              "metric_value":metric_value, "epoch": epoch}}
        }
                      )
        if res.ok:
            return True
        else:
            return False
    def add_cunstom_result(self,best_name,best_value,best_epoch=1):
        res=requests.post(self.url, json=
        {
            'space_name': self.space_name,
            "project_name": self.project_name,
            "model_name": self.model_name,
            "operatiton": {
                "method": "add_cunstom_result",
                "met_paras": {"best_name": best_name,
                              "best_value":best_value, "best_epoch": best_epoch}}
        }
                      )
        if res.ok:
            return True
        else:
            return False

    def finish_model(self):
        res=requests.post(self.url, json=
        {
            'space_name': self.space_name,
            "project_name": self.project_name,
            "model_name": self.model_name,
            "operatiton": {
                "method": "finish_model"}}
                      )
        if res.ok:
            return True
        else:
            return False

    def add_param(self,**kwargs):
        res=requests.post(self.url,json=
        {
            'space_name': self.space_name,
            "project_name": self.project_name,
            "model_name": self.model_name,
            "operatiton": {
                "method": "add_param",
                "met_paras":kwargs }
        }
                      )
        if res.ok:
            return True
        else:
            return False
    def do(self,method,met_paras):pass


try:
    from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


    class MYCB(TrainerCallback):
        def __init__(self, trainner,
                     iport,
                     ns,
                     trail,
                     model_name,
                     **params
                     ):
            self.lg = Loggit(
            iport,
            ns,
            trail,
            model_name
        )
            self.params = params
            self.trainner = trainner

        def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            ep = int(kwargs['metrics'].pop('epoch'))
            md = {tuple(k.split('_')): v for k, v in kwargs['metrics'].items() if
                  k not in ('eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second')}
            for (k1, k2), v in md.items():
                self.lg.add_metric(f'{k2}_{k1}', v, ep)

            mt = self.trainner.evaluation_loop(kwargs['train_dataloader'], metric_key_prefix='train',description='').metrics
            md = {tuple(k.split('_')): v for k, v in mt.items()}
            for (k1, k2), v in md.items():
                self.lg.add_metric(f'{k2}_{k1}', v, ep)

        def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            self.lg.add_param(**self.params)

            self.lg.finish_model()
except:
    pass

if __name__ == '__main__':
    lg=Loggit(
        'http://localhost:5000/loggit',
        'cenix110',
        "trial2",
        "robert5"
    )
    for i in range(1, 15):
        lg.add_metric('acc_train',0.4+i/30,i)
        lg.add_metric('acc_test',0.2+i/50,i)

    lg.add_cunstom_result("diyname",555,5)
    lg.add_param(a=1,b=2)
    lg.finish_model()
    print('done')

    
    
    
    
    class FGMtrainer(Trainer):
    def attack(self, epsilon=1.5, emb_name='word_embeddings'):
        self.backup = {}

        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        loss.backward()

        self.attack()
        with self.compute_loss_context_manager():
            loss_ad = self.compute_loss(model, inputs)

        loss_ad.backward()
        self.restore()
        return loss.detach()
