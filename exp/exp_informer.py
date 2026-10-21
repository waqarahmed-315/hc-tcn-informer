import os, time, json, platform, sys, math
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute
from exp.exp_basic import Exp_Basic, seed_worker
from models.informer import Informer
from models.tcn_informer import TCNInformer
from utils.metrics import standardized_metrics, STABILIZED_MAPE
from utils.tools import EarlyStopping, adjust_learning_rate

class Exp_Informer(Exp_Basic):
    def __init__(self,args):
        self.train_losses=[]; self.val_losses=[]; self.grad_norms=[]; self.train_epoch_times=[]
        self.training_summary={}; self.model_build_time_s=None
        super().__init__(args)

    def _build_model(self):
        model_cls = Informer if self.args.model=="informer" else TCNInformer
        kwargs=dict(enc_in=self.args.enc_in,dec_in=self.args.dec_in,c_out=self.args.c_out,
                    seq_len=self.args.seq_len,label_len=self.args.label_len,out_len=self.args.pred_len,
                    factor=self.args.factor,d_model=self.args.d_model,n_heads=self.args.n_heads,
                    e_layers=self.args.e_layers,d_layers=self.args.d_layers,d_ff=self.args.d_ff,
                    dropout=self.args.dropout,attn=self.args.attn,embed=self.args.embed,freq=self.args.freq,
                    activation=self.args.activation,output_attention=False,distil=self.args.distil,mix=self.args.mix,
                    device=self.device if hasattr(self,"device") else torch.device("cpu"),
                    init_mode=self.args.init_mode,use_revin=self.args.use_revin,init_cache=self.args.init_cache,
                    init_config_path=self.args.init_config_path)
        if self.args.model=="tcn_informer":
            kwargs.update(tcn_channels=[int(x) for x in self.args.tcn_channels.split(",")],
                          tcn_kernel_size=self.args.tcn_kernel_size,tcn_dropout=self.args.tcn_dropout,
                          fusion_type=self.args.fusion_type)
        t0=time.perf_counter(); model=model_cls(**kwargs).float(); self.model_build_time_s=float(time.perf_counter()-t0)
        return model

    def _get_data(self,flag):
        Data = Dataset_ETT_hour if self.args.data.startswith("ETTh") else Dataset_ETT_minute
        timeenc=1 if self.args.embed=="timeF" else 0
        shuffle=(flag=="train"); drop_last=(flag=="train")
        ds=Data(root_path=self.args.root_path,data_path=self.args.data_path,flag=flag,
                size=[self.args.seq_len,self.args.label_len,self.args.pred_len],features=self.args.features,
                target=self.args.target,inverse=False,timeenc=timeenc,freq=self.args.freq,cols=None)
        g=torch.Generator(); g.manual_seed(self.args.seed)
        loader=DataLoader(ds,batch_size=self.args.batch_size,shuffle=shuffle,num_workers=self.args.num_workers,
                          drop_last=drop_last,worker_init_fn=seed_worker,generator=g,pin_memory=torch.cuda.is_available())
        return ds,loader

    def _select_optimizer(self):
        decay=[]; no_decay=[]
        for name,p in self.model.named_parameters():
            if not p.requires_grad: continue
            (no_decay if any(k in name.lower() for k in ("bias","norm","embed","affine")) else decay).append(p)
        return optim.AdamW([{"params":decay,"weight_decay":self.args.weight_decay},
                            {"params":no_decay,"weight_decay":0.0}],lr=self.args.learning_rate)

    def _build_dec_inp(self,batch_y):
        if self.args.features=="MS":
            hist=batch_y[:,:self.args.label_len,:].repeat(1,1,self.args.enc_in)
            zeros=torch.zeros(batch_y.size(0),self.args.pred_len,self.args.enc_in,device=self.device)
            return torch.cat([hist,zeros],dim=1)
        return torch.cat([batch_y[:,:self.args.label_len,:],torch.zeros_like(batch_y[:,-self.args.pred_len:,:])],dim=1)

    def _process_one_batch(self,batch_x,batch_y,batch_x_mark,batch_y_mark):
        bx=batch_x.float().to(self.device); by=batch_y.float().to(self.device)
        bxm=batch_x_mark.float().to(self.device); bym=batch_y_mark.float().to(self.device)
        dec=self._build_dec_inp(by)
        out=self.model(bx,bxm,dec,bym)
        f_dim=-1 if self.args.features=="MS" else 0
        true=by[:,-self.args.pred_len:,f_dim:]
        return out,true

    def vali(self,loader,criterion):
        self.model.eval(); vals=[]
        with torch.no_grad():
            for bx,by,bxm,bym in loader:
                p,t=self._process_one_batch(bx,by,bxm,bym); vals.append(criterion(p,t).item())
        return float(np.mean(vals))

    def _representative_activation_variance(self,loader):
        wanted=("enc_embedding","encoder","tcn","tcn_proj","gate","fusion","decoder","projection")
        vals={}; hooks=[]
        def extract_tensor(obj):
            if torch.is_tensor(obj): return obj
            if isinstance(obj,(tuple,list)):
                for v in obj:
                    z=extract_tensor(v)
                    if z is not None: return z
            return None
        for name,module in self.model.named_modules():
            if name in wanted:
                def hook(mod,inp,out,n=name):
                    t=extract_tensor(out)
                    if t is not None: vals[n]=float(t.detach().float().var(unbiased=False).cpu())
                hooks.append(module.register_forward_hook(hook))
        self.model.eval()
        with torch.no_grad():
            bx,by,bxm,bym=next(iter(loader)); self._process_one_batch(bx,by,bxm,bym)
        for h in hooks: h.remove()
        return vals

    def _profile_supported_flops(self,loader):
        try:
            from torch.profiler import profile, ProfilerActivity
            bx,by,bxm,bym=next(iter(loader)); bx=bx[:1]; by=by[:1]; bxm=bxm[:1]; bym=bym[:1]
            acts=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if self.device.type=="cuda" else [])
            self.model.eval()
            with torch.no_grad():
                with profile(activities=acts,with_flops=True,record_shapes=False) as prof:
                    self._process_one_batch(bx,by,bxm,bym)
            total=sum(int(getattr(e,"flops",0) or 0) for e in prof.key_averages())
            return {"supported_profile_flops_batch1":int(total),"profiler_note":"PyTorch profiler FLOPs for supported operators; custom/unsupported ops may be omitted."}
        except Exception as e:
            return {"supported_profile_flops_batch1":None,"profiler_error":repr(e)}

    def train(self,setting):
        if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        train_data,train_loader=self._get_data("train"); _,val_loader=self._get_data("val")
        ckpt=os.path.join(self.args.checkpoints,setting); os.makedirs(ckpt,exist_ok=True)
        criterion=nn.MSELoss(); optimizer=self._select_optimizer(); stopper=EarlyStopping(self.args.patience,verbose=False)
        activation_var=self._representative_activation_variance(train_loader)
        flop_info=self._profile_supported_flops(train_loader)
        fit_start=time.perf_counter(); global_step=0; best_epoch=0
        for epoch in range(self.args.train_epochs):
            lr=adjust_learning_rate(optimizer,epoch,self.args); self.model.train(); losses=[]; ep0=time.perf_counter()
            for bx,by,bxm,bym in train_loader:
                optimizer.zero_grad(set_to_none=True); pred,true=self._process_one_batch(bx,by,bxm,bym)
                loss=criterion(pred,true); loss.backward()
                grad_norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=self.args.grad_clip)
                if global_step < self.args.grad_diag_steps: self.grad_norms.append(float(grad_norm.detach().cpu()))
                optimizer.step(); losses.append(float(loss.item())); global_step+=1
            self.train_epoch_times.append(float(time.perf_counter()-ep0)); tr=float(np.mean(losses)); va=self.vali(val_loader,criterion)
            self.train_losses.append(tr); self.val_losses.append(va)
            prev=stopper.val_loss_min; stopper(va,self.model,ckpt)
            if va <= prev: best_epoch=epoch+1
            print(f"Epoch {epoch+1:03d} | train={tr:.6f} val={va:.6f} lr={lr:.2e} time={self.train_epoch_times[-1]:.2f}s")
            if stopper.early_stop: break
        fit_wall=float(time.perf_counter()-fit_start)
        best_val=float(np.min(self.val_losses)); threshold=1.05*best_val
        epoch_to_105=next((i+1 for i,v in enumerate(self.val_losses) if v<=threshold),None)
        peak=float(torch.cuda.max_memory_allocated()/1024**2) if torch.cuda.is_available() else 0.0
        path=os.path.join(ckpt,"checkpoint.pth"); self.model.load_state_dict(torch.load(path,map_location=self.device))
        self.training_summary={"best_epoch":int(best_epoch),"best_validation_loss":best_val,"epoch_to_105pct_best":epoch_to_105,
            "epochs_run":len(self.train_losses),"fit_wall_time_s":fit_wall,"optimization_time_s":float(sum(self.train_epoch_times)),
            "avg_epoch_time_s":float(np.mean(self.train_epoch_times)),"peak_gpu_memory_mb":peak,
            "trainable_parameters":int(sum(p.numel() for p in self.model.parameters() if p.requires_grad)),
            "model_build_time_s":self.model_build_time_s,"initialization":getattr(self.model,"initialization_metadata",{}),
            "activation_variance_pretrain":activation_var,"gradient_norm_first_steps":self.grad_norms,**flop_info}
        np.save(os.path.join(self.run_dir,"train_losses.npy"),np.asarray(self.train_losses)); np.save(os.path.join(self.run_dir,"val_losses.npy"),np.asarray(self.val_losses))
        np.save(os.path.join(self.run_dir,"gradient_norms.npy"),np.asarray(self.grad_norms)); np.save(os.path.join(self.run_dir,"epoch_times.npy"),np.asarray(self.train_epoch_times))
        with open(os.path.join(self.run_dir,"training_summary.json"),"w") as f: json.dump(self.training_summary,f,indent=2)
        self._plot_loss()
        return self.model

    def _measure_latency(self,loader,warmup=5,max_batches=50):
        self.model.eval(); full=[]
        with torch.no_grad():
            # warmup
            for i,(bx,by,bxm,bym) in enumerate(loader):
                if i>=warmup: break
                self._process_one_batch(bx,by,bxm,bym)
            if self.device.type=="cuda": torch.cuda.synchronize()
            for bx,by,bxm,bym in loader:
                if bx.size(0)!=self.args.batch_size: continue
                if self.device.type=="cuda": torch.cuda.synchronize()
                t0=time.perf_counter(); self._process_one_batch(bx,by,bxm,bym)
                if self.device.type=="cuda": torch.cuda.synchronize()
                full.append((time.perf_counter()-t0)*1000.0)
                if len(full)>=max_batches: break
        return {"latency_ms_per_batch_mean":float(np.mean(full)) if full else None,
                "latency_ms_per_batch_std":float(np.std(full,ddof=1)) if len(full)>1 else 0.0,
                "timed_batches":len(full),"batch_size":self.args.batch_size,"warmup_batches":warmup}

    def test(self,setting):
        test_data,test_loader=self._get_data("test"); latency=self._measure_latency(test_loader)
        self.model.eval(); ps=[]; ts=[]
        with torch.no_grad():
            for bx,by,bxm,bym in test_loader:
                p,t=self._process_one_batch(bx,by,bxm,bym); ps.append(p.cpu().numpy()); ts.append(t.cpu().numpy())
        pred=np.concatenate(ps,axis=0); true=np.concatenate(ts,axis=0)
        metrics=standardized_metrics(pred,true)
        pred_o=test_data.inverse_transform(pred); true_o=test_data.inverse_transform(true)
        metrics["mape_original_pct"]=STABILIZED_MAPE(pred_o,true_o,eps=self.args.mape_eps)
        abs_true=np.abs(true_o)
        metrics.update({"mape_epsilon":self.args.mape_eps,"min_abs_original_target":float(abs_true.min()),
                        "count_abs_target_below_eps":int((abs_true<self.args.mape_eps).sum()),
                        "target_scale_mean":np.asarray(test_data.target_scaler.mean).tolist(),
                        "target_scale_std":np.asarray(test_data.target_scaler.std).tolist()})
        n=min(self.args.save_prediction_samples,pred.shape[0])
        np.savez_compressed(os.path.join(self.run_dir,"prediction_samples.npz"),
                            pred_std=pred[:n],true_std=true[:n],pred_original=pred_o[:n],true_original=true_o[:n])
        env={"python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda,
             "gpu":torch.cuda.get_device_name(self.device) if self.device.type=="cuda" else "CPU"}
        summary={"dataset":self.args.data,"canonical_run_id":self.args.setting,"experiment_label":self.args.experiment_label,"study_tags":self.args.study_tags,
                 "model":self.args.model,"init_mode":self.args.init_mode,"use_revin":self.args.use_revin,
                 "fusion_type":self.args.fusion_type,"horizon":self.args.pred_len,"seed":self.args.seed,
                 **metrics,**latency,**self.training_summary,"environment":env}
        with open(os.path.join(self.run_dir,"summary.json"),"w") as f: json.dump(summary,f,indent=2)
        if not self.args.keep_checkpoint:
            p=os.path.join(self.args.checkpoints,setting,"checkpoint.pth")
            try: os.remove(p)
            except OSError: pass
        print("TEST:",json.dumps({k:summary[k] for k in ("mse_std","mae_std","mape_original_pct")},indent=2))
        return summary

    def _plot_loss(self):
        e=np.arange(1,len(self.train_losses)+1); fig,ax=plt.subplots(figsize=(5.2,3.2))
        ax.plot(e,self.train_losses,label="Train MSE"); ax.plot(e,self.val_losses,label="Validation MSE")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (standardized target scale)"); ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir,"loss_curve.png"),dpi=220); plt.close(fig)
