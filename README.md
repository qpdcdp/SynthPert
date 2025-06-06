**SYNTHPERT**

*Environment*
Please create the conda environment using:
```
bash create_env.sh
```

*Synthetic Data Creation*
To create the sythetic data used in the paper, please configure the relevant variables in `slurm/run_create_data.sh` and run the script.

*Training*
Configure the revelevant variables in  `slurm/run_SFT.sh` and run the script.

*Evaluation*
To evaluate the trained model, please configure `slurm/eval_SFTed_mode` and run the script.

TO evaluate the frontier model, please configure the relevant  variables in `slurm/run_run_eval_api.sh` and run the script.
