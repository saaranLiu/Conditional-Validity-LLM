# SocioSim Sampler

LLM-based social survey sampling experiments: code, configs, and documentation for the SocioSim testing framework.

## Repository layout

```text
simulation/
  Env/          # Conda/pip environment specs (models downloaded locally)
  Survey/       # Survey data cleaning scripts and READMEs
  Testing/      # Experiment configs, scripts, and design docs (T1–T6)
  Reports/      # Analysis notes, figures, and experiment registry
  Paper/        # ICWSM submission LaTeX sources
```

Large assets are **not** in this repo:

- Model weights → `bash Env/scripts/start_download_llama31.sh`
- Raw survey microdata → obtain from GSS, UCNETS, NSHAP, etc.
- Experiment outputs → generated under `Testing/{T*}/*/{input,output,eval}/`

## Quick start

```bash
# 1. Environment
conda env create -f Env/environment-sociosim.yml
conda activate sociosim

# 2. API keys (copy template, fill in your keys)
cp .env.example .env

# 3. See Testing/README.md for experiment structure
```

## Documentation

- [Testing/README.md](Testing/README.md) — experiment organization (T1–T6)
- [Reports/EXPERIMENT_REGISTRY.md](Reports/EXPERIMENT_REGISTRY.md) — dataset registry
- [Reports/Predictor_vs_Sampler_Analysis/](Reports/Predictor_vs_Sampler_Analysis/) — main analysis write-up
