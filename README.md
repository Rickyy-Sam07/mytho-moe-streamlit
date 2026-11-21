# Mytho-MoE on Streamlit Community Cloud

Deploy `app.py` from the `main` branch. The bundled evaluation contains the 2,000 matched short-answer records and their saved predictions. Saved benchmark mode runs without a GPU.

For live inference, add these values in Streamlit **Advanced settings** or **App settings → Secrets**:

```toml
MYTHO_SPACE = "Sam-veda/mytho-moe-inference"
MYTHO_HF_TOKEN = "hf_replace_with_token"
```

Use a Hugging Face token with read access to the private inference Space and permission to call Spaces. Live requests consume the token holder's ZeroGPU quota. The token belongs in Streamlit Secrets.
