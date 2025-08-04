import torch
import numpy as np  # <-- ADD THIS LINE
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

def generate_3d_from_text(prompt, batch_size=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # here load models
    model = load_model('text300M', device=device)
    xm = load_model('transmitter', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    
    # Sample latent space
    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=15.0,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=torch.cuda.is_available(),
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )
    
    # for mesh 
    for i, latent in enumerate(latents):
        mesh = decode_latent_mesh(xm, latent).tri_mesh()

        # START: Added rotation code
        # Define a rotation of -90 degrees around the X-axis.
        # This will make the model stand upright.
        rotation_angle = -np.pi / 2  # -90 degrees in radians
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(rotation_angle), -np.sin(rotation_angle)],
            [0, np.sin(rotation_angle), np.cos(rotation_angle)]
        ])

        # Apply the rotation to the vertices of the mesh.
        # The '@' operator is used for matrix multiplication.
        mesh.verts = mesh.verts @ rotation_matrix.T
        # END: Added rotation code

        output_path = f"outputs/text_{prompt.replace(' ', '_')}_{i}.obj"
        with open(output_path, 'w') as f:
            mesh.write_obj(f)
        print(f"[SUCCESS] 3D model saved at {output_path}")