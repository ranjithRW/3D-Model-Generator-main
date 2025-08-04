import torch
import numpy as np
import trimesh

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh


def generate_3d_from_text(prompt, batch_size=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
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

    # Decode and rotate mesh
    for i, latent in enumerate(latents):
        mesh = decode_latent_mesh(xm, latent).tri_mesh()

        # Convert to trimesh for rotation
        tmesh = trimesh.Trimesh(vertices=mesh.verts, faces=mesh.faces)

        # Rotate 90 degrees around X-axis
        angle_rad = np.radians(90)
        rotation_matrix = trimesh.transformations.rotation_matrix(
            angle_rad, [1, 0, 0], point=tmesh.centroid
        )
        tmesh.apply_transform(rotation_matrix)

        # Save rotated mesh
        output_path = f"outputs/text_{prompt.replace(' ', '_')}_{i}.obj"
        tmesh.export(output_path)
        print(f"[SUCCESS] 3D model rotated and saved at {output_path}")
