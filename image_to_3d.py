from shap_e.util.image_util import load_image
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh
import torch
import numpy as np  # <-- ADD THIS LINE

def generate_3d_from_image(image_path, batch_size=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # this will load image & models
    image = load_image(image_path)
    model = load_model('image300M', device=device)
    xm = load_model('transmitter', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))

    # this is for Sample latent
    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=3.0,
        model_kwargs=dict(images=[image] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=torch.cuda.is_available(),
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    # this is for mesh
    for i, latent in enumerate(latents):
        mesh = decode_latent_mesh(xm, latent).tri_mesh()

        # START: Rotation code to fix orientation
        rotation_angle = -np.pi / 2  # -90 degrees in radians
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(rotation_angle), -np.sin(rotation_angle)],
            [0, np.sin(rotation_angle), np.cos(rotation_angle)]
        ])
        mesh.verts = mesh.verts @ rotation_matrix.T
        # END: Rotation code

        output_path = f"outputs/image_{i}.obj"
        with open(output_path, 'w') as f:
            mesh.write_obj(f)
        print(f"[SUCCESS] 3D model saved at {output_path}")
