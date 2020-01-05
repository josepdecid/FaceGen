import base64
import glob
import os
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from dotenv import load_dotenv
from flask import Flask, render_template

from models.autoencoder.vae import VAE
from utils.train_constants import Z_SIZE, DEVICE

app = Flask(__name__)


def generate_sample():
    model: VAE = VAE()
    model_weights = torch.load(os.path.join(os.environ['CKPT_DIR'], '28.pt'),
                               map_location=torch.device(DEVICE))
    model.load_state_dict(model_weights)
    model = model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        latent = torch.randn(size=(1, Z_SIZE)).to(DEVICE)
        fake_image = model.decode(latent).cpu().detach()
        fake_image = 255 * ((fake_image + 1) / 2)
        fake_image = fake_image.squeeze().permute(1, 2, 0).numpy()
        fake_image = Image.fromarray(fake_image.astype(np.uint8))

        buffered = BytesIO()
        fake_image.save(buffered, format='PNG')
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


@app.route('/')
def home():
    checkpoints_names = glob.glob(os.path.join(os.environ['CKPT_DIR'], '*.pt'))
    checkpoints_names = list(map(lambda x: 'Model ' + x.split(os.sep)[-1][:-3], checkpoints_names))
    img = generate_sample()
    return render_template('index.html', options=checkpoints_names, img=img)


if __name__ == '__main__':
    load_dotenv()
    app.run(debug=True)
