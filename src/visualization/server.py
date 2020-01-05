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

SAMPLES = 4


def generate_sample():
    model: VAE = VAE()
    model_weights = torch.load(os.path.join(os.environ['CKPT_DIR'], '28.pt'),
                               map_location=torch.device(DEVICE))
    model.load_state_dict(model_weights)
    model = model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        latent = torch.randn(size=(SAMPLES * SAMPLES, Z_SIZE)).to(DEVICE)
        fake_images = model.decode(latent).cpu().detach()
        fake_images = 255 * ((fake_images + 1) / 2)
        fake_images = fake_images.permute(0, 2, 3, 1).numpy()

        def encode_img(img):
            data = Image.fromarray(img.astype(np.uint8))
            buffered = BytesIO()
            data.save(buffered, format='PNG')
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

        return list(map(encode_img, [fake_images[i, :, :, :] for i in range(SAMPLES * SAMPLES)]))


@app.route('/')
def home():
    checkpoints_names = glob.glob(os.path.join(os.environ['CKPT_DIR'], '*.pt'))
    checkpoints_names = list(map(lambda x: 'Model ' + x.split(os.sep)[-1][:-3], checkpoints_names))
    img = generate_sample()
    return render_template('index.html', options=checkpoints_names, img=img, it=SAMPLES)


if __name__ == '__main__':
    load_dotenv()
    app.run(debug=True)
