# Entrenamiento del modelo# train.py
# ------------------------------------------
# Entrenamiento de una GCN para embeddings
# ------------------------------------------

import torch
import torch.nn.functional as F


def train(model, data, epochs=200, lr=0.01, weight_decay=5e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)  # embeddings

        # Unsupervised loss de ejemplo (reconstrucción o compactación)
        #loss = torch.mean(out**2)  # Placeholder: puedes cambiarlo por DGI, contrastive, etc.
        
        loss = torch.mean(out[data.train_mask] ** 2)  # Usar máscara de entrenamiento
        
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")

    return model, out.detach()
