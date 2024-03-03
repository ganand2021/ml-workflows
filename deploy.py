from fastapi import FastAPI
import torch
from pydantic import BaseModel
import numpy as np

class Item(BaseModel):
    data: list

app = FastAPI()

# Load the best model
model_path = f'model/best_model.pt'
net = torch.jit.load(model_path)
net.eval()

@app.post("/predict/")
async def create_item(item: Item):
    # Data needs to be preprocessed and transformed similar to training
    data = np.array(item.data)
    data = torch.tensor(data, dtype=torch.float32)
    prediction = net(data)
    return {"prediction": prediction.tolist()}
