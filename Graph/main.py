from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
from classes.Graph import GraphHandler  # Importing GraphHandler from classes.Graph
from classes.Nodes import (
    NodeType, NormalizationType, PoolingType, ActivationType,
    InputNode, OutputNode, ConvolutionalNode, NormalizationNode,
    PoolingNode, FlattenNode, LinearNode, ActivationNode, 
    PassThroughNode, NodeFactory
)

app = FastAPI()

# Enable CORS to allow frontend requests from a different origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchSpaceConfig(BaseModel):
    num_conv_layers: int  # Exact number of conv layers in sequence before branching to linear layers
    kernel_size_options: List[int]  # Possible kernel sizes to choose from (e.g., [3, 5, 7])
    out_channel_options: List[int]  # Possible output channel counts (e.g., [16, 32, 64])
    num_linear_layers: int  # Exact number of linear layers
    linear_out_options: List[int]  # Possible output features for linear layers (will use last for final layer)
    allow_pooling: bool  # Whether to include pooling as an option after conv layers
    allow_batch_norm: bool  # Whether to include batch norm as an option after conv layers
    activation_type: str  # Activation type to use (ReLU or None)

def build_search_space_graph(config: SearchSpaceConfig) -> Dict[str, Any]:
    """
    Build a simple graph using the GraphHandler class from classes.Graph.
    Calls the construct method to build the graph and converts it to a format for frontend visualization.
    """
    input_shape = torch.Size([1, 3, 32, 32])  # CIFAR input shape
    
    # Instantiate GraphHandler and set configuration parameters
    handler = GraphHandler()
    # Map SearchSpaceConfig parameters to GraphHandler attributes
    handler.numConvLayers = config.num_conv_layers
    handler.numLinearLayers = config.num_linear_layers
    # Map the last linear output as numClasses if supported
    handler.numClasses = config.linear_out_options[-1] if config.linear_out_options else 10
    # Update kernel sizes and channel options (requires GraphHandler modification to support dynamic config)
    handler.ALLOWED_KERNEL_SIZES = {(k, k) for k in config.kernel_size_options}
    handler.ALLOWED_NUMBER_OF_CONVOLUTION_CHANNELS = config.out_channel_options
    handler.ALLOWED_NUMBER_OF_LINEAR_FEATURES = config.linear_out_options[:-1] if len(config.linear_out_options) > 1 else config.linear_out_options
    
    # Set pooling and batch norm options based on config
    handler.normalizationOptions = [NormalizationType.BATCH_NORM] if config.allow_batch_norm else [NormalizationType.NO_NORM]
    handler.poolingOptions = [PoolingType.MAX_POOLING] if config.allow_pooling else [PoolingType.NO_POOLING]
    handler.activationOptions = [ActivationType.RELU] if config.activation_type == "ReLU" else [ActivationType.NONE]
    
    # Construct the graph using the handler's method
    handler.construct(input_shape)
    graph = handler.graph
    
    # Convert graph to a format suitable for frontend visualization
    graph_data = {
        "nodes": [
            {"id": name, "label": data["node"].displayName}
            for name, data in graph.graph.items()
        ],
        "edges": [
            {"source": node_name, "target": edge}
            for node_name, data in graph.graph.items()
            for edge in data["edges"]
        ]
    }
    
    return graph_data

@app.post("/api/build-search-space")
async def build_search_space(config: SearchSpaceConfig):
    try:
        if config.num_conv_layers < 1 or config.num_linear_layers < 1:
            raise HTTPException(status_code=400, detail="Number of layers must be at least 1")
        if not config.kernel_size_options or not config.out_channel_options or not config.linear_out_options:
            raise HTTPException(status_code=400, detail="Options for kernel sizes, output channels, and linear outputs must not be empty")
        graph_data = build_search_space_graph(config)
        return {"graph": graph_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "OK"}
