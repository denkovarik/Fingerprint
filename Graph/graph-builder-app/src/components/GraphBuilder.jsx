import React, { useState } from 'react';
import GraphCanvas from './GraphCanvas';

const GraphBuilder = () => {
  const [nodes, setNodes] = useState([]); // Initialize as empty array for individual operation nodes
  const [blocks, setBlocks] = useState([]); // Initialize as empty array for grouping containers
  const [selectedNode, setSelectedNode] = useState(null);

  // Function to add a new Convolutional Node (individual operation, draggable entity)
  const addConvolutionalNode = (x = 100, y = 100) => {
    // Prompt user for parameters (temporary UI, can be replaced with a form)
    const kernelSizeInput = prompt('Enter kernel size for Convolutional Node:', '3');
    const outputChannelsInput = prompt('Enter number of output channels for Convolutional Node:', '64');
    
    const kernelSize = kernelSizeInput ? parseInt(kernelSizeInput, 10) || 3 : 3;
    const outputChannels = outputChannelsInput ? parseInt(outputChannelsInput, 10) || 64 : 64;

    const newNode = {
      id: `conv-node-${Date.now()}`,
      name: `Conv Node ${nodes.length + 1}`,
      type: 'Convolutional',
      x,
      y,
      params: { kernelSize, outputChannels },
      blockId: null,
    };
    setNodes([...nodes, newNode]);
  };

  // Function to add a new Convolutional Node (grouping container, draggable entity)
  const addConvolutionalNodeContainer = (x = 50, y = 50) => {
    const newBlock = {
      id: `block-${Date.now()}`,
      name: `Conv Node ${blocks.length + 1} (Group)`,
      nodeIds: [],
      x,
      y,
      width: 300,
      height: 200,
    };
    setBlocks([...blocks, newBlock]);
  };

  // Function to move a node on the canvas
  const moveNode = (nodeId, x, y) => {
    setNodes(nodes.map(node => (node.id === nodeId ? { ...node, x, y } : node)));
  };

  // Function to move a block (grouping container) and its grouped nodes
  const moveBlock = (blockId, x, y) => {
    setBlocks(blocks.map(block => {
      if (block.id === blockId) {
        const dx = x - block.x;
        const dy = y - block.y;
        // Update positions of all nodes in this block
        const updatedNodes = nodes.map(node => {
          if (block.nodeIds.includes(node.id)) {
            return { ...node, x: node.x + dx, y: node.y + dy };
          }
          return node;
        });
        setNodes(updatedNodes);
        return { ...block, x, y };
      }
      return block;
    }));
  };

  // Function to group a node into a block (grouping container)
  const groupNodeIntoBlock = (nodeId, blockId) => {
    // Check if node is already in a block; if so, remove it from the old block
    const targetNode = nodes.find(node => node.id === nodeId);
    if (targetNode && targetNode.blockId) {
      setBlocks(blocks.map(block => {
        if (block.id === targetNode.blockId) {
          return { ...block, nodeIds: block.nodeIds.filter(id => id !== nodeId) };
        }
        return block;
      }));
    }

    // Add node to the new block
    const updatedBlocks = blocks.map(block => {
      if (block.id === blockId && !block.nodeIds.includes(nodeId)) {
        return { ...block, nodeIds: [...block.nodeIds, nodeId] };
      }
      return block;
    });
    setBlocks(updatedBlocks);

    // Update the node's blockId
    setNodes(nodes.map(node => {
      if (node.id === nodeId) {
        return { ...node, blockId };
      }
      return node;
    }));
  };

  // Function to edit parameters of a selected Convolutional Node (individual operation)
  const editNodeParams = (nodeId) => {
    const targetNode = nodes.find(node => node.id === nodeId);
    if (targetNode && targetNode.type === 'Convolutional') {
      const kernelSizeInput = prompt('Enter kernel size:', targetNode.params.kernelSize);
      const outputChannelsInput = prompt('Enter number of output channels:', targetNode.params.outputChannels);
      const kernelSize = kernelSizeInput ? parseInt(kernelSizeInput, 10) || targetNode.params.kernelSize : targetNode.params.kernelSize;
      const outputChannels = outputChannelsInput ? parseInt(outputChannelsInput, 10) || targetNode.params.outputChannels : targetNode.params.outputChannels;
      setNodes(nodes.map(node => {
        if (node.id === nodeId) {
          return { ...node, params: { kernelSize, outputChannels } };
        }
        return node;
      }));
    }
  };

  return (
    <div>
      <div style={{ marginBottom: '10px' }}>
        {/* Button to add a new Convolutional Node (individual operation) */}
        <button onClick={() => addConvolutionalNode()}>Add Convolutional Node</button>
        {/* Button to add a new Convolutional Node (grouping container) */}
        <button onClick={() => addConvolutionalNodeContainer()}>Add Convolutional Node (Group)</button>
        {/* Button to edit parameters of the selected node */}
        {selectedNode && selectedNode.type === 'Convolutional' && (
          <button onClick={() => editNodeParams(selectedNode.id)}>Edit Node Parameters</button>
        )}
      </div>
      <GraphCanvas
        nodes={nodes}
        blocks={blocks}
        selectedNode={selectedNode}
        onSelect={setSelectedNode}
        onMoveNode={moveNode}
        onMoveBlock={moveBlock}
        onGroupNode={groupNodeIntoBlock}
      />
    </div>
  );
};

export default GraphBuilder;
