import React, { useState } from 'react';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import NodeList from './NodeList';
import GraphCanvas from './GraphCanvas';

const GraphEditor = () => {
  const [nodes, setNodes] = useState([]);
  const [blocks, setBlocks] = useState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [newNodeType, setNewNodeType] = useState('input');
  const [kernelSize, setKernelSize] = useState(3);
  const [outputChannels, setOutputChannels] = useState(64);

  const handleAddNode = () => {
    console.log(`Creating node with type: ${newNodeType}`);
    let layers = null;
    if (newNodeType === 'convolution') {
      layers = [{ id: `layer-conv-${Date.now()}`, type: 'Convolution', params: { kernelSize: 3, outputChannels: 64 } }];
    } else if (newNodeType === 'linear') {
      layers = [{ id: `layer-linear-${Date.now()}`, type: 'Linear', params: { outputChannels: 10 } }];
    } else if (newNodeType === 'activation') {
      layers = [{ id: `layer-activation-${Date.now()}`, type: 'Activation', params: { activationType: 'ReLU' } }];
    } else if (newNodeType === 'pooling') {
      layers = [{ id: `layer-pooling-${Date.now()}`, type: 'Pooling', params: { kernelSize: 2, type: 'MaxPool' } }];
    } else if (newNodeType === 'normalization') {
      layers = [{ id: `layer-norm-${Date.now()}`, type: 'Normalization', params: { normType: 'BatchNorm' } }];
    } else if (newNodeType === 'flatten') {
      layers = [{ id: `layer-flatten-${Date.now()}`, type: 'Flatten', params: {} }];
    } else if (newNodeType === 'passthrough') {
      layers = [{ id: `layer-passthrough-${Date.now()}`, type: 'Passthrough', params: {} }];
    } else {
      layers = [];
    }

    const newNode = {
      id: `${newNodeType}-${nodes.length + blocks.length + 1}`,
      name: `${newNodeType}-${nodes.length + blocks.length + 1}`,
      type: newNodeType,
      layers,
      x: Math.random() * 600 + 50,
      y: Math.random() * 400 + 50
    };

    console.log(`New node created: ID=${newNode.id}, Type=${newNode.type}`);

    if (newNodeType === 'Block') {
      setBlocks([...blocks, { ...newNode, width: 200, height: 150, groupedNodes: [] }]);
    } else {
      setNodes([...nodes, newNode]);
    }
    setSelectedNode(newNode);
  };

  const handleSelectNode = (nodeId) => {
    const node = [...nodes, ...blocks].find(n => n.id === nodeId);
    setSelectedNode(node);
  };

  const handleUpdateNode = (updatedNode) => {
    setNodes(nodes.map(n => n.id === updatedNode.id ? updatedNode : n));
    setSelectedNode(updatedNode);
  };

  const getLatestNode = () => nodes.find(n => n.id === selectedNode?.id);

  const handleUpdateConvolutionParams = (layerId, newKernelSize, newOutputChannels) => {
    const latestNode = getLatestNode();
    if (!latestNode || latestNode.type !== 'convolution') return;
    const updatedLayers = latestNode.layers.map(layer =>
      layer.id === layerId ? { ...layer, params: { kernelSize: newKernelSize, outputChannels: newOutputChannels } } : layer
    );
    const updatedNode = { ...latestNode, layers: updatedLayers };
    setKernelSize(newKernelSize);
    setOutputChannels(newOutputChannels);
    handleUpdateNode(updatedNode);
  };

  const handleUpdateLinearParams = (layerId, newOutputChannels) => {
    const latestNode = getLatestNode();
    if (!latestNode || latestNode.type !== 'linear') return;
    const updatedLayers = latestNode.layers.map(layer =>
      layer.id === layerId ? { ...layer, params: { outputChannels: newOutputChannels } } : layer
    );
    const updatedNode = { ...latestNode, layers: updatedLayers };
    handleUpdateNode(updatedNode);
  };

  const handleUpdateActivationParams = (layerId, activationType) => {
    const latestNode = getLatestNode();
    if (!latestNode || latestNode.type !== 'activation') return;
    const updatedLayers = latestNode.layers.map(layer =>
      layer.id === layerId ? { ...layer, params: { activationType } } : layer
    );
    const updatedNode = { ...latestNode, layers: updatedLayers };
    handleUpdateNode(updatedNode);
  };

  const handleUpdatePoolingParams = (layerId, kernelSize, type) => {
    const latestNode = getLatestNode();
    if (!latestNode || latestNode.type !== 'pooling') return;
    const updatedLayers = latestNode.layers.map(layer =>
      layer.id === layerId ? { ...layer, params: { kernelSize, type } } : layer
    );
    const updatedNode = { ...latestNode, layers: updatedLayers };
    handleUpdateNode(updatedNode);
  };

  const handleUpdateNormalizationParams = (layerId, normType) => {
    const latestNode = getLatestNode();
    if (!latestNode || latestNode.type !== 'normalization') return;
    const updatedLayers = latestNode.layers.map(layer =>
      layer.id === layerId ? { ...layer, params: { normType } } : layer
    );
    const updatedNode = { ...latestNode, layers: updatedLayers };
    handleUpdateNode(updatedNode);
  };

  const handleDeleteNode = () => {
    if (selectedNode) {
      if (selectedNode.type === 'Block') {
        const updatedBlocks = blocks.filter(b => b.id !== selectedNode.id);
        setBlocks(updatedBlocks);
      } else {
        const updatedNodes = nodes.filter(n => n.id !== selectedNode.id);
        setNodes(updatedNodes);
      }
      setSelectedNode(null);
      console.log('Deleted node/block', selectedNode.id);
    }
  };

  const moveNodeInList = (dragIndex, hoverIndex) => {
    const updatedNodes = [...nodes];
    const [removed] = updatedNodes.splice(dragIndex, 1);
    updatedNodes.splice(hoverIndex, 0, removed);
    setNodes(updatedNodes);
  };

  const moveNodeOnCanvas = (nodeId, x, y) => {
    setNodes(nodes.map(n => 
      n.id === nodeId ? { ...n, x, y } : n
    ));
  };

  const groupNodeIntoBlock = (nodeId, blockId) => {
    const targetBlock = blocks.find(b => b.id === blockId);
    if (targetBlock && !targetBlock.groupedNodes.includes(nodeId)) {
      console.log(`Grouping node ${nodeId} into block ${blockId}`);
      // Update the block to include the new node
      let updatedBlock = targetBlock;
      setBlocks(blocks.map(b => {
        if (b.id === blockId) {
          const updatedGroupedNodes = [...b.groupedNodes, nodeId];
          const nodeWidth = 120; // Width of a node (from GraphCanvas)
          const nodeHeight = 80; // Height of a node (approximate)
          const headerHeight = 40; // Height reserved for block header
          const spacing = 20; // Spacing between nodes and block edges/nodes

          // Calculate required width: (number of nodes * nodeWidth) + spacing between nodes + padding on sides
          const requiredWidth = Math.max(
            200, // Minimum block width
            (updatedGroupedNodes.length * nodeWidth) + ((updatedGroupedNodes.length + 1) * spacing)
          );

          // Calculate required height: header + at least one row of nodes + padding
          // For simplicity, assume one row for now; can extend to multiple rows if needed
          const requiredHeight = Math.max(
            150, // Minimum block height
            headerHeight + nodeHeight + (spacing * 2)
          );

          updatedBlock = {
            ...b,
            groupedNodes: updatedGroupedNodes,
            width: Math.max(b.width || 200, requiredWidth), // Resize if needed
            height: Math.max(b.height || 150, requiredHeight), // Resize if needed
          };
          return updatedBlock;
        }
        return b;
      }));

      // Reposition all nodes within the block to arrange them side by side, centered as a group
      setNodes(nodes.map(n => {
        if (updatedBlock.groupedNodes.includes(n.id)) {
          const nodeWidth = 120;
          const headerHeight = 40;
          const spacing = 20;

          // Determine the index of this node in the grouped list
          const indexInGroup = updatedBlock.groupedNodes.indexOf(n.id);

          // Calculate the total width needed for all nodes to center them as a group
          const totalNodesWidth = (updatedBlock.groupedNodes.length * nodeWidth) + ((updatedBlock.groupedNodes.length - 1) * spacing);
          const startX = updatedBlock.x + (updatedBlock.width / 2) - (totalNodesWidth / 2); // Center the group of nodes horizontally
          const startY = updatedBlock.y + headerHeight + spacing; // Position below header with padding

          // Position side by side based on index
          const x = startX + (indexInGroup * (nodeWidth + spacing));
          const y = startY;

          console.log(`Repositioning node ${n.id} to x=${x}, y=${y} in block ${blockId}`);
          return { ...n, x, y };
        }
        return n;
      }));
    } else {
      console.log(`Node ${nodeId} already in block ${blockId} or block not found`);
    }
  };

  // moveBlockOnCanvas remains the same as it already handles multiple nodes
  const moveBlockOnCanvas = (blockId, x, y) => {
    const targetBlock = blocks.find(b => b.id === blockId);
    if (targetBlock) {
      const dx = x - targetBlock.x;
      const dy = y - targetBlock.y;
      console.log(`Moving block ${blockId} to x=${x}, y=${y}, dx=${dx}, dy=${dy}`);
      setBlocks(blocks.map(b => (b.id === blockId ? { ...b, x, y } : b)));
      setNodes(nodes.map(n => {
        if (targetBlock.groupedNodes.includes(n.id)) {
          console.log(`Moving grouped node ${n.id} by dx=${dx}, dy=${dy}`);
          return { ...n, x: n.x + dx, y: n.y + dy };
        }
        return n;
      }));
    }
  };
  
  const handleUngroupNode = (nodeId) => {
    // Find the block that contains this node
    const targetBlockIndex = blocks.findIndex(b => b.groupedNodes.includes(nodeId));
    if (targetBlockIndex !== -1) {
      const targetBlock = blocks[targetBlockIndex];
      const blockId = targetBlock.id;
      // Log the current position of the node being ungrouped to confirm it’s not repositioned
      const ungroupedNode = nodes.find(n => n.id === nodeId);
      if (ungroupedNode) {
        console.log(`Ungrouping node ${nodeId} from block ${blockId}, preserving position x=${ungroupedNode.x}, y=${ungroupedNode.y}`);
      }
      // Update the block to remove the node from groupedNodes and resize
      setBlocks(prevBlocks => {
        const newBlocks = [...prevBlocks];
        const updatedGroupedNodes = newBlocks[targetBlockIndex].groupedNodes.filter(id => id !== nodeId);
        const nodeWidth = 120; // Width of a node (from GraphCanvas)
        const nodeHeight = 80; // Height of a Convolutional Node (from GraphCanvas)
        const headerHeight = 40; // Height reserved for block header
        const spacing = 20; // Spacing between nodes and block edges/nodes

        // Calculate required width: (number of nodes * nodeWidth) + spacing between nodes + padding on sides
        const requiredWidth = updatedGroupedNodes.length > 0
          ? (updatedGroupedNodes.length * nodeWidth) + ((updatedGroupedNodes.length + 1) * spacing)
          : 200; // Minimum width if no nodes remain

        // Calculate required height: header + at least one row of nodes + padding (or minimum if empty)
        const requiredHeight = updatedGroupedNodes.length > 0
          ? headerHeight + nodeHeight + (spacing * 2)
          : 150; // Minimum height if no nodes remain

        newBlocks[targetBlockIndex] = {
          ...newBlocks[targetBlockIndex],
          groupedNodes: updatedGroupedNodes,
          width: requiredWidth, // Resize based on remaining nodes
          height: requiredHeight, // Resize based on remaining nodes
        };
        console.log(`Updated block ${blockId}: removed node ${nodeId}, resized to width=${requiredWidth}, height=${requiredHeight}`);

        // Immediately reposition remaining nodes based on the new block dimensions
        if (updatedGroupedNodes.length > 0) {
          setNodes(prevNodes => {
            return prevNodes.map(n => {
              if (updatedGroupedNodes.includes(n.id)) {
                const indexInGroup = updatedGroupedNodes.indexOf(n.id);
                const totalNodesWidth = (updatedGroupedNodes.length * nodeWidth) + ((updatedGroupedNodes.length - 1) * spacing);
                const startX = newBlocks[targetBlockIndex].x + (newBlocks[targetBlockIndex].width / 2) - (totalNodesWidth / 2); // Center the group of nodes
                const startY = newBlocks[targetBlockIndex].y + headerHeight + spacing; // Below header with padding

                // Position side by side based on index
                const x = startX + (indexInGroup * (nodeWidth + spacing));
                const y = startY;

                console.log(`Repositioned node ${n.id} to x=${x}, y=${y} in resized block ${blockId}`);
                return { ...n, x, y };
              }
              return n; // Explicitly do not reposition the ungrouped node
            });
          });
        }
        return newBlocks;
      });

      console.log(`Ungrouped node ${nodeId} from block ${blockId} and resized block.`);
    } else {
      console.log(`Node ${nodeId} not found in any block's groupedNodes.`);
    }
  };

  return (
    <div style={{ display: 'flex', width: '100%', height: '100%' }}>
      <div style={{ width: '300px', borderRight: '1px solid #ccc', padding: '15px', backgroundColor: '#f9f9f9', overflowY: 'auto' }}>
        <h2 style={{ marginTop: 0 }}>Graph Node Editor</h2>
        <div style={{ marginBottom: '20px' }}>
          <label style={{ display: 'block', marginBottom: '5px' }}>New Node Type:</label>
          <select
            value={newNodeType}
            onChange={(e) => setNewNodeType(e.target.value)}
            style={{ padding: '5px', width: '100%', marginBottom: '10px' }}
          >
            <option value="input">Input</option>
            <option value="output">Output</option>
            <option value="convolution">Convolution</option>
            <option value="normalization">Normalization</option>
            <option value="pooling">Pooling</option>
            <option value="flatten">Flatten</option>
            <option value="linear">Linear</option>
            <option value="activation">Activation</option>
            <option value="passthrough">Passthrough</option>
            <option value="Block">Block</option>
          </select>
          <button
            onClick={handleAddNode}
            style={{ padding: '8px 16px', backgroundColor: '#007bff', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', width: '100%' }}
          >
            Add New Node
          </button>
        </div>
        <div>
          <DndProvider backend={HTML5Backend}>
            <NodeList
              nodes={[...nodes, ...blocks]}
              selectedNode={selectedNode}
              onSelect={handleSelectNode}
              onUpdateNode={handleUpdateNode}
              moveNode={moveNodeInList}
            />
          </DndProvider>
        </div>
        {selectedNode && (
          <div style={{ marginTop: '20px', borderTop: '1px solid #ddd', paddingTop: '10px' }}>
            <h3 style={{ marginTop: 0 }}>Editing: {selectedNode.name}</h3>
            {selectedNode.type === 'Block' ? (
              <p>Grouped Nodes: {selectedNode.groupedNodes?.length || 0}</p>
            ) : selectedNode.layers !== null && selectedNode.layers.length > 0 ? (
              <div>
                {selectedNode.type === 'convolution' && selectedNode.layers.map(layer => (
                  <div key={layer.id} style={{ marginBottom: '10px', border: '1px solid #eee', padding: '5px', borderRadius: '4px' }}>
                    <h4>Convolution Layer: {layer.id.slice(-5)}</h4>
                    <label>Kernel Size:</label>
                    <input
                      type="number"
                      value={layer.params.kernelSize}
                      onChange={(e) => handleUpdateConvolutionParams(layer.id, parseInt(e.target.value), layer.params.outputChannels)}
                    />
                    <label>Output Channels:</label>
                    <input
                      type="number"
                      value={layer.params.outputChannels}
                      onChange={(e) => handleUpdateConvolutionParams(layer.id, layer.params.kernelSize, parseInt(e.target.value))}
                    />
                  </div>
                ))}

                {selectedNode.type === 'linear' && selectedNode.layers.map(layer => (
                  <div key={layer.id} style={{ marginBottom: '10px', border: '1px solid #eee', padding: '5px', borderRadius: '4px' }}>
                    <h4>Linear Layer: {layer.id.slice(-5)}</h4>
                    <label>Output Channels:</label>
                    <input
                      type="number"
                      value={layer.params.outputChannels}
                      onChange={(e) => handleUpdateLinearParams(layer.id, parseInt(e.target.value))}
                    />
                  </div>
                ))}

                {selectedNode.type === 'activation' && selectedNode.layers.map(layer => (
                  <div key={layer.id} style={{ marginBottom: '10px', border: '1px solid #eee', padding: '5px', borderRadius: '4px' }}>
                    <h4>Activation Layer: {layer.id.slice(-5)}</h4>
                    <label>Activation Type:</label>
                    <select
                      value={layer.params.activationType}
                      onChange={(e) => handleUpdateActivationParams(layer.id, e.target.value)}
                    >
                      <option value="ReLU">ReLU</option>
                      <option value="Linear">Linear</option>
                    </select>
                  </div>
                ))}

                {selectedNode.type === 'pooling' && selectedNode.layers.map(layer => (
                  <div key={layer.id} style={{ marginBottom: '10px', border: '1px solid #eee', padding: '5px', borderRadius: '4px' }}>
                    <h4>Pooling Layer: {layer.id.slice(-5)}</h4>
                    <label>Kernel Size:</label>
                    <input
                      type="number"
                      value={layer.params.kernelSize}
                      onChange={(e) => handleUpdatePoolingParams(layer.id, parseInt(e.target.value), layer.params.type)}
                    />
                    <label>Pooling Type:</label>
                    <select
                      value={layer.params.type}
                      onChange={(e) => handleUpdatePoolingParams(layer.id, layer.params.kernelSize, e.target.value)}
                    >
                      <option value="MaxPool">MaxPool</option>
                      <option value="AvgPool">AvgPool</option>
                    </select>
                  </div>
                ))}

                {selectedNode.type === 'normalization' && selectedNode.layers.map(layer => (
                  <div key={layer.id} style={{ marginBottom: '10px', border: '1px solid #eee', padding: '5px', borderRadius: '4px' }}>
                    <h4>Normalization Layer: {layer.id.slice(-5)}</h4>
                    <label>Normalization Type:</label>
                    <select
                      value={layer.params.normType}
                      onChange={(e) => handleUpdateNormalizationParams(layer.id, e.target.value)}
                    >
                      <option value="BatchNorm">BatchNorm</option>
                    </select>
                  </div>
                ))}
              </div>
            ) : (
              <p>No layers for this node type.</p>
            )}
            <p>Position: X: {Math.round(selectedNode.x)}, Y: {Math.round(selectedNode.y)}</p>
            <button onClick={handleDeleteNode} style={{ marginTop: '10px' }}>Delete Node</button>
          </div>
        )}
      </div>
      <div style={{ flex: 1, overflow: 'hidden' }}>
        <GraphCanvas
          nodes={nodes}
          blocks={blocks}
          selectedNode={selectedNode}
          onSelect={handleSelectNode}
          onMoveNode={moveNodeOnCanvas}
          onMoveBlock={moveBlockOnCanvas}
          onGroupNode={groupNodeIntoBlock}
          onUngroupNode={handleUngroupNode}
        />
      </div>
    </div>
  );
};

export default GraphEditor;
