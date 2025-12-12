import React, { useState, useMemo, useCallback } from 'react';
import { Stage, Layer, Rect, Text, Group, Line } from 'react-konva';
import { throttle } from 'lodash';

const getNodeColor = (nodeType) => {
  switch (nodeType) {
    case 'input': return '#17a2b8';
    case 'output': return '#dc3545';
    case 'convolution': return '#28a745';
    case 'linear': return '#ffc107';
    case 'activation': return '#ff4500';
    case 'pooling': return '#20b2aa';
    case 'normalization': return '#800080';
    case 'flatten': return '#a0522d';
    case 'passthrough': return '#808080';
    case 'Block': return '#007bff';
    default: return '#6c757d';
  }
};

const getNodeLabel = (node) => {
  const type = node.type?.toLowerCase();
  const params = node.params || (node.layers?.[0]?.params ?? {});

  switch (type) {
    case 'convolution':
      return {
        title: 'Conv2d',
        paramLines: [
          `out_channels = ${params.outputChannels}`,
          `kernel_size  = ${params.kernelSize}x${params.kernelSize}`
        ]
      };
    case 'linear':
      return {
        title: 'Linear',
        paramLines: [
          `out_channels = ${params.outputChannels}`
        ]
      };
    case 'activation':
      return {
        title: 'Activation',
        paramLines: [
          `type = ${params.activationType}`
        ]
      };
    case 'pooling':
      return {
        title: `${params.type} Pool`,
        paramLines: []
      };
    case 'normalization':
      return {
        title: 'Normalization',
        paramLines: [
          `type = ${params.normType}`
        ]
      };
    case 'flatten':
      return {
        title: 'Flatten',
        paramLines: []
      };
    case 'passthrough':
      return {
        title: 'Passthrough',
        paramLines: []
      };
    case 'input':
      return {
        title: 'Input',
        paramLines: []
      };
    case 'output':
      return {
        title: 'Output',
        paramLines: []
      };
    default:
      return {
        title: node.name || 'Node',
        paramLines: []
      };
  }
};

const getStrokeWidth = (isSelected) => isSelected ? 3 : 1;
const getStrokeColor = (isSelected) => isSelected ? '#007bff' : '#ccc';
const getNodeHeight = (nodeType) => (
  ['convolution', 'linear', 'activation', 'pooling', 'normalization'].includes(nodeType) ? 80 : 60
);

const NodeGroup = React.memo(({ node, isSelected, relativeX = 0, relativeY = 0, onDragStart, onDragMove, onDragEnd, onClick }) => {
  const { title, paramLines } = getNodeLabel(node);

  return (
    <Group
      x={node.x - relativeX}
      y={node.y - relativeY}
      draggable
      onDragStart={(e) => {
        onDragStart(e, node.id);
        e.cancelBubble = true;
      }}
      onDragMove={(e) => {
        onDragMove(e, node.id);
        e.cancelBubble = true;
      }}
      onDragEnd={(e) => {
        onDragEnd(e, node.id);
        e.cancelBubble = true;
      }}
      onClick={(e) => {
        onClick(e, node.id);
        e.cancelBubble = true;
      }}
      onMouseOver={(e) => {
        e.target.getStage().container().style.cursor = 'move';
      }}
      onMouseOut={(e) => {
        e.target.getStage().container().style.cursor = 'default';
      }}
    >
      <Rect
        width={120}
        height={getNodeHeight(node.type)}
        fill={getNodeColor(node.type)}
        stroke={getStrokeColor(isSelected)}
        strokeWidth={getStrokeWidth(isSelected)}
        cornerRadius={5}
        shadowBlur={0}
        listening={true}
      />

      {/* Bold Title */}
      <Text
        x={10}
        y={10}
        text={title}
        fontSize={12}
        fontStyle="bold"
        fill="white"
        listening={false}
      />

      {/* Solid Divider Line */}
      <Line
        points={[10, 25, 110, 25]}
        stroke="white"
        strokeWidth={1}
        listening={false}
      />

      {/* Param Lines */}
      <Text
        x={10}
        y={30}
        text={paramLines.join('\n')}
        fontSize={12}
        fill="white"
        lineHeight={1.2}
        listening={false}
      />
    </Group>
  );
});

const GraphCanvas = ({ nodes = [], blocks = [], selectedNode, onSelect, onMoveNode, onMoveBlock, onGroupNode, onUngroupNode }) => {
  const [draggedNode, setDraggedNode] = useState(null);
  const [tempPosition, setTempPosition] = useState({ x: 0, y: 0 });
  const [startPosition, setStartPosition] = useState({ x: 0, y: 0 }); // Store the original position for outline

  const nodeToBlockMap = useMemo(() => {
    const map = new Map();
    blocks.forEach(block => {
      (block.groupedNodes || []).forEach(id => map.set(id, block.id));
    });
    return map;
  }, [blocks]);

  const groupedNodesByBlock = useMemo(() => {
    const grouped = new Map();
    nodes.forEach(node => {
      const blockId = nodeToBlockMap.get(node.id);
      if (blockId) {
        if (!grouped.has(blockId)) grouped.set(blockId, []);
        grouped.get(blockId).push(node);
      }
    });
    return grouped;
  }, [nodes, nodeToBlockMap]);

  const ungroupedNodes = useMemo(() => nodes.filter(n => !nodeToBlockMap.has(n.id)), [nodes, nodeToBlockMap]);

  const handleDragStart = useCallback((e, id) => {
    e.target.opacity(0.7);
    e.target.moveToTop();
    const node = nodes.find(n => n.id === id) || blocks.find(b => b.id === id);
    if (node) {
      setDraggedNode({ id, type: node.type || 'Block' });
      setTempPosition({ x: node.x, y: node.y });
      setStartPosition({ x: node.x, y: node.y }); // Capture starting position for outline
      console.log(`Dragging ${node.type || 'Block'} id=${id} from x=${node.x}, y=${node.y}`);
    }
  }, [nodes, blocks]);

  // Define the throttled function using useMemo
  const throttledDragMove = useMemo(() => {
    return throttle((e, id) => {
      if (!draggedNode || draggedNode.id !== id) return;
      let newX = e.target.x();
      let newY = e.target.y();
      const node = nodes.find(n => n.id === id);
      if (node) {
        const parentBlock = blocks.find(b => b.groupedNodes?.includes(id));
        if (parentBlock) {
          newX += parentBlock.x;
          newY += parentBlock.y;
          console.log(`Adjusting grouped node position: block x=${parentBlock.x}, y=${parentBlock.y}, adjusted to x=${newX}, y=${newY}`);
        }
      }
      setTempPosition({ x: newX, y: newY });
      console.log(`Dragging to x=${newX}, y=${newY}`);
    }, 16);
  }, [draggedNode, nodes, blocks]);

  // UseCallback just wraps the memoized throttle
  const handleDragMove = useCallback((e, id) => {
    throttledDragMove(e, id);
  }, [throttledDragMove]);

  const handleDragEndNode = useCallback((e, id) => {
    e.target.opacity(1);
    let grouped = false;
    console.log(`Dropped node ${id} at x=${tempPosition.x}, y=${tempPosition.y}`);
    for (let i = 0; i < blocks.length; i++) {
      const b = blocks[i];
      const nodeCenterX = tempPosition.x + 60; // Node width=120, center at half
      const nodeCenterY = tempPosition.y + 40; // Node height~80, center at half
      if (
        nodeCenterX > b.x &&
        nodeCenterX < b.x + (b.width || 200) &&
        nodeCenterY > b.y &&
        nodeCenterY < b.y + (b.height || 150)
      ) {
        console.log(`Grouping node ${id} into block ${b.id}`);
        onGroupNode(id, b.id);
        grouped = true;
        break;
      }
    }
    if (!grouped) {
      console.log(`Node ${id} not grouped, setting position x=${tempPosition.x}, y=${tempPosition.y}`);
      onMoveNode(id, tempPosition.x, tempPosition.y);
      const parent = blocks.find(b => b.groupedNodes?.includes(id));
      if (parent) {
        const blockBounds = { left: parent.x, right: parent.x + (parent.width || 200), top: parent.y, bottom: parent.y + (parent.height || 150) };
        const nodeCenterX = tempPosition.x + 60; // Node width=120, center at half
        const nodeCenterY = tempPosition.y + 40; // Node height~80, center at half
        // Ungroup if the node's center is outside the block's bounds
        if (
          nodeCenterX < blockBounds.left ||
          nodeCenterX > blockBounds.right ||
          nodeCenterY < blockBounds.top ||
          nodeCenterY > blockBounds.bottom
        ) {
          console.log(`Ungrouping node ${id} from block ${parent.id} at position x=${tempPosition.x}, y=${tempPosition.y} (center outside block)`);
          onUngroupNode(id);
        } else {
          console.log(`Node ${id} not ungrouped, center still inside block ${parent.id} at x=${nodeCenterX}, y=${nodeCenterY}`);
        }
      }
    }
    setDraggedNode(null);
  }, [tempPosition, onMoveNode, onGroupNode, onUngroupNode, blocks]);

  const handleDragEndBlock = useCallback((e, id) => {
    e.target.opacity(1);
    console.log(`Dropped block ${id} at x=${tempPosition.x}, y=${tempPosition.y}`);
    onMoveBlock(id, tempPosition.x, tempPosition.y);
    setDraggedNode(null);
  }, [tempPosition, onMoveBlock]);

  const handleClick = useCallback((e, id) => {
    e.cancelBubble = true;
    setTimeout(() => onSelect(id), 100);
  }, [onSelect]);

  return (
    <div style={{ width: '100%', height: '100%', backgroundColor: '#eee', position: 'relative', pointerEvents: 'auto' }}>
      <Stage
        width={Math.max(800, window.innerWidth - 300)}
        height={Math.max(600, window.innerHeight - 100)}
        style={{ backgroundColor: '#fff', cursor: 'default' }}
        listening={true}
      >
        <Layer listening={true}>
          {blocks.map(block => {
            const groupedNodes = groupedNodesByBlock.get(block.id) || [];
            console.log(`Rendering block ${block.id} with ${groupedNodes.length} grouped nodes`);
            return (
              <Group
                key={block.id}
                x={block.x}
                y={block.y}
                draggable
                listening={true}
                onDragStart={(e) => handleDragStart(e, block.id)}
                onDragMove={(e) => handleDragMove(e, block.id)}
                onDragEnd={(e) => handleDragEndBlock(e, block.id)}
                onClick={(e) => handleClick(e, block.id)}
                onMouseOver={(e) => { e.target.getStage().container().style.cursor = 'move'; }}
                onMouseOut={(e) => { e.target.getStage().container().style.cursor = 'default'; }}
              >
                <Rect
                  width={block.width || 200}
                  height={block.height || 150}
                  fill="rgba(0, 123, 255, 0.1)"
                  stroke={getStrokeColor(selectedNode?.id === block.id)}
                  strokeWidth={getStrokeWidth(selectedNode?.id === block.id)}
                  dash={[5, 5]}
                  cornerRadius={5}
                  listening={true}
                />
                <Text
                  x={10}
                  y={10}
                  text={block.name}
                  fontSize={14}
                  fill="#007bff"
                  listening={true}
                />
                {groupedNodes.map(node => (
                  <NodeGroup
                    key={node.id}
                    node={node}
                    isSelected={selectedNode?.id === node.id}
                    relativeX={block.x}
                    relativeY={block.y}
                    onDragStart={handleDragStart}
                    onDragMove={handleDragMove}
                    onDragEnd={handleDragEndNode}
                    onClick={handleClick}
                  />
                ))}
              </Group>
            );
          })}
          {ungroupedNodes.map(node => (
            <NodeGroup
              key={node.id}
              node={node}
              isSelected={selectedNode?.id === node.id}
              onDragStart={handleDragStart}
              onDragMove={handleDragMove}
              onDragEnd={handleDragEndNode}
              onClick={handleClick}
            />
          ))}
          {draggedNode && (
            <Group x={startPosition.x} y={startPosition.y} listening={false}>
              <Rect
                width={draggedNode.type === 'Block' ? 200 : 120}
                height={draggedNode.type === 'Block' ? 150 : getNodeHeight(draggedNode.type)}
                stroke={getNodeColor(draggedNode.type)}
                strokeWidth={1}
                opacity={0.5}
                cornerRadius={5}
              />
            </Group>
          )}
        </Layer>
      </Stage>
    </div>
  );
};

export default React.memo(GraphCanvas);
