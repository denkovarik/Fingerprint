import React, { useState, useRef } from 'react';
import { useDrag, useDrop } from 'react-dnd';

const ItemType = 'NODE';

const NodeItem = ({ node, index, selectedNode, onSelect, moveNode }) => {
  console.log(`Rendering node: ID=${node.id}, Type=${node.type}`);
  const ref = useRef(null);

  const [{ handlerId }, drop] = useDrop({
    accept: ItemType,
    collect(monitor) {
      return {
        handlerId: monitor.getHandlerId(),
      };
    },
    hover(item, monitor) {
      if (!ref.current) return;
      const dragIndex = item.index;
      const hoverIndex = index;

      if (dragIndex === hoverIndex) return;

      const hoverBoundingRect = ref.current?.getBoundingClientRect();
      const hoverMiddleY = (hoverBoundingRect.bottom - hoverBoundingRect.top) / 2;
      const clientOffset = monitor.getClientOffset();
      const hoverClientY = clientOffset.y - hoverBoundingRect.top;

      if (dragIndex < hoverIndex && hoverClientY < hoverMiddleY) return;
      if (dragIndex > hoverIndex && hoverClientY > hoverMiddleY) return;

      moveNode(dragIndex, hoverIndex);
      item.index = hoverIndex;
    },
  });

  const [{ isDragging }, drag] = useDrag({
    type: ItemType,
    item: { id: node.id, index },
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
    }),
  });

  drag(drop(ref));

  return (
    <li
      ref={ref}
      data-handler-id={handlerId}
      style={{
        padding: '10px',
        borderBottom: '1px solid #eee',
        backgroundColor: selectedNode && selectedNode.id === node.id ? '#e6f3ff' : 'transparent',
        cursor: 'move',
        opacity: isDragging ? 0.5 : 1,
        userSelect: 'none'
      }}
      onClick={() => onSelect(node.id)}
    >
      {node.name} ({node.type})
    </li>
  );
};

const NodeList = ({ nodes, selectedNode, onSelect, onUpdateNode, moveNode }) => {
  // Removed state related to layer addition as it's no longer needed
  return (
    <div>
      <h3>Node List</h3>
      {nodes.length === 0 ? (
        <p>No nodes defined. Click "Add New Node" to start.</p>
      ) : (
        <ul style={{ listStyle: 'none', padding: 0, border: '1px solid #ddd', borderRadius: '4px', maxHeight: '300px', overflowY: 'auto' }}>
          {nodes.map((node, index) => (
            <NodeItem
              key={node.id}
              node={node}
              index={index}
              selectedNode={selectedNode}
              onSelect={onSelect}
              moveNode={moveNode}
            />
          ))}
        </ul>
      )}
    </div>
  );
};

export default NodeList;
