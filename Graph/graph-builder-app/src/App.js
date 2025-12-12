import React from 'react';
import GraphEditor from './components/GraphEditor';
import './App.css';

function App() {
  console.log('App component rendered');
  return (
    <div className="App">
      <header className="App-header">
        <h1>Neural Network Graph Builder</h1>
        <p>Design your neural network by dragging nodes on the canvas.</p>
      </header>
      <main>
        <GraphEditor />
      </main>
    </div>
  );
}

export default App;
